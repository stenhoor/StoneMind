/**
 * Turns raw network outputs into the evaluation KataGo reports (`kata-raw-nn`, analysis `rootInfo.raw*`), from White's
 * perspective. Follows KataGo's `NNEvaluator` post-processing in `cpp/neuralnet/nneval.cpp` and the output extraction
 * in `cpp/neuralnet/eigenbackend.cpp` (KataGo 1.18, MIT, © David J Wu), for model versions 8 and up.
 */
import type { ModelDesc } from '../model/types.js';
import type { RawOutputs } from './forward.js';

export type Player = 'B' | 'W';

export interface PostProcessOptions {
  /** Who moves next; outputs are relative to them and are flipped to White's view. */
  nextPlayer: Player;
  /** Legal moves: one flag per point (row-major) plus pass last. */
  legal: readonly boolean[];
  /** KataGo zeroes the no-result probability unless ko is simple or scoring is territory. */
  koRuleSimple: boolean;
  scoringTerritory: boolean;
  /** Blend toward the optimistic policy channel (0 = plain policy). */
  policyOptimism?: number;
  policyTemperature?: number;
  /** KataGo's `enablePassingHacks`: caps the pass probability at 19× the sum of the others. */
  enablePassingHacks?: boolean;
}

/** An evaluation, from White's perspective, as KataGo's NNOutput holds it. */
export interface Evaluation {
  whiteWinProb: number;
  whiteLossProb: number;
  whiteNoResultProb: number;
  whiteScoreMean: number;
  whiteScoreMeanSq: number;
  whiteLead: number;
  varTimeLeft: number;
  shorttermWinlossError: number;
  shorttermScoreError: number;
  /** Probabilities, row-major plus pass last; -1 marks an illegal move. */
  policy: Float64Array;
  /** White's ownership in [-1, 1], row-major. */
  whiteOwnership: Float64Array;
}

const softPlus = (x: number) => (x > 40 ? x : Math.log(1 + Math.exp(x)));

export function postProcess(model: ModelDesc, raw: RawOutputs, options: PostProcessOptions): Evaluation {
  const area = raw.xSize * raw.ySize;
  const params = model.postProcessParams;
  const scale = params.outputScaleMultiplier;
  const optimism = options.policyOptimism ?? 0;
  const temperature = options.policyTemperature ?? 1;
  if (options.legal.length !== area + 1) throw new Error('legal must have one flag per point plus pass');

  // Policy: blend the optimistic channel where the model has one, mask illegal moves, softmax.
  const logits = new Float64Array(area + 1);
  const p0 = raw.policy[0]!;
  // KataGo's backends: the optimistic channel is the second of 2, or of 4 from model version 16.
  const hasOptimism = model.numPolicyChannels === 2 || (model.numPolicyChannels === 4 && model.modelVersion >= 16);
  const p1 = hasOptimism ? raw.policy[1]! : null;
  for (let i = 0; i < area; i++) logits[i] = p1 ? p0[i]! + (p1[i]! - p0[i]!) * optimism : p0[i]!;
  logits[area] = p1 ? raw.pass[0]! + (raw.pass[1]! - raw.pass[0]!) * optimism : raw.pass[0]!;

  const policy = new Float64Array(area + 1);
  const scaling = scale / temperature;
  let maxLogit = -1e25;
  let legalCount = 0;
  for (let i = 0; i <= area; i++) {
    const v = options.legal[i] ? (legalCount++, Math.fround(logits[i]! * scaling)) : -1e30;
    policy[i] = v;
    if (v > maxLogit) maxLogit = v;
  }
  if (legalCount === 0) throw new Error('no legal moves');
  let sum = 0;
  if (options.enablePassingHacks) {
    for (let i = 0; i < area; i++) sum += policy[i] = Math.exp(policy[i]! - maxLogit);
    policy[area] = Math.max(1e-20, Math.min(Math.exp(policy[area]! - maxLogit), sum * 19));
    sum += policy[area]!;
  } else {
    for (let i = 0; i <= area; i++) sum += policy[i] = Math.exp(policy[i]! - maxLogit);
  }
  for (let i = 0; i <= area; i++) policy[i] = options.legal[i] ? (sum > 0 ? policy[i]! / sum : 1 / legalCount) : -1;

  // Value and score.
  const [winLogit, lossLogit, noResultLogitRaw] = [raw.value[0]! * scale, raw.value[1]! * scale, raw.value[2]! * scale];
  const noResultZeroed = !options.koRuleSimple && !options.scoringTerritory;
  const noResultLogit = noResultZeroed ? noResultLogitRaw - 100000 : noResultLogitRaw;
  const maxValue = Math.max(winLogit, lossLogit, noResultLogit);
  let win = Math.exp(winLogit - maxValue);
  let loss = Math.exp(lossLogit - maxValue);
  let noResult = noResultZeroed ? 0 : Math.exp(noResultLogit - maxValue);
  const total = win + loss + noResult;
  win /= total;
  loss /= total;
  noResult /= total;

  const sv = raw.scoreValue;
  let scoreMean = sv[0]! * scale * params.scoreMeanMultiplier;
  const scoreStdev = softPlus(sv[1]! * scale) * params.scoreStdevMultiplier;
  let scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
  let lead = sv[2]! * scale * params.leadMultiplier;
  const varTimeLeftValue = softPlus(sv[3]! * scale) * params.varianceTimeMultiplier;
  scoreMean *= 1 - noResult;
  scoreMeanSq *= 1 - noResult;
  lead *= 1 - noResult;

  let shorttermWinlossError = -1;
  let shorttermScoreError = -1;
  let varTimeLeft = -1;
  if (model.modelVersion >= 9) {
    varTimeLeft = varTimeLeftValue;
    if (model.modelVersion >= 14) {
      const a = softPlus(sv[4]! * scale * 0.5);
      shorttermWinlossError = Math.sqrt(a * a * params.shorttermValueErrorMultiplier);
      const b = softPlus(sv[5]! * scale * 0.5);
      shorttermScoreError = Math.sqrt(b * b * params.shorttermScoreErrorMultiplier);
    } else if (model.modelVersion >= 10) {
      shorttermWinlossError = Math.sqrt(softPlus(sv[4]! * scale) * params.shorttermValueErrorMultiplier);
      shorttermScoreError = Math.sqrt(softPlus(sv[5]! * scale) * params.shorttermScoreErrorMultiplier);
    } else {
      shorttermWinlossError = softPlus(sv[4]! * scale);
      shorttermScoreError = softPlus(sv[5]! * scale) * 10;
    }
  }

  const white = options.nextPlayer === 'W';
  const whiteOwnership = new Float64Array(area);
  for (let i = 0; i < area; i++) {
    const t = Math.tanh(raw.ownership[i]! * scale);
    whiteOwnership[i] = white ? t : -t;
  }

  return {
    whiteWinProb: white ? win : loss,
    whiteLossProb: white ? loss : win,
    whiteNoResultProb: noResult,
    whiteScoreMean: white ? scoreMean : -scoreMean,
    whiteScoreMeanSq: scoreMeanSq,
    whiteLead: white ? lead : -lead,
    varTimeLeft,
    shorttermWinlossError,
    shorttermScoreError,
    policy,
    whiteOwnership,
  };
}
