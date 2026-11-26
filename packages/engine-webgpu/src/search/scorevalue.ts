/**
 * KataGo's score utility: the expected value of a smoothed score function under a normal distribution of the final
 * score, from a precomputed table. Ported from `ScoreValue` in `cpp/neuralnet/nninputs.cpp` (KataGo 1.18, MIT,
 * © David J Wu), with KataGo's default maximum board size of 19.
 */

const TWO_OVER_PI = 0.63661977236758134308;
/** KataGo `NNPos::MAX_BOARD_LEN` in a default build. */
const ASSUMED_BOARD_SIZE = 19;
/** KataGo `NNPos::EXTRA_SCORE_DISTR_RADIUS`. */
const EXTRA_SCORE_DISTR_RADIUS = 60;
const MEAN_RADIUS = ASSUMED_BOARD_SIZE * ASSUMED_BOARD_SIZE + EXTRA_SCORE_DISTR_RADIUS;
const MEAN_LEN = MEAN_RADIUS * 2;
const STDEV_LEN = ASSUMED_BOARD_SIZE * ASSUMED_BOARD_SIZE + EXTRA_SCORE_DISTR_RADIUS;

/** KataGo `whiteScoreValueOfScoreSmoothNoDrawAdjust`. */
export function whiteScoreValueOfScoreSmooth(score: number, center: number, scale: number, sqrtBoardArea: number): number {
  return Math.atan((score - center) / (scale * sqrtBoardArea)) * TWO_OVER_PI;
}

/** KataGo `whiteDScoreValueDScoreSmoothNoDrawAdjust`. */
export function whiteDScoreValueDScoreSmooth(score: number, center: number, scale: number, sqrtBoardArea: number): number {
  const adjusted = score - center;
  const scaleFactor = scale * sqrtBoardArea;
  return (scaleFactor / (scaleFactor * scaleFactor + adjusted * adjusted)) * TWO_OVER_PI;
}

let table: Float64Array | null = null;

/** KataGo `ScoreValue::initTables`: numeric integration of the score value against a normal distribution. */
function buildTable(): Float64Array {
  const stepsPerUnit = 10;
  const boundStdevs = 5;
  const minStdevSteps = -boundStdevs * stepsPerUnit;
  const maxStdevSteps = boundStdevs * stepsPerUnit;
  const normalPdf = new Float64Array(maxStdevSteps - minStdevSteps + 1);
  for (let i = minStdevSteps; i <= maxStdevSteps; i++) {
    const x = i / stepsPerUnit;
    normalPdf[i - minStdevSteps] = Math.exp(-0.5 * x * x);
  }
  const minSvSteps = -(MEAN_RADIUS * stepsPerUnit + stepsPerUnit / 2 + boundStdevs * STDEV_LEN * stepsPerUnit);
  const maxSvSteps = -minSvSteps;
  const svPrecomp = new Float64Array(maxSvSteps - minSvSteps + 1);
  for (let i = minSvSteps; i <= maxSvSteps; i++) {
    svPrecomp[i - minSvSteps] = whiteScoreValueOfScoreSmooth(i / stepsPerUnit, 0, 1, ASSUMED_BOARD_SIZE);
  }
  const out = new Float64Array(MEAN_LEN * STDEV_LEN);
  for (let meanIdx = 0; meanIdx < MEAN_LEN; meanIdx++) {
    const meanSteps = (meanIdx - MEAN_RADIUS) * stepsPerUnit - stepsPerUnit / 2;
    for (let stdevIdx = 0; stdevIdx < STDEV_LEN; stdevIdx++) {
      let wSum = 0;
      let wsvSum = 0;
      for (let i = minStdevSteps; i <= maxStdevSteps; i++) {
        const w = normalPdf[i - minStdevSteps]!;
        wSum += w;
        wsvSum += w * svPrecomp[meanSteps + stdevIdx * i - minSvSteps]!;
      }
      out[meanIdx * STDEV_LEN + stdevIdx] = wsvSum / wSum;
    }
  }
  return out;
}

/** KataGo `ScoreValue::expectedWhiteScoreValue`. The table is built on first use. */
export function expectedWhiteScoreValue(scoreMean: number, scoreStdev: number, center: number, scale: number, sqrtBoardArea: number): number {
  table ??= buildTable();
  const scaleFactor = ASSUMED_BOARD_SIZE / (scale * sqrtBoardArea);
  const meanScaled = (scoreMean - center) * scaleFactor;
  const stdevScaled = scoreStdev * scaleFactor;
  // C's round() rounds halves away from zero.
  const meanRounded = Math.sign(meanScaled) * Math.round(Math.abs(meanScaled));
  const stdevFloored = Math.floor(stdevScaled);
  let meanIdx0 = meanRounded + MEAN_RADIUS;
  let stdevIdx0 = stdevFloored;
  let meanIdx1 = meanIdx0 + 1;
  let stdevIdx1 = stdevIdx0 + 1;
  if (meanIdx0 < 0) {
    meanIdx0 = 0;
    meanIdx1 = 0;
  }
  if (meanIdx1 >= MEAN_LEN) {
    meanIdx0 = MEAN_LEN - 1;
    meanIdx1 = MEAN_LEN - 1;
  }
  if (stdevIdx1 >= STDEV_LEN) {
    stdevIdx0 = STDEV_LEN - 1;
    stdevIdx1 = STDEV_LEN - 1;
  }
  const lambdaMean = meanScaled - meanRounded + 0.5;
  const lambdaStdev = stdevScaled - stdevFloored;
  const a00 = table[meanIdx0 * STDEV_LEN + stdevIdx0]!;
  const a01 = table[meanIdx0 * STDEV_LEN + stdevIdx1]!;
  const a10 = table[meanIdx1 * STDEV_LEN + stdevIdx0]!;
  const a11 = table[meanIdx1 * STDEV_LEN + stdevIdx1]!;
  const b0 = a00 + lambdaStdev * (a01 - a00);
  const b1 = a10 + lambdaStdev * (a11 - a10);
  return b0 + lambdaMean * (b1 - b0);
}

/** KataGo `ScoreValue::getScoreStdev`. */
export function getScoreStdev(scoreMean: number, scoreMeanSq: number): number {
  const variance = scoreMeanSq - scoreMean * scoreMean;
  return variance <= 0 ? 0 : Math.sqrt(variance);
}

/** KataGo `ScoreValue::whiteScoreMeanSqOfScoreGridded`, for a final score on a half-point grid. */
export function whiteScoreMeanSqOfScoreGridded(finalWhiteMinusBlackScore: number, drawEquivalentWinsForWhite: number): number {
  if (!Number.isInteger(finalWhiteMinusBlackScore)) return finalWhiteMinusBlackScore * finalWhiteMinusBlackScore;
  const lower = finalWhiteMinusBlackScore - 0.5;
  const upper = finalWhiteMinusBlackScore + 0.5;
  return lower * lower + (upper * upper - lower * lower) * drawEquivalentWinsForWhite;
}
