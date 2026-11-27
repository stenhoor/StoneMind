/**
 * The query fields of KataGo's analysis engine that change the search's settings, ported from KataGo v1.18.2:
 * `avoidMoves` and `allowMoves` (`cpp/command/analysis.cpp`), and `overrideSettings`, which reloads the search
 * parameters from the config with the overrides applied (`Setup::loadSingleParams` for the analysis engine,
 * `cpp/program/setup.cpp`, with `ConfigParser`'s value parsing).
 */
import { BLACK, WHITE, type Stone } from '../rules/board.js';
import { DEFAULT_SEARCH_PARAMS, type SearchParams } from './search.js';

/** One entry of `avoidMoves` or `allowMoves`: a player, board locations (`PASS_LOC` allowed), and a depth. */
export interface AvoidMovesEntry {
  player: Stone;
  moves: readonly number[];
  untilDepth: number;
}

/**
 * The avoid lists `Search.setAvoidMoveUntilByLoc` takes, built as KataGo's analysis engine builds them from
 * `avoidMoves` (`allow` false: each listed move is avoided until `untilDepth`) or `allowMoves` (`allow` true: every move
 * but the listed ones is avoided until `untilDepth`). A later `avoidMoves` entry for the same move wins. `arraySize` is
 * the board's `arraySize`.
 * @throws Error, with KataGo's message, for `allowMoves` with two entries for one player.
 */
export function avoidMoveUntilByLoc(
  arraySize: number,
  entries: readonly AvoidMovesEntry[],
  allow: boolean,
): { black: Int32Array | null; white: Int32Array | null } {
  const byPla: Record<Stone, Int32Array | null> = { [BLACK]: null, [WHITE]: null } as Record<Stone, Int32Array | null>;
  const gotAllowMoves = { [BLACK]: false, [WHITE]: false } as Record<Stone, boolean>;
  for (const { player, moves, untilDepth } of entries) {
    if (allow) {
      if (gotAllowMoves[player]) throw new Error('Cannot specify allowMoves more than once for the same player');
      gotAllowMoves[player] = true;
    }
    const v = (byPla[player] ??= new Int32Array(arraySize));
    if (allow) {
      v.fill(untilDepth);
      for (const loc of moves) v[loc] = 0;
    } else {
      for (const loc of moves) v[loc] = untilDepth;
    }
  }
  return { black: byPla[BLACK], white: byPla[WHITE] };
}

/** A config value KataGo's `ConfigParser` can't parse, or one outside its range. */
export class ConfigValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigValueError';
  }
}

/** A config key KataGo's search reads that this engine doesn't support (the search feature isn't ported). */
export class UnsupportedSettingError extends Error {
  constructor(readonly keys: readonly string[]) {
    super(`not supported by this engine: ${keys.join(', ')}`);
    this.name = 'UnsupportedSettingError';
  }
}

/**
 * The keys KataGo's `loadSingleParams` reads whose search features aren't ported. `subtreeValueBiasFreeProp` is listed
 * although a single search never uses it (KataGo reads it only when it frees part of a tree), as no one has checked it.
 */
const UNPORTED_KEYS = new Set([
  'maxPlayouts', 'maxTime', 'maxPlayoutsPondering', 'maxVisitsPondering', 'maxTimePondering', 'lagBuffer',
  'searchFactorAfterOnePass', 'searchFactorAfterTwoPass', 'minPlayoutsPerThread',
  'graphSearchCatchUpLeakProb', 'rootNoiseEnabled', 'rootDirichletNoiseTotalConcentration', 'rootDirichletNoiseWeight',
  'rootDesiredPerChildVisitsCoeff', 'chosenMoveTemperature', 'chosenMoveTemperatureEarly',
  'chosenMoveTemperatureOnlyBelowProb', 'fillDameBeforePass', 'alwaysComputePassAliveUnderSuicideRules',
  'excludeTerritoryAdjacentToAtari', 'playoutDoublingAdvantage', 'playoutDoublingAdvantagePla',
  'avoidRepeatedPatternUtility', 'nnPolicyTemperature', 'antiMirror', 'ignoreAllHistory', 'subtreeValueBiasFreeProp',
  'useEvalCache', 'evalCacheMinVisits', 'nodeTableShardsPowerOfTwo', 'treeReuseCarryOverTimeFactor',
  'overallocateTimeFactor', 'midgameTimeFactor', 'midgameTurnPeakTime', 'endgameTurnTimeDecay', 'obviousMovesTimeFactor',
  'obviousMovesPolicyEntropyTolerance', 'obviousMovesPolicySurpriseTolerance', 'futileVisitsThreshold', 'humanSLProfile',
  'humanSLCpuctExploration', 'humanSLCpuctPermanent', 'humanSLRootExploreProbWeightless',
  'humanSLRootExploreProbWeightful', 'humanSLPlaExploreProbWeightless', 'humanSLPlaExploreProbWeightful',
  'humanSLOppExploreProbWeightless', 'humanSLOppExploreProbWeightful', 'humanSLChosenMoveProp',
  'humanSLChosenMoveIgnorePass', 'humanSLChosenMovePiklLambda', 'numBots',
]);

// KataGo Global::tryStringToDouble and tryStringToInt read with `istream >>` after trimming; these accept what that
// reads in full.
const DOUBLE_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
const INT_RE = /^[+-]?\d+$/;

/** KataGo `ConfigParser`'s value reading, for the keys of a config held as text. */
export class Config {
  readonly #values: ReadonlyMap<string, string>;
  readonly used = new Set<string>();
  constructor(values: ReadonlyMap<string, string>) {
    this.#values = values;
  }
  contains(key: string): boolean {
    return this.#values.has(key);
  }
  #string(key: string): string {
    this.used.add(key);
    const value = this.#values.get(key);
    if (value === undefined) throw new ConfigValueError(`Could not find key '${key}'`);
    return value;
  }
  /** The value as text. */
  getString(key: string): string {
    return this.#string(key);
  }
  getBool(key: string): boolean {
    const value = this.#string(key);
    const s = value.trim().toLowerCase();
    if (s === 'false') return false;
    if (s === 'true') return true;
    throw new ConfigValueError(`Could not parse '${value}' as bool for key '${key}'`);
  }
  getDouble(key: string, min: number, max: number): number {
    const value = this.#string(key);
    if (!DOUBLE_RE.test(value.trim())) throw new ConfigValueError(`Could not parse '${value}' as double for key '${key}'`);
    const x = Number(value.trim());
    if (x < min || x > max) throw new ConfigValueError(`Key '${key}' must be in the range ${min} to ${max}`);
    return x;
  }
  getInt(key: string, min: number, max: number, what = 'int'): number {
    const value = this.#string(key);
    const x = INT_RE.test(value.trim()) ? Number(value.trim()) : NaN;
    if (!Number.isSafeInteger(x) || (what === 'int' && (x < -(2 ** 31) || x >= 2 ** 31))) {
      throw new ConfigValueError(`Could not parse '${value}' as ${what} for key '${key}'`);
    }
    if (x < min || x > max) throw new ConfigValueError(`Key '${key}' must be in the range ${min} to ${max}`);
    return x;
  }
}

/** The search parameters a config gives, and what else the loader found in it. */
export interface ConfigSearchSettings {
  params: SearchParams;
  /** `maxVisits`, if the config sets it. */
  maxVisits?: number;
  /** Keys KataGo's search parameter loader doesn't read (KataGo warns about these in `overrideSettings`). */
  unusedKeys: string[];
}

/**
 * KataGo `Setup::loadSingleParams` for the analysis engine (and its `conservativePass` default), for the parameters
 * this engine's search has: the search parameters a config (key to value, as strings) gives. With an empty config this
 * is `DEFAULT_SEARCH_PARAMS`, the analysis engine's defaults. `base` supplies the parameters KataGo's config doesn't have
 * (`seed`, `symmetry`, `analysisPVLen`), and `batchSize` unless the config sets `numSearchThreads`. A
 * `rootNumSymmetriesToSample` of 1 is accepted, as that is what this search does.
 *
 * The analysis engine handles `overrideSettings` by loading the parameters again from its config with the overrides
 * applied: pass the engine's config with the query's overrides set.
 * @throws ConfigValueError for a value KataGo would refuse; UnsupportedSettingError for keys whose search features
 *   aren't ported.
 */
export function searchParamsFromConfig(config: Readonly<Record<string, string>>, base: Partial<SearchParams> = {}): ConfigSearchSettings {
  const cfg = new Config(new Map(Object.entries(config)));
  const unsupported = Object.keys(config).filter((k) => UNPORTED_KEYS.has(k));
  if (unsupported.length > 0) throw new UnsupportedSettingError(unsupported);
  const p: SearchParams = { ...DEFAULT_SEARCH_PARAMS, ...base };
  const dbl = (key: string, min: number, max: number, def: number): number => (cfg.contains(key) ? cfg.getDouble(key, min, max) : def);
  const bool = (key: string, def: boolean): boolean => (cfg.contains(key) ? cfg.getBool(key) : def);

  const out: ConfigSearchSettings = { params: p, unusedKeys: [] };
  if (cfg.contains('maxVisits')) out.maxVisits = cfg.getInt('maxVisits', 1, 2 ** 50, 'int64');
  // KataGo's search threads each run a playout, with virtual losses, while the others wait for the network; here a
  // batch of playouts waits for one network call, so the batch stands in for the threads.
  if (cfg.contains('numSearchThreads')) p.batchSize = cfg.getInt('numSearchThreads', 1, 4096);
  // Each root evaluation uses one symmetry, KataGo's default; sampling more isn't ported.
  if (cfg.contains('rootNumSymmetriesToSample') && cfg.getInt('rootNumSymmetriesToSample', 1, 8) !== 1) {
    throw new UnsupportedSettingError(['rootNumSymmetriesToSample']);
  }

  p.winLossUtilityFactor = dbl('winLossUtilityFactor', 0.0, 1.0, 1.0);
  p.staticScoreUtilityFactor = dbl('staticScoreUtilityFactor', 0.0, 1.0, 0.1);
  p.dynamicScoreUtilityFactor = dbl('dynamicScoreUtilityFactor', 0.0, 1.0, 0.3);
  p.noResultUtilityForWhite = dbl('noResultUtilityForWhite', -1.0, 1.0, 0.0);
  p.drawEquivalentWinsForWhite = dbl('drawEquivalentWinsForWhite', 0.0, 1.0, 0.5);
  p.dynamicScoreCenterZeroWeight = dbl('dynamicScoreCenterZeroWeight', 0.0, 1.0, 0.2);
  p.dynamicScoreCenterScale = dbl('dynamicScoreCenterScale', 0.2, 5.0, 0.75);
  p.cpuctExploration = dbl('cpuctExploration', 0.0, 10.0, 1.0);
  p.cpuctExplorationLog = dbl('cpuctExplorationLog', 0.0, 10.0, 0.45);
  p.cpuctExplorationBase = dbl('cpuctExplorationBase', 10.0, 100000.0, 500.0);
  p.cpuctUtilityStdevPrior = dbl('cpuctUtilityStdevPrior', 1e-8, 10.0, 0.4);
  p.cpuctUtilityStdevPriorWeight = dbl('cpuctUtilityStdevPriorWeight', 0.0, 100.0, 2.0);
  p.cpuctUtilityStdevScale = dbl('cpuctUtilityStdevScale', 0.0, 1.0, 0.85);
  p.fpuReductionMax = dbl('fpuReductionMax', 0.0, 2.0, 0.2);
  p.fpuLossProp = dbl('fpuLossProp', 0.0, 1.0, 0.0);
  p.fpuParentWeightByVisitedPolicy = bool('fpuParentWeightByVisitedPolicy', true);
  // KataGo reads only the one of these that applies; the other keeps the parameters' built-in default.
  if (p.fpuParentWeightByVisitedPolicy) p.fpuParentWeightByVisitedPolicyPow = dbl('fpuParentWeightByVisitedPolicyPow', 0.0, 5.0, 2.0);
  else p.fpuParentWeight = dbl('fpuParentWeight', 0.0, 1.0, 0.0);
  p.policyOptimism = dbl('policyOptimism', 0.0, 1.0, 1.0);
  p.valueWeightExponent = dbl('valueWeightExponent', 0.0, 1.0, 0.25);
  p.useNoisePruning = bool('useNoisePruning', true);
  p.noisePruneUtilityScale = dbl('noisePruneUtilityScale', 0.001, 10.0, 0.15);
  p.noisePruningCap = dbl('noisePruningCap', 0.0, 1e50, 1e50);
  p.useUncertainty = bool('useUncertainty', true);
  p.uncertaintyCoeff = dbl('uncertaintyCoeff', 0.0001, 1.0, 0.25);
  p.uncertaintyExponent = dbl('uncertaintyExponent', 0.0, 2.0, 1.0);
  p.uncertaintyMaxWeight = dbl('uncertaintyMaxWeight', 1.0, 100.0, 8.0);
  p.useGraphSearch = bool('useGraphSearch', true);
  p.graphSearchRepBound = cfg.contains('graphSearchRepBound') ? cfg.getInt('graphSearchRepBound', 3, 50) : 11;
  p.rootPolicyTemperature = dbl('rootPolicyTemperature', 0.01, 100.0, 1.0);
  p.rootPolicyTemperatureEarly = dbl('rootPolicyTemperatureEarly', 0.01, 100.0, p.rootPolicyTemperature);
  // KataGo: 0 with root noise, which isn't ported (rootNoiseEnabled is refused above).
  p.rootFpuReductionMax = dbl('rootFpuReductionMax', 0.0, 2.0, 0.1);
  p.rootFpuLossProp = dbl('rootFpuLossProp', 0.0, 1.0, p.fpuLossProp);
  p.rootSymmetryPruning = bool('rootSymmetryPruning', true);
  p.rootPolicyOptimism = dbl('rootPolicyOptimism', 0.0, 1.0, Math.min(p.policyOptimism, 0.2));
  p.chosenMoveTemperatureHalflife = dbl('chosenMoveTemperatureHalflife', 0.1, 100000.0, 19);
  p.chosenMoveSubtract = dbl('chosenMoveSubtract', 0.0, 1.0e10, 0.0);
  p.chosenMovePrune = dbl('chosenMovePrune', 0.0, 1.0e10, 1.0);
  p.useLcbForSelection = bool('useLcbForSelection', true);
  p.lcbStdevs = dbl('lcbStdevs', 1.0, 12.0, 5.0);
  p.minVisitPropForLCB = dbl('minVisitPropForLCB', 0.0, 1.0, 0.15);
  p.useNonBuggyLcb = bool('useNonBuggyLcb', true);
  p.rootEndingBonusPoints = dbl('rootEndingBonusPoints', -1.0, 1.0, 0.5);
  p.rootPruneUselessMoves = bool('rootPruneUselessMoves', true);
  // KataGo's default is false; its analysis engine sets true when the config doesn't say.
  p.conservativePass = bool('conservativePass', true);
  p.wideRootNoise = dbl('wideRootNoise', 0.0, 5.0, 0.04);
  p.enablePassingHacks = bool('enablePassingHacks', true);
  p.enableMorePassingHacks = bool('enableMorePassingHacks', true);
  p.ignorePreRootHistory = bool('ignorePreRootHistory', true);
  p.subtreeValueBiasFactor = dbl('subtreeValueBiasFactor', 0.0, 1.0, 0.45);
  p.subtreeValueBiasWeightExponent = dbl('subtreeValueBiasWeightExponent', 0.0, 1.0, 0.85);
  p.numVirtualLossesPerThread = dbl('numVirtualLossesPerThread', 0.01, 1000.0, 1.0);

  out.unusedKeys = Object.keys(config).filter((k) => !cfg.used.has(k));
  return out;
}
