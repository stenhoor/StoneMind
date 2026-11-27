/**
 * The browser engine's Monte Carlo tree search: an asynchronous port of the core of KataGo's search (`cpp/search/`,
 * KataGo 1.18, MIT, © David J Wu) that keeps a batch of leaves in flight for the GPU, using virtual losses to spread
 * them across the tree, as KataGo's search threads do.
 *
 * Ported (spike 0001, phase 5, essentials):
 * - selection: KataGo's PUCT with the log-growing exploration constant and the parent-utility-stdev factor
 *   (`getExploreScaling`, `getExploreSelectionValue`), first-play urgency with the visited-policy parent weighting
 *   (`getFpuValueForChildrenAssumeVisited`), and the best-policy unexpanded move (`selectBestChildToDescend`);
 * - values: win/loss and no-result utility plus static and dynamic score utility (`getResultUtility`,
 *   `getScoreUtility`, `computeRootValues`), terminal values from the game's result, and KataGo's stats recomputation
 *   from children plus the node's own evaluation (`recomputeNodeStats`, `addLeafValue`), with the children weighted by
 *   how their values compare (`downweightBadChildrenAndNormalizeWeight`, `valueWeightExponent`) after over-explored
 *   worse children lose part of their excess weight (`pruneNoiseWeight`, `useNoisePruning`), and each evaluation
 *   weighted by the network's own short-term error estimates (`computeWeightFromNNOutput`, `useUncertainty`);
 * - subtree value bias (`subtreeValueBiasFactor`, `SubtreeValueBiasTable`): a node's evaluation is corrected by how
 *   much searching below has, on average, changed the evaluation of the same move in the same local situation;
 * - an evaluation cache keyed by the exact network inputs, and a random symmetry per evaluation;
 * - policy optimism (`policyOptimism`, `rootPolicyOptimism`): the policy blended toward the network's optimistic policy
 *   head, for networks that have one;
 * - the passing hacks (`enablePassingHacks`, `enableMorePassingHacks`): the pass's prior capped, history hidden from the
 *   network where a premature pass would lose, and an extra uncounted playout to try a pass (or a move other than a
 *   pass) where a pass would end the phase; such playouts are why a child's visits and its parent's edge visits to it
 *   (`edgeVisits`) can differ;
 * - the root's ending bonus (`rootEndingBonusPoints`, `getEndingWhiteScoreBonus`): a small score penalty on root moves
 *   that fill the player's own sure area or play into the opponent's, and on passing under territory scoring;
 * - root pruning of useless moves (`rootPruneUselessMoves`, `isAllowedRootMove`): once the opponent's last four moves
 *   were passes, the root never tries a move inside either player's pass-alive area;
 * - root symmetry pruning (`rootSymmetryPruning`, `markDuplicateMoveLocs`): where the root position is symmetric, only
 *   one of each set of mirror-image moves is searched, and the analysis lists the others with `isSymmetryOf`;
 * - graph search (`useGraphSearch`, `GraphHash`, `allocateOrFindNode`): positions reached by different move orders
 *   share one node, found by a key built as KataGo builds its graph hash, so the tree is a graph whose edges carry the
 *   move and the edge visits; cycles end the playout (`graphPath`);
 * - wide root noise (`wideRootNoise`, `maybeApplyWideRootNoise`): at the root, during the search, the policy is
 *   flattened and half the time a move's value gets a random bonus, so the search looks at more moves;
 * - the analysis's move order and principal variations from KataGo's play selection values
 *   (`getPlaySelectionValues`), with LCB selection (`useLcbForSelection`, `getSelfUtilityLCBAndRadius`) and the root's
 *   retrospective reduction of over-explored moves (`getReducedPlaySelectionWeight`).
 *
 * Not ported: tree reuse (native's analysis engine starts each search afresh).
 * Search parameters that only matter for those default to values that switch them off.
 */
import type { ModelDesc } from '../model/types.js';
import { BLACK, EMPTY, NULL_LOC, PASS_LOC, WHITE, opp, type Board, type Stone } from '../rules/board.js';
import { GameHistory, UnsupportedPositionError } from '../rules/history.js';
import { encodeHistoryV7, type LadderCache } from '../features/v7.js';
import { calculateArea } from '../rules/area.js';
import { positionHashes } from '../rules/hash.js';
import type { NetInput, RawOutputs } from '../reference/forward.js';
import { postProcess, type Evaluation } from '../reference/postprocess.js';
import { copyWithSymmetry } from '../reference/symmetry.js';
import { expectedWhiteScoreValue, getScoreStdev, whiteDScoreValueDScoreSmooth, whiteScoreMeanSqOfScoreGridded } from './scorevalue.js';
import { VALUE_WEIGHT_DISTRIBUTION } from './distribution.js';

/** Something that evaluates a batch of network inputs: the WebGPU network, or the reference forward pass in tests. */
export interface NetEvaluator {
  readonly maxBatch: number;
  /** How many `evaluate` calls it runs at once (default 1). The WebGPU network allows 2 by default. */
  readonly maxInFlight?: number;
  evaluate(inputs: readonly NetInput[]): Promise<RawOutputs[]>;
}

/** KataGo's search parameters used here, with KataGo's analysis defaults (`cpp/program/setup.cpp`). */
export interface SearchParams {
  cpuctExploration: number;
  cpuctExplorationLog: number;
  cpuctExplorationBase: number;
  cpuctUtilityStdevPrior: number;
  cpuctUtilityStdevPriorWeight: number;
  cpuctUtilityStdevScale: number;
  fpuReductionMax: number;
  rootFpuReductionMax: number;
  fpuLossProp: number;
  fpuParentWeightByVisitedPolicy: boolean;
  fpuParentWeightByVisitedPolicyPow: number;
  fpuParentWeight: number;
  winLossUtilityFactor: number;
  staticScoreUtilityFactor: number;
  dynamicScoreUtilityFactor: number;
  dynamicScoreCenterZeroWeight: number;
  dynamicScoreCenterScale: number;
  noResultUtilityForWhite: number;
  drawEquivalentWinsForWhite: number;
  numVirtualLossesPerThread: number;
  /** Leaves gathered per network call (at most the evaluator's `maxBatch`). */
  batchSize: number;
  /** Seed for the random symmetries. */
  seed: number;
  /** A fixed symmetry (0–7) for every evaluation, as KataGo with `nnRandomize = false` uses 0; or -1 for random. */
  symmetry: number;
  /**
   * KataGo `ignorePreRootHistory` (the analysis engine's default, true): the root's inputs show no past moves, and
   * deeper positions only the moves made since the root.
   */
  ignorePreRootHistory: boolean;
  /**
   * KataGo `conservativePass` (the analysis engine's default, true): a pass from the root never ends the game, and the
   * root's inputs behave as if a pass wouldn't.
   */
  conservativePass: boolean;
  /** KataGo `useLcbForSelection`: the move with the best lower confidence bound on utility is chosen, if visited enough. */
  useLcbForSelection: boolean;
  /** KataGo `lcbStdevs`: the LCB's width in standard deviations. */
  lcbStdevs: number;
  /** KataGo `minVisitPropForLCB`: a move needs this share of the best move's weight to be chosen by LCB. */
  minVisitPropForLCB: number;
  /** KataGo `useNonBuggyLcb` (the analysis default, true): LCB may also choose the first child. */
  useNonBuggyLcb: boolean;
  /** KataGo `chosenMoveSubtract` and `chosenMovePrune`: play selection values are reduced by the first and zeroed below the second. */
  chosenMoveSubtract: number;
  chosenMovePrune: number;
  /** KataGo `analysisPVLen`: moves in each principal variation after the move itself. */
  analysisPVLen: number;
  /**
   * KataGo `valueWeightExponent`: when a node's value is recomputed, each child's weight is scaled by the t
   * distribution's probability that it's better than the children's average, to this power (0 switches it off).
   */
  valueWeightExponent: number;
  /**
   * KataGo `useUncertainty`: each evaluation is weighted by the inverse of the network's short-term error estimate,
   * from `uncertaintyCoeff / uncertaintyMaxWeight` (weight `uncertaintyMaxWeight`) upward; terminal positions get
   * `uncertaintyMaxWeight`. Only for networks of model version 9 or later, as in KataGo.
   */
  useUncertainty: boolean;
  uncertaintyCoeff: number;
  uncertaintyExponent: number;
  uncertaintyMaxWeight: number;
  /**
   * KataGo `subtreeValueBiasFactor`: the share of the table's average correction added to a node's evaluation (0
   * switches it off). Each node's correction counts in its table entry with weight (children's weight) to the power
   * `subtreeValueBiasWeightExponent`.
   */
  subtreeValueBiasFactor: number;
  subtreeValueBiasWeightExponent: number;
  /**
   * KataGo `useNoisePruning`: when a node's value is recomputed, a child worse than the children before it (in child
   * order) loses part of its weight beyond twice its policy share: `1 - exp(-gap / noisePruneUtilityScale)` of the
   * excess, at most `noisePruningCap`.
   */
  useNoisePruning: boolean;
  noisePruneUtilityScale: number;
  noisePruningCap: number;
  /**
   * KataGo `policyOptimism` and `rootPolicyOptimism`: how far the policy is blended toward the network's optimistic
   * policy head (model version 12 and later), below the root and at the root. Evaluations with different optimism are
   * cached apart, as in KataGo.
   */
  policyOptimism: number;
  rootPolicyOptimism: number;
  /**
   * KataGo `enablePassingHacks`: the pass's prior is capped at 19 times the other moves' total, and the network doesn't
   * see the history where a pass would end the game and doing so now would not win.
   */
  enablePassingHacks: boolean;
  /**
   * KataGo `enableMorePassingHacks`: where a pass would end the phase and a node has 2 or more edge visits, a playout
   * that would not try a pass there yet (or would try only passes) tries one instead, as a playout that isn't counted.
   */
  enableMorePassingHacks: boolean;
  /** KataGo `rootEndingBonusPoints`: the score penalty, in points, of the root's ending bonus. */
  rootEndingBonusPoints: number;
  /**
   * KataGo `rootPruneUselessMoves` (on in its analysis engine): if the opponent's last four moves were passes, the root
   * never tries a move inside either player's strictly pass-alive area.
   */
  rootPruneUselessMoves: boolean;
  /**
   * KataGo `rootSymmetryPruning` (on in its analysis engine): where the root position is symmetric, the root searches
   * only one of each set of mirror-image moves, and the analysis lists the others as copies.
   */
  rootSymmetryPruning: boolean;
  /**
   * KataGo `useGraphSearch` (on in its analysis engine): positions reached by different move orders share one node.
   * Where the last move could be part of a repetition within `graphSearchRepBound` moves, the key also depends on the
   * parent's, so positions with different recent histories stay apart.
   */
  useGraphSearch: boolean;
  graphSearchRepBound: number;
  /**
   * KataGo `wideRootNoise` (0.04 in its analysis engine): at the root, during the search, each move's policy is raised
   * to the power 1 / (4 × this + 1), and with probability one half its value gains this × |a standard normal draw| for
   * the player to move. 0 switches it off, and the search is then deterministic for a fixed symmetry.
   */
  wideRootNoise: number;
  /**
   * KataGo `rootPolicyTemperature` and `rootPolicyTemperatureEarly`: the root's policy, as the search uses it, is raised
   * to the power 1 / temperature and renormalised. The temperature goes from the early value toward the other, halving
   * the gap every `chosenMoveTemperatureHalflife` turns (scaled to a 19x19 board). 1 leaves the policy as it is.
   */
  rootPolicyTemperature: number;
  rootPolicyTemperatureEarly: number;
  /** KataGo `chosenMoveTemperatureHalflife` (19): only the root's policy temperature uses it here. */
  chosenMoveTemperatureHalflife: number;
  /** KataGo `rootFpuLossProp`: `fpuLossProp` at the root. KataGo's config loader defaults it to `fpuLossProp`. */
  rootFpuLossProp: number;
}

export const DEFAULT_SEARCH_PARAMS: SearchParams = {
  cpuctExploration: 1.0,
  cpuctExplorationLog: 0.45,
  cpuctExplorationBase: 500,
  cpuctUtilityStdevPrior: 0.4,
  cpuctUtilityStdevPriorWeight: 2.0,
  cpuctUtilityStdevScale: 0.85,
  fpuReductionMax: 0.2,
  rootFpuReductionMax: 0.1,
  fpuLossProp: 0,
  fpuParentWeightByVisitedPolicy: true,
  fpuParentWeightByVisitedPolicyPow: 2.0,
  fpuParentWeight: 0,
  winLossUtilityFactor: 1.0,
  staticScoreUtilityFactor: 0.1,
  dynamicScoreUtilityFactor: 0.3,
  dynamicScoreCenterZeroWeight: 0.2,
  dynamicScoreCenterScale: 0.75,
  noResultUtilityForWhite: 0,
  drawEquivalentWinsForWhite: 0.5,
  numVirtualLossesPerThread: 1,
  batchSize: 64,
  seed: 1,
  symmetry: -1,
  ignorePreRootHistory: true,
  conservativePass: true,
  useLcbForSelection: true,
  lcbStdevs: 5.0,
  minVisitPropForLCB: 0.15,
  useNonBuggyLcb: true,
  chosenMoveSubtract: 0,
  chosenMovePrune: 1,
  analysisPVLen: 15,
  valueWeightExponent: 0.25,
  useUncertainty: true,
  uncertaintyCoeff: 0.25,
  uncertaintyExponent: 1.0,
  uncertaintyMaxWeight: 8.0,
  subtreeValueBiasFactor: 0.45,
  subtreeValueBiasWeightExponent: 0.85,
  useNoisePruning: true,
  noisePruneUtilityScale: 0.15,
  noisePruningCap: 1e50,
  policyOptimism: 1.0,
  rootPolicyOptimism: 0.2,
  enablePassingHacks: true,
  enableMorePassingHacks: true,
  rootEndingBonusPoints: 0.5,
  rootPruneUselessMoves: true,
  rootSymmetryPruning: true,
  useGraphSearch: true,
  graphSearchRepBound: 11,
  wideRootNoise: 0.04,
  rootPolicyTemperature: 1.0,
  rootPolicyTemperatureEarly: 1.0,
  chosenMoveTemperatureHalflife: 19,
  rootFpuLossProp: 0,
};

/** KataGo `TOTALCHILDWEIGHT_PUCT_OFFSET`. */
const TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;
const POLICY_ILLEGAL_SELECTION_VALUE = -1e50;

/** A `SubtreeValueBiasTable` entry: the summed weighted correction of the nodes that share it. */
interface SubtreeValueBiasEntry {
  deltaUtilitySum: number;
  weightSum: number;
}

/**
 * KataGo `SubtreeValueBiasTable::get`'s key, as a string in place of its 128-bit Zobrist hash: the player who moved, the
 * move before and the move, and around the move, on the board before it, each point's colour and whether its chain is
 * in atari (a 5x5 window clipped to the board), plus the ko point.
 */
function subtreeValueBiasKey(pla: Stone, parentPrevMoveLoc: number, prevMoveLoc: number, prevBoard: Board): string {
  let key = `${pla}|${parentPrevMoveLoc}|${prevMoveLoc}|`;
  const x = prevBoard.x(prevMoveLoc);
  const y = prevBoard.y(prevMoveLoc);
  const radius = 2;
  const dxMin = x < radius ? -x : -radius;
  const dxMax = x >= prevBoard.xSize - radius ? prevBoard.xSize - 1 - x : radius;
  const dyMin = y < radius ? -y : -radius;
  const dyMax = y >= prevBoard.ySize - radius ? prevBoard.ySize - 1 - y : radius;
  for (let dy = dyMin; dy <= dyMax; dy++) {
    for (let dx = dxMin; dx <= dxMax; dx++) {
      const loc2 = prevBoard.loc(x + dx, y + dy);
      const c = prevBoard.colors[loc2]!;
      key += c;
      if (c !== EMPTY && prevBoard.getNumLiberties(loc2) === 1) key += 'a';
    }
  }
  if (prevBoard.koLoc !== NULL_LOC) key += `|ko${prevBoard.koLoc}`;
  return key;
}

const UNEVALUATED = 0;
const PENDING = 1;
const EVALUATED = 2;

/** A parent's link to a child (KataGo `SearchChildPointer`): the move, and the visits the parent counts through it. */
class SearchEdge {
  /**
   * KataGo's edge visits. They are fewer than the child's visits after a playout that wasn't counted
   * (`enableMorePassingHacks`), or where the child was also reached another way (graph search), until later playouts
   * catch them up.
   */
  edgeVisits = 0;
  constructor(
    readonly node: SearchNode,
    readonly moveLoc: number,
  ) {}
}

class SearchNode {
  state = UNEVALUATED;
  eval: Evaluation | null = null;
  readonly children: SearchEdge[] = [];
  visits = 0;
  weightSum = 0;
  weightSqSum = 0;
  utilityAvg = 0;
  utilitySqAvg = 0;
  winLossValueAvg = 0;
  noResultValueAvg = 0;
  scoreMeanAvg = 0;
  scoreMeanSqAvg = 0;
  leadAvg = 0;
  virtualLosses = 0;
  /** KataGo `subtreeValueBiasTableEntry`, and this node's last contribution to it. */
  biasEntry: SubtreeValueBiasEntry | null = null;
  lastBiasDeltaSum = 0;
  lastBiasWeight = 0;

  constructor(
    /** The history of the path that created this node; other paths may reach it with other histories (graph search). */
    readonly hist: GameHistory,
    /** The node whose child this was created as (null for the root): the path through it has exactly `hist`. */
    readonly creator: SearchNode | null,
    readonly terminal: boolean,
    /** KataGo's graph hash, as a key (graph search). */
    readonly graphKey: string,
  ) {}

  get nextPla(): Stone {
    return this.hist.nextPla;
  }
}

/** One move's line in the analysis, from the root player's perspective. */
export interface SearchMoveInfo {
  /** Board location (`Board.loc`), or `PASS_LOC`. */
  loc: number;
  /** The move's own visits (KataGo's `visits`). */
  visits: number;
  /** The visits the root counts through this move (KataGo's `edgeVisits`). */
  edgeVisits: number;
  /** Win probability for the player to move at the root. */
  winrate: number;
  /** Expected score lead for the player to move at the root. */
  scoreLead: number;
  scoreMean: number;
  scoreStdev: number;
  /** The network's policy for this move at the root. */
  prior: number;
  /** Utility for the player to move at the root. */
  utility: number;
  /**
   * The move's lower confidence bound, as KataGo reports it: the win rate for the player to move at the root, less the
   * LCB radius scaled to win-rate units (`PlayUtils::getHackedLCBForWinrate`).
   */
  lcb: number;
  /** The lower confidence bound on utility, for the player to move at the root. */
  utilityLcb: number;
  /** KataGo's play selection value: the weight the move would be chosen by, after LCB and the root's reductions. */
  playSelectionValue: number;
  /** Rank, best first: by play selection value, then edge visits, then policy (KataGo's `AnalysisData` order). */
  order: number;
  /**
   * With root symmetry pruning, a move not searched because it mirrors this searched move (KataGo's `isSymmetryOf`):
   * its line is that move's line, mirrored.
   */
  isSymmetryOf?: number;
  /** Principal variation: this move, then the child with the highest play selection value at each step. */
  pv: number[];
  /** The visits of each node along the principal variation (KataGo's `pvVisits`). */
  pvVisits: number[];
  /** The edge visits of each move along the principal variation (KataGo's `pvEdgeVisits`). */
  pvEdgeVisits: number[];
  /** The child's total weight (KataGo's `weight`). */
  weight: number;
  /** The weight the root counts through this move (KataGo's `edgeWeight`). */
  edgeWeight: number;
  /** The child's average no-result value (KataGo's `noResultValue`). */
  noResultValue: number;
  /**
   * With `analysis({ movesOwnership })`: the child's ownership averaged over its subtree, for the player to move at
   * the root, row-major (KataGo's per-move `ownership`). A mirror copy's is the searched move's, mirrored.
   */
  ownership?: number[];
  /** With `analysis({ movesOwnershipStdev })`: the standard deviation of that ownership (KataGo's `ownershipStdev`). */
  ownershipStdev?: number[];
}

/**
 * KataGo `getPrunedRootValues`, for the player to move at the root: the root's children weighted by their play
 * selection values without LCB, plus the root's own evaluation. KataGo reports these as `rootInfo`.
 */
export interface SearchRootValues {
  visits: number;
  /** The root's total weight. */
  weight: number;
  winrate: number;
  /** KataGo's `scoreSelfplay`: the expected score, rather than the lead. */
  scoreSelfplay: number;
  scoreLead: number;
  scoreStdev: number;
  utility: number;
  noResultValue: number;
}

/** The root's own network evaluation, as KataGo reports it in `rootInfo` (the `raw` fields). */
export interface SearchRawValues {
  /** Win probability for the player to move. */
  winrate: number;
  /** For the player to move. */
  lead: number;
  /** For the player to move. */
  scoreSelfplay: number;
  scoreSelfplayStdev: number;
  noResultProb: number;
  /** Half the network's short-term win/loss error, as KataGo reports it. */
  stWrError: number;
  stScoreError: number;
  varTimeLeft: number;
}

export interface SearchAnalysis {
  rootVisits: number;
  /** As in moveInfos, for the whole tree. */
  rootWinrate: number;
  rootScoreLead: number;
  rootUtility: number;
  moveInfos: SearchMoveInfo[];
}

/** Which of KataGo's optional per-move outputs `analysis` computes. */
export interface AnalysisOptions {
  /** KataGo `includeMovesOwnership`. */
  movesOwnership?: boolean;
  /** KataGo `includeMovesOwnershipStdev`. */
  movesOwnershipStdev?: boolean;
}

export interface SearchStats {
  /** Visits added by `run` (KataGo counts playouts the same way). */
  visits: number;
  nnEvals: number;
  cacheHits: number;
  terminalVisits: number;
  batches: number;
  /** Playouts abandoned because they reached a leaf already waiting for the network. */
  collisions: number;
  /** Time spent waiting for the network, and in everything else (selection, encoding, backup), in ms. */
  nnMs: number;
  cpuMs: number;
}

/** A small, fast seeded PRNG (mulberry32), for the random symmetries. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** KataGo `SymmetryHelpers::getSymLoc`: `loc` under `symmetry` (bit 0 flips y, bit 1 flips x, bit 2 transposes). */
function getSymLoc(loc: number, board: Board, symmetry: number): number {
  if (loc === NULL_LOC || loc === PASS_LOC) return loc;
  let x = board.x(loc);
  let y = board.y(loc);
  if (symmetry & 0x2) x = board.xSize - x - 1;
  if (symmetry & 0x1) y = board.ySize - y - 1;
  if (symmetry & 0x4) [x, y] = [y, x];
  return board.loc(x, y);
}

/**
 * KataGo `BoardHistory::getSituationRulesAndKoHash` as a key: the position, the player to move, the encore phase, ko
 * bans (in the main phase the points banned by simple ko or superko; in the encore the once-only bans and recapture blocks,
 * and in the second encore its starting stones), the komi for the player to move (to 1/256), the rules, and the
 * history modes where they change anything. Two keys are equal exactly when KataGo's hashes are (barring a collision of
 * its hashes; `test/history.test.mjs`). The board size is left out, as it's the same throughout a search.
 */
export function situationKey(h: GameHistory, pla: Stone, drawEquivalentWinsForWhite: number): string {
  const b = h.board;
  let key = '';
  for (let y = 0; y < b.ySize; y++) for (let x = 0; x < b.xSize; x++) key += b.colors[b.loc(x, y)];
  key += `|${pla}|e${h.encorePhase}|`;
  if (h.encorePhase === 0) {
    // The simple ko point and the superko bans go into KataGo's hash alike, so only the set of banned points counts.
    for (let y = 0; y < b.ySize; y++) {
      for (let x = 0; x < b.xSize; x++) {
        const loc = b.loc(x, y);
        if (loc === b.koLoc || h.superKoBanned[loc]) key += loc.toString(36) + ',';
      }
    }
  } else {
    for (let y = 0; y < b.ySize; y++) {
      for (let x = 0; x < b.xSize; x++) {
        const loc = b.loc(x, y);
        if (h.superKoBanned[loc]) key += loc.toString(36) + ',';
        if (h.koRecapBlocked[loc]) key += 'r' + loc.toString(36) + ',';
      }
    }
    if (h.encorePhase === 2) {
      key += '|';
      for (let y = 0; y < b.ySize; y++) for (let x = 0; x < b.xSize; x++) key += h.secondEncoreStartColors[b.loc(x, y)];
    }
  }
  // The self komi discretized as KataGo does, (int64_t)(selfKomi * 256.0f), and the rules.
  const selfKomi = h.currentSelfKomi(pla, drawEquivalentWinsForWhite);
  const r = h.rules;
  key += `|k${Math.trunc(Math.fround(selfKomi * 256))}|${r.koRule}${r.scoringRule}${r.taxRule}${+r.multiStoneSuicideLegal}${+r.hasButton}${+r.friendlyPassOk}`;
  if (h.modes.alwaysComputePassAliveUnderSuicideRules && !r.multiStoneSuicideLegal) key += 's';
  if (h.modes.excludeTerritoryAdjacentToAtari && r.scoringRule === 'TERRITORY' && r.taxRule === 'NONE') key += 'x';
  return key;
}

/**
 * KataGo `GraphHash::getStateHash` as a key: the situation (`situationKey`), whether a pass would end the phase,
 * whether the game is over, and the consecutive passes.
 */
export function graphStateKey(h: GameHistory, pla: Stone, drawEquivalentWinsForWhite: number): string {
  return `${situationKey(h, pla, drawEquivalentWinsForWhite)}|${h.passWouldEndPhase(pla) ? 1 : 0}${h.isGameFinished ? 1 : 0}|${h.consecutiveEndingPasses}`;
}

/**
 * KataGo `SymmetryHelpers::markDuplicateMoveLocs`: the symmetries under which the position is unchanged (in the encore,
 * its recapture blocks too, and in the second encore its starting stones), and the moves that mirror an earlier one,
 * skipping moves the player to move avoids at the root. A position with a ko or superko ban has none.
 */
function markDuplicateMoveLocs(hist: GameHistory, avoidMoves: Int32Array | null): { isSymDupLoc: Uint8Array; validSymmetries: number[] } {
  const board = hist.board;
  const isSymDupLoc = new Uint8Array(board.arraySize);
  const validSymmetries = [0];
  if (board.koLoc !== NULL_LOC) return { isSymDupLoc, validSymmetries };
  for (let y = 0; y < board.ySize; y++) {
    for (let x = 0; x < board.xSize; x++) if (hist.superKoBanned[board.loc(x, y)]) return { isSymDupLoc, validSymmetries };
  }
  // Rectangular boards have no transposing symmetries.
  const upperBound = board.xSize === board.ySize ? 8 : 4;
  for (let symmetry = 1; symmetry < upperBound; symmetry++) {
    let isBoardSym = true;
    for (let y = 0; y < board.ySize && isBoardSym; y++) {
      for (let x = 0; x < board.xSize; x++) {
        const loc = board.loc(x, y);
        const symLoc = getSymLoc(loc, board, symmetry);
        if (
          board.colors[loc] !== board.colors[symLoc] ||
          (hist.encorePhase > 0 && hist.koRecapBlocked[loc] !== hist.koRecapBlocked[symLoc]) ||
          (hist.encorePhase === 2 && hist.secondEncoreStartColors[loc] !== hist.secondEncoreStartColors[symLoc])
        ) {
          isBoardSym = false;
          break;
        }
      }
    }
    if (isBoardSym) validSymmetries.push(symmetry);
  }
  // KataGo's order: Black keeps moves on the right, White on the left, as in common openings.
  const mark = (x: number, y: number) => {
    const loc = board.loc(x, y);
    if (avoidMoves && avoidMoves[loc]! > 0) return;
    for (const symmetry of validSymmetries) {
      if (symmetry === 0) continue;
      const symLoc = getSymLoc(loc, board, symmetry);
      if (!isSymDupLoc[loc] && loc !== symLoc) isSymDupLoc[symLoc] = 1;
    }
  };
  if (hist.nextPla === BLACK) {
    for (let x = board.xSize - 1; x >= 0; x--) for (let y = 0; y < board.ySize; y++) mark(x, y);
  } else {
    for (let x = 0; x < board.xSize; x++) for (let y = board.ySize - 1; y >= 0; y--) mark(x, y);
  }
  return { isSymDupLoc, validSymmetries };
}

interface Pending {
  /** The path: nodes[0] is the root, and edges[i] leads from nodes[i] to nodes[i + 1]. */
  nodes: SearchNode[];
  edges: SearchEdge[];
  /** The deepest edge that doesn't count this visit (`enableMorePassingHacks`), or -1: KataGo `shouldCountPlayout`. */
  uncountedIdx: number;
  input: NetInput;
  symmetry: number;
  key: string;
  encoded: ReturnType<typeof encodeHistoryV7>;
}

export class Search {
  readonly params: SearchParams;
  readonly root: SearchNode;
  readonly stats: SearchStats = { visits: 0, nnEvals: 0, cacheHits: 0, terminalVisits: 0, batches: 0, collisions: 0, nnMs: 0, cpuMs: 0 };
  readonly #model: ModelDesc;
  readonly #evaluator: NetEvaluator;
  readonly #cache: Map<string, Evaluation>;
  /** Ladder results by board, for this search's inputs (the inputs are the same without it; see `LadderCache`). */
  readonly #ladders: LadderCache = new Map();
  readonly #random: () => number;
  /** KataGo `Rand::nextGaussian` keeps the polar method's second value for the next call. */
  #storedGaussian: number | null = null;
  /** KataGo's `subtreeValueBiasTable`, one per search (native's analysis engine starts each query with it empty). */
  readonly #subtreeValueBiasTable = new Map<string, SubtreeValueBiasEntry>();
  /** KataGo's `nodeTable` (graph search): every node but the root, by graph key (and forceNonTerminal). */
  readonly #nodeTable = new Map<string, SearchNode>();
  /** Short ids for graph keys, so a key that depends on its parent's stays short. */
  readonly #graphKeyIds = new Map<string, number>();
  readonly #sqrtBoardArea: number;
  readonly #area: number;
  #recentScoreCenter = 0;
  /** KataGo `rootSafeArea`: the root's strictly pass-alive groups and strictly safe territory. */
  #rootSafeArea: Uint8Array = new Uint8Array(0);
  /** `rootPruneUselessMoves` is on and the opponent's last four moves at the root were passes. */
  #rootPrunesUselessMoves = false;
  /** KataGo `rootSymDupLoc` and `rootSymmetries`: the root's mirror-image duplicate moves, and its symmetries. */
  #rootSymDupLoc: Uint8Array = new Uint8Array(0);
  #rootSymmetries: number[] = [0];
  #rootPrepared = false;
  /**
   * KataGo's `noisedPolicyProbs` of the root: its policy with the root policy temperature, which the search uses in
   * place of the network's (`getPolicyProbsMaybeNoised`). Null when the temperature is 1.
   */
  #rootSearchPolicy: Float64Array | null = null;
  /** KataGo `avoidMoveUntilByLocBlack` and `avoidMoveUntilByLocWhite` (see `setAvoidMoveUntilByLoc`). */
  #avoidMoveUntilByLoc: Record<Stone, Int32Array | null> = { [BLACK]: null, [WHITE]: null } as Record<Stone, Int32Array | null>;

  /**
   * @param cache An evaluation cache to share with other searches (KataGo's analysis engine shares one across the turns
   *   of a query); by default each search has its own.
   */
  constructor(model: ModelDesc, evaluator: NetEvaluator, rootHistory: GameHistory, params: Partial<SearchParams> = {}, cache: Map<string, Evaluation> = new Map()) {
    this.#cache = cache;
    this.params = { ...DEFAULT_SEARCH_PARAMS, ...params };
    this.params.batchSize = Math.max(1, Math.min(this.params.batchSize, evaluator.maxBatch));
    this.#model = model;
    this.#evaluator = evaluator;
    this.#random = mulberry32(this.params.seed);
    const b = rootHistory.board;
    this.#area = b.xSize * b.ySize;
    this.#sqrtBoardArea = Math.sqrt(this.#area);
    // The root is never terminal (KataGo forceNonTerminal): searching a finished position still explores it. Like
    // KataGo's root, it isn't in the node table.
    const rootHist = rootHistory.clone();
    // KataGo Search::applyHistoryModesToRootHistory, with its params at their default (auto): the network's modes.
    rootHist.setModes({
      alwaysComputePassAliveUnderSuicideRules: model.preferPassAliveUnderSuicideRules,
      excludeTerritoryAdjacentToAtari: model.preferExcludeTerritoryAdjacentToAtari,
    });
    this.root = new SearchNode(rootHist, null, false, this.params.useGraphSearch ? this.#rootGraphKey(rootHist) : '');
  }

  /**
   * KataGo `setAvoidMoveUntilByLoc`, as its analysis engine applies `avoidMoves` and `allowMoves` (see
   * `avoidMoveUntilByLoc`): for each player, indexed by board location (`PASS_LOC` included), the depth below the root
   * until which the search doesn't try that move; 0 or less, no limit. Null for a player with no limits. KataGo clears
   * the search when these change; here they must be set before the search starts.
   */
  setAvoidMoveUntilByLoc(black: Int32Array | null, white: Int32Array | null): void {
    if (this.#rootPrepared) throw new Error('setAvoidMoveUntilByLoc must be called before the search starts');
    this.#avoidMoveUntilByLoc = { [BLACK]: black, [WHITE]: white } as Record<Stone, Int32Array | null>;
  }

  /** The policy the search uses at `node`: the root's has the root policy temperature (KataGo `getPolicyProbsMaybeNoised`). */
  #searchPolicy(node: SearchNode): Float64Array {
    return node === this.root && this.#rootSearchPolicy ? this.#rootSearchPolicy : node.eval!.policy;
  }

  /**
   * KataGo `maybeAddPolicyNoiseAndTemp` at the root, with only its policy temperature ported (root Dirichlet noise and
   * `avoidMoveUntilRescaleRoot` are off in KataGo's analysis engine). Values are 32-bit floats, as KataGo keeps them.
   */
  #rootPolicyWithTemperature(policy: Float64Array): Float64Array | null {
    const p = this.params;
    if (p.rootPolicyTemperature === 1.0 && p.rootPolicyTemperatureEarly === 1.0) return null;
    // KataGo interpolateEarly.
    const b = this.root.hist.board;
    const rawHalflives = this.root.hist.moves.length / p.chosenMoveTemperatureHalflife;
    const halflives = (rawHalflives * 19.0) / Math.sqrt(b.xSize * b.ySize);
    const temperature = p.rootPolicyTemperature + (p.rootPolicyTemperatureEarly - p.rootPolicyTemperature) * Math.pow(0.5, halflives);
    const out = Float64Array.from(policy);
    let maxValue = 0;
    for (const prob of out) if (prob > maxValue) maxValue = prob;
    const logMaxValue = Math.log(maxValue);
    const invTemp = 1.0 / temperature;
    let sum = 0;
    for (let i = 0; i < out.length; i++) {
      if (out[i]! > 0) {
        const q = Math.fround(Math.exp((Math.log(out[i]!) - logMaxValue) * invTemp));
        out[i] = q;
        sum += q;
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i]! >= 0) out[i] = Math.fround(out[i]! / sum);
    return out;
  }

  // ---------------------------------------------------------------- utilities

  #resultUtility(winLoss: number, noResult: number): number {
    return winLoss * this.params.winLossUtilityFactor + noResult * this.params.noResultUtilityForWhite;
  }

  #scoreUtility(scoreMean: number, scoreMeanSq: number): number {
    const p = this.params;
    const stdev = getScoreStdev(scoreMean, scoreMeanSq);
    const staticValue = expectedWhiteScoreValue(scoreMean, stdev, 0, 2, this.#sqrtBoardArea);
    const dynamicValue = expectedWhiteScoreValue(scoreMean, stdev, this.#recentScoreCenter, p.dynamicScoreCenterScale, this.#sqrtBoardArea);
    return staticValue * p.staticScoreUtilityFactor + dynamicValue * p.dynamicScoreUtilityFactor;
  }

  /** KataGo `getApproxScoreUtilityDerivative`: ignores scoreMeanSq's effect on the utility. */
  /** KataGo `getScoreUtilityDiff`: how much the score utility changes if White's score moves by `delta`. */
  #scoreUtilityDiff(scoreMeanAvg: number, scoreMeanSqAvg: number, delta: number): number {
    const p = this.params;
    const stdev = getScoreStdev(scoreMeanAvg, scoreMeanSqAvg);
    const staticDiff =
      expectedWhiteScoreValue(scoreMeanAvg + delta, stdev, 0, 2, this.#sqrtBoardArea) - expectedWhiteScoreValue(scoreMeanAvg, stdev, 0, 2, this.#sqrtBoardArea);
    const dynamicDiff =
      expectedWhiteScoreValue(scoreMeanAvg + delta, stdev, this.#recentScoreCenter, p.dynamicScoreCenterScale, this.#sqrtBoardArea) -
      expectedWhiteScoreValue(scoreMeanAvg, stdev, this.#recentScoreCenter, p.dynamicScoreCenterScale, this.#sqrtBoardArea);
    return staticDiff * p.staticScoreUtilityFactor + dynamicDiff * p.dynamicScoreUtilityFactor;
  }

  /**
   * KataGo `getEndingWhiteScoreBonus`, in points for White: for root moves only, a penalty of up to
   * `rootEndingBonusPoints` for a move on a point the opponent almost surely owns (unless it captures), or on one the
   * player almost surely owns that touches no opponent stone and joins no chains that aren't pass-alive; under territory
   * scoring, also two thirds of it for passing.
   */
  #endingWhiteScoreBonus(parent: SearchNode, moveLoc: number): number {
    if (parent !== this.root || moveLoc === NULL_LOC) return 0;
    const e = parent.eval;
    if (!e) return 0;
    const h = this.root.hist;
    const b = h.board;
    const rootPla = h.nextPla;
    const bonus = this.params.rootEndingBonusPoints;
    const extreme = 0.95;
    const tail = 0.05;
    const isAreaIsh = h.rules.scoringRule === 'AREA' || (h.rules.scoringRule === 'TERRITORY' && h.encorePhase >= 2);
    let extraRootPoints = 0;
    const ownershipPenalty = (): void => {
      const own = e.whiteOwnership[b.y(moveLoc) * b.xSize + b.x(moveLoc)]!;
      const plaOwnership = rootPla === WHITE ? own : -own;
      if (plaOwnership <= -extreme) {
        if (isAreaIsh ? !b.wouldBeCapture(moveLoc, rootPla) : true) extraRootPoints -= bonus * ((-extreme - plaOwnership) / tail);
      } else if (plaOwnership >= extreme) {
        if (!b.isAdjacentToPla(moveLoc, opp(rootPla)) && !b.isNonPassAliveSelfConnection(moveLoc, rootPla, this.#rootSafeArea)) {
          extraRootPoints -= bonus * ((plaOwnership - extreme) / tail);
        }
      }
    };
    if (isAreaIsh) {
      // (Rules with a button, which add a penalty for passing here, aren't supported.)
      if (moveLoc !== PASS_LOC && b.koLoc === NULL_LOC) ownershipPenalty();
    } else if (moveLoc === PASS_LOC) {
      extraRootPoints -= bonus * (2 / 3);
    } else if (b.koLoc === NULL_LOC) {
      ownershipPenalty();
    }
    return rootPla === WHITE ? extraRootPoints : -extraRootPoints;
  }

  /**
   * KataGo `isAllowedRootMove`: with `rootPruneUselessMoves`, once the opponent's last four moves were passes, no move
   * inside either player's pass-alive area; with `rootSymmetryPruning`, no move that mirrors another.
   */
  #isAllowedRootMove(moveLoc: number): boolean {
    if (moveLoc === PASS_LOC) return true;
    // The root's history doesn't change during a search, so whether the opponent's last four moves were passes is
    // worked out once, in #prepareRoot.
    if (this.#rootPrunesUselessMoves && this.#rootSafeArea[moveLoc] !== EMPTY) return false;
    if (this.params.rootSymmetryPruning && this.#rootSymDupLoc[moveLoc]) return false;
    return true;
  }

  #stateKey(h: GameHistory, pla: Stone): string {
    return graphStateKey(h, pla, this.params.drawEquivalentWinsForWhite);
  }

  /**
   * KataGo `GraphHash::getGraphHash` as a key: the state alone, unless the last move could be part of a repetition
   * within `graphSearchRepBound` moves (a pass always could), and then the state after the previous key.
   */
  #graphKey(prevKey: string, h: GameHistory, pla: Stone): string {
    const moves = h.moves;
    const prevMoveLoc = moves.length === 0 ? NULL_LOC : moves[moves.length - 1]!.loc;
    const state = this.#stateKey(h, pla);
    if (prevMoveLoc === NULL_LOC || h.board.simpleRepetitionBoundGt(prevMoveLoc, this.params.graphSearchRepBound)) return state;
    let id = this.#graphKeyIds.get(prevKey);
    if (id === undefined) {
      id = this.#graphKeyIds.size;
      this.#graphKeyIds.set(prevKey, id);
    }
    // A state key has no '>', so a key that depends on its parent's never equals one that doesn't.
    return `${id}>${state}`;
  }

  /** KataGo `GraphHash::getGraphHashFromScratch`: the key of the root, through the game's moves from the start. */
  #rootGraphKey(h: GameHistory): string {
    const replay = new GameHistory(h.initialBoard, h.moves[0]?.pla ?? h.nextPla, h.rules, h.komi, h.initialEncorePhase, h.modes);
    let key = '';
    h.moves.forEach((m, i) => {
      key = this.#graphKey(key, replay, m.pla);
      // KataGo getGraphHashFromScratch replays with makeBoardMoveTolerant, as a query's moves were played.
      if (!replay.playTolerant(m.loc, m.pla, h.preventEncoreHistory[i])) throw new Error('the root history does not replay');
    });
    return this.#graphKey(key, replay, h.nextPla);
  }

  /**
   * KataGo `allocateOrFindNode`: with graph search, the node already in the table for this position, if any; otherwise a
   * new node, with its subtree value bias entry.
   */
  #allocateOrFindNode(parent: SearchNode, hist: GameHistory, moveLoc: number, forceNonTerminal: boolean): SearchNode {
    const p = this.params;
    let graphKey = '';
    let tableKey = '';
    if (p.useGraphSearch) {
      graphKey = this.#graphKey(parent.graphKey, hist, hist.nextPla);
      tableKey = forceNonTerminal ? graphKey + '|F' : graphKey;
      // (KataGo also checks the player to move, against hash collisions; the key includes it.)
      const found = this.#nodeTable.get(tableKey);
      if (found) return found;
    }
    const child = new SearchNode(hist, parent, hist.isGameFinished && !forceNonTerminal, graphKey);
    // The subtree value bias entry: this move, after the move before it, unless it's a pass.
    if (p.subtreeValueBiasFactor !== 0 && hist.moves.length >= 2 && moveLoc !== PASS_LOC) {
      const parentPrevMoveLoc = hist.moves[hist.moves.length - 2]!.loc;
      if (parentPrevMoveLoc !== NULL_LOC) {
        const key = subtreeValueBiasKey(parent.nextPla, parentPrevMoveLoc, moveLoc, hist.recentBoard(1));
        let biasEntry = this.#subtreeValueBiasTable.get(key);
        if (!biasEntry) {
          biasEntry = { deltaUtilitySum: 0, weightSum: 0 };
          this.#subtreeValueBiasTable.set(key, biasEntry);
        }
        child.biasEntry = biasEntry;
      }
    }
    if (p.useGraphSearch) this.#nodeTable.set(tableKey, child);
    return child;
  }

  /** KataGo `getChildWeight(edgeVisits, childVisits)`: the child's weight as its parent counts it. */
  #childWeight(e: SearchEdge): number {
    return e.node.weightSum * (e.edgeVisits / Math.max(e.node.visits, 1));
  }

  #childWeightSq(e: SearchEdge): number {
    return e.node.weightSqSum * (e.edgeVisits / Math.max(e.node.visits, 1));
  }

  #approxScoreUtilityDerivative(scoreMean: number): number {
    const p = this.params;
    const staticDerivative = whiteDScoreValueDScoreSmooth(scoreMean, 0, 2, this.#sqrtBoardArea);
    const dynamicDerivative = whiteDScoreValueDScoreSmooth(scoreMean, this.#recentScoreCenter, p.dynamicScoreCenterScale, this.#sqrtBoardArea);
    return staticDerivative * p.staticScoreUtilityFactor + dynamicDerivative * p.dynamicScoreUtilityFactor;
  }

  /** Whether evaluations are weighted by uncertainty: KataGo needs short-term error outputs (`supportsShorttermError`). */
  get #usesUncertainty(): boolean {
    return this.params.useUncertainty && this.#model.modelVersion >= 9;
  }

  /** KataGo `computeWeightFromNNOutput`. */
  #weightFromNN(e: Evaluation): number {
    if (!this.#usesUncertainty) return 1;
    const p = this.params;
    const utilityUncertaintyWL = p.winLossUtilityFactor * e.shorttermWinlossError;
    const utilityUncertaintyScore = this.#approxScoreUtilityDerivative(e.whiteScoreMean) * e.shorttermScoreError;
    const utilityUncertainty = utilityUncertaintyWL + utilityUncertaintyScore;
    let powered: number;
    if (p.uncertaintyExponent === 1) powered = utilityUncertainty;
    else if (p.uncertaintyExponent === 0.5) powered = Math.sqrt(utilityUncertainty);
    else powered = Math.pow(utilityUncertainty, p.uncertaintyExponent);
    const baselineUncertainty = p.uncertaintyCoeff / p.uncertaintyMaxWeight;
    return p.uncertaintyCoeff / (powered + baselineUncertainty);
  }

  #utilityFromNN(e: Evaluation): number {
    return this.#resultUtility(e.whiteWinProb - e.whiteLossProb, e.whiteNoResultProb) + this.#scoreUtility(e.whiteScoreMean, e.whiteScoreMeanSq);
  }

  // ---------------------------------------------------------------- evaluation

  #maxHistory(node: SearchNode): number {
    if (!this.params.ignorePreRootHistory) return 5;
    return Math.max(0, node.hist.moves.length - this.root.hist.moves.length);
  }

  /**
   * KataGo's evaluation-cache key (`NNInputs::getHash`): the board size and the situation (`situationKey`), whether a
   * pass would end the phase (and friendly-pass suppression), whether the game is over or past a prevented phase end, and
   * whether history is hidden. Like KataGo, it leaves out the move history, so transpositions share an evaluation.
   */
  #cacheKey(node: SearchNode): string {
    const h = node.hist;
    const b = h.board;
    const pla = h.nextPla;
    // The cache can be shared between searches (KataGo's analysis engine shares one across a query's turns), so the
    // board size is in the key.
    let key = `${b.xSize}x${b.ySize}|${situationKey(h, pla, this.params.drawEquivalentWinsForWhite)}`;
    if (h.passWouldEndPhase(pla)) {
      key += '|p';
      if (this.params.conservativePass && node === this.root) key += 'c';
      if (h.shouldSuppressEndGameFromFriendlyPass(pla)) key += 'f';
      if (this.params.enablePassingHacks) key += 'h';
    }
    if (h.isGameFinished || h.isPastNormalPhaseEnd) key += '|over';
    if (this.#maxHistory(node) <= 0) key += '|nohist';
    // KataGo folds the policy optimism in, discretized to 1/1024.
    const optimism = this.#policyOptimism(node);
    if (optimism > 0) key += `|opt${Math.trunc(optimism * 1024)}`;
    return key;
  }

  #policyOptimism(node: SearchNode): number {
    return node === this.root ? this.params.rootPolicyOptimism : this.params.policyOptimism;
  }

  #encode(node: SearchNode): ReturnType<typeof encodeHistoryV7> {
    return encodeHistoryV7(node.hist, {
      ladders: this.#ladders,
      preferPassAliveUnderSuicideRules: this.#model.preferPassAliveUnderSuicideRules,
      maxHistory: this.#maxHistory(node),
      conservativePassAndIsRoot: this.params.conservativePass && node === this.root,
      enablePassingHacks: this.params.enablePassingHacks,
      drawEquivalentWinsForWhite: this.params.drawEquivalentWinsForWhite,
    });
  }

  /** KataGo `Rand::nextGaussian`: Marsaglia's polar method. */
  #nextGaussian(): number {
    if (this.#storedGaussian !== null) {
      const g = this.#storedGaussian;
      this.#storedGaussian = null;
      return g;
    }
    let v1: number;
    let v2: number;
    let s: number;
    do {
      v1 = this.#random() * 2 - 1;
      v2 = this.#random() * 2 - 1;
      s = v1 * v1 + v2 * v2;
    } while (s >= 1 || s === 0);
    const multiplier = Math.sqrt((-2 * Math.log(s)) / s);
    this.#storedGaussian = v2 * multiplier;
    return v1 * multiplier;
  }

  /**
   * KataGo `maybeApplyWideRootNoise`: the policy flattened (as a 32-bit float, as KataGo keeps it) and, half the time, a
   * random bonus on the value for the player to move.
   */
  #wideRootNoise(policy: number, utility: number, pla: Stone): { policy: number; utility: number } {
    const w = this.params.wideRootNoise;
    const flattened = Math.fround(Math.pow(policy, 1 / (4 * w + 1)));
    if (this.#random() < 0.5) {
      const bonus = w * Math.abs(this.#nextGaussian());
      utility += pla === WHITE ? bonus : -bonus;
    }
    return { policy: flattened, utility };
  }

  #nextSymmetry(): number {
    if (this.params.symmetry >= 0) return this.params.symmetry;
    const b = this.root.hist.board;
    return Math.floor(this.#random() * (b.xSize === b.ySize ? 8 : 4));
  }

  #postProcess(raw: RawOutputs, symmetry: number, encoded: ReturnType<typeof encodeHistoryV7>, policyOptimism: number): Evaluation {
    const { xSize, ySize } = raw;
    const back = (plane: Float32Array) => copyWithSymmetry(plane, 1, ySize, xSize, symmetry, true);
    const unrotated: RawOutputs = { ...raw, policy: raw.policy.map(back), ownership: back(raw.ownership) };
    return postProcess(this.#model, unrotated, {
      nextPlayer: encoded.nextPlayer,
      legal: encoded.legal,
      koRuleSimple: encoded.koRuleSimple,
      scoringTerritory: encoded.scoringTerritory,
      policyOptimism,
      enablePassingHacks: this.params.enablePassingHacks,
    });
  }

  async #evaluateNow(node: SearchNode): Promise<Evaluation> {
    const key = this.#cacheKey(node);
    const cached = this.#cache.get(key);
    if (cached) return cached;
    const encoded = this.#encode(node);
    const symmetry = this.#nextSymmetry();
    const { xSize, ySize } = encoded.input;
    const spatial = copyWithSymmetry(encoded.input.spatial, this.#model.numInputChannels, ySize, xSize, symmetry, false);
    const t0 = performance.now();
    const [raw] = await this.#evaluator.evaluate([{ xSize, ySize, spatial, global: encoded.input.global }]);
    this.stats.nnMs += performance.now() - t0;
    this.stats.nnEvals += 1;
    const e = this.#postProcess(raw!, symmetry, encoded, this.#policyOptimism(node));
    this.#cache.set(key, e);
    return e;
  }

  /** KataGo `computeRootValues`: the dynamic score utility is centred on the root's expected score. */
  async #prepareRoot(): Promise<void> {
    if (this.#rootPrepared) return;
    const e = await this.#evaluateNow(this.root);
    const expectedScore = this.root.visits > 0 ? this.root.scoreMeanAvg : e.whiteScoreMean;
    const p = this.params;
    let center = expectedScore * (1 - p.dynamicScoreCenterZeroWeight);
    const cap = this.#sqrtBoardArea * p.dynamicScoreCenterScale;
    center = Math.min(expectedScore + cap, Math.max(expectedScore - cap, center));
    this.#recentScoreCenter = center;
    const rootHist = this.root.hist;
    const suicideLegalForPassAlive = rootHist.rules.multiStoneSuicideLegal || this.#model.preferPassAliveUnderSuicideRules;
    this.#rootSafeArea = calculateArea(rootHist.board, false, false, false, suicideLegalForPassAlive);
    const moves = rootHist.moves;
    const lastIdx = moves.length - 1;
    const oppPassed = (i: number) => moves[i]!.loc === PASS_LOC && moves[i]!.pla === opp(rootHist.nextPla);
    this.#rootPrunesUselessMoves =
      p.rootPruneUselessMoves && lastIdx >= 6 && oppPassed(lastIdx) && oppPassed(lastIdx - 2) && oppPassed(lastIdx - 4) && oppPassed(lastIdx - 6);
    if (p.rootSymmetryPruning) {
      const marked = markDuplicateMoveLocs(rootHist, this.#avoidMoveUntilByLoc[rootHist.nextPla]);
      this.#rootSymDupLoc = marked.isSymDupLoc;
      this.#rootSymmetries = marked.validSymmetries;
    } else {
      this.#rootSymDupLoc = new Uint8Array(rootHist.board.arraySize);
      this.#rootSymmetries = [0];
    }
    this.#rootSearchPolicy = this.#rootPolicyWithTemperature(e.policy);
    if (this.root.state !== EVALUATED) {
      this.root.eval = e;
      this.root.state = EVALUATED;
      this.#addNNOutputAsLeafValue(this.root);
      this.stats.visits += 1;
    }
    this.#rootPrepared = true;
  }

  // ---------------------------------------------------------------- stats

  /** KataGo `addLeafValue`, for a node with no children yet (its first visit) or a terminal node. */
  #addLeafValue(
    node: SearchNode,
    winLoss: number,
    noResult: number,
    scoreMean: number,
    scoreMeanSq: number,
    lead: number,
    weight: number,
    isTerminal: boolean,
  ): void {
    let utility = this.#resultUtility(winLoss, noResult) + this.#scoreUtility(scoreMean, scoreMeanSq);
    const entry = node.biasEntry;
    if (this.params.subtreeValueBiasFactor !== 0 && !isTerminal && entry && entry.weightSum > 0.001) {
      utility += (this.params.subtreeValueBiasFactor * entry.deltaUtilitySum) / entry.weightSum;
    }
    const oldW = node.weightSum;
    const newW = oldW + weight;
    const mix = (avg: number, v: number) => (avg * oldW + v * weight) / newW;
    node.winLossValueAvg = mix(node.winLossValueAvg, winLoss);
    node.noResultValueAvg = mix(node.noResultValueAvg, noResult);
    node.scoreMeanAvg = mix(node.scoreMeanAvg, scoreMean);
    node.scoreMeanSqAvg = mix(node.scoreMeanSqAvg, scoreMeanSq);
    node.leadAvg = mix(node.leadAvg, lead);
    node.utilityAvg = mix(node.utilityAvg, utility);
    node.utilitySqAvg = mix(node.utilitySqAvg, utility * utility);
    node.weightSqSum += weight * weight;
    node.weightSum = newW;
    node.visits += 1;
  }

  #addNNOutputAsLeafValue(node: SearchNode): void {
    const e = node.eval!;
    this.#addLeafValue(node, e.whiteWinProb - e.whiteLossProb, e.whiteNoResultProb, e.whiteScoreMean, e.whiteScoreMeanSq, e.whiteLead, this.#weightFromNN(e), false);
  }

  /** KataGo's terminal value in `playoutDescend`, from the history of the path that reached the node. */
  #addTerminalValue(node: SearchNode, h: GameHistory): void {
    const p = this.params;
    const weight = this.#usesUncertainty ? p.uncertaintyMaxWeight : 1;
    if (h.isNoResult) {
      this.#addLeafValue(node, 0, 1, 0, 0, 0, weight, true);
      return;
    }
    const whiteWins = h.winner === WHITE ? 1 : h.winner === BLACK ? 0 : p.drawEquivalentWinsForWhite;
    // KataGo whiteScoreDrawAdjust: the draw adjustment is zero unless the result can be a draw (integer komi).
    const drawAdjust = Number.isInteger(h.komi) ? p.drawEquivalentWinsForWhite - 0.5 : 0;
    const scoreMean = h.finalWhiteMinusBlackScore + drawAdjust;
    const scoreMeanSq = whiteScoreMeanSqOfScoreGridded(h.finalWhiteMinusBlackScore, p.drawEquivalentWinsForWhite);
    this.#addLeafValue(node, 2 * whiteWins - 1, 0, scoreMean, scoreMeanSq, scoreMean, weight, true);
  }

  /**
   * KataGo `downweightBadChildrenAndNormalizeWeight`: scales `weights` (each child's, for children with visits) by
   * value weighting, after pruning those below `amountToPrune` and subtracting `amountToSubtract`, so they sum to
   * `desiredTotalWeight`. `selfUtilities` are from the point of view of the player to move.
   */
  #downweightBadChildrenAndNormalizeWeight(
    weights: number[],
    selfUtilities: readonly number[],
    currentTotalWeight: number,
    desiredTotalWeight: number,
    amountToSubtract: number,
    amountToPrune: number,
  ): void {
    const n = weights.length;
    if (n <= 0 || currentTotalWeight <= 0) return;
    /** Prunes or reduces child i's weight; true if it was pruned. */
    const pruneAndSubtract = (i: number): boolean => {
      if (weights[i]! < amountToPrune) {
        currentTotalWeight -= weights[i]!;
        weights[i] = 0;
        return true;
      }
      const newWeight = weights[i]! - amountToSubtract;
      if (newWeight <= 0) {
        currentTotalWeight -= weights[i]!;
        weights[i] = 0;
      } else {
        currentTotalWeight -= amountToSubtract;
        weights[i] = newWeight;
      }
      return false;
    };
    if (this.params.valueWeightExponent === 0) {
      for (let i = 0; i < n; i++) pruneAndSubtract(i);
      if (currentTotalWeight !== desiredTotalWeight) {
        const factor = desiredTotalWeight / currentTotalWeight;
        for (let i = 0; i < n; i++) weights[i]! *= factor;
      }
      return;
    }
    const stdevs = new Array<number>(n);
    let simpleValueSum = 0;
    for (let i = 0; i < n; i++) {
      const weight = weights[i]!;
      const precision = 1.5 * Math.sqrt(weight);
      // Some minimum variance for stability.
      stdevs[i] = Math.sqrt(0.00000001 + 1 / precision);
      simpleValueSum += selfUtilities[i]! * weight;
    }
    const simpleValue = simpleValueSum / currentTotalWeight;
    let totalNewUnnormWeight = 0;
    for (let i = 0; i < n; i++) {
      if (pruneAndSubtract(i)) continue;
      const z = (selfUtilities[i]! - simpleValue) / stdevs[i]!;
      // A tiny minimum, for numeric sanity.
      const p = VALUE_WEIGHT_DISTRIBUTION.getCdf(z) + 0.0001;
      weights[i]! *= Math.pow(p, this.params.valueWeightExponent);
      totalNewUnnormWeight += weights[i]!;
    }
    const factor = desiredTotalWeight / totalNewUnnormWeight;
    for (let i = 0; i < n; i++) weights[i]! *= factor;
  }

  /**
   * KataGo `pruneNoiseWeight`: reduces `weights` of children worse than the weighted average of those before them, if
   * they have more than twice their raw-policy share of the weight so far. Returns the new total weight.
   */
  #pruneNoiseWeight(weights: number[], selfUtilities: readonly number[], totalChildWeight: number, policyProbs: readonly number[]): number {
    const n = weights.length;
    if (n <= 1 || totalChildWeight <= 0.00001) return totalChildWeight;
    const p = this.params;
    let utilitySumSoFar = 0;
    let weightSumSoFar = 0;
    let rawPolicySumSoFar = 0;
    for (let i = 0; i < n; i++) {
      const utility = selfUtilities[i]!;
      const oldWeight = weights[i]!;
      const rawPolicy = policyProbs[i]!;
      let newWeight = oldWeight;
      if (weightSumSoFar > 0 && rawPolicySumSoFar > 0) {
        const utilityGap = utilitySumSoFar / weightSumSoFar - utility;
        if (utilityGap > 0) {
          // More than double its proper share of the weight.
          const weightShareFromRawPolicy = (weightSumSoFar * rawPolicy) / rawPolicySumSoFar;
          const lenientShare = 2 * weightShareFromRawPolicy;
          if (oldWeight > lenientShare) {
            let weightToSubtract = (oldWeight - lenientShare) * (1 - Math.exp(-utilityGap / p.noisePruneUtilityScale));
            if (weightToSubtract > p.noisePruningCap) weightToSubtract = p.noisePruningCap;
            newWeight = oldWeight - weightToSubtract;
            weights[i] = newWeight;
          }
        }
      }
      utilitySumSoFar += utility * newWeight;
      weightSumSoFar += newWeight;
      rawPolicySumSoFar += rawPolicy;
    }
    return weightSumSoFar;
  }

  /**
   * KataGo `recomputeNodeStats`: the children's averages weighted by their weights after noise pruning and value
   * weighting, plus the node's own evaluation with its uncertainty weight.
   */
  #recomputeNodeStats(node: SearchNode): void {
    const good: SearchNode[] = [];
    const weights: number[] = [];
    const selfUtilities: number[] = [];
    let totalChildWeight = 0;
    const goodMoves: number[] = [];
    for (const e of node.children) {
      const c = e.node;
      if (c.visits <= 0 || c.weightSum <= 0 || e.edgeVisits <= 0) continue;
      const w = this.#childWeight(e);
      good.push(c);
      goodMoves.push(e.moveLoc);
      weights.push(w);
      selfUtilities.push(node.nextPla === WHITE ? c.utilityAvg : -c.utilityAvg);
      totalChildWeight += w;
    }
    const origTotalChildWeight = totalChildWeight;
    if (this.params.useNoisePruning && good.length > 0) {
      const policy = this.#searchPolicy(node);
      const b = node.hist.board;
      const policyProbs = goodMoves.map((loc) => Math.max(1e-30, policy[this.#posOf(b, loc)]!));
      totalChildWeight = this.#pruneNoiseWeight(weights, selfUtilities, totalChildWeight, policyProbs);
    }
    // KataGo prunes and subtracts here only with root noise, which the analysis engine doesn't use.
    this.#downweightBadChildrenAndNormalizeWeight(weights, selfUtilities, totalChildWeight, totalChildWeight, 0, 0);

    let winLoss = 0;
    let noResult = 0;
    let scoreMean = 0;
    let scoreMeanSq = 0;
    let lead = 0;
    let utility = 0;
    let utilitySq = 0;
    let weightSqSum = 0;
    let weightSum = totalChildWeight;
    for (let i = 0; i < good.length; i++) {
      const c = good[i]!;
      const w = weights[i]!;
      const scaling = w / c.weightSum;
      winLoss += w * c.winLossValueAvg;
      noResult += w * c.noResultValueAvg;
      scoreMean += w * c.scoreMeanAvg;
      scoreMeanSq += w * c.scoreMeanSqAvg;
      lead += w * c.leadAvg;
      utility += w * c.utilityAvg;
      utilitySq += w * c.utilitySqAvg;
      weightSqSum += scaling * scaling * c.weightSqSum;
    }
    const e = node.eval!;
    let u = this.#utilityFromNN(e);
    const entry = node.biasEntry;
    if (this.params.subtreeValueBiasFactor !== 0 && entry) {
      if (totalChildWeight > 1e-10) {
        // This node's correction: how far its children's value is from its own evaluation.
        const utilityChildren = utility / totalChildWeight;
        const biasWeight = Math.pow(origTotalChildWeight, this.params.subtreeValueBiasWeightExponent);
        const biasDeltaSum = (utilityChildren - u) * biasWeight;
        entry.deltaUtilitySum += biasDeltaSum - node.lastBiasDeltaSum;
        entry.weightSum += biasWeight - node.lastBiasWeight;
        node.lastBiasDeltaSum = biasDeltaSum;
        node.lastBiasWeight = biasWeight;
      }
      if (entry.weightSum > 0.001) u += (this.params.subtreeValueBiasFactor * entry.deltaUtilitySum) / entry.weightSum;
    }
    const weight = this.#weightFromNN(e);
    winLoss += (e.whiteWinProb - e.whiteLossProb) * weight;
    noResult += e.whiteNoResultProb * weight;
    scoreMean += e.whiteScoreMean * weight;
    scoreMeanSq += e.whiteScoreMeanSq * weight;
    lead += e.whiteLead * weight;
    utility += u * weight;
    utilitySq += u * u * weight;
    weightSqSum += weight * weight;
    weightSum += weight;
    node.winLossValueAvg = winLoss / weightSum;
    node.noResultValueAvg = noResult / weightSum;
    node.scoreMeanAvg = scoreMean / weightSum;
    node.scoreMeanSqAvg = scoreMeanSq / weightSum;
    node.leadAvg = lead / weightSum;
    node.utilityAvg = utility / weightSum;
    node.utilitySqAvg = utilitySq / weightSum;
    node.weightSqSum = weightSqSum;
    node.weightSum = weightSum;
    node.visits += 1;
  }

  // ---------------------------------------------------------------- selection

  #fpuValue(node: SearchNode, isRoot: boolean, policyProbMassVisited: number): { fpu: number; stdevFactor: number } {
    const p = this.params;
    const parentUtility = node.utilityAvg;
    const variancePrior = p.cpuctUtilityStdevPrior * p.cpuctUtilityStdevPrior;
    let parentUtilityStdev: number;
    if (node.visits <= 0 || node.weightSum <= 1) parentUtilityStdev = p.cpuctUtilityStdevPrior;
    else {
      const utilitySq = parentUtility * parentUtility;
      const utilitySqAvg = Math.max(node.utilitySqAvg, utilitySq);
      parentUtilityStdev = Math.sqrt(
        Math.max(
          0,
          ((utilitySq + variancePrior) * p.cpuctUtilityStdevPriorWeight + utilitySqAvg * node.weightSum) /
            (p.cpuctUtilityStdevPriorWeight + node.weightSum - 1) -
            utilitySq,
        ),
      );
    }
    const stdevFactor = 1 + p.cpuctUtilityStdevScale * (parentUtilityStdev / p.cpuctUtilityStdevPrior - 1);

    let parentUtilityForFpu = parentUtility;
    if (p.fpuParentWeightByVisitedPolicy) {
      const avgWeight = Math.min(1, Math.pow(policyProbMassVisited, p.fpuParentWeightByVisitedPolicyPow));
      parentUtilityForFpu = avgWeight * parentUtility + (1 - avgWeight) * this.#utilityFromNN(node.eval!);
    } else if (p.fpuParentWeight > 0) {
      parentUtilityForFpu = p.fpuParentWeight * this.#utilityFromNN(node.eval!) + (1 - p.fpuParentWeight) * parentUtility;
    }
    const reductionMax = isRoot ? p.rootFpuReductionMax : p.fpuReductionMax;
    const utilityRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;
    const reduction = reductionMax * Math.sqrt(policyProbMassVisited);
    let fpu = node.nextPla === WHITE ? parentUtilityForFpu - reduction : parentUtilityForFpu + reduction;
    const lossValue = node.nextPla === WHITE ? -utilityRadius : utilityRadius;
    fpu += (lossValue - fpu) * (isRoot ? p.rootFpuLossProp : p.fpuLossProp);
    return { fpu, stdevFactor };
  }

  #selectionValue(exploreScaling: number, policy: number, childWeight: number, childUtility: number, pla: Stone): number {
    if (policy < 0) return POLICY_ILLEGAL_SELECTION_VALUE;
    const explore = (exploreScaling * policy) / (1 + childWeight);
    return explore + (pla === WHITE ? childUtility : -childUtility);
  }

  /**
   * KataGo `selectBestChildToDescend` (and the child creation of `playoutDescend`), without the features listed at the
   * top of this file. Returns the edge to the child and whether the parent counts this visit to it (`countEdgeVisit`).
   * `pathHistory` gives the history of the path to `node`, for a new child.
   */
  #selectChild(node: SearchNode, isRoot: boolean, depth: number, pathHistory: () => GameHistory): { edge: SearchEdge; countEdgeVisit: boolean } {
    const p = this.params;
    const policy = this.#searchPolicy(node);
    const b = node.hist.board;
    const posOf = (loc: number) => (loc === PASS_LOC ? this.#area : b.y(loc) * b.xSize + b.x(loc));

    let policyProbMassVisited = 0;
    let totalChildWeight = 0;
    let totalChildEdgeVisits = 0;
    for (const e of node.children) {
      const pr = policy[posOf(e.moveLoc)]!;
      if (pr < 0) continue;
      policyProbMassVisited += pr;
      totalChildWeight += this.#childWeight(e);
      totalChildEdgeVisits += e.edgeVisits;
    }
    const { fpu, stdevFactor } = this.#fpuValue(node, isRoot, policyProbMassVisited);
    const cpuct = p.cpuctExploration + p.cpuctExplorationLog * Math.log((totalChildWeight + p.cpuctExplorationBase) / p.cpuctExplorationBase);
    const exploreScaling = cpuct * Math.sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET) * stdevFactor;
    const pla = node.nextPla;
    const utilityRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;
    const avoidMoveUntilByLoc = this.#avoidMoveUntilByLoc[pla];

    let best: SearchEdge | null = null;
    let bestValue = POLICY_ILLEGAL_SELECTION_VALUE;
    const tried = new Uint8Array(this.#area + 1);
    for (const e of node.children) {
      const c = e.node;
      const pos = posOf(e.moveLoc);
      tried[pos] = 1;
      let childWeight = this.#childWeight(e);
      let childUtility: number;
      if (c.visits <= 0 || childWeight <= 0) childUtility = fpu;
      else {
        childUtility = c.utilityAvg;
        // KataGo's "tiny adjustment for passing": the root's ending bonus.
        const endingScoreBonus = this.#endingWhiteScoreBonus(node, e.moveLoc);
        if (endingScoreBonus !== 0) childUtility += this.#scoreUtilityDiff(c.scoreMeanAvg, c.scoreMeanSqAvg, endingScoreBonus);
      }
      if (c.virtualLosses > 0) {
        const vlWeight = c.virtualLosses * p.numVirtualLossesPerThread;
        const vlUtility = pla === WHITE ? -utilityRadius : utilityRadius;
        const frac = vlWeight / (vlWeight + Math.max(0.25, childWeight));
        childUtility += (vlUtility - childUtility) * frac;
        childWeight += vlWeight;
      }
      let childPolicy = policy[pos]!;
      if (isRoot && p.wideRootNoise > 0 && childPolicy >= 0) {
        const noised = this.#wideRootNoise(childPolicy, childUtility, pla);
        childPolicy = noised.policy;
        childUtility = noised.utility;
      }
      const value = this.#selectionValue(exploreScaling, childPolicy, childWeight, childUtility, pla);
      if (value > bestValue) {
        bestValue = value;
        best = e;
      }
    }
    // Of the moves with no child yet, only the one with the highest policy can be best (they share the FPU value).
    let bestNewLoc = NULL_LOC;
    let bestNewPolicy = -1;
    for (let pos = 0; pos <= this.#area; pos++) {
      if (tried[pos]) continue;
      const pr = policy[pos]!;
      if (pr < 0) continue;
      if (pr > bestNewPolicy) {
        // Checking the root's allowed moves only for a move that would be the best so far picks the same move.
        const loc = pos === this.#area ? PASS_LOC : b.loc(pos % b.xSize, Math.floor(pos / b.xSize));
        if (isRoot && !this.#isAllowedRootMove(loc)) continue;
        if (avoidMoveUntilByLoc && depth < avoidMoveUntilByLoc[loc]!) continue;
        bestNewPolicy = pr;
        bestNewLoc = loc;
      }
    }
    let newLoc = NULL_LOC;
    if (bestNewLoc !== NULL_LOC) {
      // KataGo getNewExploreSelectionValue: wide root noise applies to the best new move too.
      let newPolicy = bestNewPolicy;
      let newUtility = fpu;
      if (isRoot && p.wideRootNoise > 0) {
        const noised = this.#wideRootNoise(newPolicy, newUtility, pla);
        newPolicy = noised.policy;
        newUtility = noised.utility;
      }
      if (this.#selectionValue(exploreScaling, newPolicy, 0, newUtility, pla) > bestValue) newLoc = bestNewLoc;
    }
    const bestLoc = newLoc !== NULL_LOC ? newLoc : best ? best.moveLoc : NULL_LOC;

    // KataGo enableMorePassingHacks: where a pass would end the phase, make sure both a pass and a move other than a pass
    // get tried, by a playout the parent doesn't count. (Whether a pass would end the phase is part of the graph key, so
    // the node's own history answers it for every path to the node.) Not with avoided moves for the player, which a
    // forced playout could try.
    let countEdgeVisit = true;
    if (totalChildEdgeVisits >= 2 && p.enableMorePassingHacks && node.hist.passWouldEndPhase(pla) && !avoidMoveUntilByLoc) {
      let hasPassMove = false;
      let hasNonPassMove = false;
      for (const e of node.children) {
        if (e.moveLoc === PASS_LOC) hasPassMove = true;
        else hasNonPassMove = true;
      }
      if (!hasPassMove && bestLoc !== PASS_LOC && bestLoc !== NULL_LOC) {
        newLoc = PASS_LOC;
        countEdgeVisit = false;
      } else if (!hasNonPassMove && bestLoc === PASS_LOC && bestNewLoc !== PASS_LOC && bestNewLoc !== NULL_LOC) {
        newLoc = bestNewLoc;
        countEdgeVisit = false;
      }
    }

    if (newLoc !== NULL_LOC) {
      // KataGo playoutDescend: the move is made on the path's history. A pass that ends the game isn't terminal from the
      // root under conservativePass, or where friendly-pass rules suppress the end of the game.
      const parentHist = pathHistory();
      const friendlyPass = newLoc === PASS_LOC && parentHist.shouldSuppressEndGameFromFriendlyPass(pla);
      const hist = parentHist.clone();
      hist.play(newLoc, pla);
      const forceNonTerminal = newLoc === PASS_LOC && hist.isGameFinished && ((p.conservativePass && isRoot) || friendlyPass);
      const edge = new SearchEdge(this.#allocateOrFindNode(node, hist, newLoc, forceNonTerminal), newLoc);
      node.children.push(edge);
      return { edge, countEdgeVisit };
    }
    if (!best) throw new Error('no legal move to search');
    return { edge: best, countEdgeVisit };
  }

  // ---------------------------------------------------------------- playouts

  /**
   * KataGo's return up `playoutDescend`: each edge on the path gains a visit and each parent is recomputed, up to the
   * deepest edge that doesn't count this visit (`enableMorePassingHacks`), if any: from there up nothing changes.
   */
  #backup(nodes: SearchNode[], edges: SearchEdge[], uncountedIdx: number): void {
    for (let i = edges.length - 1; i > uncountedIdx; i--) {
      edges[i]!.edgeVisits += 1;
      this.#recomputeNodeStats(nodes[i]!);
    }
    if (uncountedIdx < 0) this.stats.visits += 1;
  }

  /**
   * Descend from the root to a leaf. Terminal leaves and cached evaluations are backed up at once; a leaf that needs the
   * network is returned with virtual losses added along its path.
   */
  #descend(): Pending | 'done' | 'collision' {
    const nodes: SearchNode[] = [this.root];
    const edges: SearchEdge[] = [];
    // The deepest node on the path whose creator chain is the path itself, so its history is the path's.
    let exactIdx = 0;
    // KataGo's thread.history at nodes[i]: the node's own history if the path created it, otherwise replayed.
    const pathHistory = (i: number): GameHistory => {
      if (exactIdx === i) return nodes[i]!.hist;
      const h = nodes[exactIdx]!.hist.clone();
      for (let j = exactIdx; j < i; j++) h.play(edges[j]!.moveLoc, h.nextPla);
      return h;
    };
    const pathHistoryHere = (): GameHistory => pathHistory(nodes.length - 1);
    let node = this.root;
    // The deepest edge on the path that doesn't count this visit, or -1.
    let uncountedIdx = -1;
    while (node.state === EVALUATED && !node.terminal) {
      const depth = nodes.length - 1;
      const sel = this.#selectChild(node, node === this.root, depth, pathHistoryHere);
      const child = sel.edge.node;
      nodes.push(child);
      edges.push(sel.edge);
      if (exactIdx === depth && child.creator === node) exactIdx = depth + 1;
      if (!sel.countEdgeVisit) uncountedIdx = depth;
      // KataGo maybeCatchUpEdgeVisits: while the parent has counted fewer visits to a child than the child has, a visit
      // just counts one more, without descending.
      else if (sel.edge.edgeVisits < child.visits) {
        this.#backup(nodes, edges, uncountedIdx);
        return 'done';
      }
      // KataGo's thread.graphPath: in a cycle (a node already entered on this path, the root aside), the edge counts a
      // visit (if this visit counts there) and the playout ends.
      if (nodes.indexOf(child, 1) < nodes.length - 1) {
        this.#backup(nodes, edges, uncountedIdx);
        return 'done';
      }
      node = child;
    }
    if (node.terminal) {
      this.#addTerminalValue(node, pathHistory(nodes.length - 1));
      this.#backup(nodes, edges, uncountedIdx);
      this.stats.terminalVisits += 1;
      return 'done';
    }
    if (node.state === PENDING) {
      this.stats.collisions += 1;
      return 'collision';
    }
    const key = this.#cacheKey(node);
    const cached = this.#cache.get(key);
    if (cached) {
      node.eval = cached;
      node.state = EVALUATED;
      this.#addNNOutputAsLeafValue(node);
      this.#backup(nodes, edges, uncountedIdx);
      this.stats.cacheHits += 1;
      return 'done';
    }
    node.state = PENDING;
    for (let i = 1; i < nodes.length; i++) nodes[i]!.virtualLosses += 1;
    const encoded = this.#encode(node);
    const symmetry = this.#nextSymmetry();
    const { xSize, ySize } = encoded.input;
    const spatial = copyWithSymmetry(encoded.input.spatial, this.#model.numInputChannels, ySize, xSize, symmetry, false);
    return { nodes, edges, uncountedIdx, input: { xSize, ySize, spatial, global: encoded.input.global }, symmetry, key, encoded };
  }

  /**
   * Run playouts until the root has `maxVisits` visits (KataGo's `maxVisits`), or `signal` aborts. Each round gathers up
   * to `batchSize` leaves and evaluates them in one network call.
   */
  async run(options: { maxVisits: number; signal?: AbortSignal; onBatch?: (search: Search) => void }): Promise<void> {
    const tStart = performance.now();
    const nnMsAtStart = this.stats.nnMs;
    await this.#prepareRoot();
    while (this.root.visits < options.maxVisits && !options.signal?.aborted) {
      const want = Math.min(this.params.batchSize, options.maxVisits - this.root.visits);
      const batch: Pending[] = [];
      // Stop gathering once a batch's worth of playouts in a row reached leaves already waiting for the network.
      let collisionsInARow = 0;
      // Only counted playouts count toward maxVisits, as in KataGo.
      let countedInBatch = 0;
      while (batch.length < want && this.root.visits + countedInBatch < options.maxVisits && collisionsInARow < want) {
        const r = this.#descend();
        if (r === 'collision') collisionsInARow++;
        else {
          collisionsInARow = 0;
          if (r !== 'done') {
            batch.push(r);
            if (r.uncountedIdx < 0) countedInBatch++;
          }
        }
      }
      // With nothing waiting, a playout can't collide, so an empty batch means every playout was backed up already.
      if (batch.length === 0) continue;
      const t0 = performance.now();
      const outputs = await this.#evaluator.evaluate(batch.map((b) => b.input));
      this.stats.nnMs += performance.now() - t0;
      this.stats.batches += 1;
      this.stats.nnEvals += batch.length;
      batch.forEach((b, i) => {
        const leaf = b.nodes[b.nodes.length - 1]!;
        const e = this.#postProcess(outputs[i]!, b.symmetry, b.encoded, this.#policyOptimism(leaf));
        this.#cache.set(b.key, e);
        leaf.eval = e;
        leaf.state = EVALUATED;
        for (let k = 1; k < b.nodes.length; k++) b.nodes[k]!.virtualLosses -= 1;
        this.#addNNOutputAsLeafValue(leaf);
        this.#backup(b.nodes, b.edges, b.uncountedIdx);
      });
      options.onBatch?.(this);
    }
    this.stats.cpuMs += performance.now() - tStart - (this.stats.nnMs - nnMsAtStart);
  }

  // ---------------------------------------------------------------- results

  #winrateFor(node: SearchNode, pla: Stone): number {
    const whiteWin = 0.5 * (1 + node.winLossValueAvg);
    return pla === WHITE ? whiteWin : 1 - whiteWin;
  }

  /** The position index of `loc` in the policy (the pass is last). */
  #posOf(board: GameHistory['board'], loc: number): number {
    return loc === PASS_LOC ? this.#area : board.y(loc) * board.xSize + board.x(loc);
  }

  /** KataGo `getExploreScaling`. */
  #exploreScaling(totalChildWeight: number, stdevFactor: number): number {
    const p = this.params;
    const cpuct = p.cpuctExploration + p.cpuctExplorationLog * Math.log((totalChildWeight + p.cpuctExplorationBase) / p.cpuctExplorationBase);
    return cpuct * Math.sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET) * stdevFactor;
  }

  /** KataGo `getSelfUtilityLCBAndRadius`, for the player to move at `parent`. */
  #selfUtilityLcbAndRadius(parent: SearchNode, edge: SearchEdge): { lcb: number; radius: number } {
    const p = this.params;
    const child = edge.node;
    const utilityRangeRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;
    let radius = 2 * utilityRangeRadius * p.lcbStdevs;
    let weightSum = this.#childWeight(edge);
    let weightSqSum = this.#childWeightSq(edge);
    if (child.visits <= 0 || weightSum <= 0 || weightSqSum <= 0) return { lcb: -radius, radius };
    const utilityAvg = child.utilityAvg;
    let ess = (weightSum * weightSum) / weightSqSum;
    // A small prior that the variance is as large as it can be, so the bound behaves at low visit counts.
    const priorWeight = weightSum / (ess * ess * ess);
    let utilitySqAvg = Math.max(child.utilitySqAvg, utilityAvg * utilityAvg + 1e-8);
    utilitySqAvg = (utilitySqAvg * weightSum + (utilitySqAvg + utilityRangeRadius * utilityRangeRadius) * priorWeight) / (weightSum + priorWeight);
    weightSum += priorWeight;
    weightSqSum += priorWeight * priorWeight;
    ess = (weightSum * weightSum) / weightSqSum;
    const endingScoreBonus = this.#endingWhiteScoreBonus(parent, edge.moveLoc);
    const utilityWithBonus = utilityAvg + this.#scoreUtilityDiff(child.scoreMeanAvg, child.scoreMeanSqAvg, endingScoreBonus);
    const selfUtility = parent.nextPla === WHITE ? utilityWithBonus : -utilityWithBonus;
    const estimateStdev = Math.sqrt((utilitySqAvg - utilityAvg * utilityAvg) / ess);
    radius = estimateStdev * p.lcbStdevs;
    return { lcb: selfUtility - radius, radius };
  }

  /**
   * KataGo `getPlaySelectionValues` for a node's children, in child order, without the parts not ported (direct policy
   * moves for a root with no children, territory scoring's pass suppression, the human-style policy).
   * Returns null where KataGo returns false. `lcb` and `radius` are filled when LCB is used or `alwaysComputeLcb`.
   */
  #playSelectionValues(node: SearchNode, alwaysComputeLcb: boolean, neverUseLcb = false): { values: number[]; lcb: number[]; radius: number[] } | null {
    const p = this.params;
    const children = node.children;
    const n = children.length;
    if (n === 0) return null;
    const policy = this.#searchPolicy(node);
    const b = node.hist.board;
    const prior = children.map((c) => policy[this.#posOf(b, c.moveLoc)]!);
    const values = children.map((c, i) => (prior[i]! < 0 ? 0 : this.#childWeight(c)));
    let totalChildWeight = 0;
    for (const c of children) totalChildWeight += this.#childWeight(c);

    // The best child before LCB: the most weighted, discounting its latest visit, with a small weight on policy.
    let nonLcbBestIdx = 0;
    let nonLcbBestChildWeight = -1e30;
    let maxGoodness = -1e30;
    for (let i = 0; i < n; i++) {
      const edgeVisits = children[i]!.edgeVisits;
      const g = (values[i]! * Math.max(0, edgeVisits - 1)) / Math.max(1, edgeVisits) + 2 * prior[i]!;
      if (g > maxGoodness) {
        maxGoodness = g;
        nonLcbBestChildWeight = values[i]!;
        nonLcbBestIdx = i;
      }
    }

    // At the root, reduce the weight of moves that in retrospect got more visits than PUCT would give them now.
    if (node === this.root) {
      const { fpu, stdevFactor } = this.#fpuValue(node, true, 1.0);
      const exploreScaling = this.#exploreScaling(totalChildWeight, stdevFactor);
      const pla = node.nextPla;
      const bestEdge = children[nonLcbBestIdx]!;
      const best = bestEdge.node;
      const utilityRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;
      let bestWeight = this.#childWeight(bestEdge);
      let bestUtility = best.visits <= 0 || bestWeight <= 0 ? fpu : best.utilityAvg;
      if (best.visits > 0 && bestWeight > 0) {
        const endingScoreBonus = this.#endingWhiteScoreBonus(node, bestEdge.moveLoc);
        if (endingScoreBonus !== 0) bestUtility += this.#scoreUtilityDiff(best.scoreMeanAvg, best.scoreMeanSqAvg, endingScoreBonus);
      }
      if (best.virtualLosses > 0) {
        const vlWeight = best.virtualLosses * p.numVirtualLossesPerThread;
        const vlUtility = pla === WHITE ? -utilityRadius : utilityRadius;
        bestUtility += (vlUtility - bestUtility) * (vlWeight / (vlWeight + Math.max(0.25, bestWeight)));
        bestWeight += vlWeight;
      }
      const bestValue = this.#selectionValue(exploreScaling, prior[nonLcbBestIdx]!, bestWeight, bestUtility, pla);
      for (let i = 0; i < n; i++) {
        if (i === nonLcbBestIdx) continue;
        const ce = children[i]!;
        const c = ce.node;
        const childWeight = this.#childWeight(ce);
        let reduced = 0;
        if (c.visits > 0 && childWeight > 0) {
          let childUtility = c.utilityAvg;
          const endingScoreBonus = this.#endingWhiteScoreBonus(node, ce.moveLoc);
          if (endingScoreBonus !== 0) childUtility += this.#scoreUtilityDiff(c.scoreMeanAvg, c.scoreMeanSqAvg, endingScoreBonus);
          // KataGo getExploreSelectionValueInverse: the weight that would give the best child's selection value.
          let wanted = 0;
          if (prior[i]! >= 0) {
            const explore = bestValue - (pla === WHITE ? childUtility : -childUtility);
            wanted = explore <= 0 ? 1e100 : Math.max(0, (exploreScaling * prior[i]!) / explore - 1);
          }
          reduced = Math.min(childWeight, wanted);
        }
        values[i] = Math.ceil(reduced);
      }
    }

    const lcb: number[] = new Array<number>(n).fill(0);
    const radius: number[] = new Array<number>(n).fill(0);
    if (!neverUseLcb && (alwaysComputeLcb || p.useLcbForSelection)) {
      let bestLcb = -1e10;
      let bestLcbIndex = -1;
      for (let i = 0; i < n; i++) {
        const r = this.#selfUtilityLcbAndRadius(node, children[i]!);
        lcb[i] = r.lcb;
        radius[i] = r.radius;
        const weight = values[i]!;
        if (weight > 0 && weight >= p.minVisitPropForLCB * nonLcbBestChildWeight && r.lcb > bestLcb) {
          bestLcb = r.lcb;
          bestLcbIndex = i;
        }
      }
      if (p.useLcbForSelection && (p.useNonBuggyLcb ? bestLcbIndex >= 0 : bestLcbIndex > 0)) {
        // The best-LCB move's weight is raised until it's large enough relative to every other child.
        let adjustedWeight = values[bestLcbIndex]!;
        for (let i = 0; i < n; i++) {
          if (i === bestLcbIndex) continue;
          const excess = bestLcb - lcb[i]!;
          if (excess < 0) continue;
          const radiusFactor = (radius[i]! + excess) / (radius[i]! + 0.2 * excess);
          const lbound = radiusFactor * radiusFactor * values[i]!;
          if (lbound > adjustedWeight) adjustedWeight = lbound;
        }
        values[bestLcbIndex] = adjustedWeight;
      }
    }

    let maxValue = 0;
    for (const v of values) if (v > maxValue) maxValue = v;
    if (maxValue <= 1e-50) {
      // Every child is weightless: weight each by its policy.
      for (let i = 0; i < n; i++) values[i] = Math.max(0, prior[i]!);
      for (const v of values) if (v > maxValue) maxValue = v;
      if (maxValue <= 1e-50) return null;
    }
    const amountToSubtract = Math.min(p.chosenMoveSubtract, maxValue / 64);
    const amountToPrune = Math.min(p.chosenMovePrune, maxValue / 64);
    for (let i = 0; i < n; i++) {
      if (values[i]! < amountToPrune) values[i] = 0;
      else values[i] = Math.max(0, values[i]! - amountToSubtract);
    }
    maxValue = 0;
    for (const v of values) if (v > maxValue) maxValue = v;
    // KataGo's scaleMaxToAtLeast of 1.0, as its analysis and principal variations use.
    if (maxValue < 1) for (let i = 0; i < n; i++) values[i] = values[i]! / maxValue;
    return { values, lcb, radius };
  }

  /**
   * KataGo `appendPV`: from `node`, the child with the highest play selection value, up to `maxDepth` moves, with each
   * child's visits and the edge visits to it.
   */
  #appendPV(pv: number[], pvVisits: number[], pvEdgeVisits: number[], node: SearchNode, maxDepth: number): void {
    for (let depth = 0; depth < maxDepth; depth++) {
      const sel = this.#playSelectionValues(node, false);
      if (!sel) return;
      let bestIdx = -1;
      let bestValue = POLICY_ILLEGAL_SELECTION_VALUE;
      sel.values.forEach((v, i) => {
        if (v > bestValue) {
          bestValue = v;
          bestIdx = i;
        }
      });
      if (bestIdx < 0) return;
      const edge = node.children[bestIdx]!;
      node = edge.node;
      pv.push(edge.moveLoc);
      pvVisits.push(node.visits);
      pvEdgeVisits.push(edge.edgeVisits);
    }
  }

  /**
   * The analysis at the root (KataGo `getAnalysisData` and `getAnalysisDataOfSingleChild`): each child of the root with
   * its values, in KataGo's order, for the player to move at the root. A child still waiting for its first evaluation
   * (during a search) is reported from its first-play-urgency value, as KataGo does.
   */
  analysis(options: AnalysisOptions = {}): SearchAnalysis {
    const root = this.root;
    const pla = root.nextPla;
    const sign = pla === WHITE ? 1 : -1;
    const b = root.hist.board;
    const p = this.params;
    const sel = this.#playSelectionValues(root, true);
    // With no play selection values, KataGo reports every move as if unvisited (getSelfUtilityLCBAndRadiusZeroVisits).
    const zeroVisitsRadius = 2 * (p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor) * p.lcbStdevs;
    // KataGo PlayUtils::getHackedLCBForWinrate: the utility radius in win-rate units.
    const radiusToWinrate = (0.5 * p.winLossUtilityFactor) / (p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor + 1e-20);
    const policy = this.#searchPolicy(root);
    // KataGo getAnalysisData: the parent's values and first-play urgency, for children with no weight yet.
    let policyProbMassVisited = 0;
    for (const e of root.children) policyProbMassVisited += Math.max(0, policy[this.#posOf(b, e.moveLoc)]!);
    const { fpu } = this.#fpuValue(root, true, policyProbMassVisited);
    const parentUtility = root.utilityAvg;
    const parentScoreStdev = getScoreStdev(root.scoreMeanAvg, root.scoreMeanSqAvg);
    const ownershipOf = new Map<SearchNode, { ownership: Float64Array; stdev: Float64Array | null }>();
    const wantOwnership = options.movesOwnership === true || options.movesOwnershipStdev === true;
    const infos = root.children.map((e, i) => {
      const c = e.node;
      const pv: number[] = [e.moveLoc];
      const pvVisits: number[] = [c.visits];
      const pvEdgeVisits: number[] = [e.edgeVisits];
      this.#appendPV(pv, pvVisits, pvEdgeVisits, c, p.analysisPVLen);
      // White's values (KataGo getAnalysisDataOfSingleChild).
      let utility: number, winLoss: number, noResult: number, scoreMean: number, scoreStdev: number, lead: number, edgeWeight: number;
      if (c.visits <= 0 || c.weightSum <= 1e-30) {
        utility = fpu;
        winLoss = p.winLossUtilityFactor === 1 ? root.winLossValueAvg + (fpu - parentUtility) : 0;
        winLoss = Math.min(1, Math.max(-1, winLoss));
        noResult = 0;
        scoreMean = root.scoreMeanAvg;
        scoreStdev = parentScoreStdev;
        lead = root.leadAvg;
        edgeWeight = 0;
      } else {
        utility = c.utilityAvg;
        winLoss = c.winLossValueAvg;
        noResult = c.noResultValueAvg;
        scoreMean = c.scoreMeanAvg;
        scoreStdev = getScoreStdev(c.scoreMeanAvg, c.scoreMeanSqAvg);
        lead = c.leadAvg;
        edgeWeight = this.#childWeight(e);
      }
      const whiteWin = 0.5 * (1 + winLoss);
      const winrate = pla === WHITE ? whiteWin : 1 - whiteWin;
      const radius = sel ? sel.radius[i]! : zeroVisitsRadius;
      const info: SearchMoveInfo = {
        loc: e.moveLoc,
        visits: c.visits,
        edgeVisits: e.edgeVisits,
        winrate,
        scoreLead: sign * lead,
        scoreMean: sign * scoreMean,
        scoreStdev,
        prior: policy[this.#posOf(b, e.moveLoc)]!,
        utility: sign * utility,
        lcb: winrate - radius * radiusToWinrate,
        utilityLcb: sel ? sel.lcb[i]! : -zeroVisitsRadius,
        playSelectionValue: sel ? sel.values[i]! : 0,
        order: 0,
        pv,
        pvVisits,
        pvEdgeVisits,
        weight: c.weightSum,
        edgeWeight,
        noResultValue: noResult,
      };
      if (wantOwnership) ownershipOf.set(c, this.#treeOwnership(c, options.movesOwnershipStdev === true));
      return { info, node: c };
    });
    // KataGo's AnalysisData order (a stable sort): moves with edge visits first, then by play selection value, edge
    // visits, policy. A move whose only visit wasn't counted (enableMorePassingHacks) is listed, with 0 edge visits.
    const sorted = infos.sort(
      (xi, yi) => {
        const x = xi.info;
        const y = yi.info;
        return (
          Number(y.edgeVisits > 0) - Number(x.edgeVisits > 0) ||
          y.playSelectionValue - x.playSelectionValue ||
          y.edgeVisits - x.edgeVisits ||
          y.prior - x.prior
        );
      },
    );
    // KataGo getAnalysisData: with root symmetry pruning, each move is followed by its mirror images, each with the
    // move's data and its line (and ownership) mirrored, in the order of the root's symmetries.
    const symmetries = p.rootSymmetryPruning && this.#rootSymmetries.length > 1 ? this.#rootSymmetries : [0];
    const rootAvoid = this.#avoidMoveUntilByLoc[pla];
    const listed: SearchMoveInfo[] = [];
    const isDone = new Set<number>();
    for (const { info: m, node } of sorted) {
      for (const symmetry of symmetries) {
        const symMove = getSymLoc(m.loc, b, symmetry);
        if (isDone.has(symMove)) continue;
        if (rootAvoid && rootAvoid[symMove]! > 0) continue;
        isDone.add(symMove);
        const copy: SearchMoveInfo = symmetry === 0 ? m : { ...m, loc: symMove, pv: m.pv.map((l) => getSymLoc(l, b, symmetry)), isSymmetryOf: m.loc };
        const own = ownershipOf.get(node);
        if (own) {
          if (options.movesOwnership) copy.ownership = this.#ownershipForOutput(own.ownership, pla, symmetry);
          if (options.movesOwnershipStdev) copy.ownershipStdev = this.#ownershipForOutput(own.stdev!, WHITE, symmetry);
        }
        listed.push(copy);
      }
    }
    listed.forEach((m, order) => (m.order = order));
    return {
      rootVisits: root.visits,
      rootWinrate: this.#winrateFor(root, pla),
      rootScoreLead: sign * root.leadAvg,
      rootUtility: sign * root.utilityAvg,
      moveInfos: listed,
    };
  }

  /**
   * KataGo `getPrunedRootValues` (falling back to `getNodeValues` with no play selection values), and the
   * `ReportedSearchValues` it builds, for the player to move at the root.
   */
  rootValues(): SearchRootValues {
    const root = this.root;
    const pla = root.nextPla;
    let winLoss: number, noResult: number, scoreMean: number, scoreMeanSq: number, lead: number, utility: number;
    const sel = this.#playSelectionValues(root, false, true);
    if (!sel) {
      winLoss = root.winLossValueAvg;
      noResult = root.noResultValueAvg;
      scoreMean = root.scoreMeanAvg;
      scoreMeanSq = root.scoreMeanSqAvg;
      lead = root.leadAvg;
      utility = root.utilityAvg;
    } else {
      let winLossSum = 0, noResultSum = 0, scoreMeanSum = 0, scoreMeanSqSum = 0, leadSum = 0, utilitySum = 0, weightSum = 0;
      root.children.forEach((e, i) => {
        const c = e.node;
        if (c.visits <= 0 || c.weightSum <= 0 || e.edgeVisits <= 0) return;
        const w = sel.values[i]!;
        winLossSum += w * c.winLossValueAvg;
        noResultSum += w * c.noResultValueAvg;
        scoreMeanSum += w * c.scoreMeanAvg;
        scoreMeanSqSum += w * c.scoreMeanSqAvg;
        leadSum += w * c.leadAvg;
        utilitySum += w * c.utilityAvg;
        weightSum += w;
      });
      // And the root's own evaluation.
      const ev = root.eval!;
      const w = this.#weightFromNN(ev);
      winLossSum += (ev.whiteWinProb - ev.whiteLossProb) * w;
      noResultSum += ev.whiteNoResultProb * w;
      scoreMeanSum += ev.whiteScoreMean * w;
      scoreMeanSqSum += ev.whiteScoreMeanSq * w;
      leadSum += ev.whiteLead * w;
      utilitySum += this.#utilityFromNN(ev) * w;
      weightSum += w;
      winLoss = winLossSum / weightSum;
      noResult = noResultSum / weightSum;
      scoreMean = scoreMeanSum / weightSum;
      scoreMeanSq = scoreMeanSqSum / weightSum;
      lead = leadSum / weightSum;
      utility = utilitySum / weightSum;
    }
    // ReportedSearchValues: clamped against floating-point error.
    winLoss = Math.min(1, Math.max(-1, winLoss));
    noResult = Math.min(1 - Math.abs(winLoss), Math.max(0, noResult));
    const sign = pla === WHITE ? 1 : -1;
    return {
      visits: root.visits,
      weight: root.weightSum,
      winrate: 0.5 + 0.5 * winLoss * sign,
      scoreSelfplay: scoreMean * sign,
      scoreLead: lead * sign,
      scoreStdev: getScoreStdev(scoreMean, scoreMeanSq),
      utility: utility * sign,
      noResultValue: noResult,
    };
  }

  /**
   * KataGo's `rootInfo` hashes of the root position (`thisHash`, `symHash`; see `rules/hash.ts`). Needs
   * `loadPositionHashes()` first.
   */
  rootHashes(): { thisHash: string; symHash: string } {
    return positionHashes(this.root.hist.board, this.root.nextPla);
  }

  /** The root's own network evaluation, as KataGo's `rootInfo` reports it (`rawWinrate` and the rest). */
  rootRawValues(): SearchRawValues {
    const ev = this.root.eval!;
    const sign = this.root.nextPla === WHITE ? 1 : -1;
    return {
      winrate: 0.5 + 0.5 * (ev.whiteWinProb - ev.whiteLossProb) * sign,
      lead: ev.whiteLead * sign,
      scoreSelfplay: ev.whiteScoreMean * sign,
      scoreSelfplayStdev: Math.sqrt(Math.max(0, ev.whiteScoreMeanSq - ev.whiteScoreMean * ev.whiteScoreMean)),
      noResultProb: ev.whiteNoResultProb,
      stWrError: ev.shorttermWinlossError * 0.5,
      stScoreError: ev.shorttermScoreError,
      varTimeLeft: ev.varTimeLeft,
    };
  }

  /**
   * KataGo `getPolicy`: the root's policy from the network (with the root's policy optimism, without its policy
   * temperature), row-major with the pass last; -1 marks an illegal move.
   */
  rootPolicy(): number[] {
    return Array.from(this.root.eval!.policy);
  }

  /**
   * KataGo `getAverageTreeOwnership` and `getAverageAndStandardDeviationTreeOwnership` at the root: ownership averaged
   * over the search tree, for the player to move, row-major, with its standard deviation if asked for.
   */
  rootOwnership(withStdev: boolean): { ownership: number[]; ownershipStdev: number[] | null } {
    const own = this.#treeOwnership(this.root, withStdev);
    return {
      ownership: this.#ownershipForOutput(own.ownership, this.root.nextPla, 0),
      ownershipStdev: own.stdev ? this.#ownershipForOutput(own.stdev, WHITE, 0) : null,
    };
  }

  /**
   * White's ownership averaged over the subtree at `node` (KataGo's `getAverageTreeOwnership(node)`), and the standard
   * deviation if asked for. Nodes are weighted as KataGo weights them; it stops deepening where a node's share of the
   * average would fall below a bound that shrinks with the visits.
   */
  #treeOwnership(node: SearchNode, withStdev: boolean): { ownership: Float64Array; stdev: Float64Array | null } {
    const area = this.#area;
    const average = new Float64Array(area);
    const sq = withStdev ? new Float64Array(area) : null;
    const accumulate = (ownership: Float64Array, prop: number): void => {
      for (let pos = 0; pos < area; pos++) {
        const v = ownership[pos]!;
        average[pos]! += prop * v;
        if (sq) sq[pos]! += prop * v * v;
      }
    };
    const minProp = 0.5 / Math.pow(Math.max(1, node.visits), 0.75);
    const pruneProp = minProp * 0.01;
    this.#traverseTreeForOwnership(minProp, pruneProp, 1, node, new Set(), accumulate);
    if (sq) for (let pos = 0; pos < area; pos++) sq[pos] = Math.sqrt(Math.max(sq[pos]! - average[pos]! * average[pos]!, 0));
    return { ownership: average, stdev: sq };
  }

  /** KataGo `traverseTreeForOwnership` and `traverseTreeForOwnershipChildren`. Returns whether anything was added. */
  #traverseTreeForOwnership(
    minProp: number,
    pruneProp: number,
    desiredProp: number,
    node: SearchNode,
    graphPath: Set<SearchNode>,
    accumulate: (ownership: Float64Array, prop: number) => void,
  ): boolean {
    const ev = node.eval;
    if (!ev) return false;
    // Base cases: a small enough share, no children, or a cycle in the graph.
    if (desiredProp < minProp || node.children.length <= 0 || graphPath.has(node)) {
      accumulate(ev.whiteOwnership, desiredProp);
      return true;
    }
    graphPath.add(node);
    const parentNNWeight = Math.max(this.#weightFromNN(ev), 1e-10);
    const childWeights = node.children.map((e) => this.#childWeight(e));
    // The children are weighted by their weight squared, sharper than the plain weight.
    let relativeChildrenWeightSum = 0;
    let childrenWeightSum = 0;
    for (const w of childWeights) {
      relativeChildrenWeightSum += w * w;
      childrenWeightSum += w;
    }
    const desiredPropFromChildren = (desiredProp * childrenWeightSum) / (childrenWeightSum + parentNNWeight);
    let selfProp = (desiredProp * parentNNWeight) / (childrenWeightSum + parentNNWeight);
    if (desiredPropFromChildren <= 0 || relativeChildrenWeightSum <= 0) selfProp += desiredPropFromChildren;
    else {
      node.children.forEach((e, i) => {
        const w = childWeights[i]!;
        const desiredPropFromChild = ((w * w) / relativeChildrenWeightSum) * desiredPropFromChildren;
        if (desiredPropFromChild < pruneProp) selfProp += desiredPropFromChild;
        else if (!this.#traverseTreeForOwnership(minProp, pruneProp, desiredPropFromChild, e.node, graphPath, accumulate)) selfProp += desiredPropFromChild;
      });
    }
    graphPath.delete(node);
    accumulate(ev.whiteOwnership, selfProp);
    return true;
  }

  /**
   * KataGo's output step for tree ownership: for `pla`'s side (white's values negated for black), each point written at
   * its image under `symmetry`.
   */
  #ownershipForOutput(values: Float64Array, pla: Stone, symmetry: number): number[] {
    const b = this.root.hist.board;
    const out = new Array<number>(this.#area).fill(0);
    const sign = pla === WHITE ? 1 : -1;
    for (let y = 0; y < b.ySize; y++) {
      for (let x = 0; x < b.xSize; x++) {
        const symLoc = getSymLoc(b.loc(x, y), b, symmetry);
        out[b.y(symLoc) * b.xSize + b.x(symLoc)] = sign * values[y * b.xSize + x]!;
      }
    }
    return out;
  }
}
