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
 * - the analysis's move order and principal variations from KataGo's play selection values
 *   (`getPlaySelectionValues`), with LCB selection (`useLcbForSelection`, `getSelfUtilityLCBAndRadius`) and the root's
 *   retrospective reduction of over-explored moves (`getReducedPlaySelectionWeight`).
 *
 * Not yet ported (each is to be added and measured separately): graph search (transpositions), wide root noise, root
 * symmetry pruning, and tree reuse.
 * Search parameters that only matter for those default to values that switch them off.
 */
import type { ModelDesc } from '../model/types.js';
import { BLACK, EMPTY, NULL_LOC, PASS_LOC, WHITE, opp, type Board, type Stone } from '../rules/board.js';
import { UnsupportedPositionError, type GameHistory } from '../rules/history.js';
import { encodeHistoryV7, type LadderCache } from '../features/v7.js';
import { calculateArea } from '../rules/area.js';
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

class SearchNode {
  state = UNEVALUATED;
  eval: Evaluation | null = null;
  readonly children: SearchNode[] = [];
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
  /**
   * KataGo's edge visits: the visits its parent counts through this node. They are fewer than `visits` after a playout
   * that wasn't counted (`enableMorePassingHacks`), until later playouts catch them up.
   */
  edgeVisits = 0;
  /** KataGo `subtreeValueBiasTableEntry`, and this node's last contribution to it. */
  biasEntry: SubtreeValueBiasEntry | null = null;
  lastBiasDeltaSum = 0;
  lastBiasWeight = 0;

  constructor(
    readonly hist: GameHistory,
    /** The move that led here (NULL for the root). */
    readonly moveLoc: number,
    readonly terminal: boolean,
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
  /** Principal variation: this move, then the child with the highest play selection value at each step. */
  pv: number[];
}

export interface SearchAnalysis {
  rootVisits: number;
  /** As in moveInfos, for the whole tree. */
  rootWinrate: number;
  rootScoreLead: number;
  rootUtility: number;
  moveInfos: SearchMoveInfo[];
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

interface Pending {
  path: SearchNode[];
  /** KataGo `shouldCountPlayout`: false for the extra playouts of `enableMorePassingHacks`, which back up nothing. */
  counted: boolean;
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
  /** KataGo's `subtreeValueBiasTable`, one per search (native's analysis engine starts each query with it empty). */
  readonly #subtreeValueBiasTable = new Map<string, SubtreeValueBiasEntry>();
  readonly #sqrtBoardArea: number;
  readonly #area: number;
  #recentScoreCenter = 0;
  /** KataGo `rootSafeArea`: the root's strictly pass-alive groups and strictly safe territory. */
  #rootSafeArea: Uint8Array = new Uint8Array(0);
  /** `rootPruneUselessMoves` is on and the opponent's last four moves at the root were passes. */
  #rootPrunesUselessMoves = false;
  #rootPrepared = false;

  /**
   * @param cache An evaluation cache to share with other searches (KataGo's analysis engine shares one across the turns
   *   of a query); by default each search has its own.
   */
  constructor(model: ModelDesc, evaluator: NetEvaluator, rootHistory: GameHistory, params: Partial<SearchParams> = {}, cache: Map<string, Evaluation> = new Map()) {
    this.#cache = cache;
    if (rootHistory.rules.scoringRule !== 'AREA' || rootHistory.rules.taxRule !== 'NONE') {
      // The territory encore and ending a game with a group tax aren't ported yet (docs/OPEN-QUESTIONS.md).
      throw new UnsupportedPositionError('the search supports area scoring with no group tax only, for now');
    }
    this.params = { ...DEFAULT_SEARCH_PARAMS, ...params };
    this.params.batchSize = Math.max(1, Math.min(this.params.batchSize, evaluator.maxBatch));
    this.#model = model;
    this.#evaluator = evaluator;
    this.#random = mulberry32(this.params.seed);
    const b = rootHistory.board;
    this.#area = b.xSize * b.ySize;
    this.#sqrtBoardArea = Math.sqrt(this.#area);
    // The root is never terminal (KataGo forceNonTerminal): searching a finished position still explores it.
    this.root = new SearchNode(rootHistory.clone(), 0, false);
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
    // The territory-scoring encore isn't supported, so only area scoring is "area-ish".
    const isAreaIsh = h.rules.scoringRule === 'AREA';
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
   * KataGo `isAllowedRootMove`, without root symmetry pruning: with `rootPruneUselessMoves`, once the opponent's last
   * four moves were passes, no move inside either player's pass-alive area.
   */
  #isAllowedRootMove(moveLoc: number): boolean {
    // The root's history doesn't change during a search, so whether the opponent's last four moves were passes is
    // worked out once, in #prepareRoot.
    return !(this.#rootPrunesUselessMoves && moveLoc !== PASS_LOC && this.#rootSafeArea[moveLoc] !== EMPTY);
  }

  /** KataGo `getChildWeight(edgeVisits, childVisits)`: the child's weight as its parent counts it. */
  #childWeight(c: SearchNode): number {
    return c.weightSum * (c.edgeVisits / Math.max(c.visits, 1));
  }

  #childWeightSq(c: SearchNode): number {
    return c.weightSqSum * (c.edgeVisits / Math.max(c.visits, 1));
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
   * KataGo's evaluation-cache key (`NNInputs::getHash`): the position, the player to move and ko bans, whether a pass
   * would end the phase (and friendly-pass suppression), whether the game is over, and whether history is hidden. Like
   * KataGo, it leaves out the move history, so transpositions share an evaluation.
   */
  #cacheKey(node: SearchNode): string {
    const h = node.hist;
    const b = h.board;
    let key = '';
    for (let y = 0; y < b.ySize; y++) for (let x = 0; x < b.xSize; x++) key += b.colors[b.loc(x, y)];
    key += `|${h.nextPla}|${b.koLoc}|`;
    for (let i = 0; i < h.superKoBanned.length; i++) if (h.superKoBanned[i]) key += i.toString(36) + ',';
    const pla = h.nextPla;
    if (h.passWouldEndPhase(pla)) {
      key += '|p';
      if (this.params.conservativePass && node === this.root) key += 'c';
      if (h.shouldSuppressEndGameFromFriendlyPass(pla)) key += 'f';
      if (this.params.enablePassingHacks) key += 'h';
    }
    if (h.isGameFinished) key += '|over';
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
    });
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

  #addTerminalValue(node: SearchNode): void {
    const h = node.hist;
    const p = this.params;
    const whiteWins = h.winner === WHITE ? 1 : h.winner === BLACK ? 0 : p.drawEquivalentWinsForWhite;
    // KataGo whiteScoreDrawAdjust: the draw adjustment is zero unless the result can be a draw (integer komi).
    const drawAdjust = Number.isInteger(h.komi) ? p.drawEquivalentWinsForWhite - 0.5 : 0;
    const scoreMean = h.finalWhiteMinusBlackScore + drawAdjust;
    const scoreMeanSq = whiteScoreMeanSqOfScoreGridded(h.finalWhiteMinusBlackScore, p.drawEquivalentWinsForWhite);
    const weight = this.#usesUncertainty ? p.uncertaintyMaxWeight : 1;
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
    for (const c of node.children) {
      if (c.visits <= 0 || c.weightSum <= 0 || c.edgeVisits <= 0) continue;
      const w = this.#childWeight(c);
      good.push(c);
      weights.push(w);
      selfUtilities.push(node.nextPla === WHITE ? c.utilityAvg : -c.utilityAvg);
      totalChildWeight += w;
    }
    const origTotalChildWeight = totalChildWeight;
    if (this.params.useNoisePruning && good.length > 0) {
      const policy = node.eval!.policy;
      const b = node.hist.board;
      const policyProbs = good.map((c) => Math.max(1e-30, policy[this.#posOf(b, c.moveLoc)]!));
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
    fpu += (lossValue - fpu) * p.fpuLossProp;
    return { fpu, stdevFactor };
  }

  #selectionValue(exploreScaling: number, policy: number, childWeight: number, childUtility: number, pla: Stone): number {
    if (policy < 0) return POLICY_ILLEGAL_SELECTION_VALUE;
    const explore = (exploreScaling * policy) / (1 + childWeight);
    return explore + (pla === WHITE ? childUtility : -childUtility);
  }

  /**
   * KataGo `selectBestChildToDescend` (and the child creation of `playoutDescend`), without the features listed at the
   * top of this file. Returns the child, whether the parent counts this visit to it (`countEdgeVisit`), and whether the
   * child was new.
   */
  #selectChild(node: SearchNode, isRoot: boolean): { child: SearchNode; countEdgeVisit: boolean; isNew: boolean } {
    const p = this.params;
    const policy = node.eval!.policy;
    const b = node.hist.board;
    const posOf = (loc: number) => (loc === PASS_LOC ? this.#area : b.y(loc) * b.xSize + b.x(loc));

    let policyProbMassVisited = 0;
    let totalChildWeight = 0;
    let totalChildEdgeVisits = 0;
    for (const c of node.children) {
      const pr = policy[posOf(c.moveLoc)]!;
      if (pr < 0) continue;
      policyProbMassVisited += pr;
      totalChildWeight += this.#childWeight(c);
      totalChildEdgeVisits += c.edgeVisits;
    }
    const { fpu, stdevFactor } = this.#fpuValue(node, isRoot, policyProbMassVisited);
    const cpuct = p.cpuctExploration + p.cpuctExplorationLog * Math.log((totalChildWeight + p.cpuctExplorationBase) / p.cpuctExplorationBase);
    const exploreScaling = cpuct * Math.sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET) * stdevFactor;
    const pla = node.nextPla;
    const utilityRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;

    let best: SearchNode | null = null;
    let bestValue = POLICY_ILLEGAL_SELECTION_VALUE;
    const tried = new Uint8Array(this.#area + 1);
    for (const c of node.children) {
      const pos = posOf(c.moveLoc);
      tried[pos] = 1;
      let childWeight = this.#childWeight(c);
      let childUtility: number;
      if (c.visits <= 0 || childWeight <= 0) childUtility = fpu;
      else {
        childUtility = c.utilityAvg;
        // KataGo's "tiny adjustment for passing": the root's ending bonus.
        const endingScoreBonus = this.#endingWhiteScoreBonus(node, c.moveLoc);
        if (endingScoreBonus !== 0) childUtility += this.#scoreUtilityDiff(c.scoreMeanAvg, c.scoreMeanSqAvg, endingScoreBonus);
      }
      if (c.virtualLosses > 0) {
        const vlWeight = c.virtualLosses * p.numVirtualLossesPerThread;
        const vlUtility = pla === WHITE ? -utilityRadius : utilityRadius;
        const frac = vlWeight / (vlWeight + Math.max(0.25, childWeight));
        childUtility += (vlUtility - childUtility) * frac;
        childWeight += vlWeight;
      }
      const value = this.#selectionValue(exploreScaling, policy[pos]!, childWeight, childUtility, pla);
      if (value > bestValue) {
        bestValue = value;
        best = c;
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
        bestNewPolicy = pr;
        bestNewLoc = loc;
      }
    }
    let newLoc = NULL_LOC;
    if (bestNewLoc !== NULL_LOC && this.#selectionValue(exploreScaling, bestNewPolicy, 0, fpu, pla) > bestValue) newLoc = bestNewLoc;
    const bestLoc = newLoc !== NULL_LOC ? newLoc : best ? best.moveLoc : NULL_LOC;

    // KataGo enableMorePassingHacks: where a pass would end the phase, make sure both a pass and a move other than a pass
    // get tried, by a playout the parent doesn't count.
    let countEdgeVisit = true;
    if (totalChildEdgeVisits >= 2 && p.enableMorePassingHacks && node.hist.passWouldEndPhase(pla)) {
      let hasPassMove = false;
      let hasNonPassMove = false;
      for (const c of node.children) {
        if (c.moveLoc === PASS_LOC) hasPassMove = true;
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
      // KataGo playoutDescend: a pass that ends the game isn't terminal from the root under conservativePass, or
      // where friendly-pass rules suppress the end of the game.
      const friendlyPass = newLoc === PASS_LOC && node.hist.shouldSuppressEndGameFromFriendlyPass(pla);
      const hist = node.hist.clone();
      hist.play(newLoc, pla);
      const forceNonTerminal = newLoc === PASS_LOC && hist.isGameFinished && ((p.conservativePass && isRoot) || friendlyPass);
      const child = new SearchNode(hist, newLoc, hist.isGameFinished && !forceNonTerminal);
      // KataGo allocateOrFindNode: the table entry for this move, after the move before it, unless it's a pass.
      if (p.subtreeValueBiasFactor !== 0 && hist.moves.length >= 2 && newLoc !== PASS_LOC) {
        const parentPrevMoveLoc = hist.moves[hist.moves.length - 2]!.loc;
        if (parentPrevMoveLoc !== NULL_LOC) {
          const key = subtreeValueBiasKey(pla, parentPrevMoveLoc, newLoc, node.hist.board);
          let biasEntry = this.#subtreeValueBiasTable.get(key);
          if (!biasEntry) {
            biasEntry = { deltaUtilitySum: 0, weightSum: 0 };
            this.#subtreeValueBiasTable.set(key, biasEntry);
          }
          child.biasEntry = biasEntry;
        }
      }
      node.children.push(child);
      return { child, countEdgeVisit, isNew: true };
    }
    if (!best) throw new Error('no legal move to search');
    return { child: best, countEdgeVisit, isNew: false };
  }

  // ---------------------------------------------------------------- playouts

  /** KataGo's return up `playoutDescend`: each edge on the path gains a visit and each parent is recomputed. */
  #backup(path: SearchNode[]): void {
    for (let i = path.length - 1; i >= 1; i--) {
      path[i]!.edgeVisits += 1;
      this.#recomputeNodeStats(path[i - 1]!);
    }
    this.stats.visits += 1;
  }

  /** A playout's end: backed up if it counts; one that doesn't (`enableMorePassingHacks`) updates only its leaf. */
  #finish(path: SearchNode[], counted: boolean): void {
    if (counted) this.#backup(path);
  }

  /**
   * Descend from the root to a leaf. Terminal leaves and cached evaluations are backed up at once; a leaf that needs the
   * network is returned with virtual losses added along its path.
   */
  #descend(): Pending | 'done' | 'collision' {
    const path: SearchNode[] = [this.root];
    let node = this.root;
    let counted = true;
    while (node.state === EVALUATED && !node.terminal) {
      const sel = this.#selectChild(node, node === this.root);
      path.push(sel.child);
      if (!sel.countEdgeVisit) counted = false;
      // KataGo maybeCatchUpEdgeVisits: while the parent has counted fewer visits to a child than the child has, a visit
      // just counts one more, without descending.
      else if (!sel.isNew && sel.child.edgeVisits < sel.child.visits) {
        this.#backup(path);
        return 'done';
      }
      node = sel.child;
    }
    if (node.terminal) {
      this.#addTerminalValue(node);
      this.#finish(path, counted);
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
      this.#finish(path, counted);
      this.stats.cacheHits += 1;
      return 'done';
    }
    node.state = PENDING;
    for (let i = 1; i < path.length; i++) path[i]!.virtualLosses += 1;
    const encoded = this.#encode(node);
    const symmetry = this.#nextSymmetry();
    const { xSize, ySize } = encoded.input;
    const spatial = copyWithSymmetry(encoded.input.spatial, this.#model.numInputChannels, ySize, xSize, symmetry, false);
    return { path, counted, input: { xSize, ySize, spatial, global: encoded.input.global }, symmetry, key, encoded };
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
            if (r.counted) countedInBatch++;
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
        const leaf = b.path[b.path.length - 1]!;
        const e = this.#postProcess(outputs[i]!, b.symmetry, b.encoded, this.#policyOptimism(leaf));
        this.#cache.set(b.key, e);
        leaf.eval = e;
        leaf.state = EVALUATED;
        for (let k = 1; k < b.path.length; k++) b.path[k]!.virtualLosses -= 1;
        this.#addNNOutputAsLeafValue(leaf);
        this.#finish(b.path, b.counted);
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
  #selfUtilityLcbAndRadius(parent: SearchNode, child: SearchNode): { lcb: number; radius: number } {
    const p = this.params;
    const utilityRangeRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;
    let radius = 2 * utilityRangeRadius * p.lcbStdevs;
    let weightSum = this.#childWeight(child);
    let weightSqSum = this.#childWeightSq(child);
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
    const endingScoreBonus = this.#endingWhiteScoreBonus(parent, child.moveLoc);
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
  #playSelectionValues(node: SearchNode, alwaysComputeLcb: boolean): { values: number[]; lcb: number[]; radius: number[] } | null {
    const p = this.params;
    const children = node.children;
    const n = children.length;
    if (n === 0) return null;
    const policy = node.eval!.policy;
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
      const best = children[nonLcbBestIdx]!;
      const utilityRadius = p.winLossUtilityFactor + p.staticScoreUtilityFactor + p.dynamicScoreUtilityFactor;
      let bestWeight = this.#childWeight(best);
      let bestUtility = best.visits <= 0 || bestWeight <= 0 ? fpu : best.utilityAvg;
      if (best.visits > 0 && bestWeight > 0) {
        const endingScoreBonus = this.#endingWhiteScoreBonus(node, best.moveLoc);
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
        const c = children[i]!;
        const childWeight = this.#childWeight(c);
        let reduced = 0;
        if (c.visits > 0 && childWeight > 0) {
          let childUtility = c.utilityAvg;
          const endingScoreBonus = this.#endingWhiteScoreBonus(node, c.moveLoc);
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
    if (alwaysComputeLcb || p.useLcbForSelection) {
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

  /** KataGo `appendPV`: from `node`, the child with the highest play selection value, up to `maxDepth` moves. */
  #appendPV(pv: number[], node: SearchNode, maxDepth: number): void {
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
      node = node.children[bestIdx]!;
      pv.push(node.moveLoc);
    }
  }

  /**
   * The analysis at the root (KataGo `getAnalysisData`): each searched move with its value, in KataGo's order. Moves
   * still waiting for their first evaluation (during a search) are left out.
   */
  analysis(): SearchAnalysis {
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
    const infos = root.children.map((c, i) => {
      const pv: number[] = [c.moveLoc];
      this.#appendPV(pv, c, p.analysisPVLen);
      const winrate = this.#winrateFor(c, pla);
      const radius = sel ? sel.radius[i]! : zeroVisitsRadius;
      return {
        loc: c.moveLoc,
        visits: c.visits,
        edgeVisits: c.edgeVisits,
        winrate,
        scoreLead: sign * c.leadAvg,
        scoreMean: sign * c.scoreMeanAvg,
        scoreStdev: getScoreStdev(c.scoreMeanAvg, c.scoreMeanSqAvg),
        prior: root.eval!.policy[this.#posOf(b, c.moveLoc)]!,
        utility: sign * c.utilityAvg,
        lcb: winrate - radius * radiusToWinrate,
        utilityLcb: sel ? sel.lcb[i]! : -zeroVisitsRadius,
        playSelectionValue: sel ? sel.values[i]! : 0,
        order: 0,
        pv,
      };
    });
    // KataGo's AnalysisData order (a stable sort): moves with edge visits first, then by play selection value, edge
    // visits, policy. A move whose only visit wasn't counted (enableMorePassingHacks) is listed, with 0 edge visits.
    const moveInfos = infos
      .filter((m) => m.visits > 0)
      .sort(
        (x, y) =>
          Number(y.edgeVisits > 0) - Number(x.edgeVisits > 0) ||
          y.playSelectionValue - x.playSelectionValue ||
          y.edgeVisits - x.edgeVisits ||
          y.prior - x.prior,
      );
    moveInfos.forEach((m, order) => (m.order = order));
    return {
      rootVisits: root.visits,
      rootWinrate: this.#winrateFor(root, pla),
      rootScoreLead: sign * root.leadAvg,
      rootUtility: sign * root.utilityAvg,
      moveInfos,
    };
  }
}
