/**
 * A search's results as KataGo's analysis engine reports them, ported from KataGo v1.18.2's `Search::getAnalysisJson`
 * (`cpp/search/searchresults.cpp`): the fields, KataGo's rounding, and the side the values are for.
 */
import { BLACK, PASS_LOC, type Stone } from '../rules/board.js';
import { gtpLocation } from '../rules/analysis-position.js';
import type { GameHistory } from '../rules/history.js';
import type { Search } from '../search/search.js';
import { JsonFloat, type JsonObject } from './json.js';

/** C `round`: halves away from zero. */
function cRound(x: number): number {
  const t = Math.trunc(x);
  const d = x - t;
  if (d >= 0.5) return t + 1;
  if (d <= -0.5) return t - 1;
  return t;
}

/** KataGo `Global::roundStatic`. */
export function roundStatic(x: number, inverseScale: number): number {
  return cRound(x * inverseScale) / inverseScale;
}

/** KataGo `Global::roundDynamic`: `x` to `precision` significant digits past the first. */
export function roundDynamic(x: number, precision: number): number {
  const absx = Math.abs(x);
  if (absx <= 1e-60) return x;
  const roundingMagnitude = Math.floor(Math.log10(absx)) - precision;
  if (roundingMagnitude >= 0) return cRound(x);
  // A power of ten read from text is correctly rounded, as glibc's pow is.
  return roundStatic(x, Number(`1e${-roundingMagnitude}`));
}

/** KataGo's `OUTPUT_PRECISION` in `getAnalysisJson`. */
const PRECISION = 8;
const r8 = (x: number): JsonFloat => new JsonFloat(roundDynamic(x, PRECISION));
/** Ownership is rounded to 10^-6. */
const r6 = (x: number): JsonFloat => new JsonFloat(roundStatic(x, 1e6));

/** What a request asks to have reported (KataGo's `AnalyzeRequest` fields). */
export interface ReportOptions {
  /** Which side the values are for: a player, or null for the player to move (`reportAnalysisWinratesAs`). */
  perspective: Stone | null;
  includePolicy: boolean;
  includeOwnership: boolean;
  includeOwnershipStdev: boolean;
  includeMovesOwnership: boolean;
  includeMovesOwnershipStdev: boolean;
  includePVVisits: boolean;
  includeNoResultValue: boolean;
  /** The engine's `preventCleanupPhase` setting: a line with a pass is cut where the phase would end. */
  preventEncore: boolean;
}

/**
 * KataGo `AnalysisData::getPVLenUpToPhaseEnd`: the moves of a line before the one that ends the root's phase (all of
 * them if none does).
 */
export function pvLenUpToPhaseEnd(root: GameHistory, pv: readonly number[]): number {
  const hist = root.clone();
  let j = 0;
  for (; j < pv.length; j++) {
    hist.play(pv[j]!, hist.nextPla);
    if (hist.encorePhase !== root.encorePhase) break;
  }
  return j;
}

/**
 * The search's results as KataGo's `getAnalysisJson` writes them (`moveInfos`, `rootInfo`, and the policy and ownership
 * asked for), added to `ret`. Returns false, as KataGo does, when the root has no results yet. The search reports values
 * for the player to move; they are turned to the other side where `perspective` asks for it.
 */
export function addAnalysisJson(search: Search, o: ReportOptions, ret: JsonObject): boolean {
  const root = search.root;
  if (!root.eval || root.visits <= 0) return false;
  const board = root.hist.board;
  const rootPla = root.nextPla;
  // KataGo flips White's values for Black; ours are for the player to move, flipped for the other player.
  const flip = (o.perspective ?? rootPla) !== rootPla;
  const side = (x: number): number => (flip ? -x : x);
  const winrate = (x: number): number => (flip ? 1 - x : x);
  const loc = (l: number): string => (l === PASS_LOC ? 'pass' : gtpLocation(board, l));

  const withOwnership = o.includeMovesOwnership || o.includeMovesOwnershipStdev;
  const a = search.analysis(withOwnership ? { movesOwnership: true, movesOwnershipStdev: o.includeMovesOwnershipStdev } : {});
  ret['moveInfos'] = a.moveInfos.map((m) => {
    const pvLen = o.preventEncore && m.pv.includes(PASS_LOC) ? pvLenUpToPhaseEnd(root.hist, m.pv) : m.pv.length;
    const info: JsonObject = {
      move: loc(m.loc),
      visits: m.visits,
      weight: r8(m.weight),
      utility: r8(side(m.utility)),
      winrate: r8(winrate(m.winrate)),
      // KataGo reports the lead as scoreMean too, for older tools, and the expected score as scoreSelfplay.
      scoreMean: r8(side(m.scoreLead)),
      scoreSelfplay: r8(side(m.scoreMean)),
      scoreLead: r8(side(m.scoreLead)),
      scoreStdev: r8(m.scoreStdev),
      prior: r8(m.prior),
      lcb: r8(winrate(m.lcb)),
      utilityLcb: r8(side(m.utilityLcb)),
      order: m.order,
      edgeVisits: m.edgeVisits,
      edgeWeight: r8(m.edgeWeight),
      playSelectionValue: r8(m.playSelectionValue),
      pv: m.pv.slice(0, pvLen).map(loc),
    };
    if (o.includeNoResultValue) info['noResultValue'] = r8(m.noResultValue);
    if (m.isSymmetryOf !== undefined) info['isSymmetryOf'] = loc(m.isSymmetryOf);
    if (o.includePVVisits) {
      info['pvVisits'] = m.pvVisits.slice(0, pvLen);
      info['pvEdgeVisits'] = m.pvEdgeVisits.slice(0, pvLen);
    }
    if (o.includeMovesOwnership) info['ownership'] = m.ownership!.map((x) => r6(side(x)));
    if (o.includeMovesOwnershipStdev) info['ownershipStdev'] = m.ownershipStdev!.map(r6);
    return info;
  });

  const v = search.rootValues();
  const raw = search.rootRawValues();
  const rootInfo: JsonObject = {
    visits: v.visits,
    // Not rounded, as in KataGo.
    weight: new JsonFloat(v.weight),
    winrate: r8(winrate(v.winrate)),
    scoreSelfplay: r8(side(v.scoreSelfplay)),
    scoreLead: r8(side(v.scoreLead)),
    scoreStdev: r8(v.scoreStdev),
    utility: r8(side(v.utility)),
    rawWinrate: r8(winrate(raw.winrate)),
    rawLead: r8(side(raw.lead)),
    rawScoreSelfplay: r8(side(raw.scoreSelfplay)),
    rawScoreSelfplayStdev: r8(raw.scoreSelfplayStdev),
    rawNoResultProb: r8(raw.noResultProb),
    rawStWrError: r8(raw.stWrError),
    rawStScoreError: r8(raw.stScoreError),
    rawVarTimeLeft: r8(raw.varTimeLeft),
    ...search.rootHashes(),
    currentPlayer: rootPla === BLACK ? 'B' : 'W',
  };
  ret['rootInfo'] = rootInfo;

  if (o.includePolicy) ret['policy'] = search.rootPolicy().map(r8);
  if (o.includeOwnership || o.includeOwnershipStdev) {
    const own = search.rootOwnership(o.includeOwnershipStdev);
    if (o.includeOwnership) ret['ownership'] = own.ownership.map((x) => r6(side(x)));
    if (o.includeOwnershipStdev) ret['ownershipStdev'] = own.ownershipStdev!.map(r6);
  }
  return true;
}
