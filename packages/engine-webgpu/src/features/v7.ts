/**
 * KataGo's V7 network inputs (22 spatial planes, 19 globals), ported from `NNInputs::fillRowV7` and `iterLadders` in
 * KataGo's `cpp/neuralnet/nninputs.cpp` (KataGo 1.18, MIT, © David J Wu): passes, territory scoring's encore, every
 * tax rule and the button included.
 * Defaults match `MiscNNInputParams` as `kata-raw-nn` uses them: no playout-doubling advantage, draws worth half a win,
 * no conservative-pass or passing hacks.
 */
import { calculateArea, calculateIndependentLifeArea } from '../rules/area.js';
import { BLACK, Board, PASS_LOC, WHITE, opp, type Stone } from '../rules/board.js';
import { GameHistory, UnsupportedPositionError } from '../rules/history.js';
import { parseRules } from '../rules/rules.js';
import { parseGtp } from './simple.js';
import type { NetInput } from '../reference/forward.js';
import type { Player } from '../reference/postprocess.js';

export interface EncodedPositionV7 {
  input: NetInput;
  /** One flag per point plus pass, for post-processing. */
  legal: boolean[];
  koRuleSimple: boolean;
  scoringTerritory: boolean;
  nextPlayer: Player;
}

export interface PositionSpec {
  xSize: number;
  ySize: number;
  /** A KataGo rule-set name, e.g. "tromp-taylor", "chinese", "japanese". */
  rules: string;
  komi: number;
  /** Stones on the board before any move: [player, x, y], y = 0 the top row. */
  initialStones?: readonly (readonly [Player, number, number])[];
  /** Moves in order: [player, x, y], or [player, 'pass']. */
  moves?: readonly SpecMove[];
  /** Who moves first when there are no moves (default Black). */
  initialPlayer?: Player;
}

/** A move in a {@link PositionSpec}. */
export type SpecMove = readonly [Player, number, number] | readonly [Player, 'pass'];

/** A GTP move (a coordinate such as "Q16", or "pass") as a {@link SpecMove}. */
export function gtpMove(player: Player, location: string, ySize: number): SpecMove {
  return location.trim().toLowerCase() === 'pass' ? [player, 'pass'] : [player, ...parseGtp(location, ySize)];
}

/** KataGo `MiscNNInputParams` options that change the inputs (all off for `kata-raw-nn`). */
export interface InputOptions {
  /**
   * Compute pass-alive area as if multi-stone suicide were legal (KataGo `passAliveSuicideRulesOverride`; the model's
   * `preferPassAliveUnderSuicideRules` flag, model version 15+). By default, the history's mode
   * (`alwaysComputePassAliveUnderSuicideRules`), as in KataGo.
   */
  preferPassAliveUnderSuicideRules?: boolean;
  /** KataGo `conservativePassAndIsRoot`: at the root, behave as if a pass would not end the game. Default false. */
  conservativePassAndIsRoot?: boolean;
  /** KataGo `enablePassingHacks`. Default false. */
  enablePassingHacks?: boolean;
  /** KataGo `maxHistory`: the most past moves the inputs may show (default 5, all of them). */
  maxHistory?: number;
  /** KataGo `drawEquivalentWinsForWhite`, folded into the komi input when the game can end in a draw (default 0.5). */
  drawEquivalentWinsForWhite?: number;
  /**
   * Ladder results already found, by board (see {@link LadderCache}). Not a KataGo option: the inputs are the same with
   * or without it. A search passes one so that a node's board, laddered when the node was encoded, isn't searched again
   * as its children's board one move ago and its grandchildren's two moves ago.
   */
  ladders?: LadderCache;
}

/**
 * Ladder results by board: the key is the board's colours and ko point, the value every laddered stone with its
 * working moves, in {@link iterLadders}' order. The ladder search reads only those two (and the chains, which the colours
 * fix), so a board with the same key gives the same results.
 */
export type LadderCache = Map<string, readonly (readonly [number, readonly number[]])[]>;

function ladderKey(board: Board): string {
  return String.fromCharCode.apply(null, board.colors as unknown as number[]) + '|' + board.koLoc;
}

/** {@link iterLadders} through a cache, if one is given. */
function iterLaddersCached(board: Board, cache: LadderCache | undefined, f: (loc: number, workingMoves: readonly number[]) => void): void {
  if (!cache) return iterLadders(board, f);
  const key = ladderKey(board);
  let found = cache.get(key);
  if (!found) {
    const list: [number, readonly number[]][] = [];
    iterLadders(board, (loc, working) => list.push([loc, working.slice()]));
    cache.set(key, list);
    found = list;
  }
  for (const [loc, working] of found) f(loc, working);
}

const stone = (p: Player): Stone => (p === 'B' ? BLACK : WHITE);

/** KataGo `iterLadders`: calls f(loc, workingMoves) for every stone of a laddered chain (1 or 2 liberties). */
function iterLadders(board: Board, f: (loc: number, workingMoves: readonly number[]) => void): void {
  const solved = new Map<number, boolean>();
  const copy = board.clone();
  const buf: number[] = [];
  for (let y = 0; y < board.ySize; y++) {
    for (let x = 0; x < board.xSize; x++) {
      const loc = board.loc(x, y);
      const c = board.colors[loc];
      if (c !== BLACK && c !== WHITE) continue;
      const libs = board.getNumLiberties(loc);
      if (libs !== 1 && libs !== 2) continue;
      const head = board.chainHead(loc);
      const known = solved.get(head);
      if (known !== undefined) {
        if (known) f(loc, []);
        continue;
      }
      const working: number[] = [];
      const laddered = libs === 1 ? copy.searchIsLadderCaptured(loc, true, buf) : copy.searchIsLadderCapturedAttackerFirst2Libs(loc, buf, working);
      solved.set(head, laddered);
      if (laddered) f(loc, working);
    }
  }
}

/** Build the game history for a position. */
export function buildHistory(spec: PositionSpec): GameHistory {
  const rules = parseRules(spec.rules);
  const board = new Board(spec.xSize, spec.ySize);
  for (const [p, x, y] of spec.initialStones ?? []) {
    if (!board.setStoneFailIfNoLibs(board.loc(x, y), stone(p))) throw new UnsupportedPositionError(`setup stone at (${x},${y}) has no liberties`);
  }
  const moves = spec.moves ?? [];
  const first = moves[0]?.[0] ?? spec.initialPlayer ?? 'B';
  const hist = new GameHistory(board, stone(first), rules, spec.komi);
  for (const m of moves) hist.play(m[1] === 'pass' ? PASS_LOC : board.loc(m[1], m[2]!), stone(m[0]));
  return hist;
}

/**
 * Encode a position as KataGo's V7 network input, for the player to move.
 * @param preferPassAliveUnderSuicideRules The model's flag (model version 15+); false for older models.
 */
export function encodePositionV7(spec: PositionSpec, preferPassAliveUnderSuicideRules = false): EncodedPositionV7 {
  return encodeHistoryV7(buildHistory(spec), { preferPassAliveUnderSuicideRules });
}

/** Encode a game history's current position as KataGo's V7 network input, for the player to move. */
export function encodeHistoryV7(hist: GameHistory, options: InputOptions = {}): EncodedPositionV7 {
  const suicideLegalForPassAlive =
    hist.rules.multiStoneSuicideLegal || (options.preferPassAliveUnderSuicideRules ?? hist.modes.alwaysComputePassAliveUnderSuicideRules);
  const excludeTerritoryAdjacentToAtari = hist.modes.excludeTerritoryAdjacentToAtari;
  const board = hist.board;
  const { xSize, ySize } = board;
  const area = xSize * ySize;
  const pla = hist.nextPla;
  const o = opp(pla);
  const rules = hist.rules;
  const spatial = new Float32Array(22 * area);
  const global = new Float32Array(19);
  const pos = (loc: number) => board.y(loc) * xSize + board.x(loc);
  const set = (plane: number, loc: number) => (spatial[plane * area + pos(loc)] = 1);

  for (let y = 0; y < ySize; y++) {
    for (let x = 0; x < xSize; x++) {
      const loc = board.loc(x, y);
      set(0, loc);
      const c = board.colors[loc];
      if (c === pla) set(1, loc);
      else if (c === o) set(2, loc);
      if (c === BLACK || c === WHITE) {
        const libs = board.getNumLiberties(loc);
        if (libs === 1) set(3, loc);
        else if (libs === 2) set(4, loc);
        else if (libs === 3) set(5, loc);
      }
    }
  }

  if (hist.encorePhase === 0) {
    // Ko in the main phase: the simple ko point, plus superko bans.
    if (board.koLoc !== 0) set(6, board.koLoc);
    for (let y = 0; y < ySize; y++) {
      for (let x = 0; x < xSize; x++) {
        const loc = board.loc(x, y);
        if (hist.superKoBanned[loc] && loc !== board.koLoc) set(6, loc);
      }
    }
  } else {
    // Ko in the encore: the ko captures already made in this position, and the stones blocked from recapture, which
    // need a pass for ko.
    for (let y = 0; y < ySize; y++) {
      for (let x = 0; x < xSize; x++) {
        const loc = board.loc(x, y);
        if (hist.superKoBanned[loc]) set(6, loc);
        if (hist.koRecapBlocked[loc]) set(7, loc);
      }
    }
  }

  // Area planes (18–19), not counting the group tax. Territory scoring has them only from the second encore, when
  // scoring matters.
  let areaMap: Uint8Array | null = null;
  let groupTaxAdjustmentForPla = 0;
  if (rules.scoringRule === 'AREA' && rules.taxRule === 'NONE') {
    areaMap = calculateArea(board, true, true, true, suicideLegalForPassAlive);
  } else {
    let keepTerritories = false;
    let keepStones = false;
    if (rules.scoringRule === 'AREA') {
      areaMap = new Uint8Array(0);
      keepStones = true;
    } else if (hist.encorePhase >= 2) {
      areaMap = new Uint8Array(0);
      keepTerritories = rules.taxRule === 'NONE';
    }
    if (areaMap) {
      const r = calculateIndependentLifeArea(board, keepTerritories, keepStones, excludeTerritoryAdjacentToAtari, suicideLegalForPassAlive);
      areaMap = r.area;
      if (rules.taxRule === 'ALL') {
        groupTaxAdjustmentForPla = pla === WHITE ? -2 * r.whiteMinusBlackIndependentLifeRegionCount : 2 * r.whiteMinusBlackIndependentLifeRegionCount;
      }
    }
  }
  let finalPhaseAndGameEndWouldNotBeWin = false;
  if (areaMap) {
    let boardScoreForPla = groupTaxAdjustmentForPla;
    for (let y = 0; y < ySize; y++) {
      for (let x = 0; x < xSize; x++) {
        const loc = board.loc(x, y);
        if (areaMap[loc] === pla) {
          set(18, loc);
          boardScoreForPla += 1;
        } else if (areaMap[loc] === o) {
          set(19, loc);
          boardScoreForPla -= 1;
        } else if (rules.scoringRule === 'TERRITORY') {
          // In the second encore: the stones that were there when it started count too.
          if (board.colors[loc] === pla && hist.secondEncoreStartColors[loc] === pla) {
            set(18, loc);
            boardScoreForPla += 1;
          } else if (board.colors[loc] === o && hist.secondEncoreStartColors[loc] === o) {
            set(19, loc);
            boardScoreForPla -= 1;
          }
        }
      }
    }
    finalPhaseAndGameEndWouldNotBeWin = Math.fround(boardScoreForPla + hist.currentSelfKomi(pla, options.drawEquivalentWinsForWhite ?? 0.5)) <= 0;
  }

  // History: the last five moves, while they alternate back from the opponent's. A pass sets global k - 1 instead of
  // plane 8 + k. History is hidden when a pass would end the game but the net is to behave as if it won't.
  let maxTurnsOfHistoryToInclude = 5;
  let suppressPassWouldEndPhase = false;
  if (
    hist.passWouldEndGame(pla) &&
    ((options.conservativePassAndIsRoot ?? false) ||
      hist.shouldSuppressEndGameFromFriendlyPass(pla) ||
      ((options.enablePassingHacks ?? false) && finalPhaseAndGameEndWouldNotBeWin))
  ) {
    maxTurnsOfHistoryToInclude = 0;
    suppressPassWouldEndPhase = true;
  } else if (hist.isGameFinished || hist.isPastNormalPhaseEnd) {
    // One of the passes that ended it.
    maxTurnsOfHistoryToInclude = 1;
  }
  maxTurnsOfHistoryToInclude = Math.min(maxTurnsOfHistoryToInclude, options.maxHistory ?? 5);
  let numTurnsOfHistoryIncluded = 0;
  const ms = hist.moves;
  const amountOfHistoryToTryToUse = Math.min(maxTurnsOfHistoryToInclude, hist.numApproxValidTurnsThisPhase, ms.length);
  for (let k = 1; k <= amountOfHistoryToTryToUse; k++) {
    const m = ms[ms.length - k]!;
    if (m.pla !== (k % 2 === 1 ? o : pla)) break;
    numTurnsOfHistoryIncluded = k;
    if (m.loc === PASS_LOC) global[k - 1] = 1;
    else set(8 + k, m.loc);
  }

  // Ladders: the current board, then the boards one and two moves ago (only as far back as history is included).
  const ladders = options.ladders;
  iterLaddersCached(board, ladders, (loc, working) => {
    set(14, loc);
    if (board.colors[loc] === o && board.getNumLiberties(loc) > 1) for (const w of working) set(17, w);
  });
  const prev = numTurnsOfHistoryIncluded < 1 ? board : hist.recentBoard(1);
  iterLaddersCached(prev, ladders, (loc) => (spatial[15 * area + prev.y(loc) * xSize + prev.x(loc)] = 1));
  const prevPrev = numTurnsOfHistoryIncluded < 2 ? prev : hist.recentBoard(2);
  iterLaddersCached(prevPrev, ladders, (loc) => (spatial[16 * area + prevPrev.y(loc) * xSize + prevPrev.x(loc)] = 1));

  // The second encore's starting stones (20–21).
  if (hist.encorePhase >= 2) {
    for (let y = 0; y < ySize; y++) {
      for (let x = 0; x < xSize; x++) {
        const loc = board.loc(x, y);
        if (hist.secondEncoreStartColors[loc] === pla) set(20, loc);
        else if (hist.secondEncoreStartColors[loc] === o) set(21, loc);
      }
    }
  }

  // Globals.
  let selfKomi = hist.currentSelfKomi(pla, options.drawEquivalentWinsForWhite ?? 0.5);
  const clip = area + 20;
  selfKomi = Math.max(-clip, Math.min(clip, selfKomi));
  global[5] = Math.fround(selfKomi) / 20;
  if (rules.koRule === 'POSITIONAL') {
    global[6] = 1;
    global[7] = 0.5;
  } else if (rules.koRule === 'SITUATIONAL') {
    global[6] = 1;
    global[7] = -0.5;
  }
  if (rules.multiStoneSuicideLegal) global[8] = 1;
  if (!suppressPassWouldEndPhase && hist.passWouldEndPhase(pla)) global[14] = 1;
  if (rules.scoringRule === 'TERRITORY') global[9] = 1;
  if (rules.taxRule === 'SEKI') global[10] = 1;
  else if (rules.taxRule === 'ALL') {
    global[10] = 1;
    global[11] = 1;
  }
  if (hist.encorePhase > 0) global[12] = 1;
  if (hist.encorePhase > 1) global[13] = 1;
  // (15–16, the playout doubling advantage, stay 0, as in kata-raw-nn and the analysis engine.)
  if (hist.hasButton) global[17] = 1;
  if (rules.scoringRule === 'AREA' || hist.encorePhase >= 2) {
    const even = area % 2 === 0;
    const komiFloor = even ? Math.floor(selfKomi / 2) * 2 : Math.floor((selfKomi - 1) / 2) * 2 + 1;
    const delta = Math.min(2, Math.max(0, selfKomi - komiFloor));
    global[18] = delta < 0.5 ? delta : delta < 1.5 ? 1 - delta : delta - 2;
  }

  // Legal moves for the player to move, plus pass.
  const legal: boolean[] = [];
  for (let y = 0; y < ySize; y++) for (let x = 0; x < xSize; x++) legal.push(hist.isLegal(board.loc(x, y), pla));
  legal.push(true);

  return {
    input: { xSize, ySize, spatial, global },
    legal,
    koRuleSimple: rules.koRule === 'SIMPLE',
    scoringTerritory: rules.scoringRule === 'TERRITORY',
    nextPlayer: pla === BLACK ? 'B' : 'W',
  };
}

