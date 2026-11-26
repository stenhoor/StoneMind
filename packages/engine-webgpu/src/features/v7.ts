/**
 * KataGo's V7 network inputs (22 spatial planes, 19 globals), ported from `NNInputs::fillRowV7` and `iterLadders` in
 * KataGo's `cpp/neuralnet/nninputs.cpp` (KataGo 1.18, MIT, © David J Wu), for main-phase positions, passes included.
 * Defaults match `MiscNNInputParams` as `kata-raw-nn` uses them: no playout-doubling advantage, draws worth half a win,
 * no conservative-pass or passing hacks.
 */
import { calculateArea } from '../rules/area.js';
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
  /** The model's `preferPassAliveUnderSuicideRules` flag (model version 15+). Default false. */
  preferPassAliveUnderSuicideRules?: boolean;
  /** KataGo `conservativePassAndIsRoot`: at the root, behave as if a pass would not end the game. Default false. */
  conservativePassAndIsRoot?: boolean;
  /** KataGo `enablePassingHacks`. Default false. */
  enablePassingHacks?: boolean;
  /** KataGo `maxHistory`: the most past moves the inputs may show (default 5, all of them). */
  maxHistory?: number;
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
    if (!board.setStone(board.loc(x, y), stone(p))) throw new UnsupportedPositionError(`setup stone at (${x},${y}) has no liberties`);
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
  const preferPassAliveUnderSuicideRules = options.preferPassAliveUnderSuicideRules ?? false;
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

  // Ko (main phase): the simple ko point, plus superko bans.
  if (board.koLoc !== 0) set(6, board.koLoc);
  for (let y = 0; y < ySize; y++) {
    for (let x = 0; x < xSize; x++) {
      const loc = board.loc(x, y);
      if (hist.superKoBanned[loc] && loc !== board.koLoc) set(6, loc);
    }
  }

  // Area planes (18–19). In the main phase only area scoring with no tax has them.
  let finalPhaseAndGameEndWouldNotBeWin = false;
  if (rules.scoringRule === 'AREA' && rules.taxRule === 'NONE') {
    const suicideLegalForPassAlive = rules.multiStoneSuicideLegal || preferPassAliveUnderSuicideRules;
    const areaMap = calculateArea(board, true, true, true, suicideLegalForPassAlive);
    let boardScoreForPla = 0;
    for (let y = 0; y < ySize; y++) {
      for (let x = 0; x < xSize; x++) {
        const loc = board.loc(x, y);
        if (areaMap[loc] === pla) {
          set(18, loc);
          boardScoreForPla += 1;
        } else if (areaMap[loc] === o) {
          set(19, loc);
          boardScoreForPla -= 1;
        }
      }
    }
    finalPhaseAndGameEndWouldNotBeWin = boardScoreForPla + hist.currentSelfKomi(pla) <= 0;
  } else if (rules.scoringRule === 'AREA') {
    throw new UnsupportedPositionError('area scoring with a group tax is not supported yet');
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
  } else if (hist.isGameFinished) {
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
  iterLadders(board, (loc, working) => {
    set(14, loc);
    if (board.colors[loc] === o && board.getNumLiberties(loc) > 1) for (const w of working) set(17, w);
  });
  const prev = numTurnsOfHistoryIncluded < 1 ? board : hist.recentBoard(1);
  iterLadders(prev, (loc) => (spatial[15 * area + prev.y(loc) * xSize + prev.x(loc)] = 1));
  const prevPrev = numTurnsOfHistoryIncluded < 2 ? prev : hist.recentBoard(2);
  iterLadders(prevPrev, (loc) => (spatial[16 * area + prevPrev.y(loc) * xSize + prevPrev.x(loc)] = 1));

  // Globals.
  let selfKomi = hist.currentSelfKomi(pla);
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
  if (rules.scoringRule === 'AREA') {
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

