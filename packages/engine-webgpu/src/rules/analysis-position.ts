/**
 * The position of a KataGo analysis query, built as KataGo v1.18.2's analysis engine (`cpp/command/analysis.cpp`) builds
 * it: `initialStones` placed with `Board::setStone`, the player to move first worked out as KataGo does, and the moves
 * replayed with `makeBoardMoveTolerant`, starting the history again wherever a player moves twice in a row.
 */
import { Board, BLACK, WHITE, NULL_LOC, PASS_LOC, type Stone } from './board.js';
import { GameHistory, IllegalMoveError, UnsupportedPositionError } from './history.js';
import type { Rules } from './rules.js';

/** A stone or move of a query: the player, and a board location (`Board.loc`) or `PASS_LOC`. */
export type QueryMove = readonly [Stone, number];

export interface AnalysisPosition {
  xSize: number;
  ySize: number;
  /** The rules, with the query's `whiteHandicapBonus` applied. */
  rules: Rules;
  komi: number;
  /** `initialStones`: stones on the board before the first move (no passes). */
  initialStones?: readonly QueryMove[];
  /** `moves`, in order. */
  moves: readonly QueryMove[];
  /** `initialPlayer`: who moves first. */
  initialPlayer?: Stone;
}

/** KataGo `BoardHistory::numHandicapStonesOnBoard`. */
function numHandicapStonesOnBoard(board: Board): number {
  let black = 0;
  let white = 0;
  for (let y = 0; y < board.ySize; y++) {
    for (let x = 0; x < board.xSize; x++) {
      const c = board.colors[board.loc(x, y)];
      if (c === BLACK) black++;
      else if (c === WHITE) white++;
    }
  }
  if (white !== 0) return 0;
  return black <= 1 ? 0 : black;
}

/**
 * The history KataGo's analysis engine searches at turn `turnNumber` of the query (default: after every move).
 * @throws IllegalMoveError, with KataGo's message, for a move `makeBoardMoveTolerant` refuses (a move on an occupied
 *   point, or a single-stone suicide); UnsupportedPositionError for an initial stone on a point already holding a stone
 *   of the other colour.
 */
export function analysisHistory(position: AnalysisPosition, turnNumber = position.moves.length): GameHistory {
  const { rules, komi, moves } = position;
  let board = new Board(position.xSize, position.ySize);
  for (const [pla, loc] of position.initialStones ?? []) {
    if (!board.setStone(loc, pla)) throw new UnsupportedPositionError('an initial stone on a stone of the other colour is not supported');
  }
  const initialPlayer = position.initialPlayer ?? moves[0]?.[0] ?? (numHandicapStonesOnBoard(board) > 0 ? WHITE : BLACK);
  let hist = new GameHistory(board, initialPlayer, rules, komi);
  for (let i = 0; i < turnNumber && i < moves.length; i++) {
    const [movePla, moveLoc] = moves[i]!;
    if (movePla !== hist.nextPla) {
      // KataGo clears the simple ko point and starts the history again from this position.
      board = hist.board.clone();
      board.koLoc = NULL_LOC;
      hist = new GameHistory(board, movePla, rules, komi);
    }
    if (!hist.playTolerant(moveLoc, movePla)) {
      const where = moveLoc === PASS_LOC ? 'pass' : gtpLocation(hist.board, moveLoc);
      throw new IllegalMoveError(`Illegal move ${i}: ${where}`);
    }
  }
  return hist;
}

/** KataGo `Location::toString`: a GTP coordinate, columns A to T without I, rows counted from the bottom. */
function gtpLocation(board: Board, loc: number): string {
  const x = board.x(loc);
  const y = board.y(loc);
  return 'ABCDEFGHJKLMNOPQRSTUVWXYZ'[x]! + String(board.ySize - y);
}
