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
  /** The analysis engine's `assumeMultipleStartingBlackMovesAreHandicap` config setting (default true). */
  assumeMultipleStartingBlackMovesAreHandicap?: boolean;
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
  const last = Math.min(turnNumber, position.moves.length);
  let found: GameHistory | undefined;
  replay(position, last, (turn, hist) => {
    if (turn === last) found = hist;
  });
  return found!;
}

/**
 * The histories KataGo's analysis engine searches for the turns in `turns` (0 to `moves.length`), by turn, each its own
 * copy. Every move is replayed, as KataGo does, so an illegal move anywhere is refused.
 * @throws as `analysisHistory`.
 */
export function analysisHistories(position: AnalysisPosition, turns: ReadonlySet<number>): Map<number, GameHistory> {
  const out = new Map<number, GameHistory>();
  replay(position, position.moves.length, (turn, hist) => {
    if (turns.has(turn)) out.set(turn, hist.clone());
  });
  return out;
}

/** KataGo's request loop: the position at each turn up to `lastTurn`, given to `onTurn` before the turn's move. */
function replay(position: AnalysisPosition, lastTurn: number, onTurn: (turn: number, hist: GameHistory) => void): void {
  const { rules, komi, moves } = position;
  const assumeHandicap = position.assumeMultipleStartingBlackMovesAreHandicap ?? true;
  let board = new Board(position.xSize, position.ySize);
  for (const [pla, loc] of position.initialStones ?? []) {
    if (!board.setStone(loc, pla)) throw new UnsupportedPositionError('an initial stone on a stone of the other colour is not supported');
  }
  const initialPlayer = position.initialPlayer ?? moves[0]?.[0] ?? (numHandicapStonesOnBoard(board) > 0 ? WHITE : BLACK);
  let hist = new GameHistory(board, initialPlayer, rules, komi);
  hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeHandicap);
  for (let i = 0; ; i++) {
    onTurn(i, hist);
    if (i >= lastTurn || i >= moves.length) break;
    const [movePla, moveLoc] = moves[i]!;
    if (movePla !== hist.nextPla) {
      // KataGo clears the simple ko point and starts the history again from this position.
      board = hist.board.clone();
      board.koLoc = NULL_LOC;
      hist = new GameHistory(board, movePla, rules, komi);
      hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeHandicap);
    }
    if (!hist.playTolerant(moveLoc, movePla)) {
      throw new IllegalMoveError(`Illegal move ${i}: ${moveLoc === PASS_LOC ? 'pass' : gtpLocation(hist.board, moveLoc)}`);
    }
  }
}

/** KataGo `Location::toString`: a GTP coordinate, columns A to T without I, rows counted from the bottom. */
export function gtpLocation(board: Board, loc: number): string {
  const x = board.x(loc);
  const y = board.y(loc);
  return 'ABCDEFGHJKLMNOPQRSTUVWXYZ'[x]! + String(board.ySize - y);
}
