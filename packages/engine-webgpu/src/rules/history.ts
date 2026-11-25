/**
 * Game history: the parts of KataGo's `BoardHistory` (`cpp/game/boardhistory.cpp`, KataGo 1.18, MIT, © David J Wu)
 * that the network's inputs depend on during the main phase of a game:
 * - legality, including simple ko and positional/situational superko,
 * - the move history and recent boards,
 * - territory scoring's "chill" of one point per stone or move, and the white handicap bonus.
 *
 * Not supported yet, and refused rather than approximated: passes (KataGo's pass-ending rules), the territory-scoring
 * encore, and rule sets with a button.
 */
import { BLACK, Board, EMPTY, PASS_LOC, WHITE, opp, type Stone } from './board.js';
import type { Rules } from './rules.js';

export class UnsupportedPositionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedPositionError';
  }
}

export class IllegalMoveError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IllegalMoveError';
  }
}

export interface HistoryMove {
  loc: number;
  pla: Stone;
}

function positionKey(board: Board, nextPla: Stone | null): string {
  let s = '';
  for (let y = 0; y < board.ySize; y++) for (let x = 0; x < board.xSize; x++) s += board.colors[board.loc(x, y)];
  return nextPla === null ? s : `${s}|${nextPla}`;
}

export class GameHistory {
  readonly rules: Rules;
  readonly komi: number;
  readonly initialBoard: Board;
  board: Board;
  readonly moves: HistoryMove[] = [];
  /** Boards before each move: boardsBefore[i] is the board before moves[i]. */
  readonly boardsBefore: Board[] = [];
  nextPla: Stone;
  whiteBonusScore = 0;
  whiteHandicapBonusScore = 0;
  #whiteHasMoved = false;
  readonly #koKeys: string[] = [];
  /** Locations the next player may not play because of superko. */
  superKoBanned: Uint8Array;

  constructor(board: Board, nextPla: Stone, rules: Rules, komi: number) {
    if (rules.hasButton) throw new UnsupportedPositionError('rule sets with a button are not supported yet');
    this.rules = rules;
    this.komi = komi;
    this.initialBoard = board.clone();
    this.board = board.clone();
    this.nextPla = nextPla;
    this.superKoBanned = new Uint8Array(board.arraySize);
    this.#koKeys.push(this.#koKey(this.board, nextPla));
    if (rules.scoringRule === 'TERRITORY') {
      for (let i = 0; i < board.arraySize; i++) {
        if (board.colors[i] === BLACK) this.whiteBonusScore += 1;
        else if (board.colors[i] === WHITE) this.whiteBonusScore -= 1;
      }
      this.whiteBonusScore -= board.numWhiteCaptures - board.numBlackCaptures;
    }
    this.whiteHandicapBonusScore = this.#computeWhiteHandicapBonus();
  }

  #koKey(board: Board, nextPla: Stone): string {
    // KataGo getKoHash: situational and simple ko include the player to move; positional doesn't.
    return positionKey(board, this.rules.koRule === 'POSITIONAL' ? null : nextPla);
  }

  #computeNumHandicapStones(): number {
    let blackTurns = 0;
    for (let i = 0; i < this.moves.length; i++) {
      const { loc, pla } = this.moves[i]!;
      if (pla !== BLACK) {
        if (i + 1 < this.moves.length && this.moves[i + 1]!.pla !== BLACK) {
          blackTurns = 0;
          break;
        }
        if (loc === PASS_LOC) continue;
        break;
      }
      if (loc !== PASS_LOC) blackTurns += 1;
    }
    let blackStones = 0;
    let whiteStones = 0;
    for (let i = 0; i < this.initialBoard.arraySize; i++) {
      if (this.initialBoard.colors[i] === BLACK) blackStones++;
      else if (this.initialBoard.colors[i] === WHITE) whiteStones++;
    }
    if (whiteStones !== 0) return 0;
    const advantage = blackStones + blackTurns;
    return advantage <= 1 ? 0 : advantage;
  }

  #computeWhiteHandicapBonus(): number {
    if (this.rules.whiteHandicapBonusRule === '0') return 0;
    const n = this.#computeNumHandicapStones();
    return this.rules.whiteHandicapBonusRule === 'N' ? n : n > 1 ? n - 1 : 0;
  }

  /** KataGo `BoardHistory::isLegal` in the main phase. */
  isLegal(loc: number, pla: Stone): boolean {
    if (pla !== this.nextPla) return false;
    if (loc === PASS_LOC) return true;
    if (this.board.isKoBanned(loc)) return false;
    if (!this.board.isLegalIgnoringKo(loc, pla, this.rules.multiStoneSuicideLegal)) return false;
    return this.superKoBanned[loc] === 0;
  }

  /**
   * Play a move.
   * @throws IllegalMoveError for an illegal move; UnsupportedPositionError for a pass.
   */
  play(loc: number, pla: Stone): void {
    if (loc === PASS_LOC) throw new UnsupportedPositionError('positions with passes are not supported yet');
    if (pla !== this.nextPla) {
      // KataGo allows a player to move twice; the history then simply continues from that player.
      this.nextPla = pla;
      this.#refreshSuperKo();
    }
    if (!this.isLegal(loc, pla)) throw new IllegalMoveError(`illegal move at (${this.board.x(loc)},${this.board.y(loc)})`);
    this.boardsBefore.push(this.board.clone());
    this.board.playMoveAssumeLegal(loc, pla);
    this.moves.push({ loc, pla });
    this.nextPla = opp(pla);
    this.#koKeys.push(this.#koKey(this.board, this.nextPla));
    if (this.rules.scoringRule === 'TERRITORY') this.whiteBonusScore += pla === BLACK ? 1 : -1;
    if (pla === WHITE) this.#whiteHasMoved = true;
    if (!this.#whiteHasMoved && pla === BLACK && this.rules.whiteHandicapBonusRule !== '0') {
      this.whiteHandicapBonusScore = this.#computeWhiteHandicapBonus();
    }
    this.#refreshSuperKo();
  }

  #refreshSuperKo(): void {
    this.superKoBanned.fill(0);
    if (this.rules.koRule === 'SIMPLE') return;
    const b = this.board;
    const keys = new Set(this.#koKeys);
    for (let y = 0; y < b.ySize; y++) {
      for (let x = 0; x < b.xSize; x++) {
        const loc = b.loc(x, y);
        if (b.colors[loc] !== EMPTY) continue;
        if (b.isIllegalSuicide(loc, this.nextPla, this.rules.multiStoneSuicideLegal) || loc === b.koLoc) continue;
        const after = b.clone();
        after.playMoveAssumeLegal(loc, this.nextPla);
        if (keys.has(this.#koKey(after, opp(this.nextPla)))) this.superKoBanned[loc] = 1;
      }
    }
  }

  /** The board as it was `n` moves ago (KataGo `getRecentBoard`). */
  recentBoard(n: number): Board {
    if (n <= 0 || this.boardsBefore.length === 0) return this.board;
    const i = this.boardsBefore.length - n;
    return i >= 0 ? this.boardsBefore[i]! : this.boardsBefore[0]!;
  }

  /** KataGo `currentSelfKomi` with draws worth half a win (the network-input default). */
  currentSelfKomi(pla: Stone): number {
    const white = this.whiteBonusScore + this.whiteHandicapBonusScore + this.komi;
    return pla === WHITE ? white : -white;
  }
}
