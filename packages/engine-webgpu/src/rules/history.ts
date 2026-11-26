/**
 * Game history: the parts of KataGo's `BoardHistory` (`cpp/game/boardhistory.cpp`, KataGo 1.18, MIT, © David J Wu)
 * that the network's inputs depend on during the main phase of a game:
 * - legality, including simple ko and positional/situational superko,
 * - the move history and recent boards,
 * - territory scoring's "chill" of one point per stone or move, and the white handicap bonus,
 * - passes: consecutive ending passes and the spight-like ending under simple ko (a player passing again in a position
 *   they passed in before). Under area scoring, the pass that ends the phase ends the game, which is then scored.
 *
 * Not supported yet, and refused rather than approximated: the territory-scoring encore (a pass that would end the main
 * phase under territory scoring), no-result long cycles, and rule sets with a button.
 */
import { calculateArea } from './area.js';
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
  /** KataGo `consecutiveEndingPasses`. */
  consecutiveEndingPasses = 0;
  /** Ko keys of the positions in which each player passed (KataGo `hashesBeforeBlackPass` / `hashesBeforeWhitePass`). */
  readonly #keysBeforePass = { [BLACK]: new Set<string>(), [WHITE]: new Set<string>() } as Record<Stone, Set<string>>;
  /** True once the game has ended (area scoring: the pass that ended the phase). */
  isGameFinished = false;
  /** The final score, White minus Black, once the game has ended. */
  finalWhiteMinusBlackScore = 0;
  /** The winner once the game has ended, or null for a draw. */
  winner: Stone | null = null;

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

  /** KataGo `newConsecutiveEndingPassesAfterPass` in the main phase (every supported ko rule counts passes). */
  #newConsecutiveEndingPassesAfterPass(): number {
    return this.consecutiveEndingPasses + 1;
  }

  /** KataGo `wouldBeSpightlikeEndingPass`: under simple ko, a pass in a position the same player passed in before. */
  #wouldBeSpightlikeEndingPass(pla: Stone): boolean {
    if (this.rules.koRule !== 'SIMPLE') return false;
    return this.#keysBeforePass[pla].has(this.#koKey(this.board, pla));
  }

  /** KataGo `passWouldEndPhase`. */
  passWouldEndPhase(pla: Stone): boolean {
    return this.#newConsecutiveEndingPassesAfterPass() >= 2 || this.#wouldBeSpightlikeEndingPass(pla);
  }

  /** KataGo `passWouldEndGame`: in the main phase, only under area scoring. */
  passWouldEndGame(pla: Stone): boolean {
    return this.passWouldEndPhase(pla) && this.rules.scoringRule === 'AREA';
  }

  /** KataGo `shouldSuppressEndGameFromFriendlyPass`. */
  shouldSuppressEndGameFromFriendlyPass(pla: Stone): boolean {
    return (
      this.rules.friendlyPassOk &&
      this.rules.scoringRule === 'AREA' &&
      this.#newConsecutiveEndingPassesAfterPass() === 2 &&
      !this.#wouldBeSpightlikeEndingPass(pla)
    );
  }

  /**
   * KataGo `countAreaScoreWhiteMinusBlack` for tax-free area scoring: pass-alive and other area, without komi.
   * @param suicideLegalForPassAlive KataGo `suicideLegalForPassAlive()`.
   */
  areaScoreWhiteMinusBlack(suicideLegalForPassAlive = this.rules.multiStoneSuicideLegal): number {
    if (this.rules.scoringRule !== 'AREA' || this.rules.taxRule !== 'NONE') {
      throw new UnsupportedPositionError('scoring is supported only for area scoring with no group tax');
    }
    const area = calculateArea(this.board, true, true, true, suicideLegalForPassAlive);
    let score = 0;
    for (let y = 0; y < this.board.ySize; y++) {
      for (let x = 0; x < this.board.xSize; x++) {
        const a = area[this.board.loc(x, y)];
        if (a === WHITE) score += 1;
        else if (a === BLACK) score -= 1;
      }
    }
    return score;
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
   * @throws IllegalMoveError for an illegal move; UnsupportedPositionError for a move after the game ended, or for a
   *   pass that would start the territory-scoring encore.
   */
  play(loc: number, pla: Stone): void {
    if (this.isGameFinished) throw new UnsupportedPositionError('moves after the end of the game are not supported');
    if (pla !== this.nextPla) {
      // KataGo allows a player to move twice; the history then simply continues from that player.
      this.nextPla = pla;
      this.#refreshSuperKo();
    }
    if (!this.isLegal(loc, pla)) throw new IllegalMoveError(`illegal move at (${this.board.x(loc)},${this.board.y(loc)})`);
    let spightlikeEndingPass = false;
    if (loc === PASS_LOC) {
      if (this.passWouldEndPhase(pla)) {
        if (this.rules.scoringRule === 'TERRITORY') throw new UnsupportedPositionError('the territory-scoring encore is not supported yet');
        if (this.rules.taxRule !== 'NONE') throw new UnsupportedPositionError('ending a game with a group tax is not supported yet');
      }
      const keyBefore = this.#koKey(this.board, pla);
      this.consecutiveEndingPasses = this.#newConsecutiveEndingPassesAfterPass();
      spightlikeEndingPass = this.#wouldBeSpightlikeEndingPass(pla);
      this.#keysBeforePass[pla].add(keyBefore);
    } else {
      this.consecutiveEndingPasses = 0;
    }
    this.boardsBefore.push(this.board.clone());
    this.board.playMoveAssumeLegal(loc, pla);
    this.moves.push({ loc, pla });
    this.nextPla = opp(pla);
    this.#koKeys.push(this.#koKey(this.board, this.nextPla));
    if (this.rules.scoringRule === 'TERRITORY' && loc !== PASS_LOC) this.whiteBonusScore += pla === BLACK ? 1 : -1;
    if (pla === WHITE && loc !== PASS_LOC) this.#whiteHasMoved = true;
    if (!this.#whiteHasMoved && pla === BLACK && this.rules.whiteHandicapBonusRule !== '0') {
      this.whiteHandicapBonusScore = this.#computeWhiteHandicapBonus();
    }
    this.#refreshSuperKo();
    if (this.consecutiveEndingPasses >= 2 || spightlikeEndingPass) {
      // Area scoring (territory scoring was refused above): KataGo `endAndScoreGameNow`.
      this.finalWhiteMinusBlackScore = this.areaScoreWhiteMinusBlack() + this.whiteBonusScore + this.whiteHandicapBonusScore + this.komi;
      this.winner = this.finalWhiteMinusBlackScore > 0 ? WHITE : this.finalWhiteMinusBlackScore < 0 ? BLACK : null;
      this.isGameFinished = true;
    }
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
