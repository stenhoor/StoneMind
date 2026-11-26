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
  private _whiteHasMoved = false;
  private _koKeys: string[] = [];
  /** Locations the next player may not play because of superko. */
  superKoBanned: Uint8Array;
  /** KataGo `consecutiveEndingPasses`. */
  consecutiveEndingPasses = 0;
  /** Ko keys of the positions in which each player passed (KataGo `hashesBeforeBlackPass` / `hashesBeforeWhitePass`). */
  private _keysBeforePass = { [BLACK]: new Set<string>(), [WHITE]: new Set<string>() } as Record<Stone, Set<string>>;
  /** True once the game has ended (area scoring: the pass that ended the phase). */
  isGameFinished = false;
  /** The final score, White minus Black, once the game has ended. */
  finalWhiteMinusBlackScore = 0;
  /** The winner once the game has ended, or null for a draw. */
  winner: Stone | null = null;
  /**
   * KataGo `numApproxValidTurnsThisPhase`: moves that the network's history inputs may show. Playing on after the end of
   * the game caps it at 1 (the move that ended the game), as KataGo does.
   */
  numApproxValidTurnsThisPhase = 0;
  /** KataGo `wasEverOccupiedOrPlayed`: a point never occupied can't repeat a position unless the move is suicide. */
  private _wasEverOccupiedOrPlayed: Uint8Array;

  constructor(board: Board, nextPla: Stone, rules: Rules, komi: number) {
    if (rules.hasButton) throw new UnsupportedPositionError('rule sets with a button are not supported yet');
    this.rules = rules;
    this.komi = komi;
    this.initialBoard = board.clone();
    this.board = board.clone();
    this.nextPla = nextPla;
    this.superKoBanned = new Uint8Array(board.arraySize);
    this._wasEverOccupiedOrPlayed = new Uint8Array(board.arraySize);
    for (let i = 0; i < board.arraySize; i++) if (board.colors[i] === BLACK || board.colors[i] === WHITE) this._wasEverOccupiedOrPlayed[i] = 1;
    this._koKeys.push(this._koKey(this.board, nextPla));
    if (rules.scoringRule === 'TERRITORY') {
      for (let i = 0; i < board.arraySize; i++) {
        if (board.colors[i] === BLACK) this.whiteBonusScore += 1;
        else if (board.colors[i] === WHITE) this.whiteBonusScore -= 1;
      }
      this.whiteBonusScore -= board.numWhiteCaptures - board.numBlackCaptures;
    }
    this.whiteHandicapBonusScore = this._computeWhiteHandicapBonus();
  }

  private _koKey(board: Board, nextPla: Stone): string {
    // KataGo getKoHash: situational and simple ko include the player to move; positional doesn't.
    return positionKey(board, this.rules.koRule === 'POSITIONAL' ? null : nextPla);
  }

  private _computeNumHandicapStones(): number {
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

  private _computeWhiteHandicapBonus(): number {
    if (this.rules.whiteHandicapBonusRule === '0') return 0;
    const n = this._computeNumHandicapStones();
    return this.rules.whiteHandicapBonusRule === 'N' ? n : n > 1 ? n - 1 : 0;
  }

  /** KataGo `newConsecutiveEndingPassesAfterPass` in the main phase (every supported ko rule counts passes). */
  private _newConsecutiveEndingPassesAfterPass(): number {
    return this.consecutiveEndingPasses + 1;
  }

  /** KataGo `wouldBeSpightlikeEndingPass`: under simple ko, a pass in a position the same player passed in before. */
  private _wouldBeSpightlikeEndingPass(pla: Stone): boolean {
    if (this.rules.koRule !== 'SIMPLE') return false;
    return this._keysBeforePass[pla].has(this._koKey(this.board, pla));
  }

  /** KataGo `passWouldEndPhase`. */
  passWouldEndPhase(pla: Stone): boolean {
    return this._newConsecutiveEndingPassesAfterPass() >= 2 || this._wouldBeSpightlikeEndingPass(pla);
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
      this._newConsecutiveEndingPassesAfterPass() === 2 &&
      !this._wouldBeSpightlikeEndingPass(pla)
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
   * A move after the end of the game clears the result and play continues, as in KataGo (the search does this past a
   * pass it treats as not ending the game).
   * @throws IllegalMoveError for an illegal move; UnsupportedPositionError for a pass that would start the
   *   territory-scoring encore.
   */
  play(loc: number, pla: Stone): void {
    if (this.isGameFinished) {
      this.numApproxValidTurnsThisPhase = Math.min(this.numApproxValidTurnsThisPhase, 1);
      this.isGameFinished = false;
      this.winner = null;
      this.finalWhiteMinusBlackScore = 0;
    }
    if (pla !== this.nextPla) {
      // KataGo allows a player to move twice; the history then simply continues from that player.
      this.nextPla = pla;
      this._refreshSuperKo();
    }
    if (!this.isLegal(loc, pla)) throw new IllegalMoveError(`illegal move at (${this.board.x(loc)},${this.board.y(loc)})`);
    let spightlikeEndingPass = false;
    if (loc === PASS_LOC) {
      if (this.passWouldEndPhase(pla)) {
        if (this.rules.scoringRule === 'TERRITORY') throw new UnsupportedPositionError('the territory-scoring encore is not supported yet');
        if (this.rules.taxRule !== 'NONE') throw new UnsupportedPositionError('ending a game with a group tax is not supported yet');
      }
      const keyBefore = this._koKey(this.board, pla);
      this.consecutiveEndingPasses = this._newConsecutiveEndingPassesAfterPass();
      spightlikeEndingPass = this._wouldBeSpightlikeEndingPass(pla);
      this._keysBeforePass[pla].add(keyBefore);
    } else {
      this.consecutiveEndingPasses = 0;
    }
    this.boardsBefore.push(this.board.clone());
    this.board.playMoveAssumeLegal(loc, pla);
    if (loc !== PASS_LOC) this._wasEverOccupiedOrPlayed[loc] = 1;
    this.moves.push({ loc, pla });
    this.numApproxValidTurnsThisPhase += 1;
    this.nextPla = opp(pla);
    this._koKeys.push(this._koKey(this.board, this.nextPla));
    if (this.rules.scoringRule === 'TERRITORY' && loc !== PASS_LOC) this.whiteBonusScore += pla === BLACK ? 1 : -1;
    if (pla === WHITE && loc !== PASS_LOC) this._whiteHasMoved = true;
    if (!this._whiteHasMoved && pla === BLACK && this.rules.whiteHandicapBonusRule !== '0') {
      this.whiteHandicapBonusScore = this._computeWhiteHandicapBonus();
    }
    this._refreshSuperKo();
    if (this.consecutiveEndingPasses >= 2 || spightlikeEndingPass) {
      // Area scoring (territory scoring was refused above): KataGo `endAndScoreGameNow`.
      this.finalWhiteMinusBlackScore = this.areaScoreWhiteMinusBlack() + this.whiteBonusScore + this.whiteHandicapBonusScore + this.komi;
      this.winner = this.finalWhiteMinusBlackScore > 0 ? WHITE : this.finalWhiteMinusBlackScore < 0 ? BLACK : null;
      this.isGameFinished = true;
    }
  }

  private _refreshSuperKo(): void {
    this.superKoBanned.fill(0);
    if (this.rules.koRule === 'SIMPLE') return;
    const b = this.board;
    const pla = this.nextPla;
    const o = opp(pla);
    let keys: Set<string> | undefined;
    // The position's key as characters, changed in place for each candidate move and put back after.
    let chars: string[] | undefined;
    const suffix = this.rules.koRule === 'POSITIONAL' ? '' : `|${o}`;
    const idx = (loc: number) => b.y(loc) * b.xSize + b.x(loc);
    for (let y = 0; y < b.ySize; y++) {
      for (let x = 0; x < b.xSize; x++) {
        const loc = b.loc(x, y);
        if (b.colors[loc] !== EMPTY) continue;
        // KataGo: a new stone where no stone ever was makes a new position, unless the move is suicide.
        if (!this._wasEverOccupiedOrPlayed[loc] && !b.isSuicide(loc, pla)) continue;
        if (b.isIllegalSuicide(loc, pla, this.rules.multiStoneSuicideLegal) || loc === b.koLoc) continue;
        keys ??= new Set(this._koKeys);
        chars ??= Array.from(positionKey(b, null));
        // The key after the move without playing it (as KataGo's getPosHashAfterMove): the new stone, minus the
        // opponent chains it captures, or, for a suicide, minus its own chain.
        const changed: number[] = [idx(loc)];
        let captured = false;
        for (const d of b.adj) {
          const a = loc + d;
          if (b.colors[a] === o && b.getNumLiberties(a) === 1) {
            captured = true;
            for (const s of b.chainStones(a)) changed.push(idx(s));
          }
        }
        let after = String(pla);
        if (!captured && b.isSuicide(loc, pla)) {
          after = String(EMPTY);
          for (const d of b.adj) if (b.colors[loc + d] === pla) for (const s of b.chainStones(loc + d)) changed.push(idx(s));
        }
        const saved = changed.map((i) => chars![i]!);
        chars[changed[0]!] = after;
        for (let i = 1; i < changed.length; i++) chars[changed[i]!] = String(EMPTY);
        if (keys.has(chars.join('') + suffix)) this.superKoBanned[loc] = 1;
        changed.forEach((i, j) => (chars![i] = saved[j]!));
      }
    }
  }

  /**
   * A copy that can be played on independently. Boards in the history are never changed once recorded, so the copy
   * shares them.
   */
  clone(): GameHistory {
    const h = Object.create(GameHistory.prototype) as GameHistory;
    Object.assign(h, this);
    const w = h as unknown as Record<string, unknown>;
    w['board'] = this.board.clone();
    w['moves'] = this.moves.slice();
    w['boardsBefore'] = this.boardsBefore.slice();
    w['superKoBanned'] = this.superKoBanned.slice();
    w['_koKeys'] = this._koKeys.slice();
    w['_keysBeforePass'] = { [BLACK]: new Set(this._keysBeforePass[BLACK]), [WHITE]: new Set(this._keysBeforePass[WHITE]) };
    w['_wasEverOccupiedOrPlayed'] = this._wasEverOccupiedOrPlayed.slice();
    return h;
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
