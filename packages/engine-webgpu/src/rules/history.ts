/**
 * Game history: KataGo's `BoardHistory` (`cpp/game/boardhistory.cpp`, KataGo 1.18, MIT, © David J Wu), ported from
 * KataGo's v1.18.2 tag:
 * - legality, including simple ko and positional/situational superko,
 * - the move history and recent boards,
 * - passes: consecutive ending passes and the spight-like ending (a player passing again in a position they passed in
 *   before) under simple ko and in the encore,
 * - territory scoring: its "chill" of one point per stone or move, and its encore. The encore's two phases follow the
 *   main phase; in them, a ko captured may not be recaptured until its capturer's opponent passes for it, and each ko
 *   capture may be made once per position. The second encore's starting stones are the ones scored.
 * - scoring at the end of the game: area or territory scoring, with no tax, the seki tax or the tax on all groups,
 * - no-result long cycles: a position repeated three times, under simple ko or in the encore, ends the game with no
 *   result,
 * - the white handicap bonus,
 * - the button: under rule sets with one, the first pass in the main phase takes it, worth half a point.
 *
 * Not kept, as only KataGo's older network input versions or its self-play read them: `numConsecValidTurnsThisGame`,
 * `numTurnsThisPhase`, resignation, and the initial turn number.
 */
import { calculateArea, calculateIndependentLifeArea } from './area.js';
import { BLACK, Board, EMPTY, NULL_LOC, PASS_LOC, WHITE, opp, type Stone } from './board.js';
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

/**
 * KataGo `BoardHistoryModes`: how a history computes pass-alive area and territory. The search sets them from the
 * network (KataGo `Search::resolveHistoryModes`).
 */
export interface HistoryModes {
  /** Compute pass-alive area as if multi-stone suicide were legal, even when the rules forbid it. */
  readonly alwaysComputePassAliveUnderSuicideRules: boolean;
  /** Territory scoring doesn't count empty points next to a chain of the territory's owner in atari. */
  readonly excludeTerritoryAdjacentToAtari: boolean;
}

/** KataGo `BoardHistoryModes()`: both off. */
export const DEFAULT_HISTORY_MODES: HistoryModes = Object.freeze({
  alwaysComputePassAliveUnderSuicideRules: false,
  excludeTerritoryAdjacentToAtari: false,
});

/** KataGo `EncoreKoCapture`: a ko capture made in the encore, which may be made only once in that position. */
interface EncoreKoCapture {
  /** The position before the capture (KataGo `posHashBeforeMove`: the stones only). */
  readonly positionBefore: string;
  readonly loc: number;
  readonly pla: Stone;
}

function positionKey(board: Board, nextPla: Stone | null): string {
  let s = '';
  for (let y = 0; y < board.ySize; y++) for (let x = 0; x < board.xSize; x++) s += board.colors[board.loc(x, y)];
  return nextPla === null ? s : `${s}|${nextPla}`;
}

/** The history `clone()` is copying, while it constructs the copy. */
let cloneSource: GameHistory | null = null;

export class GameHistory {
  readonly rules: Rules;
  readonly komi: number;
  readonly initialBoard: Board;
  /** KataGo `initialEncorePhase`: the encore phase the history started in. */
  readonly initialEncorePhase: number;
  /** KataGo `modes`. */
  modes: HistoryModes;
  board: Board;
  readonly moves: HistoryMove[] = [];
  /** KataGo `preventEncoreHistory`: for each move, whether it was played with the encore prevented. */
  readonly preventEncoreHistory: boolean[] = [];
  /** Boards before each move: boardsBefore[i] is the board before moves[i]. */
  readonly boardsBefore: Board[] = [];
  nextPla: Stone;
  whiteBonusScore = 0;
  whiteHandicapBonusScore = 0;
  private _whiteHasMoved = false;
  private _assumeMultipleStartingBlackMovesAreHandicap = true;
  /**
   * KataGo `koHashHistory`: the ko keys of the positions since the start of the phase, or since the last pass where
   * passes clear it (simple ko and the encore). Keys stand in for KataGo's hashes, so two keys are equal exactly when
   * KataGo's hashes are equal barring a hash collision.
   */
  private _koKeys: string[] = [];
  /** Locations the next player may not play because of superko, or in the encore a ko capture already made. */
  superKoBanned: Uint8Array;
  /** KataGo `consecutiveEndingPasses`. */
  consecutiveEndingPasses = 0;
  /** Ko keys of the positions in which each player passed (KataGo `hashesBeforeBlackPass` / `hashesBeforeWhitePass`). */
  private _keysBeforePass = { [BLACK]: new Set<string>(), [WHITE]: new Set<string>() } as Record<Stone, Set<string>>;
  /** KataGo `encorePhase`: 0 in the main phase; 1 and 2 in the territory-scoring encore. */
  encorePhase: number;
  /** KataGo `koRecapBlocked`: in the encore, stones that captured a ko and may not be recaptured without a pass for ko. */
  koRecapBlocked: Uint8Array;
  /** The ko recapture blocks as part of a ko key (KataGo `koRecapBlockHash`): their locations, or '' for none. */
  private _koRecapBlockKey = '';
  /** KataGo `koCapturesInEncore`. */
  private _koCapturesInEncore: EncoreKoCapture[] = [];
  /** KataGo `secondEncoreStartColors`: the board when the second encore started, whose stones are the ones scored. */
  secondEncoreStartColors: Uint8Array;
  /**
   * KataGo `isPastNormalPhaseEnd`: the main phase or first encore ended while the encore was prevented, so play went on in
   * the same phase.
   */
  isPastNormalPhaseEnd = false;
  /** KataGo `hasButton`: whether there's a button to take (rule sets with a button, in the main phase, until a pass). */
  hasButton = false;
  /** True once the game has ended (KataGo `isGameFinished`). */
  isGameFinished = false;
  /** The final score, White minus Black, once the game has ended by scoring. */
  finalWhiteMinusBlackScore = 0;
  /** The winner once the game has ended by scoring, or null for a draw or no result. */
  winner: Stone | null = null;
  /** KataGo `isScored`: the game ended by scoring. */
  isScored = false;
  /** KataGo `isNoResult`: the game ended with no result, after a long cycle. */
  isNoResult = false;
  /**
   * KataGo `numApproxValidTurnsThisPhase`: moves that the network's history inputs may show. Playing on after the end of
   * the game caps it at 1 (the move that ended the game), as KataGo does.
   */
  numApproxValidTurnsThisPhase = 0;
  /** KataGo `wasEverOccupiedOrPlayed`: a point never occupied can't repeat a position unless the move is suicide. */
  private _wasEverOccupiedOrPlayed: Uint8Array;

  /**
   * KataGo's `BoardHistory` constructor.
   * @param encorePhase The phase to start in: 0 for the main phase, or 1 or 2 for the encore (territory scoring only).
   */
  constructor(board: Board, nextPla: Stone, rules: Rules, komi: number, encorePhase = 0, modes: HistoryModes = DEFAULT_HISTORY_MODES) {
    const source = cloneSource;
    if (source) {
      // A copy (clone()): its fields, set above, give it the same shape as every other history; now their values. The
      // boards recorded in the history are never changed, so the copy shares them.
      cloneSource = null;
      Object.assign(this, source);
      this.rules = source.rules;
      this.komi = source.komi;
      this.initialBoard = source.initialBoard;
      this.initialEncorePhase = source.initialEncorePhase;
      this.modes = source.modes;
      this.board = source.board.clone();
      this.nextPla = source.nextPla;
      this.moves = source.moves.slice();
      this.preventEncoreHistory = source.preventEncoreHistory.slice();
      this.boardsBefore = source.boardsBefore.slice();
      this.superKoBanned = source.superKoBanned.slice();
      this._koKeys = source._koKeys.slice();
      this._keysBeforePass = { [BLACK]: new Set(source._keysBeforePass[BLACK]), [WHITE]: new Set(source._keysBeforePass[WHITE]) };
      this.encorePhase = source.encorePhase;
      // The recapture blocks are only written in the encore (a new array when it starts), and the second encore's
      // colours are never changed once set (a new array when it starts), so before then the copy shares them.
      this.koRecapBlocked = source.encorePhase > 0 ? source.koRecapBlocked.slice() : source.koRecapBlocked;
      this._koCapturesInEncore = source._koCapturesInEncore.slice();
      this.secondEncoreStartColors = source.secondEncoreStartColors;
      this._wasEverOccupiedOrPlayed = source._wasEverOccupiedOrPlayed.slice();
      return;
    }
    if (!(encorePhase === 0 || encorePhase === 1 || encorePhase === 2)) throw new RangeError(`encore phase ${encorePhase}: must be 0, 1 or 2`);
    if (encorePhase > 0 && rules.scoringRule !== 'TERRITORY') throw new RangeError('only territory scoring has an encore');
    this.rules = rules;
    this.komi = komi;
    this.initialBoard = board.clone();
    this.initialEncorePhase = encorePhase;
    this.modes = modes;
    this.board = board.clone();
    this.nextPla = nextPla;
    this.encorePhase = encorePhase;
    // In the main phase only. (KataGo's BoardHistory::clear reads the phase before setting it, so its constructor gives a
    // history started in the encore a button too; its analysis engine never starts one there.)
    this.hasButton = rules.hasButton && encorePhase === 0;
    this.superKoBanned = new Uint8Array(board.arraySize);
    this.koRecapBlocked = new Uint8Array(board.arraySize);
    this.secondEncoreStartColors = encorePhase === 2 ? board.colors.slice() : new Uint8Array(board.arraySize);
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

  /**
   * KataGo `getKoHash`: the position, with the player to move under situational and simple ko and in the encore, and
   * with the encore's ko recapture blocks.
   */
  private _koKey(board: Board, pla: Stone): string {
    const withPla = this.rules.koRule === 'SITUATIONAL' || this.rules.koRule === 'SIMPLE' || this.encorePhase > 0;
    const key = positionKey(board, withPla ? pla : null);
    return this._koRecapBlockKey === '' ? key : `${key}|r${this._koRecapBlockKey}`;
  }

  /** KataGo `setModes`. */
  setModes(modes: HistoryModes): void {
    this.modes = modes;
  }

  /** KataGo `suicideLegalForPassAlive`. */
  suicideLegalForPassAlive(): boolean {
    return this.rules.multiStoneSuicideLegal || this.modes.alwaysComputePassAliveUnderSuicideRules;
  }

  /**
   * KataGo `setAssumeMultipleStartingBlackMovesAreHandicap`: whether Black's moves at the start of the game, before
   * White's first move, count as handicap stones for the white handicap bonus. On by default, as in KataGo's analysis
   * engine (KataGo's own default is off; its analysis engine turns it on unless its config says otherwise).
   */
  setAssumeMultipleStartingBlackMovesAreHandicap(b: boolean): void {
    this._assumeMultipleStartingBlackMovesAreHandicap = b;
    this.whiteHandicapBonusScore = this._computeWhiteHandicapBonus();
  }

  private _computeNumHandicapStones(): number {
    let blackTurns = 0;
    for (let i = 0; this._assumeMultipleStartingBlackMovesAreHandicap && i < this.moves.length; i++) {
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

  /** KataGo `newConsecutiveEndingPassesAfterPass` (every supported ko rule counts passes). */
  private _newConsecutiveEndingPassesAfterPass(): number {
    return this.consecutiveEndingPasses + 1;
  }

  /**
   * KataGo `phaseHasSpightlikeEndingAndPassHistoryClearing`: under simple ko and in the encore, passes clear the ko
   * history, and a pass in a position the same player passed in before ends the phase.
   */
  private _phaseHasSpightlikeEndingAndPassHistoryClearing(): boolean {
    return this.encorePhase > 0 || this.rules.koRule === 'SIMPLE';
  }

  /** KataGo `wouldBeSpightlikeEndingPass`. */
  private _wouldBeSpightlikeEndingPass(pla: Stone, koKeyBeforeMove: string): boolean {
    return this._phaseHasSpightlikeEndingAndPassHistoryClearing() && this._keysBeforePass[pla].has(koKeyBeforeMove);
  }

  /**
   * KataGo `wouldBeSpightlikeEndingPass` for a pass now. The key is only made when it could matter: KataGo's answer
   * without making it, as the search asks this for every node.
   */
  private _passNowWouldBeSpightlikeEnding(pla: Stone): boolean {
    if (!this._phaseHasSpightlikeEndingAndPassHistoryClearing() || this._keysBeforePass[pla].size === 0) return false;
    return this._keysBeforePass[pla].has(this._koKey(this.board, pla));
  }

  /** KataGo `passWouldEndPhase`. */
  passWouldEndPhase(pla: Stone): boolean {
    return this._newConsecutiveEndingPassesAfterPass() >= 2 || this._passNowWouldBeSpightlikeEnding(pla);
  }

  /** KataGo `passWouldEndGame`. */
  passWouldEndGame(pla: Stone): boolean {
    return this.passWouldEndPhase(pla) && this.isFinalPhase();
  }

  /** KataGo `shouldSuppressEndGameFromFriendlyPass`. */
  shouldSuppressEndGameFromFriendlyPass(pla: Stone): boolean {
    return (
      this.rules.friendlyPassOk &&
      this.rules.scoringRule === 'AREA' &&
      this._newConsecutiveEndingPassesAfterPass() === 2 &&
      !this._passNowWouldBeSpightlikeEnding(pla)
    );
  }

  /** KataGo `isFinalPhase`: the phase whose end ends the game. */
  isFinalPhase(): boolean {
    return this.rules.scoringRule === 'AREA' || (this.rules.scoringRule === 'TERRITORY' && this.encorePhase >= 2);
  }

  /** KataGo `countAreaScoreWhiteMinusBlack`: the area score without komi or bonuses, and the area. */
  countAreaScoreWhiteMinusBlack(): { score: number; area: Uint8Array } {
    const board = this.board;
    let score = 0;
    let area: Uint8Array;
    if (this.rules.taxRule === 'NONE') {
      area = calculateArea(board, true, true, true, this.suicideLegalForPassAlive());
    } else {
      const r = calculateIndependentLifeArea(board, false, true, this.modes.excludeTerritoryAdjacentToAtari, this.suicideLegalForPassAlive());
      area = r.area;
      if (this.rules.taxRule === 'ALL') score -= 2 * r.whiteMinusBlackIndependentLifeRegionCount;
    }
    for (let y = 0; y < board.ySize; y++) {
      for (let x = 0; x < board.xSize; x++) {
        const a = area[board.loc(x, y)];
        if (a === WHITE) score += 1;
        else if (a === BLACK) score -= 1;
      }
    }
    return { score, area };
  }

  /**
   * KataGo `countTerritoryAreaScoreWhiteMinusBlack`: the territory score without komi or bonuses (stones count here, as
   * the chill has already taken a point for each), and the area, which also marks the stones scored.
   */
  countTerritoryAreaScoreWhiteMinusBlack(): { score: number; area: Uint8Array } {
    const board = this.board;
    const keepTerritories = this.rules.taxRule === 'NONE';
    const r = calculateIndependentLifeArea(board, keepTerritories, false, this.modes.excludeTerritoryAdjacentToAtari, this.suicideLegalForPassAlive());
    const area = r.area;
    let score = 0;
    for (let y = 0; y < board.ySize; y++) {
      for (let x = 0; x < board.xSize; x++) {
        const loc = board.loc(x, y);
        if (area[loc] === WHITE) score += 1;
        else if (area[loc] === BLACK) score -= 1;
        else {
          // Before the second encore, every stone on the board counts (KataGo: so a game ended early scores right).
          if (board.colors[loc] === WHITE && (this.encorePhase < 2 || this.secondEncoreStartColors[loc] === WHITE)) {
            score += 1;
            area[loc] = WHITE;
          }
          if (board.colors[loc] === BLACK && (this.encorePhase < 2 || this.secondEncoreStartColors[loc] === BLACK)) {
            score -= 1;
            area[loc] = BLACK;
          }
        }
      }
    }
    if (this.rules.taxRule === 'ALL') score -= 2 * r.whiteMinusBlackIndependentLifeRegionCount;
    return { score, area };
  }

  /** KataGo `getAreaNow`: the area the game would be scored with now. */
  getAreaNow(): Uint8Array {
    return this.rules.scoringRule === 'AREA' ? this.countAreaScoreWhiteMinusBlack().area : this.countTerritoryAreaScoreWhiteMinusBlack().area;
  }

  /** KataGo `endAndScoreGameNow`: ends the game and scores it. Returns the area it was scored with. */
  endAndScoreGameNow(): Uint8Array {
    const { score, area } = this.rules.scoringRule === 'AREA' ? this.countAreaScoreWhiteMinusBlack() : this.countTerritoryAreaScoreWhiteMinusBlack();
    // In C++ floats, as KataGo: int + float + float + float.
    const f = Math.fround;
    // A button not yet taken goes to the player to move.
    if (this.hasButton) {
      this.hasButton = false;
      this.whiteBonusScore = f(this.whiteBonusScore + (this.nextPla === WHITE ? 0.5 : -0.5));
    }
    this._setFinalScoreAndWinner(f(f(f(score + this.whiteBonusScore) + this.whiteHandicapBonusScore) + f(this.komi)));
    this.isScored = true;
    this.isNoResult = false;
    this.isGameFinished = true;
    this.isPastNormalPhaseEnd = false;
    return area;
  }

  /**
   * KataGo `whiteKomiAdjustmentForDraws`: the value of a draw for White, less half a win, when the game can end in one
   * (KataGo `gameResultWillBeInteger`: an integer komi without a button, or a half-integer one with), else 0. A C++
   * float, as KataGo's.
   */
  whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite: number): number {
    return Number.isInteger(this.komi) !== this.rules.hasButton ? Math.fround(drawEquivalentWinsForWhite - 0.5) : 0;
  }

  /** KataGo `setFinalScoreAndWinner`. */
  private _setFinalScoreAndWinner(score: number): void {
    this.finalWhiteMinusBlackScore = score;
    this.winner = score > 0 ? WHITE : score < 0 ? BLACK : null;
  }

  /** KataGo `setKoRecapBlocked`. */
  private _setKoRecapBlocked(loc: number, b: boolean): void {
    if ((this.koRecapBlocked[loc] === 1) === b) return;
    this.koRecapBlocked[loc] = b ? 1 : 0;
    let key = '';
    for (let i = 0; i < this.koRecapBlocked.length; i++) if (this.koRecapBlocked[i]) key += `${i},`;
    this._koRecapBlockKey = key;
  }

  /**
   * KataGo `isPassForKo`: in the encore, a move on a stone that captured a ko and is blocked from recapture, or a move
   * that would recapture it, is a pass for that ko: it lifts the block and does nothing else.
   */
  isPassForKo(loc: number, pla: Stone): boolean {
    if (this.encorePhase <= 0 || loc === PASS_LOC || loc === NULL_LOC || loc >= this.board.arraySize) return false;
    const b = this.board;
    const o = opp(pla);
    if (b.colors[loc] === o && this.koRecapBlocked[loc] && b.getChainSize(loc) === 1 && b.getNumLiberties(loc) === 1) return true;
    const koCaptureLoc = b.getKoCaptureLoc(loc, pla);
    return koCaptureLoc !== NULL_LOC && this.koRecapBlocked[koCaptureLoc] === 1 && b.colors[koCaptureLoc] === o;
  }

  /** KataGo `BoardHistory::isLegal`. */
  isLegal(loc: number, pla: Stone): boolean {
    if (pla !== this.nextPla) return false;
    if (this.encorePhase > 0) {
      // Ko moves blocked from recapture in the encore are passes for ko, so they're legal.
      if (this.isPassForKo(loc, pla)) return true;
    } else if (this.board.isKoBanned(loc)) {
      // Ko bans apply only in the main phase; the encore's ko rules are its own.
      return false;
    }
    if (loc === PASS_LOC) return true;
    if (!this.board.isLegalIgnoringKo(loc, pla, this.rules.multiStoneSuicideLegal)) return false;
    return this.superKoBanned[loc] === 0;
  }

  /** KataGo `isLegalTolerant`: legal for `playTolerant`. */
  isLegalTolerant(loc: number, pla: Stone): boolean {
    return this.isPassForKo(loc, pla) || loc === PASS_LOC || this.board.isLegalIgnoringKo(loc, pla, true);
  }

  /**
   * Play a move.
   * A move after the end of the game clears the result and play continues, as in KataGo (the search does this past a
   * pass it treats as not ending the game).
   * @param preventEncore KataGo's `preventEncore`: under territory scoring, the end of the main phase or of the first
   *   encore doesn't start the next phase, and play continues in the same phase.
   * @throws IllegalMoveError for an illegal move.
   */
  play(loc: number, pla: Stone, preventEncore = false): void {
    this._play(loc, pla, false, preventEncore);
  }

  /**
   * KataGo `makeBoardMoveTolerant`, as its analysis engine replays a query's moves: a move on an empty point is played
   * even if ko or superko forbids it, or it's a suicide the rules forbid. (KataGo then also resets
   * `numConsecValidTurnsThisGame`, which only its older input versions read, so it isn't kept here.) Returns false,
   * changing nothing, for a move on an occupied point (except a pass for ko) or a single-stone suicide.
   */
  playTolerant(loc: number, pla: Stone, preventEncore = false): boolean {
    if (!this.isLegalTolerant(loc, pla)) return false;
    this._play(loc, pla, true, preventEncore);
    return true;
  }

  /** KataGo `makeBoardMoveAssumeLegal`. */
  private _play(loc: number, pla: Stone, tolerant: boolean, preventEncore: boolean): void {
    if (pla !== this.nextPla) {
      // KataGo allows a player to move twice; the history then simply continues from that player.
      this.nextPla = pla;
      this._refreshSuperKo();
    }
    if (!tolerant && !this.isLegal(loc, pla)) throw new IllegalMoveError(`illegal move at (${this.board.x(loc)},${this.board.y(loc)})`);
    const board = this.board;
    const positionBefore = this.encorePhase > 0 ? positionKey(board, null) : '';

    // A move after the game or phase ended.
    if (this.isGameFinished || this.isPastNormalPhaseEnd) this.numApproxValidTurnsThisPhase = Math.min(this.numApproxValidTurnsThisPhase, 1);
    this.isGameFinished = false;
    this.isPastNormalPhaseEnd = false;
    this.winner = null;
    this.finalWhiteMinusBlackScore = 0;
    this.isScored = false;
    this.isNoResult = false;

    let isSpightlikeEndingPass = false;
    if (loc !== PASS_LOC) {
      this.consecutiveEndingPasses = 0;
    } else if (this.hasButton) {
      // The pass takes the button, worth half a point, and isn't an ending pass. Taking it clears the ko histories (as if
      // positions with and without the button were different).
      this.hasButton = false;
      this.whiteBonusScore = Math.fround(this.whiteBonusScore + (pla === WHITE ? 0.5 : -0.5));
      this.consecutiveEndingPasses = 0;
      this._keysBeforePass = { [BLACK]: new Set<string>(), [WHITE]: new Set<string>() } as Record<Stone, Set<string>>;
      this._koKeys = [];
    } else {
      // Under simple ko and in the encore, passes clear the ko history (for no-result cycles, and in the encore its
      // three-fold repetition check), but not the positions passed in, which end the phase on a repeat.
      if (this._phaseHasSpightlikeEndingAndPassHistoryClearing()) this._koKeys = [];
      const keyBefore = this._koKey(board, pla);
      this.consecutiveEndingPasses = this._newConsecutiveEndingPassesAfterPass();
      isSpightlikeEndingPass = this._wouldBeSpightlikeEndingPass(pla, keyBefore);
      this._keysBeforePass[pla].add(keyBefore);
    }

    this.boardsBefore.push(board.clone());

    // A pass for ko in the encore lifts a ko recapture block and does nothing else.
    let wasPassForKo = false;
    if (this.encorePhase > 0 && loc !== PASS_LOC) {
      if (board.colors[loc] === opp(pla) && this.koRecapBlocked[loc]) {
        this._setKoRecapBlocked(loc, false);
        wasPassForKo = true;
        // The board sees the same player move twice; clear the simple ko point.
        board.koLoc = NULL_LOC;
      } else {
        const koCaptureLoc = board.getKoCaptureLoc(loc, pla);
        if (koCaptureLoc !== NULL_LOC && this.koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] === opp(pla)) {
          this._setKoRecapBlocked(koCaptureLoc, false);
          wasPassForKo = true;
          board.koLoc = NULL_LOC;
        }
      }
    }
    if (!wasPassForKo) {
      board.playMoveAssumeLegal(loc, pla);
      if (this.encorePhase > 0) {
        // A ko capture blocks recapture, and is recorded so it isn't made again in this position. The simple ko point
        // plays no further part.
        if (board.koLoc !== NULL_LOC) {
          this._setKoRecapBlocked(loc, true);
          this._koCapturesInEncore.push({ positionBefore, loc, pla });
          board.koLoc = NULL_LOC;
        }
        for (let y = 0; y < board.ySize; y++) {
          for (let x = 0; x < board.xSize; x++) {
            const l = board.loc(x, y);
            if (board.colors[l] === EMPTY && this.koRecapBlocked[l]) this._setKoRecapBlocked(l, false);
          }
        }
      }
    }

    this.nextPla = opp(pla);
    this._koKeys.push(this._koKey(board, this.nextPla));
    this.moves.push({ loc, pla });
    this.preventEncoreHistory.push(preventEncore);
    this.numApproxValidTurnsThisPhase += 1;
    if (loc !== PASS_LOC) this._wasEverOccupiedOrPlayed[loc] = 1;

    this._refreshSuperKo();

    // Territory scoring chills one point per move in the main phase and the first encore.
    if (this.rules.scoringRule === 'TERRITORY' && this.encorePhase <= 1 && loc !== PASS_LOC && !wasPassForKo) {
      this.whiteBonusScore += pla === BLACK ? 1 : -1;
    }

    if (pla === WHITE && loc !== PASS_LOC) this._whiteHasMoved = true;
    if (this._assumeMultipleStartingBlackMovesAreHandicap && !this._whiteHasMoved && pla === BLACK && this.rules.whiteHandicapBonusRule !== '0') {
      this.whiteHandicapBonusScore = this._computeWhiteHandicapBonus();
    }

    // Phase transitions and the end of the game.
    if (this.consecutiveEndingPasses >= 2 || isSpightlikeEndingPass) {
      if (this.rules.scoringRule === 'AREA' || this.encorePhase >= 2) {
        this.endAndScoreGameNow();
      } else if (preventEncore) {
        this.isPastNormalPhaseEnd = true;
        this.numApproxValidTurnsThisPhase = Math.min(this.numApproxValidTurnsThisPhase, 1);
      } else {
        this.encorePhase += 1;
        this.numApproxValidTurnsThisPhase = 0;
        if (this.encorePhase === 2) this.secondEncoreStartColors = board.colors.slice();
        this.superKoBanned.fill(0);
        this.consecutiveEndingPasses = 0;
        this._keysBeforePass[BLACK].clear();
        this._keysBeforePass[WHITE].clear();
        this.koRecapBlocked = new Uint8Array(this.koRecapBlocked.length);
        this._koRecapBlockKey = '';
        this._koCapturesInEncore = [];
        this._koKeys = [this._koKey(board, opp(pla))];
      }
    }

    // A position repeated three times, under simple ko or in the encore, ends the game with no result.
    if (loc !== PASS_LOC && (this.encorePhase > 0 || this.rules.koRule === 'SIMPLE')) {
      const last = this._koKeys[this._koKeys.length - 1]!;
      let count = 0;
      for (const k of this._koKeys) if (k === last) count++;
      if (count >= 3) {
        this.isNoResult = true;
        this.isGameFinished = true;
      }
    }
  }

  /**
   * KataGo's marking of the locations the next player may not play, after each move: by superko in the main phase, or
   * in the encore the ko captures already made in this position.
   */
  private _refreshSuperKo(): void {
    this.superKoBanned.fill(0);
    if (this.encorePhase > 0) {
      const position = positionKey(this.board, null);
      for (const c of this._koCapturesInEncore) if (c.positionBefore === position && c.pla === this.nextPla) this.superKoBanned[c.loc] = 1;
      return;
    }
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
    // Made by the constructor, not Object.create: V8 keeps an object made by Object.create with this many fields as a
    // slow dictionary, and the search clones a history for every node.
    cloneSource = this;
    return new GameHistory(this.board, this.nextPla, this.rules, this.komi);
  }

  /** The board as it was `n` moves ago (KataGo `getRecentBoard`). */
  recentBoard(n: number): Board {
    if (n <= 0 || this.boardsBefore.length === 0) return this.board;
    const i = this.boardsBefore.length - n;
    return i >= 0 ? this.boardsBefore[i]! : this.boardsBefore[0]!;
  }

  /**
   * KataGo `currentSelfKomi`: the komi for `pla`, with White's bonus points, and with the value of a draw folded in when
   * the game can end in one (`whiteKomiAdjustmentForDraws`). Computed in C++ floats, as KataGo does.
   */
  currentSelfKomi(pla: Stone, drawEquivalentWinsForWhite = 0.5): number {
    const f = Math.fround;
    const drawAdjustment = this.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite);
    const white = f(f(f(this.whiteBonusScore + this.whiteHandicapBonusScore) + f(this.komi)) + drawAdjustment);
    return pla === WHITE ? white : -white;
  }
}
