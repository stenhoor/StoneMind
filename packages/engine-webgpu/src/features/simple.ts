/**
 * KataGo's V7 input features for "simple" positions only, so phase 2 can check the network before the full rules
 * engine exists (phase 3). Follows `NNInputs::fillRowV7` in KataGo's `cpp/neuralnet/nninputs.cpp` (KataGo 1.18, MIT,
 * © David J Wu).
 *
 * A position is simple when:
 * - no move captures, no move is suicide, and there are no passes, so there is no ko,
 * - every group, now and before each of the last two moves, has at least three liberties, so there are no ladder
 *   features (planes 14–17),
 * - under area scoring, no empty region borders stones of only one colour, so the area planes (18–19) are just the stones.
 * Anything else throws `OutsideSimpleDomainError` rather than encoding it wrongly.
 */
import type { NetInput } from '../reference/forward.js';
import type { Player } from '../reference/postprocess.js';

export class OutsideSimpleDomainError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OutsideSimpleDomainError';
  }
}

/** The rule sets the simple encoder supports, with KataGo's meanings. */
export const SIMPLE_RULES = {
  /** KataGo `Rules::getTrompTaylorish()`: area scoring, positional superko, suicide legal, no tax. */
  'tromp-taylor': { scoringTerritory: false, koRule: 'POSITIONAL', multiStoneSuicideLegal: true, tax: 'NONE' },
  /** KataGo "japanese": territory scoring, simple ko, no suicide, seki tax. */
  japanese: { scoringTerritory: true, koRule: 'SIMPLE', multiStoneSuicideLegal: false, tax: 'SEKI' },
} as const;

export type SimpleRulesName = keyof typeof SIMPLE_RULES;

/** A stone or move at (x, y), with y = 0 the top row (A19 on a 19×19 board). */
export type Placement = readonly [Player, number, number];

export interface SimplePosition {
  xSize: number;
  ySize: number;
  rules: SimpleRulesName;
  komi: number;
  initialStones?: readonly Placement[];
  moves?: readonly Placement[];
  nextPlayer: Player;
}

export interface EncodedPosition {
  input: NetInput;
  /** One flag per point plus pass, for post-processing. */
  legal: boolean[];
  koRuleSimple: boolean;
  scoringTerritory: boolean;
}

const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;
const colorOf = (p: Player) => (p === 'B' ? BLACK : WHITE);
const opposite = (c: number) => (c === BLACK ? WHITE : BLACK);

const COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';

/** Parse a GTP coordinate such as "Q16" into (x, y) with y = 0 at the top. */
export function parseGtp(location: string, ySize: number): [number, number] {
  const m = /^([A-HJ-Z])(\d{1,2})$/i.exec(location.trim());
  if (!m) throw new Error(`not a board coordinate: "${location}"`);
  const x = COLUMNS.indexOf(m[1]!.toUpperCase());
  const y = ySize - Number(m[2]);
  return [x, y];
}

class SimpleBoard {
  readonly cells: Uint8Array;

  constructor(
    readonly xSize: number,
    readonly ySize: number,
  ) {
    this.cells = new Uint8Array(xSize * ySize);
  }

  clone(): SimpleBoard {
    const b = new SimpleBoard(this.xSize, this.ySize);
    b.cells.set(this.cells);
    return b;
  }

  neighbours(i: number): number[] {
    const x = i % this.xSize;
    const y = (i - x) / this.xSize;
    const out: number[] = [];
    if (x > 0) out.push(i - 1);
    if (x < this.xSize - 1) out.push(i + 1);
    if (y > 0) out.push(i - this.xSize);
    if (y < this.ySize - 1) out.push(i + this.xSize);
    return out;
  }

  /** The liberty count of the group at each stone (0 on empty points). */
  liberties(): Int32Array {
    const libs = new Int32Array(this.cells.length);
    const seen = new Uint8Array(this.cells.length);
    for (let start = 0; start < this.cells.length; start++) {
      const color = this.cells[start]!;
      if (color === EMPTY || seen[start]) continue;
      const group: number[] = [start];
      const libSet = new Set<number>();
      seen[start] = 1;
      for (let k = 0; k < group.length; k++) {
        for (const n of this.neighbours(group[k]!)) {
          const c = this.cells[n]!;
          if (c === EMPTY) libSet.add(n);
          else if (c === color && !seen[n]) {
            seen[n] = 1;
            group.push(n);
          }
        }
      }
      for (const s of group) libs[s] = libSet.size;
    }
    return libs;
  }

  place(color: number, x: number, y: number): void {
    if (x < 0 || y < 0 || x >= this.xSize || y >= this.ySize) throw new OutsideSimpleDomainError(`(${x},${y}) is off the board`);
    const i = y * this.xSize + x;
    if (this.cells[i] !== EMPTY) throw new OutsideSimpleDomainError(`(${x},${y}) is already occupied`);
    this.cells[i] = color;
    const libs = this.liberties();
    for (let j = 0; j < this.cells.length; j++) {
      if (this.cells[j] !== EMPTY && libs[j] === 0) {
        throw new OutsideSimpleDomainError(`a stone at (${x},${y}) captures or is suicide: not a simple position`);
      }
    }
  }

  requireNoLadderCandidates(when: string): void {
    const libs = this.liberties();
    for (let i = 0; i < this.cells.length; i++) {
      if (this.cells[i] !== EMPTY && libs[i]! < 3) {
        throw new OutsideSimpleDomainError(`${when}: a group has fewer than 3 liberties, so ladder features may apply`);
      }
    }
  }

  /** Under area scoring: every empty region must border both colours (or no stones at all). */
  requireNoOneColourRegions(): void {
    const seen = new Uint8Array(this.cells.length);
    for (let start = 0; start < this.cells.length; start++) {
      if (this.cells[start] !== EMPTY || seen[start]) continue;
      const region = [start];
      seen[start] = 1;
      const borders = new Set<number>();
      for (let k = 0; k < region.length; k++) {
        for (const n of this.neighbours(region[k]!)) {
          const c = this.cells[n]!;
          if (c === EMPTY && !seen[n]) {
            seen[n] = 1;
            region.push(n);
          } else if (c !== EMPTY) borders.add(c);
        }
      }
      if (borders.size === 1) throw new OutsideSimpleDomainError('an empty region borders only one colour: area features would apply');
    }
  }
}

/**
 * Encode a simple position as KataGo's V7 network input.
 * @throws OutsideSimpleDomainError when the position is not simple (see the module comment).
 */
export function encodeSimplePosition(pos: SimplePosition): EncodedPosition {
  const { xSize, ySize } = pos;
  const rules = SIMPLE_RULES[pos.rules];
  const moves = pos.moves ?? [];
  const board = new SimpleBoard(xSize, ySize);
  for (const [player, x, y] of pos.initialStones ?? []) board.place(colorOf(player), x, y);
  const history: SimpleBoard[] = [board.clone()];
  for (const [player, x, y] of moves) {
    board.place(colorOf(player), x, y);
    history.push(board.clone());
  }
  board.requireNoLadderCandidates('now');
  if (history.length >= 2) history[history.length - 2]!.requireNoLadderCandidates('before the last move');
  if (history.length >= 3) history[history.length - 3]!.requireNoLadderCandidates('before the last two moves');
  if (!rules.scoringTerritory) board.requireNoOneColourRegions();

  const area = xSize * ySize;
  const spatial = new Float32Array(22 * area);
  const global = new Float32Array(19);
  const set = (plane: number, i: number) => (spatial[plane * area + i] = 1);
  const pla = colorOf(pos.nextPlayer);
  const opp = opposite(pla);
  const libs = board.liberties();

  for (let i = 0; i < area; i++) {
    set(0, i);
    const c = board.cells[i]!;
    if (c === pla) set(1, i);
    else if (c === opp) set(2, i);
    if (c !== EMPTY) {
      if (libs[i] === 1) set(3, i);
      else if (libs[i] === 2) set(4, i);
      else if (libs[i] === 3) set(5, i);
    }
    // Area planes 18-19: under area scoring with no tax, every stone counts as its colour's area here.
    if (!rules.scoringTerritory) {
      if (c === pla) set(18, i);
      else if (c === opp) set(19, i);
    }
  }

  // History: the last five moves, while they alternate back from the opponent's.
  for (let k = 1; k <= Math.min(5, moves.length); k++) {
    const [player, x, y] = moves[moves.length - k]!;
    const expected = k % 2 === 1 ? opp : pla;
    if (colorOf(player) !== expected) break;
    set(8 + k, y * xSize + x);
  }

  // Globals.
  // Under territory scoring KataGo "chills" one point per stone and per non-pass move (BoardHistory, KataGo 1.18):
  // White's bonus is +1 for each Black stone or move and -1 for each White one. Simple positions have no captures,
  // so the initial-capture correction is 0, and both simple rule sets have no handicap bonus.
  let whiteBonus = 0;
  if (rules.scoringTerritory) {
    for (const [player] of [...(pos.initialStones ?? []), ...moves]) whiteBonus += player === 'B' ? 1 : -1;
  }
  const whiteKomi = pos.komi + whiteBonus;
  const selfKomiRaw = pos.nextPlayer === 'W' ? whiteKomi : -whiteKomi;
  const clip = area + 20;
  const selfKomi = Math.max(-clip, Math.min(clip, selfKomiRaw));
  global[5] = selfKomi / 20;
  if (rules.koRule === 'POSITIONAL') {
    global[6] = 1;
    global[7] = 0.5;
  }
  if (rules.multiStoneSuicideLegal) global[8] = 1;
  if (rules.scoringTerritory) global[9] = 1;
  if (rules.tax === 'SEKI') global[10] = 1;
  if (!rules.scoringTerritory) {
    const even = area % 2 === 0;
    const komiFloor = even ? Math.floor(selfKomi / 2) * 2 : Math.floor((selfKomi - 1) / 2) * 2 + 1;
    const delta = Math.min(2, Math.max(0, selfKomi - komiFloor));
    global[18] = delta < 0.5 ? delta : delta < 1.5 ? 1 - delta : delta - 2;
  }

  // Legality: empty points are legal unless every neighbour is an opponent stone (single-stone suicide: nothing here
  // can be captured, since every group has at least three liberties).
  const legal: boolean[] = [];
  for (let i = 0; i < area; i++) {
    if (board.cells[i] !== EMPTY) {
      legal.push(false);
      continue;
    }
    const ns = board.neighbours(i);
    legal.push(ns.some((n) => board.cells[n] === EMPTY || board.cells[n] === pla));
  }
  legal.push(true); // pass

  return {
    input: { xSize, ySize, spatial, global },
    legal,
    koRuleSimple: rules.koRule === 'SIMPLE',
    scoringTerritory: rules.scoringTerritory,
  };
}
