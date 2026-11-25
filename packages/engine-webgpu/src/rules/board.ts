/**
 * A Go board, ported from KataGo's `cpp/game/board.cpp` (KataGo 1.18, MIT, © David J Wu): the same padded layout,
 * location numbering and move semantics, and the same ladder search, so the input features match KataGo's.
 *
 * Chains are recomputed by flood fill when the board changes rather than kept incrementally as KataGo does: simpler,
 * and fast enough for feature encoding. Results don't depend on the chain representation.
 */

export const EMPTY = 0;
export const BLACK = 1;
export const WHITE = 2;
export const WALL = 3;
export type Color = 0 | 1 | 2 | 3;
export type Stone = 1 | 2;

export const NULL_LOC = 0;
export const PASS_LOC = 1;

export const opp = (p: Stone): Stone => (p === BLACK ? WHITE : BLACK);

/** A recorded move, enough to undo it. */
export interface MoveRecord {
  colors: Uint8Array;
  koLoc: number;
  numBlackCaptures: number;
  numWhiteCaptures: number;
}

const LADDER_NODE_BUDGET = 25000; // KataGo MAX_LADDER_SEARCH_NODE_BUDGET

export class Board {
  readonly xSize: number;
  readonly ySize: number;
  readonly stride: number;
  readonly arraySize: number;
  /** Up, left, right, down: KataGo's adj_offsets[0..3]. */
  readonly adj: readonly [number, number, number, number];
  colors: Uint8Array;
  koLoc = NULL_LOC;
  /** Black stones captured (by White). */
  numBlackCaptures = 0;
  /** White stones captured (by Black). */
  numWhiteCaptures = 0;

  #dirty = true;
  #chainHead!: Int32Array;
  #chainLibs!: Int32Array; // by head
  #chainSize!: Int32Array; // by head
  #chainStones!: Map<number, number[]>; // by head

  constructor(xSize: number, ySize: number) {
    if (xSize < 1 || ySize < 1 || xSize > 25 || ySize > 25) throw new Error('board size must be 1–25');
    this.xSize = xSize;
    this.ySize = ySize;
    this.stride = xSize + 1;
    this.arraySize = (ySize + 2) * this.stride + 1;
    this.adj = [-this.stride, -1, 1, this.stride];
    this.colors = new Uint8Array(this.arraySize).fill(WALL);
    for (let y = 0; y < ySize; y++) for (let x = 0; x < xSize; x++) this.colors[this.loc(x, y)] = EMPTY;
  }

  loc(x: number, y: number): number {
    return x + 1 + (y + 1) * this.stride;
  }

  x(loc: number): number {
    return (loc % this.stride) - 1;
  }

  y(loc: number): number {
    return Math.floor(loc / this.stride) - 1;
  }

  isOnBoard(loc: number): boolean {
    return loc >= 0 && loc < this.arraySize && this.colors[loc] !== WALL;
  }

  clone(): Board {
    const b = new Board(this.xSize, this.ySize);
    b.colors.set(this.colors);
    b.koLoc = this.koLoc;
    b.numBlackCaptures = this.numBlackCaptures;
    b.numWhiteCaptures = this.numWhiteCaptures;
    return b;
  }

  // ---------------------------------------------------------------- chains

  #rebuild(): void {
    const n = this.arraySize;
    this.#chainHead = new Int32Array(n).fill(-1);
    this.#chainLibs = new Int32Array(n);
    this.#chainSize = new Int32Array(n);
    this.#chainStones = new Map();
    const libMark = new Int32Array(n).fill(-1);
    for (let start = 0; start < n; start++) {
      const c = this.colors[start]!;
      if ((c !== BLACK && c !== WHITE) || this.#chainHead[start] !== -1) continue;
      const stones = [start];
      this.#chainHead[start] = start;
      let libs = 0;
      for (let k = 0; k < stones.length; k++) {
        const s = stones[k]!;
        for (const d of this.adj) {
          const a = s + d;
          const ac = this.colors[a]!;
          if (ac === EMPTY) {
            if (libMark[a] !== start) {
              libMark[a] = start;
              libs++;
            }
          } else if (ac === c && this.#chainHead[a] === -1) {
            this.#chainHead[a] = start;
            stones.push(a);
          }
        }
      }
      this.#chainLibs[start] = libs;
      this.#chainSize[start] = stones.length;
      this.#chainStones.set(start, stones);
    }
    this.#dirty = false;
  }

  #fresh(): void {
    if (this.#dirty) this.#rebuild();
  }

  chainHead(loc: number): number {
    this.#fresh();
    return this.#chainHead[loc]!;
  }

  getNumLiberties(loc: number): number {
    this.#fresh();
    return this.#chainLibs[this.#chainHead[loc]!]!;
  }

  getChainSize(loc: number): number {
    this.#fresh();
    return this.#chainSize[this.#chainHead[loc]!]!;
  }

  chainStones(loc: number): readonly number[] {
    this.#fresh();
    return this.#chainStones.get(this.#chainHead[loc]!)!;
  }

  /** All chain heads of a colour, in location order (KataGo iterates heads in location order too). */
  chainHeads(color: Stone): number[] {
    this.#fresh();
    const out: number[] = [];
    for (const head of this.#chainStones.keys()) if (this.colors[head] === color) out.push(head);
    return out.sort((a, b) => a - b);
  }

  // ------------------------------------------------------------- legality

  getNumImmediateLiberties(loc: number): number {
    let n = 0;
    for (const d of this.adj) if (this.colors[loc + d] === EMPTY) n++;
    return n;
  }

  isSuicide(loc: number, pla: Stone): boolean {
    if (loc === PASS_LOC) return false;
    const o = opp(pla);
    for (const d of this.adj) {
      const a = loc + d;
      const c = this.colors[a];
      if (c === EMPTY) return false;
      if (c === pla && this.getNumLiberties(a) > 1) return false;
      if (c === o && this.getNumLiberties(a) === 1) return false;
    }
    return true;
  }

  isIllegalSuicide(loc: number, pla: Stone, multiStoneSuicideLegal: boolean): boolean {
    const o = opp(pla);
    for (const d of this.adj) {
      const a = loc + d;
      const c = this.colors[a];
      if (c === EMPTY) return false;
      if (c === pla && (multiStoneSuicideLegal || this.getNumLiberties(a) > 1)) return false;
      if (c === o && this.getNumLiberties(a) === 1) return false;
    }
    return true;
  }

  isKoBanned(loc: number): boolean {
    return loc === this.koLoc;
  }

  isLegal(loc: number, pla: Stone, multiStoneSuicideLegal: boolean): boolean {
    return loc === PASS_LOC || this.isLegalIgnoringKo(loc, pla, multiStoneSuicideLegal) && !this.isKoBanned(loc);
  }

  isLegalIgnoringKo(loc: number, pla: Stone, multiStoneSuicideLegal: boolean): boolean {
    return (
      loc === PASS_LOC ||
      (loc >= 0 && loc < this.arraySize && this.colors[loc] === EMPTY && !this.isIllegalSuicide(loc, pla, multiStoneSuicideLegal))
    );
  }

  wouldBeKoCapture(loc: number, pla: Stone): boolean {
    return this.getKoCaptureLoc(loc, pla) !== NULL_LOC;
  }

  getKoCaptureLoc(loc: number, pla: Stone): number {
    if (this.colors[loc] !== EMPTY) return NULL_LOC;
    const o = opp(pla);
    let capturable = NULL_LOC;
    for (const d of this.adj) {
      const a = loc + d;
      const c = this.colors[a];
      if (c !== WALL && c !== o) return NULL_LOC;
      if (c === o && this.getNumLiberties(a) === 1) {
        if (capturable !== NULL_LOC) return NULL_LOC;
        capturable = a;
      }
    }
    if (capturable === NULL_LOC) return NULL_LOC;
    return this.getChainSize(capturable) === 1 ? capturable : NULL_LOC;
  }

  // ---------------------------------------------------------------- moves

  /** KataGo `playMoveAssumeLegal`: captures, simple-ko location, and suicide removal. */
  playMoveAssumeLegal(loc: number, pla: Stone): void {
    if (loc === PASS_LOC) {
      this.koLoc = NULL_LOC;
      return;
    }
    const o = opp(pla);
    this.colors[loc] = pla;
    this.#dirty = true;
    let numCaptured = 0;
    let possibleKoLoc = NULL_LOC;
    const seen: number[] = [];
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] !== o) continue;
      const head = this.chainHead(a);
      if (seen.includes(head)) continue;
      seen.push(head);
      if (this.getNumLiberties(a) === 0) {
        numCaptured += this.#removeChain(a);
        possibleKoLoc = a;
      }
    }
    if (numCaptured === 1 && this.getChainSize(loc) === 1 && this.getNumLiberties(loc) === 1) this.koLoc = possibleKoLoc;
    else this.koLoc = NULL_LOC;
    if (pla === BLACK) this.numWhiteCaptures += numCaptured;
    else this.numBlackCaptures += numCaptured;
    if (this.getNumLiberties(loc) === 0) {
      const suicided = this.#removeChain(loc);
      if (pla === BLACK) this.numBlackCaptures += suicided;
      else this.numWhiteCaptures += suicided;
    }
  }

  #removeChain(loc: number): number {
    const stones = [...this.chainStones(loc)];
    for (const s of stones) this.colors[s] = EMPTY;
    this.#dirty = true;
    return stones.length;
  }

  playMoveRecorded(loc: number, pla: Stone): MoveRecord {
    const record: MoveRecord = {
      colors: this.colors.slice(),
      koLoc: this.koLoc,
      numBlackCaptures: this.numBlackCaptures,
      numWhiteCaptures: this.numWhiteCaptures,
    };
    this.playMoveAssumeLegal(loc, pla);
    return record;
  }

  undo(record: MoveRecord): void {
    this.colors = record.colors;
    this.koLoc = record.koLoc;
    this.numBlackCaptures = record.numBlackCaptures;
    this.numWhiteCaptures = record.numWhiteCaptures;
    this.#dirty = true;
  }

  /** A stone placed during setup (KataGo `setStoneFailIfNoLibs`): no captures; fails if it would leave a chain with no liberties. */
  setStone(loc: number, color: Stone): boolean {
    if (this.colors[loc] !== EMPTY) return false;
    this.colors[loc] = color;
    this.#dirty = true;
    for (const l of [loc, ...this.adj.map((d) => loc + d)]) {
      const c = this.colors[l];
      if ((c === BLACK || c === WHITE) && this.getNumLiberties(l) === 0) {
        this.colors[loc] = EMPTY;
        this.#dirty = true;
        return false;
      }
    }
    return true;
  }

  // --------------------------------------------------------- ladder support

  isAdjacentToPla(loc: number, pla: Stone): boolean {
    for (const d of this.adj) if (this.colors[loc + d] === pla) return true;
    return false;
  }

  /** Is loc adjacent to a stone of the chain with this head? */
  isAdjacentToPlaHead(pla: Stone, loc: number, plaHead: number): boolean {
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === pla && this.chainHead(a) === plaHead) return true;
    }
    return false;
  }

  /** The chain's liberties, appended to buf from bufIdx, skipping ones already in buf[bufStart..]. */
  findLiberties(loc: number, buf: number[], bufStart: number, bufIdx: number): number {
    let found = 0;
    for (const s of this.chainStones(loc)) {
      for (const d of this.adj) {
        const lib = s + d;
        if (this.colors[lib] !== EMPTY) continue;
        let dup = false;
        for (let j = bufStart; j < bufIdx + found; j++) if (buf[j] === lib) dup = true;
        if (!dup) buf[bufIdx + found++] = lib;
      }
    }
    return found;
  }

  findLibertyGainingCaptures(loc: number, buf: number[], bufStart: number, bufIdx: number): number {
    const o = opp(this.colors[loc] as Stone);
    const checked: number[] = [];
    let found = 0;
    for (const s of this.chainStones(loc)) {
      for (const d of this.adj) {
        const a = s + d;
        if (this.colors[a] !== o) continue;
        const head = this.chainHead(a);
        if (this.getNumLiberties(a) === 1 && !checked.includes(head)) {
          found += this.findLiberties(a, buf, bufStart, bufIdx + found);
          checked.push(head);
        }
      }
    }
    return found;
  }

  hasLibertyGainingCaptures(loc: number): boolean {
    const o = opp(this.colors[loc] as Stone);
    for (const s of this.chainStones(loc)) {
      for (const d of this.adj) {
        const a = s + d;
        if (this.colors[a] === o && this.getNumLiberties(a) === 1) return true;
      }
    }
    return false;
  }

  getBoundNumLibertiesAfterPlay(loc: number, pla: Stone): [lower: number, upper: number] {
    const o = opp(pla);
    let immediate = 0;
    let numCaps = 0;
    let potentialFromCaps = 0;
    let connectionSum = 0;
    let connectionMax = 0;
    for (const d of this.adj) {
      const a = loc + d;
      const c = this.colors[a];
      if (c === EMPTY) immediate++;
      else if (c === o) {
        if (this.getNumLiberties(a) === 1) {
          numCaps++;
          potentialFromCaps += this.getChainSize(a);
        }
      } else if (c === pla) {
        const conn = this.getNumLiberties(a) - 1;
        connectionSum += conn;
        if (conn > connectionMax) connectionMax = conn;
      }
    }
    return [numCaps + Math.max(connectionMax, immediate), immediate + potentialFromCaps + connectionSum];
  }

  getNumLibertiesAfterPlay(loc: number, pla: Stone, max: number): number {
    const o = opp(pla);
    const libs: number[] = [];
    const capturedHeads: number[] = [];
    for (const d of this.adj) {
      const a = loc + d;
      const c = this.colors[a];
      if (c === EMPTY) {
        libs.push(a);
        if (libs.length >= max) return max;
      } else if (c === o && this.getNumLiberties(a) === 1) {
        libs.push(a);
        if (libs.length >= max) return max;
        const head = this.chainHead(a);
        if (!capturedHeads.includes(head)) capturedHeads.push(head);
      }
    }
    const wouldBeEmpty = (l: number) =>
      this.colors[l] === EMPTY || (this.colors[l] === o && capturedHeads.includes(this.chainHead(l)));
    const connecting: number[] = [];
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] !== pla) continue;
      const head = this.chainHead(a);
      if (connecting.includes(head)) continue;
      connecting.push(head);
      for (const s of this.chainStones(a)) {
        for (const dd of this.adj) {
          const l = s + dd;
          if (l !== loc && wouldBeEmpty(l) && !libs.includes(l)) {
            libs.push(l);
            if (libs.length >= max) return max;
          }
        }
      }
    }
    return libs.length;
  }

  countHeuristicConnectionLibertiesX2(loc: number, pla: Stone): number {
    let n = 0;
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === pla) {
        const libs = this.getNumLiberties(a);
        if (libs > 1) n += libs * 2 - 3;
      }
    }
    return n;
  }

  isAdjacent(a: number, b: number): boolean {
    return this.adj.some((d) => a + d === b);
  }

  /** KataGo `searchIsLadderCapturedAttackerFirst2Libs`: which attacker moves on a 2-liberty chain start a working ladder. */
  searchIsLadderCapturedAttackerFirst2Libs(loc: number, buf: number[], workingMoves: number[]): boolean {
    const c = this.colors[loc];
    if (c !== BLACK && c !== WHITE) return false;
    if (this.getNumLiberties(loc) !== 2) return false;
    const pla = c as Stone;
    const o = opp(pla);
    this.findLiberties(loc, buf, 0, 0);
    const move0 = buf[0]!;
    const move1 = buf[1]!;
    let works0 = false;
    let works1 = false;
    if (this.isLegal(move0, o, false)) {
      const r = this.playMoveRecorded(move0, o);
      works0 = this.searchIsLadderCaptured(loc, true, buf);
      this.undo(r);
    }
    if (this.isLegal(move1, o, false)) {
      const r = this.playMoveRecorded(move1, o);
      works1 = this.searchIsLadderCaptured(loc, true, buf);
      this.undo(r);
    }
    if (works0 || works1) {
      workingMoves.length = 0;
      if (works0) workingMoves.push(move0);
      if (works1) workingMoves.push(move1);
      return true;
    }
    return false;
  }

  /** KataGo `searchIsLadderCaptured`: iterative ladder search with the same move ordering, cut-offs and node budget. */
  searchIsLadderCaptured(loc: number, defenderFirst: boolean, buf: number[]): boolean {
    if (loc < 0 || loc >= this.arraySize) return false;
    const c = this.colors[loc];
    if (c !== BLACK && c !== WHITE) return false;
    const libs0 = this.getNumLiberties(loc);
    if (libs0 > 2 || (defenderFirst && libs0 > 1)) return false;
    const pla = c as Stone;
    const o = opp(pla);

    const savedKo = this.koLoc;
    if (defenderFirst) this.koLoc = NULL_LOC;

    const stackSize = Math.floor((this.xSize * this.ySize * 3) / 2) + 1;
    const moveListStarts: number[] = [0];
    const moveListLens: number[] = [0];
    const moveListCur: number[] = [-1];
    const records: MoveRecord[] = [];
    let stackIdx = 0;
    let nodes = 0;
    let returnValue = false;
    let returnedFromDeeper = false;

    for (;;) {
      if (stackIdx <= -1) {
        this.koLoc = savedKo;
        return returnValue;
      }
      if (stackIdx >= stackSize - 1) {
        returnValue = true;
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }
      if (nodes >= LADDER_NODE_BUDGET) {
        stackIdx -= 1;
        while (stackIdx >= 0) {
          this.undo(records[stackIdx]!);
          stackIdx -= 1;
        }
        // As in KataGo, koLoc is not restored here: undoing the first move restores the value it had then, which was
        // already cleared when defenderFirst. iterLadders searches a copy, so this never leaks into the features.
        return false;
      }
      const isDefender = (defenderFirst && stackIdx % 2 === 0) || (!defenderFirst && stackIdx % 2 === 1);

      if (moveListCur[stackIdx] === -1) {
        const libs = this.getNumLiberties(loc);
        if (!isDefender && libs <= 1) {
          returnValue = true;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        if (!isDefender && libs >= 3) {
          returnValue = false;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        if (isDefender && libs >= 2) {
          returnValue = false;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        if (isDefender && this.koLoc !== NULL_LOC) {
          returnValue = false;
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        const start = moveListStarts[stackIdx]!;
        let len = 0;
        if (isDefender) {
          len = this.findLibertyGainingCaptures(loc, buf, start, start);
          len += this.findLiberties(loc, buf, start, start + len);
          const [lower, upper] = this.getBoundNumLibertiesAfterPlay(buf[start + len - 1]!, pla);
          if (lower >= 3) {
            returnValue = false;
            returnedFromDeeper = true;
            stackIdx--;
            continue;
          }
          if (len === 1 && upper <= 1) {
            returnValue = true;
            returnedFromDeeper = true;
            stackIdx--;
            continue;
          }
        } else {
          len = this.findLiberties(loc, buf, start, start);
          let l0 = this.getNumImmediateLiberties(buf[start]!);
          let l1 = this.getNumImmediateLiberties(buf[start + 1]!);
          if (l0 === 0 && l1 === 0 && this.wouldBeKoCapture(buf[start]!, o) && this.wouldBeKoCapture(buf[start + 1]!, o)) {
            if (this.getNumLibertiesAfterPlay(buf[start]!, pla, 3) <= 2 && this.getNumLibertiesAfterPlay(buf[start + 1]!, pla, 3) <= 2) {
              if (!this.hasLibertyGainingCaptures(loc)) {
                returnValue = true;
                returnedFromDeeper = true;
                stackIdx--;
                continue;
              }
            }
          }
          if (!this.isAdjacent(buf[start]!, buf[start + 1]!)) {
            if (l0 >= 3 && l1 >= 3) {
              returnValue = false;
              returnedFromDeeper = true;
              stackIdx--;
              continue;
            } else if (l0 >= 3) len = 1;
            else if (l1 >= 3) {
              buf[start] = buf[start + 1]!;
              len = 1;
            }
          }
          if (len > 1) {
            l0 = l0 * 2 + this.countHeuristicConnectionLibertiesX2(buf[start]!, pla);
            l1 = l1 * 2 + this.countHeuristicConnectionLibertiesX2(buf[start + 1]!, pla);
            if (l1 > l0) {
              const t = buf[start]!;
              buf[start] = buf[start + 1]!;
              buf[start + 1] = t;
            }
          }
        }
        moveListLens[stackIdx] = len;
        moveListCur[stackIdx] = 0;
      } else {
        if (returnedFromDeeper) this.undo(records[stackIdx]!);
        if (isDefender && !returnValue) {
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        if (!isDefender && returnValue) {
          returnedFromDeeper = true;
          stackIdx--;
          continue;
        }
        moveListCur[stackIdx]!++;
      }

      if (moveListCur[stackIdx]! >= moveListLens[stackIdx]!) {
        returnValue = isDefender;
        returnedFromDeeper = true;
        stackIdx--;
        continue;
      }
      const move = buf[moveListStarts[stackIdx]! + moveListCur[stackIdx]!]!;
      const p = isDefender ? pla : o;
      if (!this.isLegal(move, p, false)) {
        returnValue = isDefender;
        returnedFromDeeper = false;
        continue;
      }
      records[stackIdx] = this.playMoveRecorded(move, p);
      nodes++;
      stackIdx++;
      moveListCur[stackIdx] = -1;
      moveListStarts[stackIdx] = moveListStarts[stackIdx - 1]! + moveListLens[stackIdx - 1]!;
      moveListLens[stackIdx] = 0;
    }
  }
}
