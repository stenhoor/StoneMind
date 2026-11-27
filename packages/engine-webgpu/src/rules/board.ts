/**
 * A Go board, ported from KataGo's `cpp/game/board.cpp` (KataGo 1.18, MIT, © David J Wu): the same padded layout,
 * location numbering and move semantics, and the same ladder search, so the input features match KataGo's.
 *
 * Chains are kept incrementally, as in KataGo: a circular list of stones per chain, with the stone and liberty counts
 * held by the chain's head, updated by each placement, capture and merge, in KataGo's order. A recorded move is undone
 * as KataGo undoes it (`Board::undo`): the captured chains are filled back in and the stone taken off, which the ladder
 * search does at every node. As in KataGo, an undo leaves the same stones, chains and counts, but not always the same
 * heads or order of stones within a chain.
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

/**
 * A recorded move, KataGo's `Board::MoveRecord`: the move, the ko location before it, and which directions it captured
 * in (bits 0–3, by `adj` index) or 0x10 for a suicide. Can be passed back to be reused.
 */
export interface MoveRecord {
  pla: Stone;
  loc: number;
  koLoc: number;
  capDirs: number;
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

  /** The head of each stone's chain; only meaningful on stones. */
  #head: Int32Array;
  /** The next stone in the chain (a circular list). */
  #next: Int32Array;
  /** Stones in the chain, by head. */
  #size: Int32Array;
  /** Liberties of the chain, by head. */
  #libs: Int32Array;
  /** Reused records for the ladder search, which plays and undoes many moves (by stack depth, and for the first move). */
  #ladderRecords: MoveRecord[] = [];
  #ladderRecordFirst: MoveRecord | undefined;

  /** @param from A board to copy (see {@link clone}); its size must be xSize × ySize. */
  constructor(xSize: number, ySize: number, from?: Board) {
    if (xSize < 1 || ySize < 1 || xSize > 25 || ySize > 25) throw new Error('board size must be 1–25');
    this.xSize = xSize;
    this.ySize = ySize;
    this.stride = xSize + 1;
    this.arraySize = (ySize + 2) * this.stride + 1;
    this.adj = [-this.stride, -1, 1, this.stride];
    if (from) {
      // A copy: the arrays as they are, without building an empty board first.
      this.colors = from.colors.slice();
      this.#head = from.#head.slice();
      this.#next = from.#next.slice();
      this.#size = from.#size.slice();
      this.#libs = from.#libs.slice();
      this.koLoc = from.koLoc;
      this.numBlackCaptures = from.numBlackCaptures;
      this.numWhiteCaptures = from.numWhiteCaptures;
      return;
    }
    this.colors = new Uint8Array(this.arraySize).fill(WALL);
    for (let y = 0; y < ySize; y++) for (let x = 0; x < xSize; x++) this.colors[this.loc(x, y)] = EMPTY;
    this.#head = new Int32Array(this.arraySize).fill(-1);
    this.#next = new Int32Array(this.arraySize).fill(-1);
    this.#size = new Int32Array(this.arraySize);
    this.#libs = new Int32Array(this.arraySize);
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
    return new Board(this.xSize, this.ySize, this);
  }

  // ---------------------------------------------------------------- chains

  /** KataGo `isLibertyOf`: is the empty point loc next to a stone of the chain with this head? */
  #isLibertyOf(loc: number, head: number): boolean {
    const c = this.colors[head];
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === c && this.#head[a] === head) return true;
    }
    return false;
  }

  /** KataGo `changeSurroundingLiberties`: add delta to the liberties of each distinct chain of pla next to loc. */
  #changeSurroundingLiberties(loc: number, pla: Stone, delta: number): void {
    const [d0, d1, d2, d3] = this.adj;
    const a0 = loc + d0, a1 = loc + d1, a2 = loc + d2, a3 = loc + d3;
    const c = this.colors, h = this.#head;
    if (c[a0] === pla) this.#libs[h[a0]!]! += delta;
    if (c[a1] === pla && !(c[a0] === pla && h[a0] === h[a1])) this.#libs[h[a1]!]! += delta;
    if (c[a2] === pla && !(c[a0] === pla && h[a0] === h[a2]) && !(c[a1] === pla && h[a1] === h[a2])) this.#libs[h[a2]!]! += delta;
    if (
      c[a3] === pla && !(c[a0] === pla && h[a0] === h[a3]) && !(c[a1] === pla && h[a1] === h[a3]) && !(c[a2] === pla && h[a2] === h[a3])
    )
      this.#libs[h[a3]!]! += delta;
  }

  /** KataGo `mergeChains`: the smaller chain joins the larger (loc1's on a tie), counting its new liberties. */
  #mergeChains(loc1: number, loc2: number): void {
    let head1 = this.#head[loc1]!;
    let head2 = this.#head[loc2]!;
    if (this.#size[head1]! < this.#size[head2]!) [head1, head2] = [head2, head1];
    this.#size[head1]! += this.#size[head2]!;
    let newLibs = 0;
    let loc = head2;
    for (;;) {
      for (const d of this.adj) {
        const a = loc + d;
        if (this.colors[a] === EMPTY && !this.#isLibertyOf(a, head1)) newLibs++;
      }
      this.#head[loc] = head1;
      if (this.#next[loc] !== head2) loc = this.#next[loc]!;
      else break;
    }
    this.#libs[head1]! += newLibs;
    // head1 → head2 → … → last2 → next1 → … → last1 → head1
    this.#next[loc] = this.#next[head1]!;
    this.#next[head1] = head2;
  }

  /** KataGo `removeChain`: empty the chain's points, each giving a liberty back to the chains around it. */
  #removeChain(loc: number): number {
    const o = opp(this.colors[loc] as Stone);
    let n = 0;
    let cur = loc;
    do {
      this.colors[cur] = EMPTY;
      n++;
      this.#changeSurroundingLiberties(cur, o, +1);
      cur = this.#next[cur]!;
    } while (cur !== loc);
    return n;
  }

  /** KataGo `addChain`: fill the empty region at loc with a chain of pla whose head is loc. */
  #addChain(loc: number, pla: Stone): void {
    this.#libs[loc] = 0;
    this.#size[loc] = 0;
    const front = this.#addChainHelper(loc, loc, loc, pla);
    this.#next[loc] = front;
  }

  #addChainHelper(head: number, tailTarget: number, loc: number, pla: Stone): number {
    this.colors[loc] = pla;
    this.#head[loc] = head;
    this.#size[head]!++;
    this.#next[loc] = tailTarget;
    this.#changeSurroundingLiberties(loc, opp(pla), -1);
    let nextTailTarget = loc;
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === EMPTY) nextTailTarget = this.#addChainHelper(head, nextTailTarget, a, pla);
    }
    return nextTailTarget;
  }

  /** KataGo `rebuildChain`: relink and recount the chain of pla at loc, whose heads point nowhere; loc becomes its head. */
  #rebuildChain(loc: number, pla: Stone): void {
    this.#libs[loc] = 0;
    this.#size[loc] = 0;
    const front = this.#rebuildChainHelper(loc, loc, loc, pla);
    this.#next[loc] = front;
  }

  #rebuildChainHelper(head: number, tailTarget: number, loc: number, pla: Stone): number {
    let add = 0;
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === EMPTY && !this.#isLibertyOf(a, head)) add++;
    }
    this.#libs[head]! += add;
    this.#head[loc] = head;
    this.#size[head]!++;
    this.#next[loc] = tailTarget;
    let nextTailTarget = loc;
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === pla && this.#head[a] !== head) nextTailTarget = this.#rebuildChainHelper(head, nextTailTarget, a, pla);
    }
    return nextTailTarget;
  }

  /** The next stone in loc's chain (a circular list): KataGo's `next_in_chain`. */
  nextInChain(loc: number): number {
    return this.#next[loc]!;
  }

  chainHead(loc: number): number {
    return this.#head[loc]!;
  }

  getNumLiberties(loc: number): number {
    return this.#libs[this.#head[loc]!]!;
  }

  getChainSize(loc: number): number {
    return this.#size[this.#head[loc]!]!;
  }

  /** The chain's stones, starting from its head. */
  chainStones(loc: number): readonly number[] {
    const head = this.#head[loc]!;
    const out: number[] = [];
    let s = head;
    do {
      out.push(s);
      s = this.#next[s]!;
    } while (s !== head);
    return out;
  }

  /** All chain heads of a colour, in location order (KataGo iterates heads in location order too). */
  chainHeads(color: Stone): number[] {
    const out: number[] = [];
    for (let loc = 0; loc < this.arraySize; loc++) if (this.colors[loc] === color && this.#head[loc] === loc) out.push(loc);
    return out;
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
    this.#size[loc] = 1;
    this.#libs[loc] = this.getNumImmediateLiberties(loc);
    this.#head[loc] = loc;
    this.#next[loc] = loc;
    let numCaptured = 0;
    let possibleKoLoc = NULL_LOC;
    const oppHeadsSeen: number[] = [];
    for (const d of this.adj) {
      const a = loc + d;
      const c = this.colors[a];
      if (c === pla) {
        if (this.#head[a] === this.#head[loc]) continue;
        this.#libs[this.#head[a]!]!--;
        this.#mergeChains(a, loc);
      } else if (c === o) {
        const oppHead = this.#head[a]!;
        if (oppHeadsSeen.includes(oppHead)) continue;
        this.#libs[oppHead]!--;
        oppHeadsSeen.push(oppHead);
        if (this.getNumLiberties(a) === 0) {
          numCaptured += this.#removeChain(a);
          possibleKoLoc = a;
        }
      }
    }
    if (numCaptured === 1 && this.getChainSize(loc) === 1 && this.getNumLiberties(loc) === 1) this.koLoc = possibleKoLoc;
    else this.koLoc = NULL_LOC;
    if (pla === BLACK) this.numWhiteCaptures += numCaptured;
    else this.numBlackCaptures += numCaptured;
    if (this.getNumLiberties(loc) === 0) {
      const suicided = this.getChainSize(loc);
      this.#removeChain(loc);
      if (pla === BLACK) this.numBlackCaptures += suicided;
      else this.numWhiteCaptures += suicided;
    }
  }

  /** KataGo `playMoveRecorded`: play a move, recording what's needed to undo it; `reuse` is a record to overwrite. */
  playMoveRecorded(loc: number, pla: Stone, reuse?: MoveRecord): MoveRecord {
    const record: MoveRecord = reuse ?? { pla, loc, koLoc: 0, capDirs: 0 };
    record.pla = pla;
    record.loc = loc;
    record.koLoc = this.koLoc;
    record.capDirs = 0;
    if (loc !== PASS_LOC) {
      const o = opp(pla);
      for (let i = 0; i < 4; i++) {
        const a = loc + this.adj[i]!;
        if (this.colors[a] === o && this.getNumLiberties(a) === 1) record.capDirs |= 1 << i;
      }
      if (record.capDirs === 0 && this.isSuicide(loc, pla)) record.capDirs = 0x10;
    }
    this.playMoveAssumeLegal(loc, pla);
    return record;
  }

  /** KataGo `undo`: undo a recorded move. Moves must be undone in the reverse order they were played. */
  undo(record: MoveRecord): void {
    this.koLoc = record.koLoc;
    const loc = record.loc;
    if (loc === PASS_LOC) return;
    const o = opp(record.pla);
    // Fill the captured chains back in.
    for (let i = 0; i < 4; i++) {
      const a = loc + this.adj[i]!;
      if ((record.capDirs & (1 << i)) !== 0 && this.colors[a] === EMPTY) {
        this.#addChain(a, o);
        const n = this.#size[this.#head[a]!]!;
        if (record.pla === BLACK) this.numWhiteCaptures -= n;
        else this.numBlackCaptures -= n;
      }
    }
    // Fill a suicided chain back in.
    if (record.capDirs === 0x10) {
      this.#addChain(loc, record.pla);
      const n = this.#size[this.#head[loc]!]!;
      if (record.pla === BLACK) this.numBlackCaptures -= n;
      else this.numWhiteCaptures -= n;
    }
    // Take the stone off.
    this.colors[loc] = EMPTY;
    this.#changeSurroundingLiberties(loc, o, +1);
    if (this.#size[this.#head[loc]!]! > 1) {
      let numNeighbors = 0;
      for (const d of this.adj) if (this.colors[loc + d] === record.pla) numNeighbors++;
      if (numNeighbors <= 1) {
        // One neighbour: taking the stone off can't split the chain.
        let head = this.#head[loc]!;
        if (head === loc) {
          const newHead = this.#next[loc]!;
          let cur = loc;
          do {
            this.#head[cur] = newHead;
            cur = this.#next[cur]!;
          } while (cur !== loc);
          this.#size[newHead] = this.#size[head]!;
          this.#libs[newHead] = this.#libs[head]!;
          head = newHead;
        }
        let cur = head;
        while (this.#next[cur] !== loc) cur = this.#next[cur]!;
        this.#next[cur] = this.#next[loc]!;
        let delta = 0;
        for (const d of this.adj) {
          const a = loc + d;
          if (this.colors[a] === EMPTY && !this.#isLibertyOf(a, head)) delta--;
        }
        delta++;
        this.#libs[head]! += delta;
        this.#size[head]!--;
      } else {
        // More than one: the chain may have split, so rebuild the chains around the point.
        let cur = loc;
        do {
          this.#head[cur] = NULL_LOC;
          cur = this.#next[cur]!;
        } while (cur !== loc);
        for (const d of this.adj) {
          const a = loc + d;
          if (this.colors[a] === record.pla && this.#head[a] === NULL_LOC) this.#rebuildChain(a, record.pla);
        }
      }
    }
  }

  /**
   * A stone placed during setup (KataGo `setStoneFailIfNoLibs` on an empty point): fails, changing nothing, if the stone
   * would capture or have no liberties. The ko location is kept.
   */
  setStone(loc: number, color: Stone): boolean {
    if (this.colors[loc] !== EMPTY) return false;
    if (this.isSuicide(loc, color) || this.wouldBeCapture(loc, color)) return false;
    const ko = this.koLoc;
    this.playMoveAssumeLegal(loc, color);
    this.koLoc = ko;
    return true;
  }

  /** KataGo `wouldBeCapture`. */
  wouldBeCapture(loc: number, pla: Stone): boolean {
    if (this.colors[loc] !== EMPTY) return false;
    const o = opp(pla);
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === o && this.getNumLiberties(a) === 1) return true;
    }
    return false;
  }

  /**
   * KataGo `isNonPassAliveSelfConnection`: would a stone of `pla` at the empty `loc`, outside `pla`'s pass-alive area,
   * join a chain of `pla`'s that isn't pass-alive to another of `pla`'s chains?
   */
  isNonPassAliveSelfConnection(loc: number, pla: Stone, passAliveArea: Uint8Array): boolean {
    if (this.colors[loc] !== EMPTY || passAliveArea[loc] === pla) return false;
    let nonPassAliveAdjHead = NULL_LOC;
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === pla && passAliveArea[a] === EMPTY) {
        nonPassAliveAdjHead = this.chainHead(a);
        break;
      }
    }
    if (nonPassAliveAdjHead === NULL_LOC) return false;
    for (const d of this.adj) {
      const a = loc + d;
      if (this.colors[a] === pla && this.chainHead(a) !== nonPassAliveAdjHead) return true;
    }
    return false;
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
    let s = loc;
    do {
      for (const d of this.adj) {
        const lib = s + d;
        if (this.colors[lib] !== EMPTY) continue;
        let dup = false;
        for (let j = bufStart; j < bufIdx + found; j++) if (buf[j] === lib) dup = true;
        if (!dup) buf[bufIdx + found++] = lib;
      }
      s = this.#next[s]!;
    } while (s !== loc);
    return found;
  }

  findLibertyGainingCaptures(loc: number, buf: number[], bufStart: number, bufIdx: number): number {
    const o = opp(this.colors[loc] as Stone);
    const checked: number[] = [];
    let found = 0;
    const chain = loc;
    let s = chain;
    do {
      for (const d of this.adj) {
        const a = s + d;
        if (this.colors[a] !== o) continue;
        const head = this.#head[a]!;
        if (this.#libs[head] === 1 && !checked.includes(head)) {
          found += this.findLiberties(a, buf, bufStart, bufIdx + found);
          checked.push(head);
        }
      }
      s = this.#next[s]!;
    } while (s !== chain);
    return found;
  }

  hasLibertyGainingCaptures(loc: number): boolean {
    const o = opp(this.colors[loc] as Stone);
    const chain = loc;
    let s = chain;
    do {
      for (const d of this.adj) {
        const a = s + d;
        if (this.colors[a] === o && this.getNumLiberties(a) === 1) return true;
      }
      s = this.#next[s]!;
    } while (s !== chain);
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
      const r = (this.#ladderRecordFirst = this.playMoveRecorded(move0, o, this.#ladderRecordFirst));
      works0 = this.searchIsLadderCaptured(loc, true, buf);
      this.undo(r);
    }
    if (this.isLegal(move1, o, false)) {
      const r = (this.#ladderRecordFirst = this.playMoveRecorded(move1, o, this.#ladderRecordFirst));
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
    const records = this.#ladderRecords;
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
      records[stackIdx] = this.playMoveRecorded(move, p, records[stackIdx]);
      nodes++;
      stackIdx++;
      moveListCur[stackIdx] = -1;
      moveListStarts[stackIdx] = moveListStarts[stackIdx - 1]! + moveListLens[stackIdx - 1]!;
      moveListLens[stackIdx] = 0;
    }
  }
}
