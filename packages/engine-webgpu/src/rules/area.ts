/**
 * Pass-alive area (Benson's algorithm with KataGo's extensions), ported from `Board::calculateArea` and
 * `Board::calculateAreaForPla` in KataGo's `cpp/game/board.cpp` (KataGo 1.18, MIT, © David J Wu).
 */
import { BLACK, Board, EMPTY, NULL_LOC, WHITE, opp, type Stone } from './board.js';

/**
 * KataGo `calculateArea`: result[loc] is BLACK, WHITE or EMPTY for every location of the board's array.
 * @param nonPassAliveStones Also count stones that aren't pass-alive as their colour's area.
 * @param safeBigTerritories Count empty regions bordered only by pass-alive stones of one colour.
 * @param unsafeBigTerritories Count empty regions bordered only by stones of one colour (pass-alive or not).
 */
export function calculateArea(
  board: Board,
  nonPassAliveStones: boolean,
  safeBigTerritories: boolean,
  unsafeBigTerritories: boolean,
  multiStoneSuicideLegal: boolean,
): Uint8Array {
  const result = new Uint8Array(board.arraySize);
  calculateAreaForPla(board, BLACK, safeBigTerritories, unsafeBigTerritories, multiStoneSuicideLegal, result);
  calculateAreaForPla(board, WHITE, safeBigTerritories, unsafeBigTerritories, multiStoneSuicideLegal, result);
  if (nonPassAliveStones) {
    for (let y = 0; y < board.ySize; y++) {
      for (let x = 0; x < board.xSize; x++) {
        const loc = board.loc(x, y);
        if (result[loc] === EMPTY) result[loc] = board.colors[loc]!;
      }
    }
  }
  return result;
}

/** Scratch arrays for one board array size, reused between calls (KataGo keeps them on the stack). */
interface AreaScratch {
  regionIdxByLoc: Int16Array;
  nextEmptyOrOpp: Int32Array;
  bordersNonPassAliveByHead: Uint8Array;
  vitalLists: Int32Array;
  regionHeads: Int32Array;
  vitalStart: Int32Array;
  vitalLen: Int32Array;
  numInternalSpacesMax2: Uint8Array;
  containsOpp: Uint8Array;
  queue: Int32Array;
  plaHeads: Int32Array;
  killed: Uint8Array;
  vitalCountByHead: Int32Array;
}
const scratchBySize = new Map<number, AreaScratch>();
function scratchFor(n: number): AreaScratch {
  let s = scratchBySize.get(n);
  if (!s) {
    const maxRegions = Math.floor((n + 1) / 2) + 1;
    s = {
      regionIdxByLoc: new Int16Array(n),
      nextEmptyOrOpp: new Int32Array(n),
      bordersNonPassAliveByHead: new Uint8Array(n),
      vitalLists: new Int32Array(maxRegions * 4),
      regionHeads: new Int32Array(maxRegions),
      vitalStart: new Int32Array(maxRegions),
      vitalLen: new Int32Array(maxRegions),
      numInternalSpacesMax2: new Uint8Array(maxRegions),
      containsOpp: new Uint8Array(maxRegions),
      queue: new Int32Array(n),
      plaHeads: new Int32Array(n),
      killed: new Uint8Array(n),
      vitalCountByHead: new Int32Array(n),
    };
    scratchBySize.set(n, s);
  }
  return s;
}

/** KataGo `calculateAreaForPla`, with KataGo's arrays and linked lists. */
function calculateAreaForPla(
  board: Board,
  pla: Stone,
  safeBigTerritories: boolean,
  unsafeBigTerritories: boolean,
  multiStoneSuicideLegal: boolean,
  result: Uint8Array,
): void {
  const o = opp(pla);
  const colors = board.colors;
  const adj = board.adj;
  const n = board.arraySize;
  const S = scratchFor(n);
  const { regionIdxByLoc, nextEmptyOrOpp, bordersNonPassAliveByHead, vitalLists, regionHeads, vitalStart, vitalLen, numInternalSpacesMax2, containsOpp, queue } = S;
  regionIdxByLoc.fill(-1);
  nextEmptyOrOpp.fill(NULL_LOC);
  bordersNonPassAliveByHead.fill(0);
  let vitalTotal = 0;
  let numRegions = 0;

  // Breadth-first search of one maximal empty-or-opp region; links it through nextEmptyOrOpp. Returns the list's tip.
  const buildRegion = (initialLoc: number, regionIdx: number): number => {
    let tailTarget = initialLoc;
    let vNonZero = vitalLen[regionIdx]! > 0;
    let qHead = 0;
    let qTail = 1;
    queue[0] = initialLoc;
    regionIdxByLoc[initialLoc] = regionIdx;
    while (qHead !== qTail) {
      const loc = queue[qHead++]!;
      if (vNonZero && (multiStoneSuicideLegal || colors[loc] === EMPTY)) {
        const vs = vitalStart[regionIdx]!;
        const oldLen = vitalLen[regionIdx]!;
        let newLen = 0;
        for (let i = 0; i < oldLen; i++) {
          if (board.isAdjacentToPlaHead(pla, loc, vitalLists[vs + i]!)) vitalLists[vs + newLen++] = vitalLists[vs + i]!;
        }
        vitalLen[regionIdx] = newLen;
        vNonZero = newLen > 0;
      }
      if (numInternalSpacesMax2[regionIdx]! < 2 && !board.isAdjacentToPla(loc, pla)) numInternalSpacesMax2[regionIdx]!++;
      if (colors[loc] === o) containsOpp[regionIdx] = 1;
      nextEmptyOrOpp[loc] = tailTarget;
      tailTarget = loc;
      for (const d of adj) {
        const a = loc + d;
        if ((colors[a] === EMPTY || colors[a] === o) && regionIdxByLoc[a] === -1) {
          queue[qTail++] = a;
          regionIdxByLoc[a] = regionIdx;
        }
      }
    }
    return tailTarget;
  };

  let atLeastOnePla = false;
  for (let y = 0; y < board.ySize; y++) {
    for (let x = 0; x < board.xSize; x++) {
      const loc = board.loc(x, y);
      if (regionIdxByLoc[loc] !== -1) continue;
      if (colors[loc] !== EMPTY) {
        atLeastOnePla ||= colors[loc] === pla;
        continue;
      }
      const regionIdx = numRegions++;
      regionHeads[regionIdx] = loc;
      vitalStart[regionIdx] = vitalTotal;
      vitalLen[regionIdx] = 0;
      numInternalSpacesMax2[regionIdx] = 0;
      containsOpp[regionIdx] = 0;
      // Every adjacent pla head starts as vital; buildRegion filters them.
      const vs = vitalTotal;
      let initialLen = 0;
      for (const d of adj) {
        const a = loc + d;
        if (colors[a] === pla) {
          const head = board.chainHead(a);
          let present = false;
          for (let j = 0; j < initialLen; j++) if (vitalLists[vs + j] === head) { present = true; break; }
          if (!present) vitalLists[vs + initialLen++] = head;
        }
      }
      vitalLen[regionIdx] = initialLen;
      const tailTarget = buildRegion(loc, regionIdx);
      nextEmptyOrOpp[loc] = tailTarget;
      vitalTotal += vitalLen[regionIdx]!;
    }
  }

  const { plaHeads, killed, vitalCountByHead } = S;
  let numPlaHeads = 0;
  for (let loc = 0; loc < n; loc++) if (colors[loc] === pla && board.chainHead(loc) === loc) plaHeads[numPlaHeads++] = loc;
  killed.fill(0, 0, numPlaHeads);
  for (let i = 0; i < numPlaHeads; i++) vitalCountByHead[plaHeads[i]!] = 0;
  for (let i = 0; i < numRegions; i++) {
    const vs = vitalStart[i]!;
    const vl = vitalLen[i]!;
    for (let j = 0; j < vl; j++) vitalCountByHead[vitalLists[vs + j]!]!++;
  }

  // Benson's iteration.
  for (;;) {
    let killedAnything = false;
    for (let i = 0; i < numPlaHeads; i++) {
      if (killed[i]) continue;
      const plaHead = plaHeads[i]!;
      if (vitalCountByHead[plaHead]! < 2) {
        killed[i] = 1;
        killedAnything = true;
        let cur = plaHead;
        do {
          for (const d of adj) {
            const a = cur + d;
            const regionIdx = regionIdxByLoc[a]!;
            if (regionIdx >= 0 && !bordersNonPassAliveByHead[regionHeads[regionIdx]!] && (colors[a] === EMPTY || colors[a] === o)) {
              bordersNonPassAliveByHead[regionHeads[regionIdx]!] = 1;
              const vs = vitalStart[regionIdx]!;
              const vl = vitalLen[regionIdx]!;
              for (let k = 0; k < vl; k++) vitalCountByHead[vitalLists[vs + k]!]!--;
            }
          }
          cur = board.nextInChain(cur);
        } while (cur !== plaHead);
      }
    }
    if (!killedAnything) break;
  }

  // Pass-alive groups.
  for (let i = 0; i < numPlaHeads; i++) {
    if (killed[i]) continue;
    const plaHead = plaHeads[i]!;
    let cur = plaHead;
    do {
      result[cur] = pla;
      cur = board.nextInChain(cur);
    } while (cur !== plaHead);
  }

  // Territory.
  for (let i = 0; i < numRegions; i++) {
    const head = regionHeads[i]!;
    let shouldMark = numInternalSpacesMax2[i]! <= 1 && !bordersNonPassAliveByHead[head] && atLeastOnePla;
    shouldMark ||= safeBigTerritories && !containsOpp[i] && !bordersNonPassAliveByHead[head] && atLeastOnePla;
    if (shouldMark) {
      let cur = head;
      do {
        result[cur] = pla;
        cur = nextEmptyOrOpp[cur]!;
      } while (cur !== head);
    } else if (unsafeBigTerritories && !containsOpp[i] && atLeastOnePla) {
      let cur = head;
      do {
        if (result[cur] === EMPTY) result[cur] = pla;
        cur = nextEmptyOrOpp[cur]!;
      } while (cur !== head);
    }
  }
}
