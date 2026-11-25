/**
 * Pass-alive area (Benson's algorithm with KataGo's extensions), ported from `Board::calculateArea` and
 * `Board::calculateAreaForPla` in KataGo's `cpp/game/board.cpp` (KataGo 1.18, MIT, © David J Wu).
 */
import { BLACK, Board, EMPTY, WHITE, opp, type Stone } from './board.js';

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
  const n = board.arraySize;
  const regionIdxByLoc = new Int32Array(n).fill(-1);
  const regionMembers: number[][] = [];
  const vital: number[][] = []; // per region: pla chain heads adjacent to every (relevant) location of the region
  const numInternalSpacesMax2: number[] = [];
  const containsOpp: boolean[] = [];
  let atLeastOnePla = false;

  for (let y = 0; y < board.ySize; y++) {
    for (let x = 0; x < board.xSize; x++) {
      const loc = board.loc(x, y);
      if (regionIdxByLoc[loc] !== -1) continue;
      if (colors[loc] !== EMPTY) {
        atLeastOnePla ||= colors[loc] === pla;
        continue;
      }
      const regionIdx = regionMembers.length;
      // Initial vital heads: pla chains adjacent to the region's first location.
      const initialVital: number[] = [];
      for (const d of board.adj) {
        const a = loc + d;
        if (colors[a] === pla) {
          const head = board.chainHead(a);
          if (!initialVital.includes(head)) initialVital.push(head);
        }
      }
      let v = initialVital;
      let internal = 0;
      let hasOpp = false;
      const members: number[] = [];
      const queue = [loc];
      regionIdxByLoc[loc] = regionIdx;
      for (let qi = 0; qi < queue.length; qi++) {
        const l = queue[qi]!;
        if (v.length > 0 && (multiStoneSuicideLegal || colors[l] === EMPTY)) {
          v = v.filter((head) => board.isAdjacentToPlaHead(pla, l, head));
        }
        if (internal < 2 && !board.isAdjacentToPla(l, pla)) internal++;
        if (colors[l] === o) hasOpp = true;
        members.push(l);
        for (const d of board.adj) {
          const a = l + d;
          if ((colors[a] === EMPTY || colors[a] === o) && regionIdxByLoc[a] === -1) {
            regionIdxByLoc[a] = regionIdx;
            queue.push(a);
          }
        }
      }
      regionMembers.push(members);
      vital.push(v);
      numInternalSpacesMax2.push(internal);
      containsOpp.push(hasOpp);
    }
  }

  const plaHeads = board.chainHeads(pla);
  const vitalCount = new Map<number, number>();
  for (const h of plaHeads) vitalCount.set(h, 0);
  for (const v of vital) for (const h of v) vitalCount.set(h, (vitalCount.get(h) ?? 0) + 1);

  const bordersNonPassAlive = new Array<boolean>(regionMembers.length).fill(false);
  const killed = new Set<number>();
  for (;;) {
    let killedAnything = false;
    for (const head of plaHeads) {
      if (killed.has(head) || (vitalCount.get(head) ?? 0) >= 2) continue;
      killed.add(head);
      killedAnything = true;
      for (const s of board.chainStones(head)) {
        for (const d of board.adj) {
          const a = s + d;
          const r = regionIdxByLoc[a]!;
          if (r >= 0 && !bordersNonPassAlive[r] && (colors[a] === EMPTY || colors[a] === o)) {
            bordersNonPassAlive[r] = true;
            for (const h of vital[r]!) vitalCount.set(h, vitalCount.get(h)! - 1);
          }
        }
      }
    }
    if (!killedAnything) break;
  }

  for (const head of plaHeads) if (!killed.has(head)) for (const s of board.chainStones(head)) result[s] = pla;

  for (let r = 0; r < regionMembers.length; r++) {
    let shouldMark = numInternalSpacesMax2[r]! <= 1 && !bordersNonPassAlive[r] && atLeastOnePla;
    shouldMark ||= safeBigTerritories && !containsOpp[r] && !bordersNonPassAlive[r] && atLeastOnePla;
    if (shouldMark) {
      for (const l of regionMembers[r]!) result[l] = pla;
    } else if (unsafeBigTerritories && !containsOpp[r] && atLeastOnePla) {
      for (const l of regionMembers[r]!) if (result[l] === EMPTY) result[l] = pla;
    }
  }
}
