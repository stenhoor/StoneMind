// A search golden query's position, as native KataGo's analysis engine builds it (the fields of part B, step 11c, are
// in q.extra). Shared by test/search.test.mjs and test/hash.test.mjs.
import { Board, PASS_LOC, BLACK, WHITE, gtpMove, parseRules, analysisHistory } from '../dist/index.js';

export const stoneOf = (c) => (c === 'B' ? BLACK : WHITE);

/** A GTP vertex ("C3" or "pass") of a size × size board as a board location. */
export const queryLoc = (size, v) => (v === 'pass' ? PASS_LOC : new Board(size, size).loc(...gtpMove('B', v, size).slice(1)));

export function queryHistory(q) {
  const x = q.extra ?? {};
  const rules = { ...parseRules(q.rules), ...(x.whiteHandicapBonus ? { whiteHandicapBonusRule: x.whiteHandicapBonus } : {}) };
  const position = { xSize: q.size, ySize: q.size, rules, komi: q.komi, moves: q.moves.map(([c, v]) => [stoneOf(c), queryLoc(q.size, v)]) };
  if (x.initialStones) position.initialStones = x.initialStones.map(([c, v]) => [stoneOf(c), queryLoc(q.size, v)]);
  if (x.initialPlayer) position.initialPlayer = stoneOf(x.initialPlayer);
  return analysisHistory(position);
}
