// Reusing ladder results across boards (encodeHistoryV7's `ladders` option) gives the same inputs as searching each board.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { buildHistory, encodeHistoryV7, gtpMove } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const positions = JSON.parse(readFileSync(here('./goldens/positions.json'), 'utf8')).positions;
const game = JSON.parse(readFileSync(here('./goldens/agreement-game.json'), 'utf8'));

const histories = [];
for (const p of positions) {
  histories.push(buildHistory({ xSize: p.boardSize, ySize: p.boardSize, rules: p.rules, komi: p.komi, moves: p.moves.map(([c, l]) => gtpMove(c, l, p.boardSize)) }));
}
for (let t = 0; t <= game.moves.length; t += 1) {
  histories.push(buildHistory({ xSize: 19, ySize: 19, rules: game.rules, komi: game.komi, moves: game.moves.slice(0, t).map(([c, l]) => gtpMove(c, l, 19)) }));
}

test('inputs with a ladder cache equal inputs without, in any order, and the cache is reused', () => {
  const plain = histories.map((h) => encodeHistoryV7(h).input);
  for (const order of [histories.map((_, i) => i), histories.map((_, i) => histories.length - 1 - i)]) {
    const ladders = new Map();
    for (const i of order) {
      const cached = encodeHistoryV7(histories[i], { ladders }).input;
      assert.deepEqual(cached.spatial, plain[i].spatial, `position ${i}`);
      assert.deepEqual(cached.global, plain[i].global, `position ${i}`);
    }
    // Three boards per input, but consecutive turns share two of them.
    assert.ok(ladders.size < 1.5 * histories.length, `${ladders.size} boards for ${histories.length} inputs`);
  }
  // Some of these boards have ladders (fight-19 covers the ladder search).
  assert.ok(plain.some((x) => x.spatial.subarray(14 * 361, 15 * 361).some((v) => v === 1)));
});
