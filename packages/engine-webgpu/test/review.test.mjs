// Reviewing a game: the shared evaluator that combines concurrent searches' leaves into one network call, and
// reviewGame. Uses the b6 test network and the reference forward pass.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, forward, buildHistory, gtpMove, encodePositionV7, SharedEvaluator, reviewGame } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
let calls = 0;
const reference = { maxBatch: 16, evaluate: async (inputs) => (calls++, inputs.map((i) => forward(model, i))) };

test('the shared evaluator combines concurrent requests and gives each caller its own outputs', async () => {
  const inputs = ['E5', 'C3', 'G7', 'D4', 'F6', 'C7'].map((m) => encodePositionV7({ xSize: 9, ySize: 9, rules: 'tromp-taylor', komi: 7.5, moves: [gtpMove('B', m, 9)] }).input);
  const shared = new SharedEvaluator(reference);
  calls = 0;
  // Three callers asking at the same time, with 1, 2 and 3 inputs.
  const [a, b, c] = await Promise.all([shared.evaluate(inputs.slice(0, 1)), shared.evaluate(inputs.slice(1, 3)), shared.evaluate(inputs.slice(3, 6))]);
  assert.equal(calls, 1, 'one network call');
  assert.equal(shared.inputs, 6);
  const direct = inputs.map((i) => forward(model, i));
  for (const [got, want] of [[a, direct.slice(0, 1)], [b, direct.slice(1, 3)], [c, direct.slice(3, 6)]]) {
    assert.equal(got.length, want.length);
    got.forEach((g, k) => {
      assert.deepEqual(g.value, want[k].value);
      assert.deepEqual(g.ownership, want[k].ownership);
    });
  }
});

test('the shared evaluator splits requests larger than the network batch across calls', async () => {
  const input = encodePositionV7({ xSize: 9, ySize: 9, rules: 'tromp-taylor', komi: 7.5, moves: [] }).input;
  const shared = new SharedEvaluator(reference);
  calls = 0;
  const outs = await Promise.all(Array.from({ length: 5 }, () => shared.evaluate(Array.from({ length: 6 }, () => input))));
  assert.equal(outs.flat().length, 30);
  assert.ok(calls >= 2 && calls <= 3, `calls ${calls}`);
});

test('reviewGame searches every turn, several at once, with combined network calls', async () => {
  const game = JSON.parse(readFileSync(here('./goldens/search-late-9.json'), 'utf8')).moves.slice(0, 8);
  const histories = [];
  for (let t = 0; t < game.length; t++) {
    histories.push(buildHistory({ xSize: 9, ySize: 9, rules: 'tromp-taylor', komi: 7.5, moves: game.slice(0, t).map(([c, l]) => gtpMove(c, l, 9)) }));
  }
  calls = 0;
  const seen = [];
  const results = await reviewGame(model, reference, histories, { visits: 24, batchSize: 4, concurrency: 4, onTurn: (t) => seen.push(t) });
  assert.equal(results.length, 8);
  for (const r of results) {
    assert.equal(r.rootVisits, 24);
    assert.equal(r.moveInfos.reduce((s, m) => s + m.visits, 0), 23);
  }
  assert.deepEqual([...seen].sort((x, y) => x - y), [0, 1, 2, 3, 4, 5, 6, 7]);
  // 8 searches × 23 visits after the root, in batches of 4 per search: alone they would need at least 8 × 6 calls.
  assert.ok(calls < 8 * 6, `calls ${calls}`);
});
