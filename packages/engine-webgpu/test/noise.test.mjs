// Wide root noise against native KataGo, statistically: the noise is random and the two engines' random numbers differ,
// so single runs can't match. Native's visits over many runs (scripts/capture-noise-goldens.mjs) are compared with
// ours over as many runs, each with its own seed.
//
// The pass rule was fixed before either engine was run: for every move searched in either engine, our mean visits must
// be within 4 standard errors of native's (the two means' combined standard error), plus half a visit, so that a move
// whose visits never vary has to match exactly. With 30 moves compared (27 and 3), 4 standard errors keeps the chance of a
// failure from sampling alone near 0.2%.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, forward, buildHistory, gtpMove, Search, PASS_LOC } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const COLUMNS = 'ABCDEFGHJKLMNOPQRST';
const golden = JSON.parse(readFileSync(here('./goldens/noise-g170-b6c96-s175395328-d26788732.json'), 'utf8'));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
const reference = { maxBatch: 1, evaluate: async (inputs) => inputs.map((i) => forward(model, i)) };
const STANDARD_ERRORS = 4;
const SLACK = 0.5;

const meanAndVariance = (xs) => {
  const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
  const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / (xs.length - 1);
  return { mean, variance };
};

for (const pos of golden.positions) {
  test(`wide root noise matches native KataGo over ${pos.visits.length} runs: ${pos.id} (${pos.maxVisits} visits)`, async () => {
    const hist = buildHistory({ xSize: pos.size, ySize: pos.size, rules: pos.rules, komi: pos.komi, moves: pos.moves.map(([c, l]) => gtpMove(c, l, pos.size)) });
    const name = (loc) => (loc === PASS_LOC ? 'pass' : COLUMNS[hist.board.x(loc)] + (pos.size - hist.board.y(loc)));
    // One evaluation cache for all runs: every run has the same history, and the evaluations don't depend on the seed.
    const cache = new Map();
    const ours = [];
    for (let i = 0; i < pos.visits.length; i++) {
      const search = new Search(model, reference, hist, { symmetry: 0, batchSize: 1, seed: i + 1 }, cache);
      await search.run({ maxVisits: pos.maxVisits });
      ours.push(Object.fromEntries(search.analysis().moveInfos.filter((m) => m.isSymmetryOf === undefined).map((m) => [name(m.loc), m.visits])));
    }
    const moves = new Set([...pos.visits, ...ours].flatMap((r) => Object.keys(r)));
    const failures = [];
    for (const move of moves) {
      const n = meanAndVariance(pos.visits.map((r) => r[move] ?? 0));
      const o = meanAndVariance(ours.map((r) => r[move] ?? 0));
      const se = Math.sqrt(n.variance / pos.visits.length + o.variance / ours.length);
      if (Math.abs(o.mean - n.mean) > STANDARD_ERRORS * se + SLACK) failures.push(`${move}: ours ${o.mean.toFixed(2)}, native ${n.mean.toFixed(2)} (se ${se.toFixed(2)})`);
    }
    assert.deepEqual(failures, [], 'mean visits per move');
  });
}
