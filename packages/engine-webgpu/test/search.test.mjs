// The batched search's bookkeeping. The search goldens (native KataGo's analysis) are in test/search-goldens-*.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, forward, buildHistory, gtpMove, Search } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
const golden = JSON.parse(readFileSync(here('./goldens/search-g170-b6c96-s175395328-d26788732.json'), 'utf8'));
const reference = { maxBatch: 64, evaluate: async (inputs) => inputs.map((i) => forward(model, i)) };
const historyOf = (q) => buildHistory({ xSize: q.size, ySize: q.size, rules: q.rules, komi: q.komi, moves: q.moves.map(([c, l]) => gtpMove(c, l, q.size)) });

test('batched search: every visit is accounted for and no leaf is left waiting', async () => {
  const q = golden.queries.find((x) => x.id === 'opening-9');
  const search = new Search(model, reference, historyOf(q), { batchSize: 16, seed: 3 });
  await search.run({ maxVisits: 80 });
  const a = search.analysis();
  assert.equal(a.rootVisits, 80);
  // Mirror-image copies (root symmetry pruning) repeat a searched move's visits, so they're left out of the sum.
  assert.equal(a.moveInfos.filter((m) => m.isSymmetryOf === undefined).reduce((s, m) => s + m.visits, 0), 79);
  // With graph search a node can be reached more than one way, so each is checked once.
  const seen = new Set();
  const walk = (n) => {
    if (seen.has(n)) return;
    seen.add(n);
    assert.equal(n.virtualLosses, 0);
    assert.notEqual(n.state, 1, 'no node left waiting for the network');
    for (const e of n.children) walk(e.node);
  };
  walk(search.root);
  assert.ok(search.stats.batches < 79, 'leaves were batched');
});

