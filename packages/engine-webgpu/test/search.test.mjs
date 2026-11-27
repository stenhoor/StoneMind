// The search against native KataGo's analysis (scripts/capture-search-goldens.mjs): one search thread, symmetry 0, and
// the features our search doesn't have yet switched off in both, so the two should visit the same moves the same number
// of times. Uses the reference forward pass, so it runs without a GPU: the b6 test network on the full set of positions,
// and b18c384nbt on a short set when STONEMIND_B18_MODEL points at its model file (it isn't downloaded in CI). b6 is
// model version 8, which has no short-term error outputs, so only b18 checks uncertainty weighting.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, forward, buildHistory, gtpMove, Search, PASS_LOC, UnsupportedPositionError } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
const golden = JSON.parse(readFileSync(here('./goldens/search-g170-b6c96-s175395328-d26788732.json'), 'utf8'));
const reference = { maxBatch: 64, evaluate: async (inputs) => inputs.map((i) => forward(model, i)) };
const NETS = [
  { golden: './goldens/search-g170-b6c96-s175395328-d26788732.json', model: here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz') },
  { golden: './goldens/search-kata1-b18c384nbt-s9996604416-d4316597426.json', model: process.env.STONEMIND_B18_MODEL ?? '' },
];
const COLUMNS = 'ABCDEFGHJKLMNOPQRST';

// Raw outputs match native within the goldens test's tolerances (2e-4 policy, 2e-3 lead), so values may differ that much.
// Utility adds score utility to the win/loss value (twice the win rate's error), so its tolerance is wider; the play
// selection value is compared relative to its size, since its LCB adjustment scales with the moves' weights.
const TOL = { winrate: 2e-4, scoreLead: 2e-3, utility: 1e-3, playSelectionValue: 1e-3 };

const historyOf = (q) => buildHistory({ xSize: q.size, ySize: q.size, rules: q.rules, komi: q.komi, moves: q.moves.map(([c, l]) => gtpMove(c, l, q.size)) });

for (const net of NETS) {
  const netName = net.golden.replace(/^.*search-|\.json$/g, '');
  if (!net.model || !existsSync(net.model)) {
    test(`search matches native KataGo: ${netName}`, { skip: 'model file not present (set STONEMIND_B18_MODEL for b18c384nbt)' }, () => {});
    continue;
  }
  const netModel = net.model === here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz') ? model : await loadModel(new Uint8Array(readFileSync(net.model)));
  const netReference = { maxBatch: 64, evaluate: async (inputs) => inputs.map((i) => forward(netModel, i)) };
  const netGolden = JSON.parse(readFileSync(here(net.golden), 'utf8'));
  for (const q of netGolden.queries) {
    test(`search matches native KataGo: ${netName} ${q.id} (${q.maxVisits} visits)`, async () => {
      const hist = historyOf(q);
      const name = (loc) => (loc === PASS_LOC ? 'pass' : COLUMNS[hist.board.x(loc)] + (q.size - hist.board.y(loc)));
      const search = new Search(netModel, netReference, hist, { symmetry: 0, batchSize: 1, analysisPVLen: 6, wideRootNoise: 0 });
      await search.run({ maxVisits: q.maxVisits });
      const a = search.analysis();
      assert.equal(a.rootVisits, q.rootVisits);
      const ours = new Map(a.moveInfos.map((m) => [name(m.loc), m]));
      assert.deepEqual([...ours.keys()].sort(), q.moveInfos.map((m) => m.move).sort(), 'the same moves searched');
      for (const want of q.moveInfos) {
        const got = ours.get(want.move);
        assert.equal(got.visits, want.visits, `${want.move} visits`);
        assert.equal(got.edgeVisits, want.edgeVisits, `${want.move} edge visits`);
        assert.equal(got.isSymmetryOf === undefined ? undefined : name(got.isSymmetryOf), want.isSymmetryOf, `${want.move} isSymmetryOf`);
        assert.equal(got.order, want.order, `${want.move} order`);
        assert.ok(Math.abs(got.winrate - want.winrate) <= TOL.winrate, `${want.move} winrate ${got.winrate} vs ${want.winrate}`);
        assert.ok(Math.abs(got.scoreLead - want.scoreLead) <= TOL.scoreLead, `${want.move} scoreLead ${got.scoreLead} vs ${want.scoreLead}`);
        assert.ok(Math.abs(got.lcb - want.lcb) <= TOL.winrate, `${want.move} lcb ${got.lcb} vs ${want.lcb}`);
        assert.ok(Math.abs(got.utilityLcb - want.utilityLcb) <= TOL.utility, `${want.move} utilityLcb ${got.utilityLcb} vs ${want.utilityLcb}`);
        assert.ok(Math.abs(got.playSelectionValue - want.playSelectionValue) <= TOL.playSelectionValue * Math.max(1, want.playSelectionValue), `${want.move} playSelectionValue ${got.playSelectionValue} vs ${want.playSelectionValue}`);
        assert.deepEqual(got.pv.map(name), want.pv, `${want.move} principal variation`);
      }
    });
  }
}

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

test('the search refuses territory scoring for now', () => {
  assert.throws(() => new Search(model, reference, historyOf({ size: 9, rules: 'japanese', komi: 6.5, moves: [] })), UnsupportedPositionError);
});
