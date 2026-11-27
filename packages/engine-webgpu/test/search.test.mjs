// The search against native KataGo's analysis (scripts/capture-search-goldens.mjs): one search thread, symmetry 0, and
// the features our search doesn't have yet switched off in both, so the two should visit the same moves the same number
// of times. Uses the reference forward pass, so it runs without a GPU: the b6 test network on the full set of positions,
// and b18c384nbt on a short set when STONEMIND_B18_MODEL points at its model file (it isn't downloaded in CI). b6 is
// model version 8, which has no short-term error outputs, so only b18 checks uncertainty weighting.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  loadModel, forward, buildHistory, gtpMove, Search, PASS_LOC, UnsupportedPositionError, loadPositionHashes, avoidMoveUntilByLoc,
  searchParamsFromConfig,
} from '../dist/index.js';
import { queryHistory, queryLoc, stoneOf } from './query-position.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
const golden = JSON.parse(readFileSync(here('./goldens/search-g170-b6c96-s175395328-d26788732.json'), 'utf8'));
const reference = { maxBatch: 64, evaluate: async (inputs) => inputs.map((i) => forward(model, i)) };
const NETS = [
  { golden: './goldens/search-g170-b6c96-s175395328-d26788732.json', model: here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz') },
  { golden: './goldens/search-kata1-b18c384nbt-s9996604416-d4316597426.json', model: process.env.STONEMIND_B18_MODEL ?? '' },
];
const COLUMNS = 'ABCDEFGHJKLMNOPQRST';

// Raw outputs match native within the goldens test's tolerances (2e-4 probabilities, policy and ownership, 2e-3 lead), so
// values may differ that much.
// - Utility adds score utility to the win/loss value (twice the win rate's error), so its tolerance is wider.
// - The play selection value and the weights are compared relative to their size: the LCB adjustment scales with the
//   moves' weights, and with uncertainty weighting (b18) each evaluation's weight comes from the network's short-term
//   error outputs.
// - Score standard deviations and the network's short-term errors and time left are score-like, relative above 1.
// - Ownership averages values each within 2e-4, and native rounds it to 1e-6. Its standard deviation is the square root
//   of E[x²] − E[x]², which amplifies small errors where the deviation is near 0, so the variance is compared: each of
//   x² and E[x]² may be off by up to 2 × 2e-4 for |x| ≤ 1, so 1e-3 with the rounding.
const TOL = { winrate: 2e-4, scoreLead: 2e-3, utility: 1e-3, relative: 1e-3, ownership: 2e-4 + 1e-6, ownershipVariance: 1e-3 };
const near = (got, want, tol, what) => assert.ok(Math.abs(got - want) <= tol, `${what}: ${got} vs ${want}`);
const nearScore = (got, want, what) => near(got, want, TOL.scoreLead * Math.max(1, Math.abs(want)), what);
const nearRelative = (got, want, what) => near(got, want, TOL.relative * Math.max(1, Math.abs(want)), what);
const nearOwnership = (got, want, what) => {
  assert.equal(got.length, want.length, `${what} length`);
  want.forEach((w, i) => near(got[i], w, TOL.ownership, `${what}[${i}]`));
};
const nearOwnershipStdev = (got, want, what) => {
  assert.equal(got.length, want.length, `${what} length`);
  want.forEach((w, i) => near(got[i] * got[i], w * w, TOL.ownershipVariance, `${what}[${i}] squared`));
};
// Every field native reports, so a field we don't check can't slip by.
const MOVE_FIELDS = ['move', 'visits', 'edgeVisits', 'order', 'isSymmetryOf', 'pv', 'pvVisits', 'pvEdgeVisits', 'winrate', 'lcb', 'scoreLead', 'scoreMean', 'scoreSelfplay', 'scoreStdev', 'utility', 'utilityLcb', 'playSelectionValue', 'prior', 'noResultValue', 'weight', 'edgeWeight', 'ownership', 'ownershipStdev'];
const ROOT_FIELDS = ['visits', 'weight', 'winrate', 'scoreSelfplay', 'scoreLead', 'scoreStdev', 'utility', 'rawWinrate', 'rawLead', 'rawScoreSelfplay', 'rawScoreSelfplayStdev', 'rawNoResultProb', 'rawStWrError', 'rawStScoreError', 'rawVarTimeLeft', 'currentPlayer', 'thisHash', 'symHash'];
const RESPONSE_FIELDS = ['moveInfos', 'rootInfo', 'policy', 'ownership', 'ownershipStdev'];

const historyOf = (q) => buildHistory({ xSize: q.size, ySize: q.size, rules: q.rules, komi: q.komi, moves: q.moves.map(([c, l]) => gtpMove(c, l, q.size)) });

// A query's position, search parameters and avoided moves, as native's analysis engine reads them (the fields of part
// B, step 11c, are in q.extra). Native's config switches wide root noise off (the capture script's overrides): that is
// the config overrideSettings applies to.
const searchSetup = (q) => {
  const x = q.extra ?? {};
  const hist = queryHistory(q);
  const { params, unusedKeys } = searchParamsFromConfig(
    { wideRootNoise: '0', ...Object.fromEntries(Object.entries(x.overrideSettings ?? {}).map(([k, v]) => [k, String(v)])) },
    { symmetry: 0, batchSize: 1, analysisPVLen: 6 },
  );
  assert.deepEqual(unusedKeys, [], 'every override is a search parameter');
  if (x.rootFpuReductionMax !== undefined) params.rootFpuReductionMax = x.rootFpuReductionMax;
  if (x.rootPolicyTemperature !== undefined) params.rootPolicyTemperature = params.rootPolicyTemperatureEarly = x.rootPolicyTemperature;
  const entries = x.allowMoves ?? x.avoidMoves;
  const avoid = entries
    ? avoidMoveUntilByLoc(hist.board.arraySize, entries.map((e) => ({ player: stoneOf(e.player), moves: e.moves.map((v) => queryLoc(q.size, v)), untilDepth: e.untilDepth })), !!x.allowMoves)
    : null;
  return { hist, params, avoid };
};

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
      const { hist, params, avoid } = searchSetup(q);
      const name = (loc) => (loc === PASS_LOC ? 'pass' : COLUMNS[hist.board.x(loc)] + (q.size - hist.board.y(loc)));
      const search = new Search(netModel, netReference, hist, params);
      if (avoid) search.setAvoidMoveUntilByLoc(avoid.black, avoid.white);
      await search.run({ maxVisits: q.maxVisits });
      const r = q.response;
      assert.deepEqual(Object.keys(r).filter((k) => !RESPONSE_FIELDS.includes(k)), [], 'fields of the response the test knows');
      const a = search.analysis({ movesOwnership: true, movesOwnershipStdev: true });
      assert.equal(a.rootVisits, r.rootInfo.visits);
      const ours = new Map(a.moveInfos.map((m) => [name(m.loc), m]));
      assert.deepEqual([...ours.keys()].sort(), r.moveInfos.map((m) => m.move).sort(), 'the same moves searched');
      for (const want of r.moveInfos) {
        const m = want.move;
        assert.deepEqual(Object.keys(want).filter((k) => !MOVE_FIELDS.includes(k)), [], `${m}: fields the test knows`);
        const got = ours.get(m);
        assert.equal(got.visits, want.visits, `${m} visits`);
        assert.equal(got.edgeVisits, want.edgeVisits, `${m} edge visits`);
        assert.equal(got.isSymmetryOf === undefined ? undefined : name(got.isSymmetryOf), want.isSymmetryOf, `${m} isSymmetryOf`);
        assert.equal(got.order, want.order, `${m} order`);
        assert.deepEqual(got.pv.map(name), want.pv, `${m} principal variation`);
        assert.deepEqual(got.pvVisits, want.pvVisits, `${m} pvVisits`);
        assert.deepEqual(got.pvEdgeVisits, want.pvEdgeVisits, `${m} pvEdgeVisits`);
        near(got.winrate, want.winrate, TOL.winrate, `${m} winrate`);
        near(got.lcb, want.lcb, TOL.winrate, `${m} lcb`);
        // KataGo reports the lead as scoreMean too, for older tools, and the expected score as scoreSelfplay.
        near(got.scoreLead, want.scoreLead, TOL.scoreLead, `${m} scoreLead`);
        near(got.scoreLead, want.scoreMean, TOL.scoreLead, `${m} scoreMean`);
        near(got.scoreMean, want.scoreSelfplay, TOL.scoreLead, `${m} scoreSelfplay`);
        nearScore(got.scoreStdev, want.scoreStdev, `${m} scoreStdev`);
        near(got.utility, want.utility, TOL.utility, `${m} utility`);
        near(got.utilityLcb, want.utilityLcb, TOL.utility, `${m} utilityLcb`);
        nearRelative(got.playSelectionValue, want.playSelectionValue, `${m} playSelectionValue`);
        near(got.prior, want.prior, TOL.winrate, `${m} prior`);
        near(got.noResultValue, want.noResultValue, TOL.winrate, `${m} noResultValue`);
        nearRelative(got.weight, want.weight, `${m} weight`);
        nearRelative(got.edgeWeight, want.edgeWeight, `${m} edgeWeight`);
        nearOwnership(got.ownership, want.ownership, `${m} ownership`);
        nearOwnershipStdev(got.ownershipStdev, want.ownershipStdev, `${m} ownershipStdev`);
      }

      const root = r.rootInfo;
      assert.deepEqual(Object.keys(root).filter((k) => !ROOT_FIELDS.includes(k)), [], 'rootInfo fields the test knows');
      const v = search.rootValues();
      assert.equal(v.visits, root.visits, 'rootInfo visits');
      nearRelative(v.weight, root.weight, 'rootInfo weight');
      near(v.winrate, root.winrate, TOL.winrate, 'rootInfo winrate');
      near(v.scoreSelfplay, root.scoreSelfplay, TOL.scoreLead, 'rootInfo scoreSelfplay');
      near(v.scoreLead, root.scoreLead, TOL.scoreLead, 'rootInfo scoreLead');
      nearScore(v.scoreStdev, root.scoreStdev, 'rootInfo scoreStdev');
      near(v.utility, root.utility, TOL.utility, 'rootInfo utility');
      const raw = search.rootRawValues();
      near(raw.winrate, root.rawWinrate, TOL.winrate, 'rawWinrate');
      near(raw.lead, root.rawLead, TOL.scoreLead, 'rawLead');
      near(raw.scoreSelfplay, root.rawScoreSelfplay, TOL.scoreLead, 'rawScoreSelfplay');
      nearScore(raw.scoreSelfplayStdev, root.rawScoreSelfplayStdev, 'rawScoreSelfplayStdev');
      near(raw.noResultProb, root.rawNoResultProb, TOL.winrate, 'rawNoResultProb');
      near(raw.stWrError, root.rawStWrError, TOL.winrate, 'rawStWrError');
      nearScore(raw.stScoreError, root.rawStScoreError, 'rawStScoreError');
      nearScore(raw.varTimeLeft, root.rawVarTimeLeft, 'rawVarTimeLeft');
      assert.equal(hist.nextPla === 1 ? 'B' : 'W', root.currentPlayer, 'currentPlayer');
      await loadPositionHashes();
      assert.deepEqual(search.rootHashes(), { thisHash: root.thisHash, symHash: root.symHash }, 'thisHash and symHash');

      const policy = search.rootPolicy();
      assert.equal(policy.length, r.policy.length, 'policy length');
      r.policy.forEach((p, i) => (p === -1 ? assert.equal(policy[i], -1, `policy[${i}] illegal`) : near(policy[i], p, TOL.winrate, `policy[${i}]`)));
      const own = search.rootOwnership(true);
      nearOwnership(own.ownership, r.ownership, 'ownership');
      nearOwnershipStdev(own.ownershipStdev, r.ownershipStdev, 'ownershipStdev');
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
