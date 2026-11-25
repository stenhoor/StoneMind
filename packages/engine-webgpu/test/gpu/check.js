// The GPU check page's logic. Query parameters: model (URL of the .bin.gz), golden (URL of its golden outputs).
import {
  loadModel, encodePositionV7, parseGtp, forward, postProcess, requestEngineDevice, GpuNetwork,
} from '../../dist/index.js';

// The same tolerances as test/goldens.test.mjs.
const TOL = { prob: 2e-4, lead: 2e-3, policy: 2e-4, ownership: 2e-4 };

const params = new URLSearchParams(location.search);
const modelUrl = params.get('model') ?? '../fixtures/g170-b6c96-s175395328-d26788732.bin.gz';
const goldenUrl = params.get('golden') ?? '../goldens/g170-b6c96-s175395328-d26788732.json';
// The plain-TypeScript reference is slow on big nets, so it runs by default only for the test network; ?reference=1
// forces it. Native KataGo's goldens are the correctness check either way.
const useReference = params.get('reference') !== null ? params.get('reference') === '1' : !params.has('model');

const getJson = async (url) => (await fetch(url)).json();

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

/** Compare a post-processed evaluation with KataGo's golden output for symmetry 0; returns the failures. */
function compareGolden(got, want, size) {
  const fails = [];
  const near = (a, b, tol, what) => { if (!(Math.abs(a - b) <= tol)) fails.push(`${what} ${a} vs KataGo ${b}`); };
  near(got.whiteWinProb, want.whiteWin, TOL.prob, 'whiteWin');
  near(got.whiteLossProb, want.whiteLoss, TOL.prob, 'whiteLoss');
  near(got.whiteLead, want.whiteLead, TOL.lead, 'whiteLead');
  near(got.whiteScoreMean, want.whiteScoreSelfplay, TOL.lead, 'whiteScoreSelfplay');
  near(got.policy[size * size], want.policyPass, TOL.policy, 'policyPass');
  for (let i = 0; i < size * size; i++) {
    const p = want.policy[i];
    if (p === null) { if (got.policy[i] !== -1) fails.push(`point ${i} should be illegal`); }
    else near(got.policy[i], p, TOL.policy, `policy[${i}]`);
    near(got.whiteOwnership[i], want.whiteOwnership[i], TOL.ownership, `ownership[${i}]`);
  }
  return fails;
}

export async function runCheck({ bench = false, batches = [1, 2, 4, 8, 16, 32, 64], runs = 5 } = {}) {
  const report = { modelUrl, referenceCompared: useReference, userAgent: navigator.userAgent, date: new Date().toISOString() };
  const engine = await requestEngineDevice();
  report.adapter = engine.adapterInfo;
  report.features = [...engine.features];
  const model = await loadModel(new Uint8Array(await (await fetch(modelUrl)).arrayBuffer()));
  report.model = model.name;

  const positions = (await getJson('../goldens/positions.json')).positions;
  const golden = await fetch(goldenUrl).then((r) => (r.ok ? r.json() : null)).catch(() => null);

  const cases = positions.map((position) => {
    const size = position.boardSize;
    const moves = position.moves.map(([color, loc]) => [color, ...parseGtp(loc, size)]);
    const encoded = encodePositionV7({ xSize: size, ySize: size, rules: position.rules, komi: position.komi, moves }, model.preferPassAliveUnderSuicideRules);
    return { position, size, encoded };
  });

  report.variants = [];
  for (const variant of [{ winograd: true, fuse: true }, { winograd: true, fuse: false }, { winograd: false }]) {
    report.variants.push(await checkVariant({ engine, model, cases, golden, variant, bench, batches, runs }));
  }
  report.goldensCompared = Boolean(golden);
  report.ok = Boolean(golden) && report.variants.every((v) => v.ok);
  return report;
}

/** Check one network variant (GpuNetwork options) on every case; optionally benchmark it. */
async function checkVariant({ engine, model, cases, golden, variant, bench, batches, runs }) {
  const report = { variant };
  const net = await GpuNetwork.create(engine, model, { ...variant, maxBatch: Math.max(cases.length, bench ? Math.max(...batches) : 1) });
  report.dispatches = net.dispatchCount;
  const batched = await net.evaluate(cases.map((c) => c.encoded.input));
  const single = await net.evaluate([cases[0].encoded.input]);

  report.positions = [];
  let ok = true;
  cases.forEach(({ position, size, encoded }, i) => {
    const gpu = batched[i];
    const row = { id: position.id };
    if (useReference) {
      cases[i].reference ??= forward(model, encoded.input);
      const ref = cases[i].reference;
      row.maxAbsDiffVsReference = {
        policy: Math.max(...gpu.policy.map((p, c) => maxAbsDiff(p, ref.policy[c]))),
        pass: maxAbsDiff(gpu.pass, ref.pass),
        value: maxAbsDiff(gpu.value, ref.value),
        scoreValue: maxAbsDiff(gpu.scoreValue, ref.scoreValue),
        ownership: maxAbsDiff(gpu.ownership, ref.ownership),
      };
    }
    const want = golden?.positions.find((p) => p.id === position.id)?.evaluations.find((e) => e.symmetry === 0);
    if (want) {
      const nextPlayer = encoded.nextPlayer;
      const got = postProcess(model, gpu, { nextPlayer, legal: encoded.legal, koRuleSimple: encoded.koRuleSimple, scoringTerritory: encoded.scoringTerritory });
      const fails = compareGolden(got, want, size);
      row.matchesKataGo = fails.length === 0;
      if (fails.length) { row.failures = fails.slice(0, 5); ok = false; }
    }
    report.positions.push(row);
  });
  report.batchOfOneMatchesBatched = maxAbsDiff(single[0].ownership, batched[0].ownership) === 0 && maxAbsDiff(single[0].policy[0], batched[0].policy[0]) === 0;
  report.ok = ok;

  if (bench) {
    report.bench = [];
    const input = cases.find((c) => c.size === 19).encoded.input;
    for (const n of batches) {
      const batch = Array.from({ length: n }, () => input);
      await net.evaluate(batch); // warm-up
      const times = [];
      for (let r = 0; r < runs; r++) {
        const t0 = performance.now();
        await net.evaluate(batch);
        times.push(performance.now() - t0);
      }
      times.sort((a, b) => a - b);
      const median = times[times.length >> 1];
      report.bench.push({ batch: n, medianMs: +median.toFixed(2), evalsPerSecond: +((n * 1000) / median).toFixed(1) });
    }
    if (net.canProfile) report.profile = await profileNetwork(net, input, batches.filter((n) => [1, 16, 64].includes(n)), runs);
    else report.profile = 'the device has no timestamp-query';
  }
  net.destroy();
  return report;
}

const median = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
const ms = (ns) => +(ns / 1e6).toFixed(3);
const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));

/**
 * GPU timestamp profile at a few batch sizes. 'pass': the normal plan's GPU time against its wall time (median of
 * `runs`). 'ops': each dispatch timed in its own pass (one run), summed by kernel. `timerTickNs` is the greatest common
 * divisor of every non-zero measured interval: a large value means the browser coarsens its timestamps.
 */
async function profileNetwork(net, input, batches, runs) {
  const rows = [];
  const intervals = [];
  for (const n of batches) {
    const batch = Array.from({ length: n }, () => input);
    await net.profile(batch, 'pass'); // warm-up
    const passes = [];
    for (let r = 0; r < runs; r++) passes.push(await net.profile(batch, 'pass'));
    const ops = await net.profile(batch, 'ops');
    const byKind = {};
    for (const op of ops.ops) {
      byKind[op.kind] ??= { count: 0, ms: 0 };
      byKind[op.kind].count++;
      byKind[op.kind].ms += op.ns / 1e6;
      if (op.ns > 0) intervals.push(op.ns);
    }
    for (const k of Object.values(byKind)) k.ms = +k.ms.toFixed(3);
    for (const p of passes) if (p.passNs > 0) intervals.push(p.passNs);
    rows.push({
      batch: n,
      wallMsMedian: +median(passes.map((p) => p.wallMs)).toFixed(3),
      gpuPassMsMedian: ms(median(passes.map((p) => p.passNs))),
      perDispatch: {
        dispatches: ops.ops.length,
        sumMs: ms(ops.ops.reduce((a, o) => a + o.ns, 0)),
        wallMs: +ops.wallMs.toFixed(3),
        byKind,
        slowest: ops.ops.map((o, i) => ({ index: i, kind: o.kind, ms: ms(o.ns) })).sort((a, b) => b.ms - a.ms).slice(0, 10),
      },
    });
  }
  return { timerTickNs: intervals.reduce((a, b) => gcd(a, b), 0), rows };
}

const out = document.getElementById('out');
const show = (r) => {
  out.textContent = JSON.stringify(r, null, 2);
  // Served by gpu-check.mjs --serve, the report is also saved on disk; elsewhere this POST just fails.
  fetch('/report', { method: 'POST', body: JSON.stringify(r) }).then(
    (res) => { if (res.ok) out.textContent = `Report saved by the server.\n\n${out.textContent}`; },
    () => {},
  );
};
const run = (bench) => () => { out.textContent = 'Running…'; runCheck({ bench }).then(show, (e) => { out.textContent = String(e?.stack ?? e); }); };
document.getElementById('check').addEventListener('click', run(false));
document.getElementById('bench').addEventListener('click', run(true));
window.runCheck = runCheck;
