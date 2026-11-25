// The GPU check page's logic. Query parameters: model (URL of the .bin.gz), golden (URL of its golden outputs).
import {
  loadModel, encodePositionV7, parseGtp, forward, postProcess, requestEngineDevice, GpuNetwork,
} from '../../dist/index.js';

// The same tolerances as test/goldens.test.mjs.
const TOL = { prob: 2e-4, lead: 2e-3, policy: 2e-4, ownership: 2e-4 };

const params = new URLSearchParams(location.search);
const modelUrl = params.get('model') ?? '../fixtures/g170-b6c96-s175395328-d26788732.bin.gz';
const goldenUrl = params.get('golden') ?? '../goldens/g170-b6c96-s175395328-d26788732.json';

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
  const report = { modelUrl };
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
  for (const winograd of [true, false]) {
    report.variants.push(await checkVariant({ engine, model, cases, golden, winograd, bench, batches, runs }));
  }
  report.goldensCompared = Boolean(golden);
  report.ok = Boolean(golden) && report.variants.every((v) => v.ok);
  return report;
}

/** Check one network variant (Winograd on or off) on every case; optionally benchmark it. */
async function checkVariant({ engine, model, cases, golden, winograd, bench, batches, runs }) {
  const report = { winograd };
  const net = await GpuNetwork.create(engine, model, { winograd, maxBatch: Math.max(cases.length, bench ? Math.max(...batches) : 1) });
  report.dispatches = net.dispatchCount;
  const batched = await net.evaluate(cases.map((c) => c.encoded.input));
  const single = await net.evaluate([cases[0].encoded.input]);

  report.positions = [];
  let ok = true;
  cases.forEach(({ position, size, encoded }, i) => {
    const gpu = batched[i];
    const ref = forward(model, encoded.input);
    const diff = {
      policy: Math.max(...gpu.policy.map((p, c) => maxAbsDiff(p, ref.policy[c]))),
      pass: maxAbsDiff(gpu.pass, ref.pass),
      value: maxAbsDiff(gpu.value, ref.value),
      scoreValue: maxAbsDiff(gpu.scoreValue, ref.scoreValue),
      ownership: maxAbsDiff(gpu.ownership, ref.ownership),
    };
    const row = { id: position.id, maxAbsDiffVsReference: diff };
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
  }
  net.destroy();
  return report;
}

const out = document.getElementById('out');
const show = (r) => { out.textContent = JSON.stringify(r, null, 2); };
const run = (bench) => () => { out.textContent = 'Running…'; runCheck({ bench }).then(show, (e) => { out.textContent = String(e?.stack ?? e); }); };
document.getElementById('check').addEventListener('click', run(false));
document.getElementById('bench').addEventListener('click', run(true));
window.runCheck = runCheck;
