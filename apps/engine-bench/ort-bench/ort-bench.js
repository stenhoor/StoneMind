// The onnxruntime-web baseline (spike 0001, pass mark 1): KataGo's ONNX graph under onnxruntime-web's WebGPU and WASM
// execution providers, checked against native KataGo's goldens and benchmarked by batch size with the same method as
// the WebGPU engine's GPU check page (packages/engine-webgpu/test/gpu/check.js): the first 19x19 golden position
// repeated to the batch size, one warm-up, then the median of `runs` timed runs, each including upload and readback.
import { loadModel, encodePositionV7, gtpMove, postProcess } from '../gpu-check/dist/index.js';

const params = new URLSearchParams(location.search);
// The same tolerances as the engine's goldens test.
const TOL = { prob: 2e-4, lead: 2e-3, policy: 2e-4, ownership: 2e-4 };
const DEFAULT_BATCHES = { webgpu: [1, 2, 4, 8, 16, 32, 64], wasm: [1, 2, 4, 8, 16] };
const steps = [];
const stepsEl = document.getElementById('steps');
function step(message) {
  steps.push(`${new Date().toISOString().slice(11, 23)} ${message}`);
  stepsEl.textContent = steps.join('\n');
}
const getJson = async (url) => (await fetch(url)).json();

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

/** Feed a batch of encoded inputs to the session; returns the outputs per position in the engine's RawOutputs shape. */
async function run(ort, session, model, inputs) {
  const n = inputs.length;
  const { xSize, ySize } = inputs[0];
  const area = xSize * ySize;
  const cIn = model.numInputChannels;
  const cGlobal = model.numInputGlobalChannels;
  const spatial = new Float32Array(n * cIn * area);
  const global = new Float32Array(n * cGlobal);
  const mask = new Float32Array(n * area);
  inputs.forEach((input, i) => {
    spatial.set(input.spatial, i * cIn * area);
    global.set(input.global, i * cGlobal);
    mask.set(input.spatial.subarray(0, area), i * area);
  });
  const feeds = {
    InputSpatial: new ort.Tensor('float32', spatial, [n, cIn, ySize, xSize]),
    InputGlobal: new ort.Tensor('float32', global, [n, cGlobal, 1, 1]),
    InputMask: new ort.Tensor('float32', mask, [n, 1, ySize, xSize]),
  };
  const out = await session.run(feeds);
  const policy = out.OutputPolicy.data;
  const pass = out.OutputPolicyPass.data;
  const value = out.OutputValue.data;
  const score = out.OutputScoreValue.data;
  const own = out.OutputOwnership.data;
  const cPolicy = out.OutputPolicy.dims[1];
  const cScore = out.OutputScoreValue.dims[1];
  return inputs.map((_, i) => ({
    xSize,
    ySize,
    policy: Array.from({ length: cPolicy }, (_, c) => policy.slice((i * cPolicy + c) * area, (i * cPolicy + c + 1) * area)),
    pass: pass.slice(i * cPolicy, (i + 1) * cPolicy),
    value: value.slice(i * 3, i * 3 + 3),
    scoreValue: score.slice(i * cScore, (i + 1) * cScore),
    ownership: own.slice(i * area, (i + 1) * area),
  }));
}

/** Run the baseline for each execution provider. */
export async function runOrtBench({ onnx, modelBytes, eps = (params.get('eps') ?? 'webgpu,wasm').split(','), runs = 5 } = {}) {
  steps.length = 0;
  const report = { kind: 'ort-baseline', userAgent: navigator.userAgent, date: new Date().toISOString(), crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency };
  step('loading the KataGo model file');
  const model = await loadModel(new Uint8Array(modelBytes));
  report.model = model.name;
  report.onnxBytes = onnx.byteLength;
  const positions = (await getJson('../gpu-check/test/goldens/positions.json')).positions.filter((p) => p.boardSize === 19);
  const golden = await fetch(`../gpu-check/test/goldens/${model.name}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  report.goldensCompared = Boolean(golden);
  const cases = positions.map((position) => {
    const moves = position.moves.map(([c, l]) => gtpMove(c, l, 19));
    return { position, encoded: encodePositionV7({ xSize: 19, ySize: 19, rules: position.rules, komi: position.komi, moves }, model.preferPassAliveUnderSuicideRules) };
  });
  report.providers = [];
  for (const ep of eps) {
    const r = { executionProvider: ep };
    report.providers.push(r);
    try {
      step(`${ep}: loading onnxruntime-web`);
      // The WebGPU build carries both providers, but each provider is measured with its own build, as a page using only
      // that provider would load it.
      const ort = await import(ep === 'webgpu' ? './vendor/ort.webgpu.min.mjs' : './vendor/ort.wasm.min.mjs');
      ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
      r.ortVersion = ort.env.versions?.web ?? null;
      step(`${ep}: creating the session`);
      const t0 = performance.now();
      const session = await ort.InferenceSession.create(new Uint8Array(onnx), { executionProviders: [ep], graphOptimizationLevel: 'all' });
      r.sessionCreateMs = +(performance.now() - t0).toFixed(1);
      r.wasmNumThreads = ort.env.wasm.numThreads ?? null;

      step(`${ep}: checking ${cases.length} positions against the goldens`);
      const raws = await run(ort, session, model, cases.map((c) => c.encoded.input));
      r.positions = [];
      let ok = Boolean(golden);
      cases.forEach(({ position, encoded }, i) => {
        const got = postProcess(model, raws[i], { nextPlayer: encoded.nextPlayer, legal: encoded.legal, koRuleSimple: encoded.koRuleSimple, scoringTerritory: encoded.scoringTerritory });
        const want = golden?.positions.find((p) => p.id === position.id)?.evaluations.find((e) => e.symmetry === 0);
        const row = { id: position.id };
        if (want) {
          const fails = compareGolden(got, want, 19);
          row.matchesKataGo = fails.length === 0;
          if (fails.length) row.failures = fails.slice(0, 5);
          if (fails.length) ok = false;
        } else row.matchesKataGo = null;
        r.positions.push(row);
      });
      r.ok = ok;

      const batches = params.get(`${ep}Batches`)?.split(',').map(Number) ?? DEFAULT_BATCHES[ep] ?? DEFAULT_BATCHES.wasm;
      const input = cases[0].encoded.input;
      r.benchPosition = cases[0].position.id;
      r.bench = [];
      for (const n of batches) {
        step(`${ep}: batch ${n}`);
        const batch = Array.from({ length: n }, () => input);
        await run(ort, session, model, batch);
        const times = [];
        for (let k = 0; k < runs; k++) {
          const s = performance.now();
          await run(ort, session, model, batch);
          times.push(performance.now() - s);
        }
        times.sort((a, b) => a - b);
        const median = times[times.length >> 1];
        r.bench.push({ batch: n, medianMs: +median.toFixed(2), evalsPerSecond: +((n * 1000) / median).toFixed(1) });
        step(`  ${r.bench.at(-1).evalsPerSecond} evals/s`);
      }
      await session.release();
    } catch (e) {
      r.ok = false;
      r.error = String(e?.stack ?? e);
      r.failedAt = steps.at(-1);
    }
  }
  report.ok = report.providers.every((p) => p.ok);
  report.steps = [...steps];
  return report;
}

function summarise(r) {
  const lines = [`ok: ${r.ok} · model ${r.model} · crossOriginIsolated ${r.crossOriginIsolated}`];
  for (const p of r.providers) {
    const matched = p.positions?.filter((x) => x.matchesKataGo).length ?? 0;
    lines.push(`${p.executionProvider}: ok ${p.ok}, ${matched}/${p.positions?.length ?? 0} positions match KataGo${p.wasmNumThreads != null ? `, wasm threads ${p.wasmNumThreads}` : ''}`);
    for (const b of p.bench ?? []) lines.push(`  batch ${b.batch}: ${b.evalsPerSecond} evals/s`);
    if (p.error) lines.push(`  FAILED at "${p.failedAt}": ${p.error.split('\n')[0]}`);
  }
  return lines.join('\n');
}

const summary = document.getElementById('summary');
const out = document.getElementById('out');
const download = document.getElementById('download');
let lastReport = null;
document.getElementById('run').addEventListener('click', async () => {
  const onnxFile = document.getElementById('onnx').files?.[0];
  const modelFile = document.getElementById('model').files?.[0];
  if (!onnxFile || !modelFile) {
    summary.textContent = 'Pick both files first.';
    return;
  }
  summary.textContent = 'Running… (the WASM provider is slow on b18; several minutes)';
  out.textContent = '';
  download.hidden = true;
  try {
    lastReport = await runOrtBench({ onnx: await onnxFile.arrayBuffer(), modelBytes: await modelFile.arrayBuffer() });
  } catch (e) {
    lastReport = { failed: true, error: String(e?.stack ?? e), steps: [...steps], date: new Date().toISOString(), model: 'failed' };
  }
  summary.textContent = lastReport.failed ? `Failed: ${lastReport.error.split('\n')[0]}` : summarise(lastReport);
  out.textContent = JSON.stringify(lastReport, null, 2);
  download.hidden = false;
  // Served by the bench's serve script, the report is also saved on disk; elsewhere this POST just fails.
  fetch('/report', { method: 'POST', body: JSON.stringify(lastReport) }).then(
    (res) => { if (res.ok) summary.textContent = `Report saved by the server.\n${summary.textContent}`; },
    () => {},
  );
});
download.addEventListener('click', () => {
  if (!lastReport) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(lastReport, null, 2) + '\n'], { type: 'application/json' }));
  a.download = `ort-bench-${lastReport.model}-${lastReport.date.replace(/[:.]/g, '-')}.json`;
  document.body.append(a);
  a.click();
  a.remove();
});
window.runOrtBench = runOrtBench;
