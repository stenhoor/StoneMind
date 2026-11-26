// The GPU check page's logic. The model comes from the file picker, or the `model` query parameter (a URL of the
// .bin.gz), or else the test network. Goldens: the `golden` parameter, or else ../goldens/<model name>.json.
import {
  loadModel, encodePositionV7, gtpMove, forward, postProcess, requestEngineDevice, GpuNetwork, ACT, ACTIVATION_PROBE,
  buildHistory, Search, PASS_LOC, reviewGame,
} from '../../dist/index.js';
import { runGemmBench } from './gemm-bench.js';

// The same tolerances as test/goldens.test.mjs.
const TOL = { prob: 2e-4, lead: 2e-3, policy: 2e-4, ownership: 2e-4 };

const params = new URLSearchParams(location.search);
const modelUrl = params.get('model') ?? '../fixtures/g170-b6c96-s175395328-d26788732.bin.gz';
// The plain-TypeScript reference is slow on big nets, so it runs by default only for the test network; ?reference=1
// forces it. Native KataGo's goldens are the correctness check either way.
const referenceDefault = (file) => (params.get('reference') !== null ? params.get('reference') === '1' : !params.has('model') && !file);

const getJson = async (url) => (await fetch(url)).json();

/** Steps reached so far in this run, shown on the page and kept in the report, so a failure shows where it stopped. */
const steps = [];
function step(message) {
  steps.push(`${new Date().toISOString().slice(11, 23)} ${message}`);
  const el = document.getElementById('steps');
  if (el) el.textContent = steps.join('\n');
}

/** Everything an error carries, as text: some errors (DOMExceptions, GPU errors) have an empty stack. */
function describeError(e) {
  if (e === null || typeof e !== 'object') return `thrown value: ${String(e)}`;
  const lines = [`${e.name ?? e.constructor?.name ?? 'Error'}: ${e.message ?? ''}`];
  if (e.reason) lines.push(`reason: ${e.reason}`);
  if (e.cause) lines.push(`cause: ${describeError(e.cause)}`);
  if (e.stack) lines.push(String(e.stack));
  return lines.join('\n');
}
let useReference = referenceDefault(null);

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

/**
 * Run the check. `file` is a model picked on the page (a File); without it the model is fetched from `modelUrl`.
 */
export async function runCheck({ bench = false, batches = [1, 2, 4, 8, 16, 32, 64], runs = 5, file = null, tune = false } = {}) {
  useReference = referenceDefault(file);
  const report = {
    modelUrl: file ? `file: ${file.name}` : modelUrl,
    referenceCompared: useReference,
    userAgent: navigator.userAgent,
    date: new Date().toISOString(),
  };
  steps.length = 0;
  step('requesting a WebGPU device');
  const engine = await requestEngineDevice();
  // A lost device makes every later GPU call fail; record why, as the browser reports it.
  let lost = null;
  engine.device.lost.then((info) => {
    lost = { reason: info.reason, message: info.message };
    step(`DEVICE LOST: reason ${info.reason}: ${info.message}`);
  });
  step(`device: ${JSON.stringify(engine.adapterInfo)}, features ${[...engine.features].join(', ') || 'none'}`);
  report.adapter = engine.adapterInfo;
  report.features = [...engine.features];
  report.adapterFeatures = engine.adapterFeatures;
  step(`adapter offers: ${engine.adapterFeatures.join(', ') || 'none'}`);
  const bytes = file ? await file.arrayBuffer() : await (await fetch(modelUrl)).arrayBuffer();
  step(`loading the model (${bytes.byteLength} bytes)`);
  const model = await loadModel(new Uint8Array(bytes));
  step(`model ${model.name}`);
  report.model = model.name;

  const positions = (await getJson('../goldens/positions.json')).positions;
  const goldenUrl = params.get('golden') ?? `../goldens/${model.name}.json`;
  report.goldenUrl = goldenUrl;
  const golden = await fetch(goldenUrl).then((r) => (r.ok ? r.json() : null)).catch(() => null);

  const cases = positions.map((position) => {
    const size = position.boardSize;
    const moves = position.moves.map(([color, loc]) => gtpMove(color, loc, size));
    const encoded = encodePositionV7({ xSize: size, ySize: size, rules: position.rules, komi: position.komi, moves }, model.preferPassAliveUnderSuicideRules);
    return { position, size, encoded };
  });

  step('checking activations');
  report.activations = await checkActivations(engine.device);
  step(`activations ok: ${report.activations.ok}`);
  report.variants = [];
  // The implicit GEMM runs last: it is the slowest variant, and losing the device there (as happened once in Safari on
  // an iPad, while profiling it) should not cost the other variants' results.
  // Every variant names its GEMM, so the report doesn't depend on the default (v2 since it was measured faster).
  const variants = [
    { winograd: true, fuse: true, gemm: 'v1' },
    { winograd: true, fuse: false, gemm: 'v1' },
    // GEMM v2: the unrolled matrix multiply for the Winograd GEMMs and 1×1 convolutions.
    { winograd: true, fuse: true, gemm: 'v2' },
    { winograd: true, fuse: false, gemm: 'v2' },
    // Winograd transforms v2: staged through workgroup memory.
    { winograd: true, fuse: true, gemm: 'v2', winoTransforms: 'v2' },
    { winograd: true, fuse: false, gemm: 'v2', winoTransforms: 'v2' },
    // GEMM v3: GEMM v2's tiles with vec4 workgroup reads and the next chunk prefetched.
    { winograd: true, fuse: true, gemm: 'v3' },
    // scale8 in fp32: the same function as the unscaled network, so it must pass the strict goldens check too.
    { winograd: true, fuse: true, scale8: true, gemm: 'v2' },
  ];
  if (engine.features.has('shader-f16')) {
    variants.push(
      { winograd: true, fuse: true, precision: 'f16', gemm: 'v1' },
      { winograd: true, fuse: false, precision: 'f16', gemm: 'v1' },
      { winograd: true, fuse: true, precision: 'f16', gemm: 'v2' },
      { winograd: true, fuse: false, precision: 'f16', gemm: 'v2' },
      { winograd: true, fuse: true, precision: 'f16', gemm: 'v2', winoTransforms: 'v2' },
      { winograd: true, fuse: false, precision: 'f16', gemm: 'v2', winoTransforms: 'v2' },
      { winograd: true, fuse: true, precision: 'f16', gemm: 'v3' },
      { winograd: true, fuse: false, precision: 'f16', gemm: 'v3' },
    );
  }
  // The implicit-GEMM convolution for every size: the first kernels, as a baseline.
  variants.push({ winograd: false, gemm: 'v1' });
  if (tune) {
    // Tuning: every GEMM v2 shape, in the storage precision this device runs best (f16 where it can), each checked for
    // correctness and benchmarked at batches 16 and 64. No profile, to keep the run short.
    variants.length = 0;
    const precision = engine.features.has('shader-f16') ? 'f16' : 'f32';
    for (const arith of precision === 'f16' ? ['f32', 'f16', 'f16x2'] : ['f32']) {
      for (const rm of [4, 8]) for (const rn of [4, 8]) for (const tk of [8, 16]) {
        variants.push({ winograd: true, fuse: true, gemm: 'v2', precision, gemmConfig: { rm, rn, tk, arith } });
      }
    }
    report.tuning = { precision, batches };
  }
  if (!engine.features.has('shader-f16')) report.f16 = 'not checked: the device has no shader-f16';
  // ?variants=<text>: run only the variants whose JSON contains the text (for example "v2"; default: all).
  const only = params.get('variants');
  for (const variant of variants) {
    if (only && !JSON.stringify(variant).includes(only)) continue;
    if (lost) {
      report.variants.push({ variant, ok: false, skipped: 'the device was lost earlier in this run' });
      continue;
    }
    step(`variant ${JSON.stringify(variant)}`);
    try {
      report.variants.push(await checkVariant({ engine, model, cases, golden, variant, bench, batches, runs, tune }));
    } catch (e) {
      // Keep going: record where and why this variant failed, so the other variants' results are not lost.
      const failedAt = steps.at(-1);
      currentNet?.destroy();
      currentNet = null;
      report.variants.push({ variant, ok: false, failedAt, error: describeError(e) });
      step(`  FAILED: ${String(e?.message ?? e)}`);
    }
  }
  if (lost) report.deviceLost = lost;
  report.goldensCompared = Boolean(golden);
  report.ok = Boolean(golden) && report.activations.ok && report.variants.every((v) => v.ok);
  return report;
}

/** Check one network variant (GpuNetwork options) on every case; optionally benchmark it. */
/**
 * The WGSL `activate` function against KataGo's formulas, computed in double precision: mish as in the Eigen
 * backend, mish_scale8 as in the OpenCL kernels (ACTIVATION == 12). Inputs cover -30..30, with a fine grid near 0
 * and around mish_scale8's switch at 2.5. Tolerance: 1e-5 · max(1, |y|), well above f32 rounding and well below what
 * the goldens test could notice.
 */
async function checkActivations(device) {
  const formulas = {
    identity: (x) => x,
    relu: (x) => Math.max(x, 0),
    mish: (x) => x * Math.tanh(Math.log1p(Math.exp(Math.min(x, 20))) + Math.max(x, 20) - 20),
    silu: (x) => x / (1 + Math.exp(-x)),
    mish_scale8: (x) => (x < 2.5 ? x * Math.tanh(Math.log1p(Math.exp(8 * x))) : x),
  };
  const xs = [];
  for (let x = -30; x <= 30; x += 0.25) xs.push(x);
  for (let x = -3; x <= 3; x += 1 / 128) xs.push(x);
  for (let x = 2.49; x <= 2.51; x += 0.0005) xs.push(x);
  const input = new Float32Array(xs);
  const module = device.createShaderModule({ code: ACTIVATION_PROBE });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }).catch(async (e) => {
    const info = await module.getCompilationInfo();
    const messages = info.messages.map((m) => `${m.type} at ${m.lineNum}:${m.linePos}: ${m.message}`).join('; ');
    throw new Error(`the activation probe shader failed to compile: ${messages || describeError(e)}`);
  });
  const STORAGE = 0x80, COPY_DST = 0x08, COPY_SRC = 0x04, UNIFORM = 0x40, MAP_READ = 0x01;
  const xsBuf = device.createBuffer({ size: input.byteLength, usage: STORAGE | COPY_DST });
  const ysBuf = device.createBuffer({ size: input.byteLength, usage: STORAGE | COPY_SRC });
  const readBuf = device.createBuffer({ size: input.byteLength, usage: MAP_READ | COPY_DST });
  const actBuf = device.createBuffer({ size: 16, usage: UNIFORM | COPY_DST });
  device.queue.writeBuffer(xsBuf, 0, input);
  const result = { ok: true, maxError: {} };
  for (const [name, f] of Object.entries(formulas)) {
    device.queue.writeBuffer(actBuf, 0, new Uint32Array([ACT[name], 0, 0, 0]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [actBuf, xsBuf, ysBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    pass.dispatchWorkgroups(Math.ceil(input.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(ysBuf, 0, readBuf, 0, input.byteLength);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(1);
    const ys = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    let worst = 0;
    let at = 0;
    for (let i = 0; i < input.length; i++) {
      const want = f(input[i]);
      const err = Math.abs(ys[i] - want) / Math.max(1, Math.abs(want));
      if (err > worst) { worst = err; at = i; }
    }
    result.maxError[name] = worst;
    result.worstAt ??= {};
    result.worstAt[name] = { x: input[at], gpu: ys[at], want: f(input[at]) };
    if (!(worst <= 1e-5)) result.ok = false;
  }
  for (const b of [xsBuf, ysBuf, readBuf, actBuf]) b.destroy();
  return result;
}

/**
 * KataGo's GPU error statistics (`GpuErrorStats` in cpp/tests/testnnevalcanary.cpp, KataGo 1.18) of our outputs
 * against the goldens, at symmetry 0, and KataGo's limits for an fp16 configuration against an fp32 reference
 * (`runBackendErrorTest`: checkStats99(2.0, 1.00, 2.50, 0.0020) and checkStatsMax(5.0, 3.00, 6.00, 0.0040)).
 * The goldens have no no-result probability (taken as 1 − win − loss), score stdev or short-term error outputs, so the
 * three metrics built on those last two are not checked.
 */
function katagoErrorStats(evaluated) {
  const stats = { winrateError: [], leadError: [], scoreMeanError: [], topPolicyDiff: [], policyKLDiv: [], ownershipError: [] };
  for (const { got, want, size } of evaluated) {
    const area = size * size;
    const baseNoResult = 1 - want.whiteWin - want.whiteLoss;
    stats.winrateError.push(
      Math.abs(0.5 * (want.whiteWin - want.whiteLoss) - 0.5 * (got.whiteWinProb - got.whiteLossProb)) + Math.abs(baseNoResult - got.whiteNoResultProb),
    );
    stats.leadError.push(Math.abs(want.whiteLead - got.whiteLead));
    stats.scoreMeanError.push(Math.abs(want.whiteScoreSelfplay - got.whiteScoreMean));
    const base = [...want.policy.slice(0, area).map((p) => (p === null ? -1 : p)), want.policyPass];
    const other = Array.from(got.policy);
    let top = 0;
    for (let i = 0; i < base.length; i++) if (base[i] > base[top]) top = i;
    stats.topPolicyDiff.push(Math.abs(base[top] - other[top]));
    let kl = 0;
    for (let i = 0; i < base.length; i++) if (base[i] > 1e-30) kl += base[i] * (Math.log(base[i]) - Math.log(other[i]));
    stats.policyKLDiv.push(kl);
    for (let i = 0; i < area; i++) stats.ownershipError.push(Math.abs(want.whiteOwnership[i] - got.whiteOwnership[i]));
  }
  const sorted = Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, [...v].sort((a, b) => a - b)]));
  const p99 = (v) => v[Math.floor(((v.length - 1) * 99) / 100)];
  const max = (v) => v[v.length - 1];
  const at = (pick) => Object.fromEntries(Object.entries(sorted).map(([k, v]) => [k, pick(v)]));
  const q99 = at(p99);
  const qMax = at(max);
  // KataGo's checks, as written: winrate, top-policy and ownership are compared in percent.
  const check = (q, wr, score, tpd, pkld, ownershipMult) => ({
    winrateError: 100 * q.winrateError / wr,
    leadError: q.leadError / score,
    scoreMeanError: q.scoreMeanError / score,
    topPolicyDiff: 100 * q.topPolicyDiff / tpd,
    policyKLDiv: q.policyKLDiv / pkld,
    ownershipError: 100 * q.ownershipError / (wr * ownershipMult),
  });
  const ratio99 = check(q99, 2.0, 1.0, 2.5, 0.002, 1.75);
  const ratioMax = check(qMax, 5.0, 3.0, 6.0, 0.004, 4.0);
  const worst = Math.max(...Object.values(ratio99), ...Object.values(ratioMax));
  return { percentile99: q99, max: qMax, fp16LimitRatio99: ratio99, fp16LimitRatioMax: ratioMax, worstFp16LimitRatio: worst, fp16LimitsPass: worst <= 1 };
}

/** The network of the variant being checked, so a failure part-way can still free it. */
let currentNet = null;

async function checkVariant({ engine, model, cases, golden, variant, bench, batches, runs, tune }) {
  const report = { variant };
  step('  building the network');
  const net = await GpuNetwork.create(engine, model, { ...variant, maxBatch: Math.max(cases.length, bench ? Math.max(...batches) : 1) });
  currentNet = net;
  // ?fail=<text>: make matching variants fail here, to test that the page records the failure and carries on.
  if (params.get('fail') && JSON.stringify(variant).includes(params.get('fail'))) throw new Error('forced failure (?fail=)');
  // ?lose=<text>: destroy the device in matching variants, to test the device-lost handling.
  if (params.get('lose') && JSON.stringify(variant).includes(params.get('lose'))) engine.device.destroy();
  step('  evaluating the positions');
  report.dispatches = net.dispatchCount;
  report.gemm = net.gemm;
  report.gemmConfig = net.gemm === 'v3' ? net.gemmV3Config : net.gemmConfig;
  const batched = await net.evaluate(cases.map((c) => c.encoded.input));
  const single = await net.evaluate([cases[0].encoded.input]);
  // Calls in flight together (GpuNetwork's maxInFlight): the positions split across three calls made at once, so with
  // two allowed in flight the third waits for a readback buffer. Each must give exactly the batched call's outputs.
  const third = Math.ceil(cases.length / 3);
  const parts = [0, 1, 2].map((k) => cases.slice(k * third, (k + 1) * third).map((c) => c.encoded.input)).filter((p) => p.length > 0);
  const concurrent = (await Promise.all(parts.map((p) => net.evaluate(p)))).flat();

  report.positions = [];
  let ok = true;
  const evaluated = [];
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
      evaluated.push({ got, want, size });
      const fails = compareGolden(got, want, size);
      row.matchesKataGo = fails.length === 0;
      if (fails.length) row.failures = fails.slice(0, 5);
      // fp16 is judged by KataGo's own fp16 limits (below), not by the strict fp32 tolerances.
      if (fails.length && variant.precision !== 'f16') ok = false;
    } else if (golden) {
      row.matchesKataGo = null;
      row.note = 'no native golden captured for this position yet';
    }
    report.positions.push(row);
  });
  report.batchOfOneMatchesBatched = maxAbsDiff(single[0].ownership, batched[0].ownership) === 0 && maxAbsDiff(single[0].policy[0], batched[0].policy[0]) === 0;
  report.maxInFlight = net.maxInFlight;
  report.concurrentCallsMatchBatched =
    concurrent.length === batched.length &&
    concurrent.every((c, i) => {
      const b = batched[i];
      return c.policy.every((p, ch) => maxAbsDiff(p, b.policy[ch]) === 0) && ['pass', 'value', 'scoreValue', 'ownership'].every((k) => maxAbsDiff(c[k], b[k]) === 0);
    });
  if (!report.concurrentCallsMatchBatched) ok = false;
  if (evaluated.length) {
    report.katagoErrorStats = katagoErrorStats(evaluated);
    if (variant.precision === 'f16' && !report.katagoErrorStats.fp16LimitsPass) ok = false;
  }
  report.ok = ok;

  if (bench) {
    step('  benchmarking');
    report.bench = [];
    const input = cases.find((c) => c.size === 19).encoded.input;
    for (const n of batches) {
      const batch = Array.from({ length: n }, () => input);
      await net.evaluate(batch); // warm-up
      const times = [];
      const parts = [];
      for (let r = 0; r < runs; r++) {
        const t0 = performance.now();
        await net.evaluate(batch);
        times.push(performance.now() - t0);
        parts.push(net.lastCallTiming);
      }
      times.sort((a, b) => a - b);
      const median = times[times.length >> 1];
      // The median of each part of the call across the runs (GpuNetwork.lastCallTiming).
      const callTiming = {};
      for (const key of ['packMs', 'writeMs', 'encodeMs', 'submitMs', 'waitMs', 'readMs', 'unpackMs', 'totalMs']) {
        const values = parts.map((p) => p[key]).sort((a, b) => a - b);
        callTiming[key] = +values[values.length >> 1].toFixed(2);
      }
      report.bench.push({ batch: n, medianMs: +median.toFixed(2), evalsPerSecond: +((n * 1000) / median).toFixed(1), callTiming });
    }
    if (!tune) step('  profiling');
    if (tune) report.profile = 'skipped while tuning';
    else if (net.canProfile) report.profile = await profileNetwork(net, input, batches.filter((n) => [1, 16, 64].includes(n)), runs);
    else report.profile = 'the device has no timestamp-query';
  }
  net.destroy();
  currentNet = null;
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
const summary = document.getElementById('summary');
const download = document.getElementById('download');
const picker = document.getElementById('model');
let lastReport = null;

/**
 * The search benchmark: visits per second on one position at several batch sizes ("live"), and a fixed game reviewed at
 * a fixed number of visits per turn ("review"), in several modes. The game's moves are in the report so native KataGo
 * can review the same game (scripts/native-search-bench.mjs).
 * Network: Winograd fused, GEMM v2, fp16 storage where the device has shader-f16 (the fastest measured on the owner's
 * NVIDIA GPU and iPad), or `?searchVariant=<GpuNetwork options JSON>`.
 */
export async function runSearchBench({
  file = null,
  liveVisits = 2000,
  liveBatches = [16, 32, 64],
  reviewMoves = 250,
  reviewVisits = 100,
  reviewModes = [{ batchSize: 64, concurrency: 1 }, { batchSize: 16, concurrency: 1 }, { batchSize: 16, concurrency: 2 }, { batchSize: 16, concurrency: 4 }],
} = {}) {
  const report = { kind: 'search', modelUrl: file ? `file: ${file.name}` : modelUrl, userAgent: navigator.userAgent, date: new Date().toISOString() };
  steps.length = 0;
  step('requesting a WebGPU device');
  const engine = await requestEngineDevice();
  engine.device.lost.then((info) => {
    report.deviceLost = { reason: info.reason, message: info.message };
    step(`DEVICE LOST: reason ${info.reason}: ${info.message}`);
  });
  report.adapter = engine.adapterInfo;
  report.features = [...engine.features];
  const bytes = file ? await file.arrayBuffer() : await (await fetch(modelUrl)).arrayBuffer();
  step(`loading the model (${bytes.byteLength} bytes)`);
  const model = await loadModel(new Uint8Array(bytes));
  report.model = model.name;
  const variant = params.get('searchVariant')
    ? JSON.parse(params.get('searchVariant'))
    : { winograd: true, fuse: true, precision: engine.features.has('shader-f16') ? 'f16' : 'f32' };
  report.variant = variant;
  step(`building the network ${JSON.stringify(variant)}`);
  const maxBatch = Math.max(...liveBatches, ...reviewModes.map((m) => m.batchSize * m.concurrency));
  const net = await GpuNetwork.create(engine, model, { ...variant, maxBatch });
  report.gemm = net.gemm;
  report.gemmConfig = net.gemm === 'v3' ? net.gemmV3Config : net.gemmConfig;
  report.maxInFlight = net.maxInFlight;
  const cols = 'ABCDEFGHJKLMNOPQRST';
  const gtpOf = (board, loc) => (loc === PASS_LOC ? 'pass' : cols[board.x(loc)] + (board.ySize - board.y(loc)));
  const statsOf = (search, ms) => {
    const s = search.stats;
    return {
      visits: s.visits,
      ms: +ms.toFixed(1),
      visitsPerSecond: +((s.visits * 1000) / ms).toFixed(1),
      nnEvals: s.nnEvals,
      cacheHits: s.cacheHits,
      terminalVisits: s.terminalVisits,
      batches: s.batches,
      meanBatch: s.batches ? +((s.nnEvals - 1) / s.batches).toFixed(1) : 0,
      collisions: s.collisions,
      nnMs: +s.nnMs.toFixed(1),
      cpuMs: +s.cpuMs.toFixed(1),
    };
  };
  try {
    // Warm-up: the first network calls include one-off driver work.
    const warm = buildHistory({ xSize: 19, ySize: 19, rules: 'tromp-taylor', komi: 7.5, moves: [] });
    await new Search(model, net, warm, { batchSize: 64 }).run({ maxVisits: 200 });

    // Live: one middle-game position (fight-19 from the goldens list), a fresh search per batch size.
    const positions = (await getJson('../goldens/positions.json')).positions;
    const pos = positions.find((p) => p.id === 'fight-19');
    report.live = { position: pos.id, rules: pos.rules, komi: pos.komi, maxVisits: liveVisits, runs: [] };
    for (const batchSize of liveBatches) {
      step(`live: ${liveVisits} visits at batch ${batchSize}`);
      const hist = buildHistory({ xSize: 19, ySize: 19, rules: pos.rules, komi: pos.komi, moves: pos.moves.map(([c, l]) => gtpMove(c, l, 19)) });
      const search = new Search(model, net, hist, { batchSize });
      const t0 = performance.now();
      await search.run({ maxVisits: liveVisits });
      const run = { batch: batchSize, ...statsOf(search, performance.now() - t0) };
      const top = search.analysis().moveInfos[0];
      run.bestMove = gtpOf(hist.board, top.loc);
      run.bestMoveVisits = top.visits;
      report.live.runs.push(run);
      step(`  ${run.visitsPerSecond} visits/s (best ${run.bestMove})`);
    }

    // Review: the fixed game of test/goldens/agreement-game.json (the first reviewMoves turns), reviewVisits visits a
    // turn, in each mode: turns one at a time at batch 64 (the first benchmark's setting), one at a time at batch 16,
    // and several at once at batch 16 (reviewGame's default is native KataGo's 2) with their leaves sent through a
    // shared evaluator, up to the network's maxInFlight calls at once. Every mode shares one evaluation cache across
    // turns, as native KataGo's analysis does.
    const agreementGame = await getJson('../goldens/agreement-game.json');
    const game = agreementGame.moves.slice(0, reviewMoves);
    const histories = [];
    for (let t = 0; t < game.length; t++) {
      histories.push(buildHistory({ xSize: 19, ySize: 19, rules: agreementGame.rules, komi: agreementGame.komi, moves: game.slice(0, t).map(([c, l]) => gtpMove(c, l, 19)) }));
    }
    report.review = { rules: agreementGame.rules, komi: agreementGame.komi, visitsPerMove: reviewVisits, moves: game.length, game, modes: [] };
    for (const mode of reviewModes) {
      step(`review: ${game.length} turns × ${reviewVisits} visits, batch ${mode.batchSize}, ${mode.concurrency} at once`);
      const t0 = performance.now();
      const results = await reviewGame(model, net, histories, { visits: reviewVisits, batchSize: mode.batchSize, concurrency: mode.concurrency });
      const totalMs = performance.now() - t0;
      const visits = results.reduce((sum, r) => sum + r.rootVisits, 0);
      const run = {
        ...mode,
        visits,
        totalMs: +totalMs.toFixed(1),
        visitsPerSecond: +((visits * 1000) / totalMs).toFixed(1),
        minutesFor250Moves: +((totalMs / game.length) * 250 / 60000).toFixed(2),
        bestMoves: results.map((r) => gtpOf(histories[0].board, r.moveInfos[0].loc)),
      };
      report.review.modes.push(run);
      step(`  ${(totalMs / 1000).toFixed(1)} s (${run.visitsPerSecond} visits/s)`);
    }
  } finally {
    net.destroy();
  }
  report.steps = [...steps];
  report.ok = !report.deviceLost;
  return report;
}

/**
 * Top-move agreement (phase 5's exit criterion): the search's recommended move (most visits) on every `every`th
 * position of test/goldens/agreement-game.json, at each visit count and batch size, with the default search settings. The
 * report feeds scripts/native-agreement.mjs, which runs native KataGo on the same positions and compares.
 */
export async function runAgreement({ file = null, visits = [100, 500], batches = [16, 64], every = 5 } = {}) {
  const report = { kind: 'agreement', modelUrl: file ? `file: ${file.name}` : modelUrl, userAgent: navigator.userAgent, date: new Date().toISOString() };
  steps.length = 0;
  step('requesting a WebGPU device');
  const engine = await requestEngineDevice();
  engine.device.lost.then((info) => {
    report.deviceLost = { reason: info.reason, message: info.message };
    step(`DEVICE LOST: reason ${info.reason}: ${info.message}`);
  });
  report.adapter = engine.adapterInfo;
  const bytes = file ? await file.arrayBuffer() : await (await fetch(modelUrl)).arrayBuffer();
  step(`loading the model (${bytes.byteLength} bytes)`);
  const model = await loadModel(new Uint8Array(bytes));
  report.model = model.name;
  const variant = params.get('searchVariant')
    ? JSON.parse(params.get('searchVariant'))
    : { winograd: true, fuse: true, precision: engine.features.has('shader-f16') ? 'f16' : 'f32' };
  report.variant = variant;
  const net = await GpuNetwork.create(engine, model, { ...variant, maxBatch: Math.max(...batches) });
  const game = await getJson('../goldens/agreement-game.json');
  const cols = 'ABCDEFGHJKLMNOPQRST';
  report.game = { rules: game.rules, komi: game.komi, moves: game.moves.length };
  report.every = every;
  report.batches = batches;
  report.positions = [];
  try {
    for (let turn = 0; turn < game.moves.length; turn += every) {
      const hist = buildHistory({ xSize: 19, ySize: 19, rules: game.rules, komi: game.komi, moves: game.moves.slice(0, turn).map(([c, l]) => gtpMove(c, l, 19)) });
      const name = (loc) => (loc === PASS_LOC ? 'pass' : cols[hist.board.x(loc)] + (19 - hist.board.y(loc)));
      const row = { turn, byVisits: {} };
      // Keyed "<visits>@<batch>": with few visits, a large batch spreads the first leaves across many moves before any
      // result comes back, so the batch size can change the recommended move.
      for (const v of visits) for (const batchSize of batches) {
        const search = new Search(model, net, hist, { batchSize });
        await search.run({ maxVisits: v });
        const infos = search.analysis().moveInfos;
        row.byVisits[`${v}@${batchSize}`] = { best: name(infos[0].loc), top: infos.slice(0, 3).map((m) => ({ move: name(m.loc), visits: m.visits, winrate: +m.winrate.toFixed(4) })) };
      }
      report.positions.push(row);
      if (report.positions.length % 10 === 0) step(`  ${report.positions.length} positions`);
    }
  } finally {
    net.destroy();
  }
  report.visits = visits;
  report.batches = batches;
  report.steps = [...steps];
  report.ok = !report.deviceLost;
  return report;
}

/** One line per variant: whether it passed, and its evaluations per second at the largest batch benchmarked. */
function summarise(r) {
  if (r.kind === 'gemm') {
    const lines = [`GEMM benchmark · ok: ${r.ok} · adapter ${JSON.stringify(r.adapter)} · ${r.elem} storage · ${r.timing}`];
    if (r.deviceLost) lines.push(`DEVICE LOST: ${JSON.stringify(r.deviceLost)}`);
    for (const p of r.perNetwork) lines.push(`batch ${p.batch}: ${p.ms.map((k) => `${k.kernel} ${JSON.stringify(k.config)} ${k.ms} ms`).join(' · ')}`);
    return lines.join('\n');
  }
  if (r.kind === 'agreement') {
    return `agreement run · ok: ${r.ok} · model ${r.model} · ${r.positions.length} positions at ${r.visits.join(', ')} visits, batches ${r.batches.join(', ')}. Feed this report to scripts/native-agreement.mjs.`;
  }
  if (r.kind === 'search') {
    const lines = [`search benchmark · ok: ${r.ok} · model ${r.model} · adapter ${JSON.stringify(r.adapter)} · network ${JSON.stringify(r.variant)}`];
    if (r.deviceLost) lines.push(`DEVICE LOST: ${JSON.stringify(r.deviceLost)}`);
    for (const run of r.live?.runs ?? []) {
      lines.push(`live (${r.live.position}), batch ${run.batch}: ${run.visitsPerSecond} visits/s (${run.visits} visits, mean batch ${run.meanBatch}, network ${run.nnMs} ms, other ${run.cpuMs} ms)`);
    }
    for (const m of r.review?.modes ?? []) {
      lines.push(`review: ${r.review.moves} turns × ${r.review.visitsPerMove} visits, batch ${m.batchSize}, ${m.concurrency} at once: ${(m.totalMs / 1000).toFixed(1)} s (${m.visitsPerSecond} visits/s; ${m.minutesFor250Moves} min for 250 moves)`);
    }
    return lines.join('\n');
  }
  const lines = [`ok: ${r.ok} · model ${r.model} · adapter ${JSON.stringify(r.adapter)} · features ${r.features.join(', ') || 'none'}`];
  if (r.adapterFeatures) lines.push(`adapter offers: ${r.adapterFeatures.join(', ') || 'none'}`);
  if (r.f16) lines.push(`fp16: ${r.f16}`);
  lines.push(`activations ok: ${r.activations?.ok}`);
  if (r.deviceLost) lines.push(`DEVICE LOST: ${JSON.stringify(r.deviceLost)}`);
  for (const v of r.variants ?? []) {
    const top = v.bench?.at(-1);
    const why = v.error ? ` (FAILED at "${v.failedAt?.trim()}": ${v.error.split('\n')[0]})` : v.skipped ? ` (skipped: ${v.skipped})` : '';
    lines.push(`${JSON.stringify(v.variant)}: ok ${v.ok}${top ? `, ${top.evalsPerSecond} evals/s at batch ${top.batch}` : ''}${why}`);
  }
  return lines.join('\n');
}

const show = (r) => {
  r.steps = [...steps];
  lastReport = r;
  summary.textContent = summarise(r);
  out.textContent = JSON.stringify(r, null, 2);
  download.hidden = false;
  // Served by gpu-check.mjs --serve, the report is also saved on disk; elsewhere this POST just fails.
  fetch('/report', { method: 'POST', body: JSON.stringify(r) }).then(
    (res) => { if (res.ok) summary.textContent = `Report saved by the server.\n${summary.textContent}`; },
    () => {},
  );
};
const run = (bench) => () => {
  summary.textContent = 'Running…';
  out.textContent = '';
  download.hidden = true;
  const file = picker.files?.[0] ?? null;
  runCheck({ bench, file }).then(show, (e) => {
    const error = describeError(e);
    summary.textContent = `Failed at: ${steps.at(-1) ?? 'the start'}\n${error}`;
    out.textContent = '';
    lastReport = { failed: true, error, steps: [...steps], userAgent: navigator.userAgent, date: new Date().toISOString(), model: 'failed' };
    download.hidden = false;
  });
};
document.getElementById('check').addEventListener('click', run(false));
document.getElementById('agreement').addEventListener('click', () => {
  summary.textContent = 'Running the agreement positions… (a few minutes)';
  out.textContent = '';
  download.hidden = true;
  const file = picker.files?.[0] ?? null;
  runAgreement({ file }).then(show, (e) => {
    const error = describeError(e);
    summary.textContent = `Failed at: ${steps.at(-1) ?? 'the start'}\n${error}`;
    lastReport = { failed: true, error, steps: [...steps], userAgent: navigator.userAgent, date: new Date().toISOString(), model: 'failed' };
    download.hidden = false;
  });
});
document.getElementById('search').addEventListener('click', () => {
  summary.textContent = 'Running the search benchmark… (several minutes)';
  out.textContent = '';
  download.hidden = true;
  const file = picker.files?.[0] ?? null;
  runSearchBench({ file }).then(show, (e) => {
    const error = describeError(e);
    summary.textContent = `Failed at: ${steps.at(-1) ?? 'the start'}\n${error}`;
    lastReport = { failed: true, error, steps: [...steps], userAgent: navigator.userAgent, date: new Date().toISOString(), model: 'failed' };
    download.hidden = false;
  });
});
document.getElementById('bench').addEventListener('click', run(true));
document.getElementById('tune').addEventListener('click', () => {
  summary.textContent = 'Running…';
  out.textContent = '';
  download.hidden = true;
  const file = picker.files?.[0] ?? null;
  runCheck({ bench: true, tune: true, batches: [16, 64], file }).then(show, (e) => {
    const error = describeError(e);
    summary.textContent = `Failed at: ${steps.at(-1) ?? 'the start'}\n${error}`;
    lastReport = { failed: true, error, steps: [...steps], userAgent: navigator.userAgent, date: new Date().toISOString(), model: 'failed' };
    download.hidden = false;
  });
});
download.addEventListener('click', () => {
  if (!lastReport) return;
  const blob = new Blob([JSON.stringify(lastReport, null, 2) + '\n'], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `gpu-check-${lastReport.model}-${lastReport.date.replace(/[:.]/g, '-')}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
});
document.getElementById('gemm').addEventListener('click', () => {
  summary.textContent = 'Running…';
  out.textContent = '';
  download.hidden = true;
  steps.length = 0;
  runGemmBench({ step }).then((r) => show({ ...r, model: 'gemm' }), (e) => {
    const error = describeError(e);
    summary.textContent = `Failed at: ${steps.at(-1) ?? 'the start'}\n${error}`;
    lastReport = { failed: true, error, steps: [...steps], userAgent: navigator.userAgent, date: new Date().toISOString(), model: 'failed' };
    download.hidden = false;
  });
});
window.runCheck = runCheck;
window.runSearchBench = runSearchBench;
window.runAgreement = runAgreement;
