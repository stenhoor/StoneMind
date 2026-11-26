// The budget benchmark: what each kind of work in the network costs on this device, against what the device can do.
// - Device ceilings, measured here: copy bandwidth (vec4 f32 copy, bytes read plus bytes written), vector FMA
//   throughput in f32 and (with shader-f16) f16, matrix throughput (with chromium-experimental-subgroup-matrix), and the
//   fixed cost of a near-empty dependent dispatch.
// - The network, at each batch: its whole pass, and for each kind of dispatch the pass without that kind ("without")
//   and with only that kind ("only"), each timed back to back after a warm-up (timing.js). "Full minus without" is the
//   kind's share of the real pass; with kinds left out the outputs are meaningless, but the work and its timing are the
//   same. Each kind's least work (flops and bytes, GpuNetwork.opWork) is recorded alongside.
// The network is built for maxBatch 64, as the search benchmark builds it.
// The GPU is warmed for 2 s first. Every timing is repeated `repeats` times: the median, with the least and greatest.
// For each kind, the whole pass, the pass without it and the pass with only it are measured next to each other, so the
// kind's share comes from adjacent measurements. Under this sustained load the owner's laptop GPU runs at its power
// cap (docs/spikes/results/2026-11-b18-budget.md), as it does through a review. Each row records when it was measured
// (ISO time), to line it up with an nvidia-smi clock log.
import { requestEngineDevice, loadModel, GpuNetwork, encodePositionV7, gtpMove } from '../../dist/index.js';
import { timePasses } from './timing.js';
import { warmGpu, measureCeilings, hasMatrix16 } from './ceilings.js';

export async function runBudgetBench({ modelBytes, variant = null, batches = [1, 4, 8, 16, 32, 64], maxBatch = 64, runs = 7, warmUpMs = 300, repeats = 3, step = () => {} } = {}) {
  const engine = await requestEngineDevice();
  const { device } = engine;
  if (!engine.features.has('timestamp-query')) throw new Error('the budget benchmark needs timestamp-query');
  const hasF16 = engine.features.has('shader-f16');
  const net0 = variant ?? { winograd: true, fuse: true, precision: hasF16 ? 'f16' : 'f32' };
  const report = {
    kind: 'budget', adapter: engine.adapterInfo, features: [...engine.features], userAgent: navigator.userAgent,
    date: new Date().toISOString(), variant: net0, batches, maxBatch, runs, warmUpMs, repeats,
    timing: 'gpu timestamps, passes submitted back to back after a warm-up (timing.js)',
  };
  let lost = null;
  device.lost.then((info) => { lost = { reason: info.reason, message: info.message }; });
  /** ms per pass: the median of `repeats` timings, with the least and greatest. */
  const time = async (record) => {
    const xs = [];
    for (let r = 0; r < repeats; r++) xs.push((await timePasses(device, record, { runs, warmUpMs })).msPerPass);
    xs.sort((a, b) => a - b);
    return { msPerPass: xs[xs.length >> 1], min: xs[0], max: xs[xs.length - 1] };
  };
  const r4 = (x) => +x.toFixed(4);
  const spread = (t) => ({ ms: r4(t.msPerPass), minMs: r4(t.min), maxMs: r4(t.max) });

  step('warming the GPU for 2 s');
  await warmGpu(device, 2000);
  step('ceilings');
  report.subgroupMatrixConfigs = engine.subgroupMatrixConfigs;
  report.ceilings = { at: new Date().toISOString(), ...(await measureCeilings(device, { hasF16, matrix: hasMatrix16(engine), time, spread })) };

  // The network.
  step('loading the model');
  const model = await loadModel(new Uint8Array(modelBytes));
  report.model = model.name;
  step(`building the network ${JSON.stringify(net0)} for maxBatch ${maxBatch}`);
  const net = await GpuNetwork.create(engine, model, { ...net0, maxBatch });
  report.gemm = net.gemm;
  report.gemmConfig = net.gemm === 'v3' ? net.gemmV3Config : net.gemmConfig;
  report.winoTransforms = net.winoTransforms;
  if (net.winoGemmMma) report.winoGemmMma = net.winoGemmMma;
  report.dispatches = net.dispatchCount;
  // Real inputs in the network's buffers, from one real evaluation.
  const input = encodePositionV7({ xSize: 19, ySize: 19, rules: 'tromp-taylor', komi: 7.5, moves: [gtpMove('B', 'Q16', 19), gtpMove('W', 'D4', 19)] }, model.preferPassAliveUnderSuicideRules).input;
  await net.evaluate(Array.from({ length: maxBatch }, () => input));
  const kinds = [...new Set(net.opWork(1).map((o) => o.kind))];
  report.kinds = kinds;
  report.network = [];
  for (const n of batches) {
    if (lost) throw new Error(`device lost: ${JSON.stringify(lost)}`);
    step(`network at batch ${n}`);
    const row = { batch: n, at: new Date().toISOString(), full: spread(await time((p) => net.recordPass(p, n))), kinds: [] };
    const work = net.opWork(n);
    for (const kind of kinds) {
      step(`network at batch ${n}: ${kind}`);
      const at = new Date().toISOString();
      const full = await time((p) => net.recordPass(p, n));
      const without = await time((p) => net.recordPass(p, n, new Set([kind])));
      const only = await time((p) => net.recordPass(p, n, new Set(kinds.filter((k) => k !== kind))));
      const mine = work.filter((w) => w.kind === kind);
      row.kinds.push({
        kind,
        at,
        dispatches: mine.length,
        flops: mine.reduce((a, w) => a + w.flops, 0),
        bytes: mine.reduce((a, w) => a + w.bytes, 0),
        full: spread(full),
        without: spread(without),
        only: spread(only),
        shareMs: r4(full.msPerPass - without.msPerPass),
      });
    }
    report.network.push(row);
  }
  net.destroy();
  report.ok = !lost;
  if (lost) report.deviceLost = lost;
  device.destroy();
  return report;
}
