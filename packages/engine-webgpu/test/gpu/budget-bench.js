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
import { measureMatrixPeak } from './gemm-bench.js';

const NOOP = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) { v[i] = v[i] + 1.0; }
`;

const COPY = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.y * nw.x * 256u + g.x;
  if (i < arrayLength(&dst)) { dst[i] = src[i]; }
}
`;

/** Vector FMA throughput: 8 independent vec4 accumulators per thread, `iter` fma steps each. */
const fmaKernel = (t, iter) => `${t === 'f16' ? 'enable f16;\n' : ''}
@group(0) @binding(0) var<storage, read_write> out: array<vec4<${t}>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let s = ${t}(g.x % 13u) * ${t}(0.001);
  var a0 = vec4<${t}>(s);
  var a1 = a0 + ${t}(0.01);
  var a2 = a0 + ${t}(0.02);
  var a3 = a0 + ${t}(0.03);
  var a4 = a0 + ${t}(0.04);
  var a5 = a0 + ${t}(0.05);
  var a6 = a0 + ${t}(0.06);
  var a7 = a0 + ${t}(0.07);
  let m = vec4<${t}>(${t}(0.999));
  let c = vec4<${t}>(${t}(0.0001));
  for (var i = 0u; i < ${iter}u; i++) {
    a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
    a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
  }
  out[g.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
`;

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
  const pipelineOf = (code) => device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  /** ms per pass: the median of `repeats` timings, with the least and greatest. */
  const time = async (record) => {
    const xs = [];
    for (let r = 0; r < repeats; r++) xs.push((await timePasses(device, record, { runs, warmUpMs })).msPerPass);
    xs.sort((a, b) => a - b);
    return { msPerPass: xs[xs.length >> 1], min: xs[0], max: xs[xs.length - 1] };
  };
  const r4 = (x) => +x.toFixed(4);
  const spread = (t) => ({ ms: r4(t.msPerPass), minMs: r4(t.min), maxMs: r4(t.max) });

  // Warm the GPU for 2 s with the f32 FMA kernel, so the first measurements don't run while its clocks rise.
  {
    step('warming the GPU for 2 s');
    const pipeline = await pipelineOf(fmaKernel('f32', 256));
    const out = device.createBuffer({ size: 6144 * 256 * 16, usage: GPUBufferUsage.STORAGE });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
    const t0 = performance.now();
    while (performance.now() - t0 < 2000) {
      const enc = device.createCommandEncoder();
      for (let i = 0; i < 20; i++) {
        const p = enc.beginComputePass();
        p.setPipeline(pipeline);
        p.setBindGroup(0, bind);
        p.dispatchWorkgroups(6144);
        p.end();
      }
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    out.destroy();
  }

  // Ceilings.
  report.ceilings = { at: new Date().toISOString() };
  step('copy bandwidth');
  {
    const pipeline = await pipelineOf(COPY);
    report.ceilings.copy = [];
    for (const mib of [64, 256]) {
      const size = mib * 1024 * 1024;
      const src = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      const dst = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: dst } }] });
      const groups = size / 16 / 256;
      const x = Math.min(groups, 32768);
      const y = Math.ceil(groups / x);
      const t = await time((p) => { p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(x, y); });
      report.ceilings.copy.push({ mib, bytesMoved: 2 * size, ...spread(t), gbPerSecond: +((2 * size) / (t.msPerPass / 1000) / 1e9).toFixed(1) });
      src.destroy();
      dst.destroy();
    }
  }
  for (const t of hasF16 ? ['f32', 'f16'] : ['f32']) {
    step(`vector FMA throughput, ${t}`);
    const ITER = 1024;
    const WGS = 6144;
    const pipeline = await pipelineOf(fmaKernel(t, ITER));
    const out = device.createBuffer({ size: WGS * 256 * 16, usage: GPUBufferUsage.STORAGE });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
    const timed = await time((p) => { p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(WGS); });
    const flops = WGS * 256 * ITER * 8 * 4 * 2;
    report.ceilings[`fma${t}`] = { workgroups: WGS, iterations: ITER, ...spread(timed), tflops: +(flops / (timed.msPerPass / 1000) / 1e12).toFixed(2) };
    out.destroy();
  }
  report.subgroupMatrixConfigs = engine.subgroupMatrixConfigs;
  if (hasF16 && engine.subgroupMatrixConfigs.some((c) => c.componentType === 'f16' && c.resultComponentType === 'f32' && c.M === 16 && c.N === 16 && c.K === 16)) {
    step('matrix throughput');
    report.ceilings.matrix = await measureMatrixPeak(device, { runs });
  }
  step('dispatch fixed cost');
  {
    const COUNT = 500;
    const pipeline = await pipelineOf(NOOP);
    const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const t = await time((p) => { for (let i = 0; i < COUNT; i++) { p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(1); } });
    report.ceilings.dispatch = { count: COUNT, msPerDispatch: +(t.msPerPass / COUNT).toFixed(5), minMsPerDispatch: +(t.min / COUNT).toFixed(5), maxMsPerDispatch: +(t.max / COUNT).toFixed(5) };
    buf.destroy();
  }

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
