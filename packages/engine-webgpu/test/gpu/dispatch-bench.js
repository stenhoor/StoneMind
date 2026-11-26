// The dispatch benchmark: how much a dispatch costs beyond its own work, on this device.
// - Fixed cost: a kernel that does almost nothing (one workgroup adds 1 to 64 values), dispatched `count` times in one
//   compute pass. "Dependent": every dispatch writes the same buffer, so each must wait for the one before it (the
//   browser puts a barrier between them). "Independent": each writes its own buffer, so none has to wait.
// - On a real kernel: GEMM v3 (the default shape) at b18's Winograd GEMM shape for a batch, `repeat` dispatches in one
//   pass, dependent (the same output) and independent (each its own output).
// - The network: the b18 plan's GPU time for one batch in its single pass, with its dispatch count.
// Each timing is the median of `runs` passes, from GPU timestamps, after a warm-up.
import { requestEngineDevice, loadModel, GpuNetwork, encodePositionV7, gtpMove, toFloat16Array } from '../../dist/index.js';
import { GEMM_V3, GEMM_V3_DEFAULT, gemmV3Tile } from '../../dist/gpu/kernels.js';
import { timePasses } from './timing.js';

const NOOP = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) { v[i] = v[i] + 1.0; }
`;

export async function runDispatchBench({ modelBytes, batches = [16, 32, 64], count = 500, repeat = 30, runs = 7, warmUpMs = 2000, step = () => {} } = {}) {
  const engine = await requestEngineDevice();
  const { device } = engine;
  const report = { kind: 'dispatch', adapter: engine.adapterInfo, features: [...engine.features], userAgent: navigator.userAgent, date: new Date().toISOString(), batches, count, repeat, runs };
  let lost = null;
  device.lost.then((info) => { lost = { reason: info.reason, message: info.message }; });
  if (!engine.features.has('timestamp-query')) throw new Error('the dispatch benchmark needs timestamp-query');
  const elem = engine.features.has('shader-f16') ? 'f16' : 'f32';
  report.elem = elem;
  const query = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const tsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  /** One pass of the given (pipeline, bind group, workgroups) dispatches; GPU ms between its start and end. */
  const pass = async (dispatches, timed = true) => {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass(timed ? { timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    for (const [pipeline, bind, groups] of dispatches) {
      p.setPipeline(pipeline);
      p.setBindGroup(0, bind);
      p.dispatchWorkgroups(...groups);
    }
    p.end();
    if (timed) {
      enc.resolveQuerySet(query, 0, 2, resolve, 0);
      enc.copyBufferToBuffer(resolve, 0, tsRead, 0, 16);
    }
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (!timed) return 0;
    await tsRead.mapAsync(GPUMapMode.READ);
    const [begin, end] = new BigUint64Array(tsRead.getMappedRange().slice(0));
    tsRead.unmap();
    return Number(end - begin) / 1e6;
  };
  // GPU ms per pass of these dispatches, with the GPU kept busy (timing.js).
  const median = async (dispatches) =>
    (await timePasses(device, (p) => {
      for (const [pipeline, bind, groups] of dispatches) {
        p.setPipeline(pipeline);
        p.setBindGroup(0, bind);
        p.dispatchWorkgroups(...groups);
      }
    }, { runs })).msPerPass;
  report.timing = 'gpu timestamps, passes submitted back to back after a warm-up (timing.js); the network pass is awaited per call';

  // Fixed cost per dispatch.
  step(`fixed cost: ${count} near-empty dispatches`);
  const noop = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: NOOP }), entryPoint: 'main' } });
  const small = Array.from({ length: count }, () => device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE }));
  const binds = small.map((b) => device.createBindGroup({ layout: noop.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b } }] }));
  {
    const t0 = performance.now();
    step(`warming up for ${warmUpMs} ms`);
    while (performance.now() - t0 < warmUpMs) await pass(Array.from({ length: count }, (_, i) => [noop, binds[i], [1]]), false);
  }
  report.warmUpMs = warmUpMs;
  const dependent = await median(Array.from({ length: count }, () => [noop, binds[0], [1]]));
  const independent = await median(Array.from({ length: count }, (_, i) => [noop, binds[i], [1]]));
  report.fixed = {
    dependentMsPerDispatch: +(dependent / count).toFixed(5),
    independentMsPerDispatch: +(independent / count).toFixed(5),
  };

  // On a real kernel: GEMM v3 at the Winograd GEMM shape (3×3 192→192).
  const gemm = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: GEMM_V3(elem, elem, elem, GEMM_V3_DEFAULT) }), entryPoint: 'main' } });
  const tile = gemmV3Tile(GEMM_V3_DEFAULT);
  const maxBatch = Math.max(...batches);
  const tileStride = maxBatch * 25;
  const ic = 192;
  const oc = 192;
  const bytes = elem === 'f16' ? 2 : 4;
  const filled = (n) => {
    const data = new Float32Array(n).map((_, i) => ((i * 7919) % 129 - 64) / 64);
    const raw = elem === 'f16' ? toFloat16Array(data) : data;
    const buf = device.createBuffer({ size: Math.ceil(raw.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const padded = new Uint8Array(buf.size);
    padded.set(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
    device.queue.writeBuffer(buf, 0, padded);
    return buf;
  };
  const uniform = (u32s) => {
    const buf = device.createBuffer({ size: Math.max(16, u32s.length * 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Uint32Array(u32s));
    return buf;
  };
  const A = filled(36 * ic * oc);
  const B = filled(36 * ic * tileStride);
  const nbias = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  const params = uniform([oc, ic, 0, 25, ic * oc, oc, ic * tileStride, tileStride, oc * tileStride, tileStride, 0, 0]);
  const outputs = Array.from({ length: repeat }, () => device.createBuffer({ size: Math.ceil((36 * oc * tileStride * bytes) / 4) * 4, usage: GPUBufferUsage.STORAGE }));
  report.gemm = [];
  for (const b of batches) {
    if (lost) throw new Error(`device lost: ${JSON.stringify(lost)}`);
    step(`GEMM v3 at batch ${b}: ${repeat} dependent and ${repeat} independent dispatches`);
    const dyn = uniform([b, 0, 0, 0]);
    const gb = outputs.map((C) => device.createBindGroup({
      layout: gemm.getBindGroupLayout(0),
      entries: [params, dyn, A, B, C, nbias].map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    const groups = [Math.ceil((b * 25) / tile.tn), Math.ceil(oc / tile.tm), 36];
    const dep = await median(Array.from({ length: repeat }, () => [gemm, gb[0], groups]));
    const ind = await median(Array.from({ length: repeat }, (_, i) => [gemm, gb[i], groups]));
    report.gemm.push({ batch: b, dependentMsPerDispatch: +(dep / repeat).toFixed(4), independentMsPerDispatch: +(ind / repeat).toFixed(4) });
  }

  // The network: the b18 plan's single pass.
  if (modelBytes) {
    step('loading the model');
    const model = await loadModel(new Uint8Array(modelBytes));
    report.model = model.name;
    const input = encodePositionV7({ xSize: 19, ySize: 19, rules: 'tromp-taylor', komi: 7.5, moves: [gtpMove('B', 'Q16', 19)] }, model.preferPassAliveUnderSuicideRules).input;
    report.network = [];
    // Each batch on a plan built for exactly that batch, and on one built for the largest (as the search pages do).
    for (const [b, planBatch] of batches.flatMap((b) => (b === maxBatch ? [[b, b]] : [[b, b], [b, maxBatch]]))) {
      if (lost) throw new Error(`device lost: ${JSON.stringify(lost)}`);
      step(`network pass at batch ${b}, plan for ${planBatch}`);
      const net = await GpuNetwork.create(engine, model, { maxBatch: planBatch, precision: elem });
      const inputs = Array.from({ length: b }, () => input);
      await net.profile(inputs, 'pass');
      const ms = [];
      for (let r = 0; r < runs; r++) {
        const p = await net.profile(inputs, 'pass');
        ms.push(p.passNs / 1e6);
      }
      ms.sort((x, y) => x - y);
      const gpuPassMs = ms[Math.floor(ms.length / 2)];
      report.network.push({ batch: b, planMaxBatch: planBatch, dispatches: net.dispatchCount, gemm: net.gemm, gpuPassMs: +gpuPassMs.toFixed(3) });
      net.destroy();
    }
  }
  report.ok = !lost;
  if (lost) report.deviceLost = lost;
  device.destroy();
  return report;
}
