// The Winograd transform benchmark: each version of the input and output transforms on its own, at b18's 192 channels
// on a 19 × 19 board, with GpuNetwork's parameters and dispatch. The input transform runs fused (batch norm and
// activation applied to each input value, as in every fused 3×3 convolution of b18), and the output transform adds the
// residual (as every second one does). Each timing is one compute pass of `repeat` identical dispatches after a
// warm-up, median of `runs`, per dispatch. Every version's output is compared with v1's.
import { requestEngineDevice, toFloat16Array, fromFloat16Bits } from '../../dist/index.js';
import { ACT, WINO_IN, WINO_IN_V2, WINO_OUT, WINO_OUT_V2, winoTransformChannels } from '../../dist/gpu/kernels.js';

const SIZE = 19;
const AREA = SIZE * SIZE;
const TILES_X = 5;
const TILES = 25;
const C = 192;

function values(n, seed, lo = -1, hi = 1) {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = lo + ((s >>> 8) % 129) / 128 * (hi - lo);
  }
  return out;
}

export async function runTransformBench({ batches = [16, 32, 64], repeat = 30, runs = 7, warmUpMs = 2000, step = () => {} } = {}) {
  const engine = await requestEngineDevice();
  const { device } = engine;
  const report = { kind: 'transforms', adapter: engine.adapterInfo, features: [...engine.features], userAgent: navigator.userAgent, date: new Date().toISOString(), batches, repeat, runs, channels: C };
  let lost = null;
  device.lost.then((info) => { lost = { reason: info.reason, message: info.message }; });
  const elem = engine.features.has('shader-f16') ? 'f16' : 'f32';
  const size = elem === 'f16' ? 2 : 4;
  // With f16 storage GpuNetwork applies scale8, so the fused activation is mish_scale8.
  const act = elem === 'f16' ? ACT.mish_scale8 : ACT.mish;
  report.elem = elem;
  report.act = act;
  if (!engine.features.has('timestamp-query')) throw new Error('the transform benchmark needs timestamp-query');
  const maxBatch = Math.max(...batches);
  const tileStride = maxBatch * TILES;
  const buffer = (bytes, usage) => device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: usage | GPUBufferUsage.COPY_DST });
  const upload = (data, asElem = true) => {
    const bytes = asElem && elem === 'f16' ? toFloat16Array(data) : data;
    const buf = buffer(bytes.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const padded = new Uint8Array(buf.size);
    padded.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    device.queue.writeBuffer(buf, 0, padded);
    return buf;
  };
  const uniform = (u32s) => {
    const buf = device.createBuffer({ size: Math.max(16, u32s.length * 4), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Uint32Array(u32s));
    return buf;
  };
  const readBack = async (buf, count) => {
    const bytes = count * size;
    const staging = device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, staging.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = staging.getMappedRange().slice(0, bytes);
    staging.unmap();
    staging.destroy();
    if (elem === 'f32') return new Float32Array(raw);
    const h = new Uint16Array(raw);
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) out[i] = fromFloat16Bits(h[i]);
    return out;
  };
  const query = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const tsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  // Shared inputs, sized for the largest batch.
  const board = upload(values(maxBatch * C * AREA, 1));
  const sb = upload(new Float32Array([...values(C, 2, 0.5, 1.5), ...values(C, 3, -0.5, 0.5)]), false);
  const pre = upload(values(maxBatch * C, 4, -0.2, 0.2), false);
  const mask = upload(new Float32Array(maxBatch * AREA).fill(1), false);
  const M = upload(values(36 * C * tileStride, 5));
  const nbias = upload(new Float32Array(4), false);

  const kernels = [
    { kind: 'in', name: 'v1', code: WINO_IN(elem, elem), groups: (b) => [Math.ceil((b * TILES) / 64), C] },
    { kind: 'in', name: 'v2', code: WINO_IN_V2(elem, elem, SIZE, SIZE), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] },
    { kind: 'out', name: 'v1', code: WINO_OUT(elem, elem), groups: (b) => [Math.ceil((b * TILES) / 64), C] },
    { kind: 'out', name: 'v2', code: WINO_OUT_V2(elem, elem, SIZE, SIZE), groups: (b) => [Math.ceil(C / winoTransformChannels('out', SIZE, SIZE)), b] },
  ];
  step(`compiling ${kernels.length} kernels`);
  for (const k of kernels) {
    k.pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: k.code }), entryPoint: 'main' } });
  }
  const inParams = uniform([C, SIZE, SIZE, TILES_X, TILES, tileStride, 1, act, AREA, 0, 0, 0]);
  const outParams = uniform([C, SIZE, SIZE, TILES_X, TILES, tileStride, 1, 0]);
  const setup = (k, dyn) => {
    const out = k.kind === 'in' ? buffer(36 * C * tileStride * size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : upload(values(maxBatch * C * AREA, 6));
    const resources = k.kind === 'in' ? [inParams, dyn, board, out, sb, pre, mask] : [outParams, dyn, M, out, nbias];
    const bind = device.createBindGroup({ layout: k.pipeline.getBindGroupLayout(0), entries: resources.map((b, binding) => ({ binding, resource: { buffer: b } })) });
    return { out, bind };
  };
  const pass = async (k, bind, groups, count, timed) => {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass(timed ? { timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    p.setPipeline(k.pipeline);
    p.setBindGroup(0, bind);
    for (let i = 0; i < count; i++) p.dispatchWorkgroups(...groups);
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

  step(`warming up for ${warmUpMs} ms`);
  {
    const dyn = uniform([maxBatch, 0, 0, 0]);
    const { bind } = setup(kernels[0], dyn);
    const t0 = performance.now();
    while (performance.now() - t0 < warmUpMs) await pass(kernels[0], bind, kernels[0].groups(maxBatch), 50, false);
  }
  report.warmUpMs = warmUpMs;

  report.rows = [];
  for (const b of batches) {
    const dyn = uniform([b, 0, 0, 0]);
    for (const kind of ['in', 'out']) {
      step(`batch ${b}, ${kind}put transform`);
      // Bytes each dispatch must move at least: the board (read or written, plus the residual read for 'out') and V or M.
      const bytes = kind === 'in' ? (b * C * AREA + 36 * C * b * TILES) * size : (2 * b * C * AREA + 36 * C * b * TILES) * size;
      const row = { batch: b, transform: kind, minBytes: bytes, kernels: [] };
      let first = null;
      for (const k of kernels.filter((x) => x.kind === kind)) {
        if (lost) throw new Error(`device lost: ${JSON.stringify(lost)}`);
        const { out, bind } = setup(k, dyn);
        const groups = k.groups(b);
        // For 'out' (which adds into its output), the checked output is after exactly one dispatch from the same start.
        await pass(k, bind, groups, 1, false);
        const count = kind === 'in' ? 36 * C * tileStride : maxBatch * C * AREA;
        const got = await readBack(out, count);
        const ms = [];
        for (let r = 0; r < runs; r++) ms.push((await pass(k, bind, groups, repeat, true)) / repeat);
        ms.sort((x, y) => x - y);
        const median = ms[Math.floor(ms.length / 2)];
        const entry = { kernel: k.name, msPerDispatch: +median.toFixed(4), gbPerSecond: +(bytes / (median / 1000) / 1e9).toFixed(1) };
        // Compare only what a batch of b writes: V[e][c][t < b·25], or the first b boards.
        let d = 0;
        if (first === null) first = got;
        else if (kind === 'in') {
          for (let e = 0; e < 36; e++)
            for (let c = 0; c < C; c++)
              for (let t = 0; t < b * TILES; t++) { const i = (e * C + c) * tileStride + t; d = Math.max(d, Math.abs(got[i] - first[i])); }
          entry.maxAbsDiffFromV1 = d;
        } else {
          for (let i = 0; i < b * C * AREA; i++) d = Math.max(d, Math.abs(got[i] - first[i]));
          entry.maxAbsDiffFromV1 = d;
        }
        row.kernels.push(entry);
        out.destroy();
      }
      report.rows.push(row);
    }
  }
  // The ceiling: a plain copy of vec4<u32>s (read and write the same number of bytes), at the input transform's size
  // (board + V) for each batch, timed the same way.
  {
    step('copy baseline');
    const copy = await device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: `@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = gid.x + gid.y * nw.x * 256u;
  if (i < arrayLength(&dst)) { dst[i] = src[i]; }
}`,
        }),
        entryPoint: 'main',
      },
    });
    report.copy = [];
    for (const b of batches) {
      const bytes = (b * C * AREA + 36 * C * b * TILES) * size; // the input transform's minimum
      const half = Math.ceil(bytes / 2 / 16) * 16;
      const src = buffer(half, GPUBufferUsage.STORAGE);
      const dst = buffer(half, GPUBufferUsage.STORAGE);
      const bind = device.createBindGroup({ layout: copy.getBindGroupLayout(0), entries: [src, dst].map((b2, binding) => ({ binding, resource: { buffer: b2 } })) });
      const n = half / 16;
      const groups = Math.ceil(n / 256);
      const k = { pipeline: copy };
      const ms = [];
      await pass(k, bind, [Math.min(groups, 65535), Math.ceil(groups / 65535)], 1, false);
      for (let r = 0; r < runs; r++) ms.push((await pass(k, bind, [Math.min(groups, 65535), Math.ceil(groups / 65535)], repeat, true)) / repeat);
      ms.sort((x, y) => x - y);
      const median = ms[Math.floor(ms.length / 2)];
      report.copy.push({ batch: b, bytes: 2 * half, msPerDispatch: +median.toFixed(4), gbPerSecond: +((2 * half) / (median / 1000) / 1e9).toFixed(1) });
      src.destroy();
      dst.destroy();
    }
  }
  // b18 has 78 Winograd convolutions, each one input and one output transform (most at 192 channels).
  report.perNetwork = batches.map((b) => {
    const rows = report.rows.filter((r) => r.batch === b);
    const entry = { batch: b };
    for (const r of rows) for (const k of r.kernels) entry[`${r.transform} ${k.kernel}`] = +(78 * k.msPerDispatch).toFixed(3);
    return entry;
  });
  report.ok = report.rows.every((r) => r.kernels.every((k) => (k.maxAbsDiffFromV1 ?? 0) === 0));
  if (lost) report.deviceLost = lost;
  device.destroy();
  return report;
}
