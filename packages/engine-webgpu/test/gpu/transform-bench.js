// The Winograd transform benchmark: each version of the input and output transforms on its own, at b18's 192 channels
// on a 19 × 19 board, with GpuNetwork's parameters and dispatch. The input transform runs fused (batch norm and
// activation applied to each input value, as in every fused 3×3 convolution of b18), and the output transform adds the
// residual (as every second one does). Each timing is one compute pass of `repeat` identical dispatches after a
// warm-up, median of `runs`, per dispatch. Every version's output is compared with v1's.
//
// ladder (?transformLadder=1): each version also by stage (WinoTransformStage, measurement only: 'move' without the
// transform arithmetic, 'reads' only the global reads, 'writes' only the global writes), and every timing is compared
// with a stream kernel that reads and writes the same numbers of bytes, contiguously, at the same buffer sizes, timed
// right after it (docs/spikes/results/2026-11-b18-transform-ladder.md).
import { requestEngineDevice, toFloat16Array, fromFloat16Bits } from '../../dist/index.js';
import { ACT, WINO_IN, WINO_IN_V2, WINO_IN_V3, WINO_OUT, WINO_OUT_V2, WINO_OUT_V3, winoTransformChannels } from '../../dist/gpu/kernels.js';
import { timePasses } from './timing.js';

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

export async function runTransformBench({ batches = [16, 32, 64], repeat = 30, runs = 7, warmUpMs = 2000, ladder = false, repeats = 3, rounds = 7, step = () => {} } = {}) {
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
  report.timing = 'gpu timestamps, passes submitted back to back after a warm-up (timing.js)';
  report.ladder = ladder;
  if (ladder) Object.assign(report, { repeats, rounds });
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

  const stages = ladder ? [undefined, 'move', 'reads', 'writes'] : [undefined];
  const kernels = stages.flatMap((stage) => [
    { kind: 'in', name: 'v1', stage, code: WINO_IN(elem, elem, stage), groups: (b) => [Math.ceil((b * TILES) / 64), C] },
    { kind: 'in', name: 'v2', stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { stage }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] },
    ...(ladder ? [{ kind: 'in', name: 'v2 unroll', stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { stage, unroll: true }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] }] : []),
    // v3 (the network's default) and the variants measured for it.
    ...(ladder ? [{ kind: 'in', name: 'v3', stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { ...WINO_IN_V3, stage }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] }] : []),
    ...(ladder ? [{ kind: 'in', name: 'v3 cpw4', stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { stage, linear: true, cpw: 4 }), groups: (b) => [Math.ceil(C / 4), b] }] : []),
    ...(ladder ? [{ kind: 'in', name: 'v3 4x2', stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { stage, linear: true, cpw: 4, bpw: 2 }), groups: (b) => [Math.ceil(C / 4), Math.ceil(b / 2)] }] : []),
    ...(ladder ? [{ kind: 'in', name: 'v3 words', stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { stage, linear: true, maskBits: 'words' }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] }] : []),
    // Diagnostic: the same, with no fused batch norm and activation (flags 0), for the cost of the fusion.
    // (v2's loop, whose branches skip the fusion's reads and arithmetic when it is off; the unrolled load always does them.)
    ...(ladder && stage === 'reads' ? [{ kind: 'in', name: 'v2', unfused: true, stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { stage }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] }] : []),
    ...(ladder && stage === 'reads' ? [{ kind: 'in', name: 'v3', identity: true, stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { ...WINO_IN_V3, stage }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] }] : []),
    ...(ladder && stage === 'reads' ? [{ kind: 'in', name: 'v3', unfused: true, stage, code: WINO_IN_V2(elem, elem, SIZE, SIZE, { ...WINO_IN_V3, stage }), groups: (b) => [Math.ceil(C / winoTransformChannels('in', SIZE, SIZE)), b] }] : []),
    { kind: 'out', name: 'v1', stage, code: WINO_OUT(elem, elem, stage), groups: (b) => [Math.ceil((b * TILES) / 64), C] },
    { kind: 'out', name: 'v2', stage, code: WINO_OUT_V2(elem, elem, SIZE, SIZE, { stage }), groups: (b) => [Math.ceil(C / winoTransformChannels('out', SIZE, SIZE)), b] },
    ...(ladder ? [{ kind: 'out', name: 'v3 8x1', stage, code: WINO_OUT_V2(elem, elem, SIZE, SIZE, { stage, linear: true, cpw: 8 }), groups: (b) => [Math.ceil(C / 8), b] }] : []),
    ...(ladder ? [{ kind: 'out', name: 'v3', stage, code: WINO_OUT_V2(elem, elem, SIZE, SIZE, { ...WINO_OUT_V3, stage }), groups: (b) => [Math.ceil(C / WINO_OUT_V3.cpw), Math.ceil(b / WINO_OUT_V3.bpw)] }] : []),
  ]).sort((x, y) => (x.kind + x.name).localeCompare(y.kind + y.name));
  // Explicit layouts, so a stage kernel that leaves a binding unused still takes the same bind group.
  const entriesOf = (types) => types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } }));
  const layouts = {
    in: device.createBindGroupLayout({ entries: entriesOf(['uniform', 'uniform', 'read-only-storage', 'storage', 'read-only-storage', 'read-only-storage', 'read-only-storage']) }),
    out: device.createBindGroupLayout({ entries: entriesOf(['uniform', 'uniform', 'read-only-storage', 'storage', 'read-only-storage']) }),
  };
  step(`compiling ${kernels.length} kernels`);
  for (const k of kernels) {
    k.layout = layouts[k.kind];
    k.pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [k.layout] }), compute: { module: device.createShaderModule({ code: k.code }), entryPoint: 'main' } });
  }
  // The stream kernel for the ladder: thread i reads 16 bytes at i (i < nRead) and writes 16 bytes at i (i < nWrite).
  const STREAM = `@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.y * nw.x * 256u + g.x;
  var v = vec4<u32>(i);
  if (i < U.x) { v = src[i]; }
  if (i < U.y) { dst[i] = v; } else if (i < U.x && v.x == 0xdeadbeefu) { dst[0] = v; } // keeps the read live
}`;
  const streamPipeline = ladder ? await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: STREAM }), entryPoint: 'main' } }) : null;
  /** ms per dispatch of the stream kernel reading `read` bytes and writing `write` bytes: the median of `reps` timings. */
  const stream = async (read, write, reps = repeats) => {
    const nR = Math.ceil(read / 16);
    const nW = Math.ceil(write / 16);
    const src = buffer(Math.max(16, nR * 16), GPUBufferUsage.STORAGE);
    const dst = buffer(Math.max(16, nW * 16), GPUBufferUsage.STORAGE);
    const groups = Math.ceil(Math.max(nR, nW) / 256);
    const x = Math.min(groups, 32768);
    const u = uniform([nR, nW, x, 0]);
    const bind = device.createBindGroup({ layout: streamPipeline.getBindGroupLayout(0), entries: [u, src, dst].map((b2, binding) => ({ binding, resource: { buffer: b2 } })) });
    const ms = [];
    for (let r = 0; r < reps; r++) {
      const t = await timePasses(device, (p) => {
        p.setPipeline(streamPipeline);
        p.setBindGroup(0, bind);
        for (let i = 0; i < repeat; i++) p.dispatchWorkgroups(x, Math.ceil(groups / x));
      }, { runs });
      ms.push(t.msPerPass / repeat);
    }
    src.destroy();
    dst.destroy();
    u.destroy();
    return ms.sort((a, b2) => a - b2)[ms.length >> 1];
  };
  /** The global bytes a dispatch of each stage reads and writes (the small batch-norm, bias and mask reads left out). */
  const stageBytes = (kind, stage, b) => {
    const board = b * C * AREA * size;
    const vm = 36 * C * b * TILES * size;
    const one = C * b * TILES * size; // one value per (tile, channel)
    if (kind === 'in') return stage === 'reads' ? { read: board, write: one } : stage === 'writes' ? { read: 0, write: vm } : { read: board, write: vm };
    return stage === 'reads' ? { read: vm, write: one } : stage === 'writes' ? { read: board, write: board } : { read: vm + board, write: board };
  };
  const inParams = uniform([C, SIZE, SIZE, TILES_X, TILES, tileStride, 1, act, AREA, 0, 0, 0]);
  const inParamsUnfused = uniform([C, SIZE, SIZE, TILES_X, TILES, tileStride, 0, act, AREA, 0, 0, 0]);
  const inParamsIdentity = uniform([C, SIZE, SIZE, TILES_X, TILES, tileStride, 1, ACT.identity, AREA, 0, 0, 0]);
  const outParams = uniform([C, SIZE, SIZE, TILES_X, TILES, tileStride, 1, 0]);
  const setup = (k, dyn) => {
    const out = k.kind === 'in' ? buffer(36 * C * tileStride * size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) : upload(values(maxBatch * C * AREA, 6));
    const resources = k.kind === 'in' ? [k.unfused ? inParamsUnfused : k.identity ? inParamsIdentity : inParams, dyn, board, out, sb, pre, mask] : [outParams, dyn, M, out, nbias];
    const bind = device.createBindGroup({ layout: k.layout, entries: resources.map((b, binding) => ({ binding, resource: { buffer: b } })) });
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
        // Stage kernels' outputs are meaningless and not checked.
        await pass(k, bind, groups, 1, false);
        const count = kind === 'in' ? 36 * C * tileStride : maxBatch * C * AREA;
        const got = k.stage ? null : await readBack(out, count);
        // The span of all timed passes, per dispatch (see timing.js); in the ladder, the median of `repeats` such timings.
        const ms = [];
        for (let r = 0; r < (ladder ? repeats : 1); r++) {
          const timed = await timePasses(device, (p) => {
            p.setPipeline(k.pipeline);
            p.setBindGroup(0, bind);
            for (let i = 0; i < repeat; i++) p.dispatchWorkgroups(...groups);
          }, { runs });
          ms.push(timed.msPerPass / repeat);
        }
        ms.sort((x, y) => x - y);
        const median = ms[Math.floor(ms.length / 2)];
        const entry = { kernel: k.name, ...(k.stage ? { stage: k.stage } : {}), ...(k.unfused ? { unfused: true } : {}), ...(k.identity ? { identity: true } : {}), msPerDispatch: +median.toFixed(4), gbPerSecond: +(bytes / (median / 1000) / 1e9).toFixed(1) };
        if (ladder) {
          const { read, write } = stageBytes(kind, k.stage, b);
          const streamMs = await stream(read, write);
          Object.assign(entry, { readBytes: read, writeBytes: write, streamMs: +streamMs.toFixed(4), ofStream: +(streamMs / median).toFixed(3) });
        }
        // Compare only what a batch of b writes: V[e][c][t < b·25], or the first b boards.
        let d = 0;
        if (k.stage) { /* not checked */ } else if (first === null) first = got;
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
  // ladder: the whole kernels again, interleaved. For `rounds` rounds, every whole kernel of a transform and the stream
  // kernel for its bytes are timed once each, in an order that rotates by one each round, so clock drift under the
  // power cap affects all of them alike. Median, least and greatest over the rounds.
  if (ladder) {
    report.interleaved = [];
    for (const b of batches) {
      const dyn = uniform([b, 0, 0, 0]);
      for (const kind of ['in', 'out']) {
        step(`batch ${b}, ${kind}put transform, interleaved`);
        const { read, write } = stageBytes(kind, undefined, b);
        const entries = kernels.filter((k) => k.kind === kind && !k.stage).map((k) => {
          const { out, bind } = setup(k, dyn);
          const groups = k.groups(b);
          return { name: k.name, out, ms: [], time: async () => (await timePasses(device, (p) => { p.setPipeline(k.pipeline); p.setBindGroup(0, bind); for (let i = 0; i < repeat; i++) p.dispatchWorkgroups(...groups); }, { runs })).msPerPass / repeat };
        });
        entries.push({ name: 'stream', ms: [], time: () => stream(read, write, 1) });
        for (let round = 0; round < rounds; round++) {
          for (let j = 0; j < entries.length; j++) {
            const e = entries[(j + round) % entries.length];
            e.ms.push(await e.time());
          }
        }
        const streamMs = entries.at(-1).ms.slice().sort((x, y) => x - y)[rounds >> 1];
        for (const e of entries) {
          const sorted = e.ms.slice().sort((x, y) => x - y);
          const med = sorted[rounds >> 1];
          report.interleaved.push({ batch: b, transform: kind, kernel: e.name, readBytes: read, writeBytes: write, msPerDispatch: +med.toFixed(4), minMs: +sorted[0].toFixed(4), maxMs: +sorted.at(-1).toFixed(4), ofStream: +(streamMs / med).toFixed(3) });
          e.out?.destroy();
        }
      }
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
      const ms = [(await timePasses(device, (p) => {
        p.setPipeline(k.pipeline);
        p.setBindGroup(0, bind);
        for (let i = 0; i < repeat; i++) p.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      }, { runs })).msPerPass / repeat];
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
    for (const r of rows) for (const k of r.kernels) if (!k.stage) entry[`${r.transform} ${k.kernel}`] = +(78 * k.msPerDispatch).toFixed(3);
    return entry;
  });
  report.ok = report.rows.every((r) => r.kernels.every((k) => (k.maxAbsDiffFromV1 ?? 0) === 0));
  if (lost) report.deviceLost = lost;
  device.destroy();
  return report;
}
