// The GEMM benchmark: GEMM v2 and GEMM v3 shapes on their own, at the b18 network's three GEMM shapes (the Winograd
// GEMM of a 3×3 192→192 convolution, and the 1×1 convolutions 384→192 and 192→384), with the same parameters and
// dispatch as GpuNetwork. Each timing is one compute pass of `repeat` identical dispatches, so it is far longer than
// the browser's timestamp resolution; the median over `runs` passes is reported, per dispatch. Every kernel's output is
// compared with GEMM v2's and, for the first batch position, with a float64 product on the CPU.
import { requestEngineDevice, toFloat16Array, fromFloat16Bits } from '../../dist/index.js';
import { GEMM_V2, GEMM_V3, GEMM_DEFAULT, GEMM_MMA, gemmTile, gemmV3Tile, gemmMmaTile } from '../../dist/gpu/kernels.js';
import { timePasses } from './timing.js';
import { measureMatrixPeak, measureCeilings, measureCacheRead } from './ceilings.js';

const TILES_PER_BOARD = 25; // 19 × 19 in 4 × 4 tiles
const AREA = 361;

/** The b18 GEMM shapes at batch `b`, with GpuNetwork's parameters (see #winogradConv and #conv1x1). */
function shapes(b, maxBatch) {
  const tileStride = maxBatch * TILES_PER_BOARD;
  const wino = (ic, oc) => ({
    name: `wino ${ic}→${oc}`, M: oc, K: ic, N: b * TILES_PER_BOARD, Z: 36, perNetwork: 62,
    params: [oc, ic, 0, TILES_PER_BOARD, ic * oc, oc, ic * tileStride, tileStride, oc * tileStride, tileStride, 0, 0],
    sizes: { A: 36 * ic * oc, B: 36 * ic * tileStride, C: 36 * oc * tileStride },
    strides: { aBatch: ic * oc, aRow: oc, bBatch: ic * tileStride, bRow: tileStride, cBatch: oc * tileStride, cRow: tileStride },
  });
  const conv1x1 = (ic, oc) => ({
    name: `1x1 ${ic}→${oc}`, M: oc, K: ic, N: AREA, Z: b, perNetwork: 18,
    params: [oc, ic, AREA, 0, 0, oc, ic * AREA, AREA, oc * AREA, AREA, 0, oc],
    sizes: { A: ic * oc, B: maxBatch * ic * AREA, C: maxBatch * oc * AREA },
    strides: { aBatch: 0, aRow: oc, bBatch: ic * AREA, bRow: AREA, cBatch: oc * AREA, cRow: AREA },
  });
  return [wino(192, 192), conv1x1(384, 192), conv1x1(192, 384)];
}

/** Values exactly representable in f16: multiples of 1/64 in [-1, 1]. */
function values(n, seed) {
  const out = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = ((s >>> 8) % 129 - 64) / 64;
  }
  return out;
}

/** The product for batch position z = 0, in float64: C[m][n] = Σ_k A[k][m] · B[k][n]. */
function reference(shape, A, B) {
  const { M, K, N, strides } = shape;
  const C = new Float64Array(M * N);
  for (let k = 0; k < K; k++) {
    for (let m = 0; m < M; m++) {
      const a = A[k * strides.aRow + m];
      if (a === 0) continue;
      const row = k * strides.bRow;
      for (let n = 0; n < N; n++) C[m * N + n] += a * B[row + n];
    }
  }
  return C;
}

export async function runGemmBench({ batches = [16, 32, 64], repeat = 30, runs = 7, warmUpMs = 2000, reverse = false, smallShapes = false, ladder = false, ladderParams = new URLSearchParams(), step = () => {} } = {}) {
  const engine = await requestEngineDevice();
  const { device } = engine;
  const report = { kind: 'gemm', adapter: engine.adapterInfo, features: [...engine.features], userAgent: navigator.userAgent, date: new Date().toISOString(), batches, repeat, runs };
  let lost = null;
  device.lost.then((info) => { lost = { reason: info.reason, message: info.message }; });
  const elem = engine.features.has('shader-f16') ? 'f16' : 'f32';
  const timestamps = engine.features.has('timestamp-query');
  report.elem = elem;
  report.timing = timestamps ? 'gpu timestamps, passes submitted back to back after a warm-up (timing.js)' : 'wall time (no timestamp-query)';
  const kernels = [
    { name: 'v2', config: GEMM_DEFAULT, code: GEMM_V2(elem, elem, elem, GEMM_DEFAULT), tile: gemmTile(GEMM_DEFAULT) },
  ];
  const wgBytes = device.limits.maxComputeWorkgroupStorageSize;
  report.maxComputeWorkgroupStorageSize = wgBytes;
  // [rm, rn, tk, wy, double]: rows and columns per thread, the chunk, threads along M, and double buffering.
  // [rm, rn, tk, wy, wx, double]: rows and columns per thread, the chunk, threads along M and along N, and double
  // buffering. smallShapes: narrower tiles, for small batches.
  // ladder: v3 shapes whole and by stage (GemmV3Config.stage), with this device's ceilings measured first
  // (docs/spikes/results/2026-11-b18-gemm-ladder.md). ?gemmLadderShapes=<JSON list of [rm, rn, tk, wy, wx]> (default:
  // the default shape) and ?gemmLadderVariants=<comma list of vload, blocks, prefetch, double>: each shape also with
  // that option, whole (and for blocks and prefetch, its loop: stage 'inner' at 1× and 8×).
  const ladderShapes = ladderParams.get('gemmLadderShapes') ? JSON.parse(ladderParams.get('gemmLadderShapes')) : [[4, 8, 16, 16, 16]];
  const ladderVariants = (ladderParams.get('gemmLadderVariants') ?? '').split(',').filter(Boolean);
  // ?gemmMmaLadder=1: the matrix GEMM's ladder (GemmMmaConfig.stage and kRepeat), with only v2 and the default v3
  // alongside. ?gemmMmaShapes=<JSON list of [fm, fn, ta, extra options]> (default [[2, 1, true], [2, 2, true]]).
  // ?gemmMmaLadder=whole: the same kernels, whole only (no stages).
  const mmaLadder = ladderParams.get('gemmMmaLadder') === '1' || ladderParams.get('gemmMmaLadder') === 'whole';
  const mmaStages = ladderParams.get('gemmMmaLadder') === '1';
  report.mmaLadder = mmaLadder;
  const shapesToTry = ladder || mmaLadder
    ? (mmaLadder ? [[4, 8, 16, 16, 16]] : ladderShapes)
    : smallShapes
    ? [[4, 8, 16, 16, 16], [4, 4, 16, 16, 16], [4, 4, 16, 16, 8], [4, 4, 16, 8, 8], [4, 4, 8, 8, 8], [4, 4, 16, 8, 16], [4, 4, 16, 16, 4], [4, 4, 32, 8, 8]]
    : [[4, 8, 16, 16, 16], [4, 4, 16, 16, 16], [4, 8, 8, 16, 16], [4, 8, 32, 16, 16], [8, 8, 16, 16, 16], [8, 8, 16, 8, 16], [8, 4, 16, 8, 16], [4, 8, 16, 16, 16, true]];
  report.smallShapes = smallShapes;
  report.ladder = ladder;
  if (ladder) report.ladderVariants = ladderVariants;
  for (const [rm, rn, tk, wy, wx, double] of shapesToTry) {
    const config = { rm, rn, tk, wy, ...(wx === 16 ? {} : { wx }), ...(double ? { double } : {}) };
    const tile = gemmV3Tile(config);
    if ((double ? 2 : 1) * tk * (tile.tm + tile.tn) * 4 > wgBytes) continue;
    kernels.push({ name: 'v3', config, code: GEMM_V3(elem, elem, elem, config), tile });
    if (!ladder) continue;
    const add = (c) => kernels.push({ name: 'v3', config: c, code: GEMM_V3(elem, elem, elem, c), tile });
    // The stages: the loop from workgroup memory and from registers, each at 1× and 8× K, and the loads alone.
    for (const stage of [{ stage: 'inner' }, { stage: 'inner', innerRepeat: 8 }, { stage: 'innerRegs' }, { stage: 'innerRegs', innerRepeat: 8 }, { stage: 'loads' }]) add({ ...config, ...stage });
    const loops = [{}, { stage: 'inner' }, { stage: 'inner', innerRepeat: 8 }];
    if (ladderVariants.includes('vload')) for (const extra of [{}, { stage: 'loads' }]) add({ ...config, vload: true, ...extra });
    if (ladderVariants.includes('blocks') && (config.wx ?? 16) % 8 === 0 && (config.wy ?? 16) % 4 === 0) for (const extra of loops) add({ ...config, layout: 'blocks', ...extra });
    if (ladderVariants.includes('prefetch')) for (const extra of loops) add({ ...config, prefetch: true, ...extra });
    if (ladderVariants.includes('double') && !double && 2 * tk * (tile.tm + tile.tn) * 4 <= wgBytes) for (const extra of [{}, { vload: true }]) add({ ...config, double: true, ...extra });
  }
  // The matrix GEMM (experimental), where the device has subgroup matrices with an f16 → f32 16 × 16 × 16 shape.
  const mma = elem === 'f16' && engine.subgroupMatrixConfigs.some((c) => c.componentType === 'f16' && c.resultComponentType === 'f32' && c.M === 16 && c.N === 16 && c.K === 16);
  report.subgroupMatrixConfigs = engine.subgroupMatrixConfigs;
  if (mma) {
    // Direct loads, with A as GpuNetwork stores it and transposed (as winoGemm: 'mma' stores it). In the matrix ladder,
    // each shape whole at K and 8 × K, its multiply-accumulates alone at K and 8 × K, and its fixed part.
    const mmaShapes = mmaLadder ? JSON.parse(ladderParams.get('gemmMmaShapes') ?? '[[2,1,true],[2,2,true]]') : [[2, 1, false], [2, 2, false], [2, 1, true], [2, 2, true]];
    for (const [fm, fn, ta, extra = {}] of mmaShapes) {
      const base = { ...(ta ? { fm, fn, direct: true, ta } : { fm, fn, direct: true }), ...extra };
      // Per pass of K: whole at 8 × K ÷ 8 (loop and loads) and mmaOnly at 8 × K ÷ 8 (multiply-accumulates). A difference
      // of 1 × and 8 × K isn't used: workgroups overlap one another's fixed part, so it underestimates the loop.
      const stages = mmaStages ? [{}, { kRepeat: 8 }, { stage: 'mmaOnly', kRepeat: 8 }, { stage: 'fixed' }] : [{}];
      for (const st of stages) {
        const config = { ...base, ...st };
        kernels.push({ name: 'mma', config, code: GEMM_MMA(elem, elem, elem, config), tile: gemmMmaTile(config) });
      }
    }
  }
  // Raw matrix throughput (measureMatrixPeak).
  if (mma && mmaLadder) {
    // The read bandwidth the matrix GEMM's direct loads are compared with: L2-sized working sets and a DRAM-sized one.
    const time = async (record) => {
      const xs = [];
      for (let r = 0; r < 3; r++) xs.push((await timePasses(device, record, { runs })).msPerPass);
      xs.sort((a, b) => a - b);
      return { msPerPass: xs[1], min: xs[0], max: xs[2] };
    };
    const spread = (t) => ({ ms: +t.msPerPass.toFixed(4), minMs: +t.min.toFixed(4), maxMs: +t.max.toFixed(4) });
    report.cacheRead = await measureCacheRead(device, { time, spread });
    step(`reads: ${report.cacheRead.map((c) => `${c.mib} MiB ${c.gbPerSecond} GB/s`).join(', ')}`);
  }
  if (mma) {
    report.matrixPeak = await measureMatrixPeak(device, { runs });
    step(`matrix units: ${report.matrixPeak.variants.map((v) => `${v.subgroupsPerWorkgroup}×${v.accumulators} ${v.tflops}`).join(', ')} TFLOPS`);
  }
  // reverse: measure the v3 shapes in the opposite order (v2 stays first, as the reference for the output check).
  if (reverse) kernels.splice(1, kernels.length - 1, ...kernels.slice(1).reverse());
  report.reverse = reverse;
  step(`compiling ${kernels.length} kernels`);
  for (const k of kernels) {
    const module = device.createShaderModule({ code: k.code });
    k.pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  const maxBatch = Math.max(...batches);
  const upload = (data, usage = GPUBufferUsage.STORAGE) => {
    const bytes = elem === 'f16' && data instanceof Float32Array ? toFloat16Array(data) : data;
    const size = Math.ceil(bytes.byteLength / 4) * 4;
    const buf = device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST });
    const padded = new Uint8Array(size);
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
    const bytes = count * (elem === 'f16' ? 2 : 4);
    const size = Math.ceil(bytes / 4) * 4;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, size);
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
  const query = timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const resolve = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const tsRead = timestamps ? device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
  const nbias = upload(new Float32Array(4), GPUBufferUsage.STORAGE);

  // Warm-up: keep the GPU busy for warmUpMs before anything is timed, so the first timings aren't taken while its clocks
  // are still rising (assumption: the first run's batch-16 figures, taken first, looked low for that reason).
  {
    step(`warming up for ${warmUpMs} ms`);
    const shape = shapes(maxBatch, maxBatch)[0];
    const A = upload(values(shape.sizes.A, 1));
    const B = upload(values(shape.sizes.B, 2));
    const C = device.createBuffer({ size: shape.sizes.C * 4, usage: GPUBufferUsage.STORAGE });
    const bind = device.createBindGroup({
      layout: kernels[0].pipeline.getBindGroupLayout(0),
      entries: [uniform(shape.params), uniform([maxBatch, 0, 0, 0]), A, B, C, nbias].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const t0 = performance.now();
    while (performance.now() - t0 < warmUpMs) {
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(kernels[0].pipeline);
      p.setBindGroup(0, bind);
      for (let i = 0; i < 20; i++) p.dispatchWorkgroups(Math.ceil(shape.N / kernels[0].tile.tn), Math.ceil(shape.M / kernels[0].tile.tm), shape.Z);
      p.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    report.warmUpMs = warmUpMs;
    A.destroy();
    B.destroy();
    C.destroy();
  }
  if (ladder) {
    // The ceilings the stages are compared with, in this session: each the median of 3 timings.
    step('ceilings');
    const time = async (record) => {
      const xs = [];
      for (let r = 0; r < 3; r++) xs.push((await timePasses(device, record, { runs })).msPerPass);
      xs.sort((a, b) => a - b);
      return { msPerPass: xs[1], min: xs[0], max: xs[2] };
    };
    const spread = (t) => ({ ms: +t.msPerPass.toFixed(4), minMs: +t.min.toFixed(4), maxMs: +t.max.toFixed(4) });
    report.ceilings = await measureCeilings(device, { hasF16: elem === 'f16', matrix: false, time, spread, runs });
  }
  report.rows = [];
  for (const b of batches) {
    for (const shape of shapes(b, maxBatch)) {
      step(`batch ${b}, ${shape.name}`);
      const Avals = values(shape.sizes.A, 1);
      const Bvals = values(shape.sizes.B, 2);
      const A = upload(Avals);
      // B is followed by 256 values, for the matrix GEMM's overreadN (read, never used).
      const Bpadded = new Float32Array(Bvals.length + 256);
      Bpadded.set(Bvals);
      const B = upload(Bpadded);
      // For kernels that take A transposed ([M][K] per matrix): the same values, and the row length K.
      let At = null;
      let paramsT = null;
      if (kernels.some((k) => k.config?.ta)) {
        const t = new Float32Array(Avals.length);
        for (let base = 0; base < Avals.length; base += shape.M * shape.K)
          for (let k = 0; k < shape.K; k++) for (let m = 0; m < shape.M; m++) t[base + m * shape.K + k] = Avals[base + k * shape.M + m];
        At = upload(t);
        const pt = [...shape.params];
        pt[5] = shape.K;
        paramsT = uniform(pt);
      }
      const dyn = uniform([b, 0, 0, 0]);
      const params = uniform(shape.params);
      const flops = 2 * shape.M * shape.K * shape.N * shape.Z;
      const row = { batch: b, shape: shape.name, M: shape.M, K: shape.K, N: shape.N, Z: shape.Z, perNetwork: shape.perNetwork, gflop: +(flops / 1e9).toFixed(3), kernels: [] };
      let v2Out = null;
      const ref = b === batches[0] ? reference(shape, Avals, Bvals) : null;
      for (const k of kernels) {
        if (lost) throw new Error(`device lost: ${JSON.stringify(lost)}`);
        const C = device.createBuffer({ size: Math.ceil((shape.sizes.C * (elem === 'f16' ? 2 : 4)) / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bind = device.createBindGroup({
          layout: k.pipeline.getBindGroupLayout(0),
          entries: (k.config?.ta ? [paramsT, dyn, At, B, C, nbias] : [params, dyn, A, B, C, nbias]).map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const groups = [Math.ceil(shape.N / k.tile.tn), Math.ceil(shape.M / k.tile.tm), shape.Z];
        const pass = async (count, timed) => {
          const enc = device.createCommandEncoder();
          const p = enc.beginComputePass(timed && query ? { timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
          p.setPipeline(k.pipeline);
          p.setBindGroup(0, bind);
          for (let i = 0; i < count; i++) p.dispatchWorkgroups(...groups);
          p.end();
          if (timed && query) {
            enc.resolveQuerySet(query, 0, 2, resolve, 0);
            enc.copyBufferToBuffer(resolve, 0, tsRead, 0, 16);
          }
          const t0 = performance.now();
          device.queue.submit([enc.finish()]);
          await device.queue.onSubmittedWorkDone();
          const wall = performance.now() - t0;
          if (!(timed && query)) return wall;
          await tsRead.mapAsync(GPUMapMode.READ);
          const [begin, end] = new BigUint64Array(tsRead.getMappedRange().slice(0));
          tsRead.unmap();
          return Number(end - begin) / 1e6;
        };
        await pass(1, false); // warm-up, and the output checked below
        const out = await readBack(C, shape.sizes.C);
        let ms = [];
        const passMedians = [];
        if (timestamps) {
          const timed = await timePasses(device, (p) => {
            p.setPipeline(k.pipeline);
            p.setBindGroup(0, bind);
            for (let i = 0; i < repeat; i++) p.dispatchWorkgroups(...groups);
          }, { runs });
          // The span of all timed passes, per dispatch (see timing.js); the per-pass median is kept for comparison.
          ms = [timed.msPerPass / repeat];
          passMedians.push(timed.ms[Math.floor(timed.ms.length / 2)] / repeat);
        } else {
          for (let r = 0; r < runs; r++) ms.push((await pass(repeat, true)) / repeat);
          ms.sort((x, y) => x - y);
        }
        const median = ms[Math.floor(ms.length / 2)];
        const entry = { kernel: k.name, config: k.config, msPerDispatch: +median.toFixed(4), tflops: +(flops / (median / 1000) / 1e12).toFixed(2) };
        if (passMedians.length) entry.passMedianMsPerDispatch = +passMedians[0].toFixed(4);
        // The outputs that are written: z < Z, m < M, n < N.
        const cmp = (other) => {
          let d = 0;
          for (let z = 0; z < shape.Z; z++)
            for (let m = 0; m < shape.M; m++) {
              const base = z * shape.strides.cBatch + m * shape.strides.cRow;
              for (let n = 0; n < shape.N; n++) d = Math.max(d, Math.abs(out[base + n] - other[base + n]));
            }
          return d;
        };
        if (v2Out === null) v2Out = out;
        else entry.maxAbsDiffFromV2 = cmp(v2Out);
        // The matrix GEMM sums in another order, so it is held to the reference check's tolerance against v2 instead.
        if (k.name === 'mma' && v2Out !== out) {
          let worst = 0;
          for (let z = 0; z < shape.Z; z++)
            for (let m = 0; m < shape.M; m++) {
              const base = z * shape.strides.cBatch + m * shape.strides.cRow;
              for (let n = 0; n < shape.N; n++) worst = Math.max(worst, Math.abs(out[base + n] - v2Out[base + n]) / (1e-2 + 2e-3 * Math.abs(v2Out[base + n])));
            }
          entry.worstV2Ratio = +worst.toFixed(4);
        }
        if (ref) {
          let worst = 0;
          for (let m = 0; m < shape.M; m++)
            for (let n = 0; n < shape.N; n++) {
              const want = ref[m * shape.N + n];
              const got = out[m * shape.strides.cRow + n];
              worst = Math.max(worst, Math.abs(got - want) / (1e-2 + 2e-3 * Math.abs(want)));
            }
          entry.worstReferenceRatio = +worst.toFixed(4);
          entry.matchesReference = worst <= 1;
        }
        row.kernels.push(entry);
        C.destroy();
      }
      report.rows.push(row);
      A.destroy();
      B.destroy();
      At?.destroy();
    }
  }
  // Per network, for each kernel: Σ over the three shapes of perNetwork × ms per dispatch, by batch.
  report.perNetwork = batches.map((b) => {
    const rows = report.rows.filter((r) => r.batch === b);
    return {
      batch: b,
      ms: kernels.map((k, i) => ({ kernel: k.name, config: k.config, ms: +rows.reduce((s, r) => s + r.perNetwork * r.kernels[i].msPerDispatch, 0).toFixed(3) })),
    };
  });
  // A stage of a kernel (GemmV3Config.stage, GemmMmaConfig.stage or kRepeat) computes nothing correct, so it isn't checked.
  report.ok = report.rows.every((r) => r.kernels.every((e) => e.config?.stage !== undefined || e.config?.kRepeat !== undefined || ((e.kernel === 'mma' ? e.worstV2Ratio <= 1 : (e.maxAbsDiffFromV2 ?? 0) === 0) && e.matchesReference !== false)));
  if (lost) report.deviceLost = lost;
  device.destroy();
  return report;
}

