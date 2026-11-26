// Device ceilings for the benchmarks: what this GPU reaches through WebGPU on simple kernels, measured in the same
// session as the work they're compared with. Copy bandwidth (vec4 f32 copy; bytes read plus bytes written), vector FMA
// throughput (8 independent vec4 accumulators per thread), matrix throughput (subgroup matrices, where the device has
// them) and the fixed cost of a near-empty dependent dispatch.
import { timePasses } from './timing.js';

export const NOOP = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) i: u32) { v[i] = v[i] + 1.0; }
`;

export const COPY = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.y * nw.x * 256u + g.x;
  if (i < arrayLength(&dst)) { dst[i] = src[i]; }
}
`;

/** Vector FMA throughput: 8 independent vec4 accumulators per thread, `iter` fma steps each. */
export const fmaKernel = (t, iter) => `${t === 'f16' ? 'enable f16;\n' : ''}
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


const pipelineOf = (device, code) => device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });

/** Whether the device has subgroup matrices with an f16 → f32 16 × 16 × 16 shape. */
export const hasMatrix16 = (engine) =>
  engine.features.has('shader-f16') && engine.subgroupMatrixConfigs.some((c) => c.componentType === 'f16' && c.resultComponentType === 'f32' && c.M === 16 && c.N === 16 && c.K === 16);

/** Keep the GPU busy with the f32 FMA kernel for `ms`, so measurements after it don't run while its clocks rise. */
export async function warmGpu(device, ms) {
  const pipeline = await pipelineOf(device, fmaKernel('f32', 256));
  const out = device.createBuffer({ size: 6144 * 256 * 16, usage: GPUBufferUsage.STORAGE });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
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

/**
 * Every ceiling. `time(record)` times a pass and returns { msPerPass, min, max }; `spread(t)` formats it for the report.
 * @returns { copy: [{ mib, bytesMoved, ms…, gbPerSecond }], fmaf32, fmaf16?, matrix?, dispatch }
 */
export async function measureCeilings(device, { hasF16, matrix, time, spread, runs = 7 }) {
  const out = {};
  {
    const pipeline = await pipelineOf(device, COPY);
    out.copy = [];
    for (const mib of [64, 256]) {
      const size = mib * 1024 * 1024;
      const src = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      const dst = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: dst } }] });
      const groups = size / 16 / 256;
      const x = Math.min(groups, 32768);
      const y = Math.ceil(groups / x);
      const t = await time((p) => { p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(x, y); });
      out.copy.push({ mib, bytesMoved: 2 * size, ...spread(t), gbPerSecond: +((2 * size) / (t.msPerPass / 1000) / 1e9).toFixed(1) });
      src.destroy();
      dst.destroy();
    }
  }
  for (const t of hasF16 ? ['f32', 'f16'] : ['f32']) {
    const ITER = 1024;
    const WGS = 6144;
    const pipeline = await pipelineOf(device, fmaKernel(t, ITER));
    const buf = device.createBuffer({ size: WGS * 256 * 16, usage: GPUBufferUsage.STORAGE });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const timed = await time((p) => { p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(WGS); });
    const flops = WGS * 256 * ITER * 8 * 4 * 2;
    out[`fma${t}`] = { workgroups: WGS, iterations: ITER, ...spread(timed), tflops: +(flops / (timed.msPerPass / 1000) / 1e12).toFixed(2) };
    buf.destroy();
  }
  if (matrix) out.matrix = await measureMatrixPeak(device, { runs });
  {
    const COUNT = 500;
    const pipeline = await pipelineOf(device, NOOP);
    const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const t = await time((p) => { for (let i = 0; i < COUNT; i++) { p.setPipeline(pipeline); p.setBindGroup(0, bind); p.dispatchWorkgroups(1); } });
    out.dispatch = { count: COUNT, msPerDispatch: +(t.msPerPass / COUNT).toFixed(5), minMsPerDispatch: +(t.min / COUNT).toFixed(5), maxMsPerDispatch: +(t.max / COUNT).toFixed(5) };
    buf.destroy();
  }
  return out;
}

/**
 * Raw matrix throughput: only subgroupMatrixMultiplyAccumulate on values already in registers (no loads), with 4
 * independent accumulators, so the figure is what the matrix units reach through WebGPU on this device. Needs the
 * device feature chromium-experimental-subgroup-matrix with an f16 → f32 16 × 16 × 16 shape.
 */
export async function measureMatrixPeak(device, { runs = 7 } = {}) {
  const ITER = 1024;
  const WGS = 48 * 64;
  const code = `enable f16;
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
var<workgroup> L: array<f16, 256>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  for (var i = tid; i < 256u; i += 32u) { L[i] = f16(f32(i % 7u) * 0.001); }
  workgroupBarrier();
  let l = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&L, 0u, false, 16u);
  let r = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&L, 0u, false, 16u);
  var a0 = subgroup_matrix_result<f32, 16, 16>();
  var a1 = subgroup_matrix_result<f32, 16, 16>();
  var a2 = subgroup_matrix_result<f32, 16, 16>();
  var a3 = subgroup_matrix_result<f32, 16, 16>();
  for (var i = 0u; i < ${ITER}u; i++) {
    a0 = subgroupMatrixMultiplyAccumulate(l, r, a0);
    a1 = subgroupMatrixMultiplyAccumulate(l, r, a1);
    a2 = subgroupMatrixMultiplyAccumulate(l, r, a2);
    a3 = subgroupMatrixMultiplyAccumulate(l, r, a3);
  }
  let base = wg.x * 1024u;
  subgroupMatrixStore(&out, base, a0, false, 16u);
  subgroupMatrixStore(&out, base + 256u, a1, false, 16u);
  subgroupMatrixStore(&out, base + 512u, a2, false, 16u);
  subgroupMatrixStore(&out, base + 768u, a3, false, 16u);
}
`;
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const out = device.createBuffer({ size: WGS * 1024 * 4, usage: GPUBufferUsage.STORAGE });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: out } }] });
  const timed = await timePasses(device, (p) => {
    p.setPipeline(pipeline);
    p.setBindGroup(0, bind);
    p.dispatchWorkgroups(WGS);
  }, { runs });
  const flops = WGS * ITER * 4 * 2 * 16 * 16 * 16;
  const result = { workgroups: WGS, iterations: ITER, msPerDispatch: +timed.msPerPass.toFixed(4), tflops: +(flops / (timed.msPerPass / 1000) / 1e12).toFixed(2) };
  out.destroy();
  return result;
}
