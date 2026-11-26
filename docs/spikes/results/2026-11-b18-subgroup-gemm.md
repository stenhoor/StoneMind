# Spike 0001: subgroup shuffles in the GEMM (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Code: branch `spike-0001/subgroup-gemm`, on top of
`spike-0001/pass-gap` (#78), with the experimental change below. That change was then taken out again, because it was
slower. All values below are quoted from the reports in this directory.

## What was tried

Edge on this machine exposes the `subgroups` feature without any flag, and the engine requests it, but no kernel used
it. GEMM v3 got an experimental option, `shuffleB`:
- **Normally:** each thread reads its own B operands from workgroup memory, as `vec4`s.
- **With `shuffleB`:** each of a subgroup's 32 lanes reads one `vec4` of the chunk's B row, and the threads take their
  operands from those with `subgroupShuffle`.

It assumes 32-lane subgroups made of consecutive threads. The GEMM benchmark's output check would catch a wrong result.

The change, as tested:

```diff
diff --git a/packages/engine-webgpu/src/gpu/kernels.ts b/packages/engine-webgpu/src/gpu/kernels.ts
index bcd3a8e..f995374 100644
--- a/packages/engine-webgpu/src/gpu/kernels.ts
+++ b/packages/engine-webgpu/src/gpu/kernels.ts
@@ -676,6 +676,13 @@ export interface GemmV3Config {
   double?: boolean;
   /** Threads along M (default 16): the workgroup is 16 × wy threads and its tile wy·rm × 16·rn. */
   wy?: 8 | 16;
+  /**
+   * Experimental (needs the `subgroups` feature): each lane reads one vec4 of the chunk's B row from workgroup memory
+   * and the threads take their B operands from it with `subgroupShuffle`, where otherwise each thread reads its own.
+   * Assumes 32-lane subgroups made of consecutive threads (lane = thread index mod 32), as on NVIDIA; anywhere else
+   * the result is wrong, which the GEMM benchmark's output check reports.
+   */
+  shuffleB?: boolean;
 }
 
 /** GEMM v3's tile: rows (M) and columns (N) of C per workgroup. */
@@ -699,6 +706,7 @@ export const GEMM_V3_DEFAULT: GemmV3Config = { rm: 4, rn: 8, tk: 16 };
 export const GEMM_V3 = (aT: Elem, bT: Elem, cT: Elem, config: GemmV3Config = GEMM_V3_DEFAULT) => {
   const { rm, rn, tk: TK } = config;
   const DB = config.double === true;
+  const SB = config.shuffleB === true;
   const WY = config.wy ?? 16;
   const THREADS = 16 * WY;
   const TM = WY * rm;
@@ -742,7 +750,10 @@ export const GEMM_V3 = (aT: Elem, bT: Elem, cT: Elem, config: GemmV3Config = GEM
   for (let kk = 0; kk < TK; kk++) {
     inner += '    {\n';
     for (let g = 0; g < GM; g++) inner += `      let a${g} = As[${DB ? 'ab + ' : ''}${kk * (TM / 4) + g * WY}u + ty];\n`;
-    for (let h = 0; h < GN; h++) inner += `      let b${h} = Bs[${DB ? 'bb + ' : ''}${kk * (TN / 4) + h * 16}u + tx];\n`;
+    if (SB) {
+      inner += `      let bv = Bs[${DB ? 'bb + ' : ''}${kk * (TN / 4)}u + lane % ${TN / 4}u];\n`;
+      for (let h = 0; h < GN; h++) inner += `      let b${h} = subgroupShuffle(bv, ${h * 16}u + tx);\n`;
+    } else for (let h = 0; h < GN; h++) inner += `      let b${h} = Bs[${DB ? 'bb + ' : ''}${kk * (TN / 4) + h * 16}u + tx];\n`;
     for (let g = 0; g < GM; g++)
       for (let i = 0; i < 4; i++)
         for (let h = 0; h < GN; h++) inner += `      c${g * 4 + i}_${h} = fma(vec4<f32>(a${g}.${comps[i]}), b${h}, c${g * 4 + i}_${h});\n`;
@@ -761,7 +772,7 @@ export const GEMM_V3 = (aT: Elem, bT: Elem, cT: Elem, config: GemmV3Config = GEM
         }
       store += `    }\n  }\n`;
     }
-  return /* wgsl */ `${enableFor(aT, bT, cT)}
+  return /* wgsl */ `${enableFor(aT, bT, cT)}${SB ? 'enable subgroups;\n' : ''}
 ${DYN}
 struct Params {
   M: u32, K: u32, nFixed: u32, nPerBatch: u32,
@@ -779,7 +790,7 @@ var<workgroup> As: array<vec4<f32>, ${DB ? 2 * NA : NA}>; // [${DB ? '2][' : ''}
 var<workgroup> Bs: array<vec4<f32>, ${DB ? 2 * NB : NB}>; // [${DB ? '2][' : ''}TK][TN / 4]
 
 @compute @workgroup_size(16, ${WY})
-fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
+fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>${SB ? ', @builtin(subgroup_invocation_id) lane: u32' : ''}) {
   let tx = l.x;
   let ty = l.y;
   let tid = ty * 16u + tx;
```

The benchmark added two kernels when the device has `subgroups`: `{rm:4, rn:8, tk:16, wy:16, shuffleB:true}` and
`{rm:4, rn:4, tk:16, wy:16, shuffleB:true}`.

## How it was run

- **Browser:** the separate Edge instance with the launch used in #48, no flags added, a fresh instance per run, one run
  after another.
- **Page:** the GEMM benchmark, with the steady timing from #76.
- **Reports:** [`…-gemm-edge-nvidia-1.json`](2026-11-b18-subgroup-gemm-gemm-edge-nvidia-1.json) (2026-11-26T14:34:53.945Z) and
  [`…-gemm-edge-nvidia-2.json`](2026-11-b18-subgroup-gemm-gemm-edge-nvidia-2.json) (2026-11-26T14:36:07.500Z).
- **Adapter:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}` in both, f16 storage, features
  `shader-f16, subgroups, timestamp-query`.
- **Both reports:** `ok: true`. The shuffle kernels' outputs equal v2's exactly (`maxAbsDiffFromV2: 0`).

## Results: `perNetwork` (ms)

| Kernel | Batch 16 (runs 1 / 2) | Batch 32 | Batch 64 |
| --- | --- | --- | --- |
| v2 4×8, chunk 16 | 21.075 / 21.456 | 39.364 / 38.746 | 74.004 / 74.534 |
| v3 4×8, chunk 16 (default) | 15.083 / 15.44 | 27.277 / 27.879 | 51.576 / 51.56 |
| v3 4×8, chunk 16, B by shuffle | 20.922 / 19.808 | 35.395 / 36.889 | 81.531 / 82.977 |
| v3 4×4, chunk 16 | 15.735 / 15.779 | 28.728 / 28.817 | 60.487 / 55.865 |
| v3 4×4, chunk 16, B by shuffle | 21.976 / 22.155 | 41.573 / 41.989 | 88.145 / 88.415 |

The shuffle kernels were slower than the same shape without shuffles in both runs, at every batch.

## Why (an explanation, not measured)

- **Instruction count:** a `subgroupShuffle` of a `vec4<f32>` moves 32-bit values, and on NVIDIA each lane gets one
  per shuffle instruction. A `vec4` read from workgroup memory is one instruction, and threads that read the same
  address share it.
- **What changed:** with `shuffleB`, each thread still reads workgroup memory once per step, then shuffles once per B
  operand. The kernel does more work for the same data.
- **Elsewhere:** subgroups may behave differently on other GPUs (Intel, AMD, Apple). They haven't been measured.
- **Matrix hardware:** that is the other subgroup route, and it needs `chromium-experimental-subgroup-matrix`, which
  is behind a browser flag.
