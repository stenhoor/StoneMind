# Spike 0001: Phase 5, item 1, the Winograd input transform inside the matrix GEMM (b18): closed by its stop rule

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Browser: the separate test Edge with
`--enable-unsafe-webgpu` (matrix hardware). Code: branch `spike-0001/wino-fused`, on top of `spike-0001/bn-act` (#91),
with the experimental change below. **That change was taken out again, because it was slower.** Every figure below is
quoted from, or computed from, the reports in this directory (prefix `2026-11-b18-wino-fused-`).

## Outcome

**Closed by its stop rule.** The goal was for the combined kernel to take at least 25% less time than the separate
input transform and GEMM. The rule was to stop if the first version wasn't faster at all.

- **The separate steps today**, each timed alone in the network at batch 16:
  - input transform 2.6843 and 2.7601 ms;
  - matrix GEMM 8.4306 and 8.4089 ms;
  - together 11.1 ms.

  The input transform also applies the batch norm before the convolution.
- **The combined kernel**, timed the same way, took 25.845 and 25.9802 ms at batch 16. That's about 2.3 times as long.
  - **Batch norm:** it needs the batch norm as a step of its own, and the network's batch-norm steps took 1.932–2.262
    ms in all, against 0.8752–0.9279 ms with the separate steps.
  - **Batch 64:** 92.2473 and 92.7713 ms, against 9.2668–13.5411 ms for the input transform plus 26.6136–27.0921 ms for
    the GEMM.
- **The whole batch-16 pass** took 34.6419–34.9257 ms with the combined kernel, against 17.9318–18.438 ms.
- **Results:** correct. The goldens check passed, including identical results across call sizes and concurrent calls
  (`-check.json`).
- **No code change is kept.** No review was run, as the kernel was slower on its own.

## Why it's slower

- **The combined kernel's layout.** Each workgroup computes one of the 36 Winograd elements for 64 tiles and all 192
  output channels. Each element of a tile's transform takes up to 16 of the tile's 36 input values. The 36
  workgroups for the same tiles each read the same inputs and compute their own element. The separate input transform
  reads each input value once per tile and computes all 36 elements together.
- **A diagnostic run:** the transform was replaced by a single input read, so the result was wrong. The combined
  kernel then took 12.5797 ms at batch 16 (`-budget-diag.json`).
  - **Overhead without the transform:** even then, it's slower than the separate input transform and GEMM together
    (11.1 ms). That's from writing each step's B values to workgroup memory one thread at a time, where the GEMM loads
    them as matrices.
  - **The transform's own cost:** the other 13.3 ms (25.845 − 12.5797) is the repeated transform work.
- **The design that would avoid the repeated work** is one workgroup computing all 36 elements for its tiles. That needs
  36 sets of accumulators per workgroup. With 32 output channels × 32 tiles per workgroup (an estimate, not built),
  batch 16 would give 6 × 13 = 78 workgroups for the GPU's 48 SMs. It wasn't tried.

## How it was run

- **The check:** `?variants=winoFused` on the check page, with the flag.
- **The budget benchmark:** `?budgetBatches=16,64`, with `budgetVariant` set to the default network (`sep`) or to it
  with `winoFused: true` (`fus`), in the order sep, fus, fus, sep. Each run used a fresh browser.

## The change, as tested

<details>
<summary>Diff against <code>spike-0001/bn-act</code></summary>

```diff
diff --git a/packages/engine-webgpu/src/gpu/kernels.ts b/packages/engine-webgpu/src/gpu/kernels.ts
index 94d3f95..ad976ee 100644
--- a/packages/engine-webgpu/src/gpu/kernels.ts
+++ b/packages/engine-webgpu/src/gpu/kernels.ts
@@ -1626,3 +1626,126 @@ ${direct ? '  }\n' : ''}`}${store}${epilogue}}
 `;
 };
 
+/** The combined Winograd input transform and matrix GEMM's shape: fm × fn 16 × 16 accumulators per subgroup, sgm × sgn subgroups. */
+export interface WinoFusedConfig { fm: number; fn: number; sgm: number; sgn: number }
+export const WINO_FUSED_DEFAULT: WinoFusedConfig = { fm: 4, fn: 2, sgm: 3, sgn: 2 };
+export const winoFusedTile = (c: WinoFusedConfig) => ({ tm: 16 * c.fm * c.sgm, tn: 16 * c.fn * c.sgn });
+/** Whether the combined kernel can run a layer: whole tiles of output channels, and input channels in steps of 32. */
+export const winoFusedFits = (c: WinoFusedConfig, outC: number, inC: number) => outC % winoFusedTile(c).tm === 0 && inC % 32 === 0;
+
+/**
+ * The Winograd input transform inside the matrix GEMM (needs the subgroup-matrix feature with an f16 → f32 16 × 16 × 16
+ * configuration): M[e][oc][t] = Σ_ic U[e][ic][oc] · V[e][ic][t], where each step's V values are computed from the input
+ * board, V[e][ic][t] = (Bᵀ d B)[e] for the tile's 6 × 6 input d (padding 0), into workgroup memory, instead of being
+ * read from a V written by a separate input transform. One workgroup per (column tile, row tile, e). The input must
+ * already be batch-normalised (no fused batch norm). U is [e][ic][oc], as the plan's GEMMs store it.
+ */
+export const WINO_FUSED_MMA = (inT: Elem, uT: Elem, mT: Elem, config: WinoFusedConfig = WINO_FUSED_DEFAULT) => {
+  const { fm, fn, sgm: SGM, sgn: SGN } = config;
+  const TM = 16 * fm;
+  const TN = 16 * fn;
+  const SG = SGM * SGN;
+  const WTM = TM * SGM;
+  const WTN = TN * SGN;
+  const FBN = WTN / 16;
+  const THREADS = 32 * SG;
+  const UK = 2;
+  // Bᵀ's rows as up to 4 (column, coefficient) pairs; unused pairs have coefficient 0 and column 0.
+  const idx = BT.map((row) => { const r = row.flatMap((c, k) => (c === 0 ? [] : [k])); while (r.length < 4) r.push(0); return r; });
+  const coef = BT.map((row) => { const r = row.filter((c) => c !== 0); while (r.length < 4) r.push(0); return r; });
+  const arr = (rows: number[][], t: string) => `array<array<${t}, 4>, 6>(${rows.map((r) => `array<${t}, 4>(${r.map((v) => (t === 'f32' ? v.toFixed(1) : `${v}`)).join(', ')})`).join(', ')})`;
+  let decl = '';
+  for (let i = 0; i < fm; i++) for (let j = 0; j < fn; j++) decl += `  var acc${i}_${j} = subgroup_matrix_result<f32, 16, 16>();\n`;
+  let body = '';
+  for (let kk = 0; kk < UK; kk++) {
+    for (let i = 0; i < fm; i++)
+      body += `      let l${kk}_${i} = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&U, aBase + (k0 + ${kk * 16}u) * P.aRow + m0 + ${i * 16}u, true, P.aRow);\n`;
+    for (let j = 0; j < fn; j++)
+      body += `      let r${kk}_${j} = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&Bsh, ${kk * FBN * 256}u + (sn * ${fn}u + ${j}u) * 256u, false, 16u);\n`;
+  }
+  for (let kk = 0; kk < UK; kk++)
+    for (let i = 0; i < fm; i++)
+      for (let j = 0; j < fn; j++) body += `      acc${i}_${j} = subgroupMatrixMultiplyAccumulate(l${kk}_${i}, r${kk}_${j}, acc${i}_${j});\n`;
+  let epilogue = '';
+  for (let i = 0; i < fm; i++)
+    for (let j = 0; j < fn; j++)
+      epilogue += `  subgroupMatrixStore(&Cs, sgid * 256u, acc${i}_${j}, false, 16u);
+  workgroupBarrier();
+  for (var f = lane; f < 256u; f += 32u) {
+    let m = m0 + ${i * 16}u + f / 16u;
+    let n = n0 + ${j * 16}u + f % 16u;
+    if (m < P.M && n < N) { M[e * P.cBatch + m * P.cRow + n] = ${mT}(Cs[sgid * 256u + f]); }
+  }
+  workgroupBarrier();
+`;
+  return /* wgsl */ `enable f16;
+enable subgroups;
+enable chromium_experimental_subgroup_matrix;
+${DYN}
+struct Params {
+  M: u32, K: u32, xSize: u32, ySize: u32,
+  tilesX: u32, tilesPerBoard: u32, aBatch: u32, aRow: u32,
+  cBatch: u32, cRow: u32, _pad0: u32, _pad1: u32,
+}
+@group(0) @binding(0) var<uniform> P: Params;
+@group(0) @binding(1) var<uniform> dyn: Dyn;
+@group(0) @binding(2) var<storage, read> U: array<${uT}>;
+@group(0) @binding(3) var<storage, read> inp: array<${inT}>;
+@group(0) @binding(4) var<storage, read_write> M: array<${mT}>;
+
+const BTI = ${arr(idx, 'u32')};
+const BTC = ${arr(coef, 'f32')};
+
+var<workgroup> Bsh: array<f16, ${UK * 16 * WTN}>; // one step's V: ${UK * FBN} matrices of 16 × 16, each contiguous
+var<workgroup> Cs: array<f32, ${256 * SG}>; // one 16 × 16 accumulator per subgroup
+
+@compute @workgroup_size(${THREADS})
+fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32, @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32) {
+  let e = wg.z;
+  let ei = e / 6u;
+  let ej = e % 6u;
+  let wm0 = wg.y * ${WTM}u;
+  let wn0 = wg.x * ${WTN}u;
+  let sm = sgid % ${SGM}u;
+  let sn = sgid / ${SGM}u;
+  let m0 = wm0 + sm * ${TM}u;
+  let n0 = wn0 + sn * ${TN}u;
+  let N = dyn.n * P.tilesPerBoard;
+  let area = P.xSize * P.ySize;
+  let aBase = e * P.aBatch;
+${decl}  for (var k0 = 0u; k0 < P.K; k0 += ${16 * UK}u) {
+    for (var q = tid; q < ${UK * 16 * WTN}u; q += ${THREADS}u) {
+      let kr = q / ${WTN}u;
+      let n = q % ${WTN}u;
+      let t = wn0 + n;
+      var v = 0.0;
+      if (t < N) {
+        let b = t / P.tilesPerBoard;
+        let r = t % P.tilesPerBoard;
+        let y0 = i32(r / P.tilesX) * 4 - 1;
+        let x0 = i32(r % P.tilesX) * 4 - 1;
+        let base = (b * P.K + k0 + kr) * area;
+        // V = Σ_b Bᵀ[ej][cb] · (Σ_a Bᵀ[ei][ra] · d[ra][cb]), the order of the separate input transform's sums.
+        for (var cbi = 0u; cbi < 4u; cbi++) {
+          let cc = BTC[ej][cbi];
+          if (cc == 0.0) { continue; }
+          let x = x0 + i32(BTI[ej][cbi]);
+          if (x < 0 || x >= i32(P.xSize)) { continue; }
+          var s = 0.0;
+          for (var rai = 0u; rai < 4u; rai++) {
+            let rc = BTC[ei][rai];
+            if (rc == 0.0) { continue; }
+            let y = y0 + i32(BTI[ei][rai]);
+            if (y >= 0 && y < i32(P.ySize)) { s += rc * f32(inp[base + u32(y) * P.xSize + u32(x)]); }
+          }
+          v += cc * s;
+        }
+      }
+      Bsh[((kr / 16u) * ${FBN}u + n / 16u) * 256u + (kr % 16u) * 16u + n % 16u] = f16(v);
+    }
+    workgroupBarrier();
+${body}    workgroupBarrier();
+  }
+${epilogue}}
+`;
+};
diff --git a/packages/engine-webgpu/src/gpu/network.ts b/packages/engine-webgpu/src/gpu/network.ts
index b8b8852..f105795 100644
--- a/packages/engine-webgpu/src/gpu/network.ts
+++ b/packages/engine-webgpu/src/gpu/network.ts
@@ -11,7 +11,7 @@ import { applyScale8 } from '../model/scale8.js';
 import type { NetInput, RawOutputs } from '../reference/forward.js';
 import type { EngineDevice } from './device.js';
 import { toFloat16Array } from './f16.js';
-import { ACT, BN_ACT, BN_ACT_V2, CONV, CONV_TILE_M, CONV_TILE_N, GEMM_DEFAULT, GEMM_MMA, gemmMmaTile, MATRIX_GEMM_BY_BATCH, matrixGemmFits, type GemmMmaConfig, GEMM_V2, GEMM_V3, GEMM_V3_DEFAULT, gemmV3Tile, type GemmV3Config, defaultGemm, defaultGemmConfig, gemmTile, type GemmConfig, GPOOL, MATVEC, WINO_GEMM, WINO_IN, WINO_IN_V2, WINO_IN_V3, WINO_OUT, WINO_OUT_V2, WINO_OUT_V3, winoTransformChannels, winogradWeights, type Elem } from './kernels.js';
+import { ACT, BN_ACT, BN_ACT_V2, CONV, CONV_TILE_M, CONV_TILE_N, GEMM_DEFAULT, GEMM_MMA, gemmMmaTile, MATRIX_GEMM_BY_BATCH, matrixGemmFits, WINO_FUSED_DEFAULT, WINO_FUSED_MMA, winoFusedFits, winoFusedTile, type WinoFusedConfig, type GemmMmaConfig, GEMM_V2, GEMM_V3, GEMM_V3_DEFAULT, gemmV3Tile, type GemmV3Config, defaultGemm, defaultGemmConfig, gemmTile, type GemmConfig, GPOOL, MATVEC, WINO_GEMM, WINO_IN, WINO_IN_V2, WINO_IN_V3, WINO_OUT, WINO_OUT_V2, WINO_OUT_V3, winoTransformChannels, winogradWeights, type Elem } from './kernels.js';
 import { BufferUsage, MapMode, type GPUBindGroup, type GPUBuffer, type GPUComputePassEncoder, type GPUComputePipeline, type GPUDevice, type GPUQuerySet } from './webgpu.js';
 
 export interface GpuNetworkOptions {
@@ -69,6 +69,11 @@ export interface GpuNetworkOptions {
    * (docs/spikes/results/2026-11-b18-matrix-ladder.md). \`winoGemm: 'mma'\` takes precedence for the Winograd GEMMs.
    */
   matrix?: 'auto' | 'off';
+  /**
+   * Experimental (default false): with matrix hardware in use (\`matrix\`), run each Winograd input transform inside its
+   * GEMM (\`WINO_FUSED_MMA\`), for layers it fits, with the batch norm before it as its own step.
+   */
+  winoFused?: boolean;
   /**
    * The Winograd input and output transforms together. 'v2' stages each board through workgroup memory (`WINO_IN_V2`,
    * `WINO_OUT_V2`): adjacent reads and writes, and a fused batch norm applied once per point. 'v3' is v2 reading
@@ -308,6 +313,7 @@ export class GpuNetwork {
     return this.#winoGemmMma;
   }
   #matrix = false;
+  #winoFused: WinoFusedConfig | null = null;
   /** Whether the plan runs its Winograd GEMMs and 1×1 convolutions on matrix hardware where they fit (option \`matrix\`). */
   get matrix(): boolean {
     return this.#matrix;
@@ -375,6 +381,7 @@ export class GpuNetwork {
     }
     net.#matrix = (options.matrix ?? 'auto') === 'auto' && net.#elem === 'f16' && net.#gemm !== 'v1' &&
       engine.subgroupMatrixConfigs.some((c) => c.componentType === 'f16' && c.resultComponentType === 'f32' && c.M === 16 && c.N === 16 && c.K === 16);
+    if (options.winoFused === true && net.#matrix) net.#winoFused = WINO_FUSED_DEFAULT;
     const scale8 = options.scale8 ?? net.#elem === 'f16';
     net.#desc = scale8 ? applyScale8(model) : model;
     net.#outputFactor = scale8 ? 8 : 1;
@@ -685,7 +692,12 @@ export class GpuNetwork {
   }
 
   #fusesInto(layer: ConvLayer): boolean {
-    return this.#fuse && this.#winogradEligible(layer);
+    return this.#fuse && this.#winogradEligible(layer) && !this.#winoFusedFor(layer);
+  }
+
+  /** Whether a Winograd layer runs its input transform inside its GEMM (option \`winoFused\`). */
+  #winoFusedFor(layer: ConvLayer): boolean {
+    return this.#winoFused !== null && this.#winogradEligible(layer) && winoFusedFits(this.#winoFused, layer.outChannels, layer.inChannels);
   }
 
   /**
@@ -721,7 +733,24 @@ export class GpuNetwork {
     const { inChannels: ic, outChannels: oc } = layer;
     const inFlags = fused ? 1 | (fused.pre ? 2 : 0) : 0;
     const act = fused ? ACT[fused.activation] : 0;
-    this.#specs.push({
+    const combined = !fused && this.#winoFusedFor(layer) ? this.#winoFused : null;
+    if (combined) {
+      const U = this.#upload(winogradWeights(layer.weights, ic, oc), this.#elem);
+      const tile = winoFusedTile(combined);
+      this.#specs.push({
+        kind: 'winoFused',
+        code: WINO_FUSED_MMA(this.#elemOf(input), this.#elemOf(U), this.#elemOf(M), combined),
+        buffers: [
+          this.#uniform([oc, ic, this.xSize, this.ySize, tilesX, tilesPerBoard, ic * oc, oc, oc * tileStride, tileStride, 0, 0]),
+          this.#dyn, U, input, M,
+        ],
+        workgroups: (n) => [Math.ceil((n * tilesPerBoard) / tile.tn), oc / tile.tm, 36],
+        work: (n: number) => ({
+          flops: 2 * 36 * oc * ic * n * tilesPerBoard + n * tilesPerBoard * ic * 2 * (2 * 6 * 6 * 6),
+          bytes: 36 * ic * oc * this.#eb(U) + n * ic * this.#area * this.#eb(input) + 36 * oc * n * tilesPerBoard * this.#eb(M),
+        }),
+      });
+    } else this.#specs.push({
       kind: 'winoIn',
       code: this.#winoIn === 'v1'
         ? WINO_IN(this.#elemOf(input), this.#elemOf(V))
@@ -741,6 +770,31 @@ export class GpuNetwork {
         bytes: n * ic * this.#area * this.#eb(input) + 36 * ic * n * tilesPerBoard * this.#eb(V),
       }),
     });
+    if (!combined) this.#winogradGemm(layer, V, M, ic, oc, tilesPerBoard, tileStride);
+    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
+    this.#specs.push({
+      kind: 'winoOut',
+      code: this.#winoOut === 'v1'
+        ? WINO_OUT(this.#elemOf(M), this.#elemOf(output))
+        : WINO_OUT_V2(this.#elemOf(M), this.#elemOf(output), this.xSize, this.ySize, this.#winoOut === 'v3' ? WINO_OUT_V3 : {}),
+      buffers: [
+        this.#uniform([oc, this.xSize, this.ySize, tilesX, tilesPerBoard, tileStride, flags, 0]),
+        this.#dyn, M, output, opts.nbias ?? this.#dummy,
+      ],
+      workgroups: this.#winoOut === 'v1'
+        ? (n) => [Math.ceil((n * tilesPerBoard) / 64), oc]
+        : this.#winoOut === 'v3'
+          ? (n) => [Math.ceil(oc / WINO_OUT_V3.cpw!), Math.ceil(n / WINO_OUT_V3.bpw!)]
+          : (n) => [Math.ceil(oc / winoTransformChannels('out', this.xSize, this.ySize)), n],
+      work: (n) => ({
+        flops: n * tilesPerBoard * oc * 2 * (4 * 6 * 6 + 4 * 4 * 6),
+        bytes: 36 * oc * n * tilesPerBoard * this.#eb(M) + n * oc * this.#area * this.#eb(output) * (opts.accumulate ? 2 : 1),
+      }),
+    });
+  }
+
+  /** The Winograd GEMM, M[e][oc][t] = Σ_ic U[e][ic][oc] · V[e][ic][t]: on matrix hardware or GEMM v1, v2 or v3. */
+  #winogradGemm(layer: ConvLayer, V: GPUBuffer, M: GPUBuffer, ic: number, oc: number, tilesPerBoard: number, tileStride: number): void {
     const mma = this.#winoGemmMma;
     if (mma) {
       // M[e][oc][t] = Σ_ic U[e][oc][ic] · V[e][ic][t] on matrix hardware, with U stored transposed ([e][oc][ic]).
@@ -791,26 +845,6 @@ export class GpuNetwork {
         }),
       });
     }
-    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
-    this.#specs.push({
-      kind: 'winoOut',
-      code: this.#winoOut === 'v1'
-        ? WINO_OUT(this.#elemOf(M), this.#elemOf(output))
-        : WINO_OUT_V2(this.#elemOf(M), this.#elemOf(output), this.xSize, this.ySize, this.#winoOut === 'v3' ? WINO_OUT_V3 : {}),
-      buffers: [
-        this.#uniform([oc, this.xSize, this.ySize, tilesX, tilesPerBoard, tileStride, flags, 0]),
-        this.#dyn, M, output, opts.nbias ?? this.#dummy,
-      ],
-      workgroups: this.#winoOut === 'v1'
-        ? (n) => [Math.ceil((n * tilesPerBoard) / 64), oc]
-        : this.#winoOut === 'v3'
-          ? (n) => [Math.ceil(oc / WINO_OUT_V3.cpw!), Math.ceil(n / WINO_OUT_V3.bpw!)]
-          : (n) => [Math.ceil(oc / winoTransformChannels('out', this.xSize, this.ySize)), n],
-      work: (n) => ({
-        flops: n * tilesPerBoard * oc * 2 * (4 * 6 * 6 + 4 * 4 * 6),
-        bytes: 36 * oc * n * tilesPerBoard * this.#eb(M) + n * oc * this.#area * this.#eb(output) * (opts.accumulate ? 2 : 1),
-      }),
-    });
   }
 
   #bnAct(bn: BatchNormLayer, activation: Activation, input: GPUBuffer, output: GPUBuffer, pre?: GPUBuffer): void {
diff --git a/packages/engine-webgpu/test/gpu/check.js b/packages/engine-webgpu/test/gpu/check.js
index f6e6e8f..11dfa3b 100644
--- a/packages/engine-webgpu/test/gpu/check.js
+++ b/packages/engine-webgpu/test/gpu/check.js
@@ -169,6 +169,8 @@ export async function runCheck({ bench = false, batches = [1, 2, 4, 8, 16, 32, 6
       { winograd: true, fuse: true, precision: 'f16', gemm: 'v3', matrix: 'auto' },
       { winograd: true, fuse: false, precision: 'f16', gemm: 'v3', matrix: 'auto' },
       { winograd: true, fuse: true, precision: 'f16', gemm: 'v3', matrix: 'auto', bnAct: 'v2' },
+      // The Winograd input transform inside the matrix GEMM (experimental).
+      { winograd: true, fuse: true, precision: 'f16', gemm: 'v3', matrix: 'auto', winoFused: true },
       { winograd: true, fuse: true, precision: 'f16', matrix: 'off', gemm: 'v3', winoGemm: 'mma' },
       { winograd: true, fuse: false, precision: 'f16', matrix: 'off', gemm: 'v3', winoGemm: 'mma' },
     );
```

</details>

## Next

Item 2 of Phase 5: faster search code.
