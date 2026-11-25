/**
 * WGSL kernels for the network. Tensors are [batch][channel][y][x], with the board's points contiguous.
 *
 * The kernels that touch large tensors are generators: each storage buffer's element type is a parameter (`f32` or
 * `f16`). Values are converted to f32 as they are loaded and back to the buffer's type as they are stored, so all
 * arithmetic is f32 and fp16 only changes storage and memory traffic. The same generated code serves both, with the
 * conversions becoming identities in the f32 build.
 *
 * Every kernel reads the batch size from `dyn.n`, so one set of bind groups serves every batch size up to the plan's.
 */

/** A storage buffer's element type. */
export type Elem = 'f32' | 'f16';

/** WGSL's `enable f16;` when any of the types is f16 (it needs the device feature `shader-f16`). */
const enableFor = (...types: Elem[]) => (types.includes('f16') ? 'enable f16;\n' : '');

/** Activation codes, shared with the plan builder. */
export const ACT = { identity: 0, relu: 1, mish: 2, silu: 3, mish_scale8: 4 } as const;

const ACTIVATE = /* wgsl */ `
// mish(x) = x · tanh(softplus(x)). Computed as x · n / (n + 2) with n = eˣ(eˣ + 2), which equals tanh(log(1 + eˣ))
// exactly (tanh y = (e²ʸ − 1)/(e²ʸ + 1) with eʸ = 1 + eˣ). It needs only exp, which WGSL specifies with relative
// error; log(1 + eˣ) loses most of its precision for negative x in f32, and WGSL has no log1p.
// For x ≥ 20 both KataGo and this return x (tanh(softplus(x)) is 1 in f32 well before 20).
fn mish_(x: f32) -> f32 {
  if (x >= 20.0) { return x; }
  let e = exp(x);
  let n = e * (e + 2.0);
  return x * n / (n + 2.0);
}

fn activate(x: f32, act: u32) -> f32 {
  switch act {
    case 1u: { return max(x, 0.0); }
    case 2u: { return mish_(x); }
    case 3u: { return x / (exp(-x) + 1.0); }
    case 4u: {
      // KataGo's mish_scale8 (OpenCL ACTIVATION == 12): x < 20/8 ? x · tanh(log1p(exp(8x))) : x, i.e. mish(8x)/8.
      if (x >= 2.5) { return x; }
      let e = exp(8.0 * x);
      let n = e * (e + 2.0);
      return x * n / (n + 2.0);
    }
    default: { return x; }
  }
}
`;

const DYN = /* wgsl */ `struct Dyn { n: u32 }`;

/** Evaluates `activate` on each input, for checking it against KataGo's formulas (the GPU check uses it). */
export const ACTIVATION_PROBE = /* wgsl */ `
${ACTIVATE}
@group(0) @binding(0) var<uniform> act: vec4<u32>;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read_write> ys: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&xs)) { ys[gid.x] = activate(xs[gid.x], act.x); }
}
`;

/** Convolution tile sizes: 64 output channels × 64 board points per workgroup, 16 of the reduction at a time. */
export const CONV_TILE_M = 64;
export const CONV_TILE_N = 64;

/**
 * Convolution as an implicit GEMM: out[b][oc][p] = Σ_k w[k][oc] · in[b][ic(k)][p + offset(k)], k = (ic, ky, kx).
 * Zero padding. flags bit 0: add to the output instead of overwriting it (the residual add); bit 1: add a per-(batch,
 * output channel) bias, `nbias[b][oc]`.
 */
export const CONV = (inT: Elem, wT: Elem, outT: Elem) => /* wgsl */ `${enableFor(inT, wT, outT)}
${DYN}
struct Params {
  inC: u32, outC: u32, kh: u32, kw: u32,
  dilY: u32, dilX: u32, xSize: u32, ySize: u32,
  K: u32, flags: u32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<${inT}>;
@group(0) @binding(3) var<storage, read> w: array<${wT}>;
@group(0) @binding(4) var<storage, read_write> outp: array<${outT}>;
@group(0) @binding(5) var<storage, read> nbias: array<f32>;

const TM = 64u;
const TN = 64u;
const TK = 16u;
var<workgroup> As: array<f32, 1024>; // [TK][TM]
var<workgroup> Bs: array<f32, 1024>; // [TK][TN]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let area = P.xSize * P.ySize;
  let N = dyn.n * area;
  let m0 = wg.y * TM;
  let n0 = wg.x * TN;
  let tid = l.y * 16u + l.x;
  let khw = P.kh * P.kw;
  let ry = i32((P.kh - 1u) / 2u);
  let rx = i32((P.kw - 1u) / 2u);

  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

  for (var k0 = 0u; k0 < P.K; k0 += TK) {
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      let kk = idx / TM;
      let col = idx % TM;
      let k = k0 + kk;
      // Weights, [K][outC].
      let m = m0 + col;
      var a = 0.0;
      if (k < P.K && m < P.outC) { a = f32(w[k * P.outC + m]); }
      As[idx] = a;
      // Inputs, gathered for this reduction index.
      let p = n0 + col;
      var v = 0.0;
      if (k < P.K && p < N) {
        let b = p / area;
        let pos = p % area;
        let y = i32(pos / P.xSize);
        let x = i32(pos % P.xSize);
        let ic = k / khw;
        let r = k % khw;
        let yy = y + (i32(r / P.kw) - ry) * i32(P.dilY);
        let xx = x + (i32(r % P.kw) - rx) * i32(P.dilX);
        if (yy >= 0 && yy < i32(P.ySize) && xx >= 0 && xx < i32(P.xSize)) {
          v = f32(inp[(b * P.inC + ic) * area + u32(yy) * P.xSize + u32(xx)]);
        }
      }
      Bs[idx] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < TK; kk++) {
      var av: array<f32, 4>;
      var bv: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { av[i] = As[kk * TM + i * 16u + l.y]; }
      for (var j = 0u; j < 4u; j++) { bv[j] = Bs[kk * TN + j * 16u + l.x]; }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += av[i] * bv[j]; }
      }
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < 4u; i++) {
    let m = m0 + i * 16u + l.y;
    if (m >= P.outC) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let p = n0 + j * 16u + l.x;
      if (p >= N) { continue; }
      let b = p / area;
      let o = (b * P.outC + m) * area + p % area;
      var v = acc[i * 4u + j];
      if ((P.flags & 2u) != 0u) { v += nbias[b * P.outC + m]; }
      if ((P.flags & 1u) != 0u) { v += f32(outp[o]); }
      outp[o] = ${outT}(v);
    }
  }
}
`;

/**
 * Batch norm, activation, then the board mask: out = mask ? act(x · scale + bias) : 0. flags bit 0: first add a
 * per-(batch, channel) bias, `pre[b][c]` (the global-pooling bias, added before the batch norm as KataGo does).
 * `sb` holds the merged scales, then the merged biases. The mask is channel 0 of the network's input.
 */
export const BN_ACT = (inT: Elem, outT: Elem) => /* wgsl */ `${enableFor(inT, outT)}
${DYN}
${ACTIVATE}
struct Params { C: u32, area: u32, flags: u32, act: u32, maskStride: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<${inT}>;
@group(0) @binding(3) var<storage, read> sb: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<${outT}>;
@group(0) @binding(5) var<storage, read> pre: array<f32>;
@group(0) @binding(6) var<storage, read> mask: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let idx = (wg.y * nwg.x + wg.x) * 256u + li;
  let perBatch = P.C * P.area;
  if (idx >= dyn.n * perBatch) { return; }
  let b = idx / perBatch;
  let c = (idx / P.area) % P.C;
  let pos = idx % P.area;
  var x = f32(inp[idx]);
  if ((P.flags & 1u) != 0u) { x += pre[b * P.C + c]; }
  let y = activate(x * sb[c] + sb[P.C + c], P.act);
  outp[idx] = ${outT}(select(0.0, y, mask[b * P.maskStride + pos] == 1.0));
}
`;

/**
 * Global pooling, one workgroup per (channel, batch). Writes out[b][3C]: [means, means · (√n − 14)/10, third], where
 * n counts the on-board points and the third is the max over them (mode 0) or the mean · ((√n − 14)²/100 − 0.1) the
 * value head uses (mode 1). Inputs are already masked to 0 off the board.
 */
export const GPOOL = (inT: Elem) => /* wgsl */ `${enableFor(inT)}
${DYN}
struct Params { C: u32, area: u32, mode: u32, maskStride: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<${inT}>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;

var<workgroup> sSum: array<f32, 64>;
var<workgroup> sMax: array<f32, 64>;
var<workgroup> sCount: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = wg.x;
  let b = wg.y;
  if (b >= dyn.n) { return; }
  var s = 0.0;
  var m = -1.0;
  var count = 0.0;
  for (var i = li; i < P.area; i += 64u) {
    let x = f32(inp[(b * P.C + c) * P.area + i]);
    let k = mask[b * P.maskStride + i];
    s += x;
    m = max(m, x + (k - 1.0));
    count += k;
  }
  sSum[li] = s;
  sMax[li] = m;
  sCount[li] = count;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (li < stride) {
      sSum[li] += sSum[li + stride];
      sMax[li] = max(sMax[li], sMax[li + stride]);
      sCount[li] += sCount[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let n = sCount[0];
    let mean = sSum[0] / n;
    let k = sqrt(n) - 14.0;
    let o = b * 3u * P.C;
    outp[o + c] = mean;
    outp[o + P.C + c] = mean * k * 0.1;
    if (P.mode == 0u) { outp[o + 2u * P.C + c] = sMax[0]; } else { outp[o + 2u * P.C + c] = mean * (k * k * 0.01 - 0.1); }
  }
}
`;

/**
 * Matrix-vector products for the small dense layers: out[b][outOffset + oc] = act(Σ_ic in[b][ic] · w[ic][oc] + bias).
 * flags bit 0: add `bias[oc]`.
 */
export const MATVEC = /* wgsl */ `
${DYN}
${ACTIVATE}
struct Params { inC: u32, outC: u32, inStride: u32, outStride: u32, outOffset: u32, flags: u32, act: u32, _pad0: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let oc = wg.x * 64u + li;
  let b = wg.y;
  if (oc >= P.outC || b >= dyn.n) { return; }
  var s = 0.0;
  for (var ic = 0u; ic < P.inC; ic++) { s += inp[b * P.inStride + ic] * w[ic * P.outC + oc]; }
  if ((P.flags & 1u) != 0u) { s += bias[oc]; }
  outp[b * P.outStride + P.outOffset + oc] = activate(s, P.act);
}
`;

// ------------------------------------------------------------------ Winograd F(4×4, 3×3)
//
// A 3×3 convolution on 4×4 output tiles: Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A, with d the 6×6 input tile (padding 1).
// The matrices are Lavin & Gray's ("Fast Algorithms for Convolutional Neural Networks", 2015), interpolation points
// 0, ±1, ±2, ∞; KataGo's OpenCL backend uses the same algorithm. The weight transform G g Gᵀ is done once, on the CPU.
// Transformed tensors are [36][channel][tile], tile = b · tilesPerBoard + ty · tilesX + tx, with a fixed tile stride
// (the plan's largest batch) so the layout doesn't depend on the batch size.

const BT = [
  [4, 0, -5, 0, 1, 0],
  [0, -4, -4, 1, 1, 0],
  [0, 4, -4, -1, 1, 0],
  [0, -2, -1, 2, 1, 0],
  [0, 2, -1, -2, 1, 0],
  [0, 4, 0, -5, 0, 1],
];
const G = [
  [1 / 4, 0, 0],
  [-1 / 6, -1 / 6, -1 / 6],
  [-1 / 6, 1 / 6, -1 / 6],
  [1 / 24, 1 / 12, 1 / 6],
  [1 / 24, -1 / 12, 1 / 6],
  [0, 0, 1],
];
const AT = [
  [1, 1, 1, 1, 1, 0],
  [0, 1, -1, 2, -2, 0],
  [0, 1, 1, 4, 4, 0],
  [0, 1, -1, 8, -8, 1],
];

/** WGSL for out = M · in · Mᵀ with constant M (rows × 6 or rows × 3), fully unrolled, skipping zero terms. */
function sandwich(m: number[][], inName: string, inCols: number, tmpName: string, outName: string): string {
  const term = (coef: number, expr: string) => (coef === 1 ? expr : coef === -1 ? `-${expr}` : `${coef.toFixed(9)} * ${expr}`);
  const sum = (terms: string[]) => (terms.length ? terms.join(' + ').replace(/\+ -/g, '- ') : '0.0');
  const rows = m.length;
  let s = '';
  // tmp[i][j] = Σ_k m[i][k] · in[k][j]
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < inCols; j++)
      s += `  let ${tmpName}${i}_${j} = ${sum(m[i]!.flatMap((c, k) => (c === 0 ? [] : [term(c, `${inName}[${k * inCols + j}]`)])))};\n`;
  // out[i][j] = Σ_k tmp[i][k] · m[j][k]
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < rows; j++)
      s += `  ${outName}[${i * rows + j}] = ${sum(m[j]!.flatMap((c, k) => (c === 0 ? [] : [term(c, `${tmpName}${i}_${k}`)])))};\n`;
  return s;
}

/** The weight transform, on the CPU in float64: weights [oc][ic][3][3] → U [36][ic][oc]. */
export function winogradWeights(weights: Float32Array, inChannels: number, outChannels: number): Float32Array {
  const U = new Float32Array(36 * inChannels * outChannels);
  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < inChannels; ic++) {
      const g = weights.subarray((oc * inChannels + ic) * 9, (oc * inChannels + ic) * 9 + 9);
      const t: number[][] = G.map((row) => [0, 1, 2].map((j) => row[0]! * g[j]! + row[1]! * g[3 + j]! + row[2]! * g[6 + j]!));
      for (let i = 0; i < 6; i++)
        for (let j = 0; j < 6; j++) {
          const v = t[i]![0]! * G[j]![0]! + t[i]![1]! * G[j]![1]! + t[i]![2]! * G[j]![2]!;
          U[((i * 6 + j) * inChannels + ic) * outChannels + oc] = v;
        }
    }
  }
  return U;
}

/**
 * Input transform: one thread per (tile, channel). V[e][c][t] = (Bᵀ d B)[e].
 * flags bit 0: apply BN_ACT's batch norm, activation and mask to each input value first (fusing the preceding layer);
 * bit 1: with it, first add `pre[b][c]`. Padding stays 0, as it would be after a separate BN_ACT.
 */
export const WINO_IN = (inT: Elem, vT: Elem) => /* wgsl */ `${enableFor(inT, vT)}
${DYN}
${ACTIVATE}
struct Params {
  C: u32, xSize: u32, ySize: u32, tilesX: u32,
  tilesPerBoard: u32, tileStride: u32, flags: u32, act: u32,
  maskStride: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<${inT}>;
@group(0) @binding(3) var<storage, read_write> V: array<${vT}>;
@group(0) @binding(4) var<storage, read> sb: array<f32>;
@group(0) @binding(5) var<storage, read> pre: array<f32>;
@group(0) @binding(6) var<storage, read> mask: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let c = gid.y;
  if (t >= dyn.n * P.tilesPerBoard || c >= P.C) { return; }
  let b = t / P.tilesPerBoard;
  let r = t % P.tilesPerBoard;
  let y0 = i32(r / P.tilesX) * 4 - 1;
  let x0 = i32(r % P.tilesX) * 4 - 1;
  let base = (b * P.C + c) * P.xSize * P.ySize;
  let fused = (P.flags & 1u) != 0u;
  var scale = 1.0;
  var shift = 0.0;
  var preBias = 0.0;
  if (fused) {
    scale = sb[c];
    shift = sb[P.C + c];
    if ((P.flags & 2u) != 0u) { preBias = pre[b * P.C + c]; }
  }
  var d: array<f32, 36>;
  for (var i = 0; i < 6; i++) {
    for (var j = 0; j < 6; j++) {
      let y = y0 + i;
      let x = x0 + j;
      var v = 0.0;
      if (y >= 0 && y < i32(P.ySize) && x >= 0 && x < i32(P.xSize)) {
        let pos = u32(y) * P.xSize + u32(x);
        v = f32(inp[base + pos]);
        if (fused) { v = select(0.0, activate((v + preBias) * scale + shift, P.act), mask[b * P.maskStride + pos] == 1.0); }
      }
      d[i * 6 + j] = v;
    }
  }
  var o: array<f32, 36>;
${sandwich(BT, 'd', 6, 't', 'o')}
  for (var e = 0u; e < 36u; e++) { V[(e * P.C + c) * P.tileStride + t] = ${vT}(o[e]); }
}
`;

/** 36 GEMMs, one per Winograd element: M[e][oc][t] = Σ_ic U[e][ic][oc] · V[e][ic][t]. Tiled like CONV. */
export const WINO_GEMM = (uT: Elem, vT: Elem, mT: Elem) => /* wgsl */ `${enableFor(uT, vT, mT)}
${DYN}
struct Params { inC: u32, outC: u32, tilesPerBoard: u32, tileStride: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> U: array<${uT}>;
@group(0) @binding(3) var<storage, read> V: array<${vT}>;
@group(0) @binding(4) var<storage, read_write> M: array<${mT}>;

const TM = 64u;
const TN = 64u;
const TK = 16u;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let N = dyn.n * P.tilesPerBoard;
  let e = wg.z;
  let m0 = wg.y * TM;
  let n0 = wg.x * TN;
  let tid = l.y * 16u + l.x;
  let uBase = e * P.inC * P.outC;
  let vBase = e * P.inC * P.tileStride;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
  for (var k0 = 0u; k0 < P.inC; k0 += TK) {
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      let k = k0 + idx / 64u;
      let col = idx % 64u;
      var a = 0.0;
      if (k < P.inC && m0 + col < P.outC) { a = f32(U[uBase + k * P.outC + m0 + col]); }
      As[idx] = a;
      var v = 0.0;
      if (k < P.inC && n0 + col < N) { v = f32(V[vBase + k * P.tileStride + n0 + col]); }
      Bs[idx] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < TK; kk++) {
      var av: array<f32, 4>;
      var bv: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) { av[i] = As[kk * TM + i * 16u + l.y]; }
      for (var j = 0u; j < 4u; j++) { bv[j] = Bs[kk * TN + j * 16u + l.x]; }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] += av[i] * bv[j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + i * 16u + l.y;
    if (m >= P.outC) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let t = n0 + j * 16u + l.x;
      if (t < N) { M[(e * P.outC + m) * P.tileStride + t] = ${mT}(acc[i * 4u + j]); }
    }
  }
}
`;

/**
 * Output transform: one thread per (tile, output channel), Y = Aᵀ m A, written to the board with CONV's flags
 * (bit 0 accumulate, bit 1 per-(batch, channel) bias).
 */
export const WINO_OUT = (mT: Elem, outT: Elem) => /* wgsl */ `${enableFor(mT, outT)}
${DYN}
struct Params { outC: u32, xSize: u32, ySize: u32, tilesX: u32, tilesPerBoard: u32, tileStride: u32, flags: u32, _pad0: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> M: array<${mT}>;
@group(0) @binding(3) var<storage, read_write> outp: array<${outT}>;
@group(0) @binding(4) var<storage, read> nbias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let c = gid.y;
  if (t >= dyn.n * P.tilesPerBoard || c >= P.outC) { return; }
  var m: array<f32, 36>;
  for (var e = 0u; e < 36u; e++) { m[e] = f32(M[(e * P.outC + c) * P.tileStride + t]); }
  var y: array<f32, 16>;
${sandwich(AT, 'm', 6, 't', 'y')}
  let b = t / P.tilesPerBoard;
  let r = t % P.tilesPerBoard;
  let y0 = (r / P.tilesX) * 4u;
  let x0 = (r % P.tilesX) * 4u;
  var bias = 0.0;
  if ((P.flags & 2u) != 0u) { bias = nbias[b * P.outC + c]; }
  let base = (b * P.outC + c) * P.xSize * P.ySize;
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let yy = y0 + i;
      let xx = x0 + j;
      if (yy < P.ySize && xx < P.xSize) {
        let o = base + yy * P.xSize + xx;
        var v = y[i * 4u + j] + bias;
        if ((P.flags & 1u) != 0u) { v += f32(outp[o]); }
        outp[o] = ${outT}(v);
      }
    }
  }
}
`;

// ------------------------------------------------------------------ GEMM v2
//
// A generic batched matrix multiply, C[z][m][n] = Σ_k A[z][k][m] · B[z][k][n], used for the Winograd GEMMs (z = the
// 36 elements) and for 1×1 convolutions (z = the position in the batch; A = weights [ic][oc], B = input [ic][point],
// C = output [oc][point]). Each workgroup of 16 × 16 threads computes a GEMM_TILE_M × GEMM_TILE_N tile of C, each
// thread a 4 × 8 block, reading GEMM_TILE_K of the reduction at a time through workgroup memory.
//
// The inner product is fully unrolled by this generator and the accumulators are scalars, so no index in it is a
// run-time value except the thread's own offsets into workgroup memory. (WebKit's WGSL compiler clamps every array
// index that is not a compile-time constant: Source/WebGPU/WGSL/BoundsCheck.cpp.)

export const GEMM_TILE_M = 64;
export const GEMM_TILE_N = 128;
const GEMM_TILE_K = 16;
const GEMM_RM = 4; // rows per thread: m = ty + i·16
const GEMM_RN = 8; // columns per thread: n = tx + j·16

/**
 * GEMM v2. Params: M, K; N is `nFixed` when non-zero, else dyn.n · nPerBatch; z is the workgroup's z index (the
 * dispatch covers exactly the batches wanted). Strides are in elements. flags bit 0: add to C (the residual add);
 * bit 1: add `nbias[z · nbiasStride + m]` (per batch position and output channel, for 1×1 convolutions).
 */
export const GEMM_V2 = (aT: Elem, bT: Elem, cT: Elem) => {
  const TM = GEMM_TILE_M;
  const TN = GEMM_TILE_N;
  const TK = GEMM_TILE_K;
  let decl = '';
  for (let i = 0; i < GEMM_RM; i++) for (let j = 0; j < GEMM_RN; j++) decl += `  var c${i}_${j} = 0.0;\n`;
  let inner = '';
  // One block per reduction step, so only that step's 4 + 8 operands are live at a time.
  for (let kk = 0; kk < TK; kk++) {
    inner += '    {\n';
    for (let i = 0; i < GEMM_RM; i++) inner += `      let a${i} = As[${kk * TM + i * 16}u + ty];\n`;
    for (let j = 0; j < GEMM_RN; j++) inner += `      let b${j} = Bs[${kk * TN + j * 16}u + tx];\n`;
    for (let i = 0; i < GEMM_RM; i++) for (let j = 0; j < GEMM_RN; j++) inner += `      c${i}_${j} = fma(a${i}, b${j}, c${i}_${j});\n`;
    inner += '    }\n';
  }
  let store = '';
  for (let i = 0; i < GEMM_RM; i++) {
    store += `  { let m = m0 + ty + ${i * 16}u;\n    if (m < P.M) {\n`;
    store += `      var bias = 0.0;\n      if ((P.flags & 2u) != 0u) { bias = nbias[z * P.nbiasStride + m]; }\n`;
    for (let j = 0; j < GEMM_RN; j++) {
      store += `      { let n = n0 + tx + ${j * 16}u;\n        if (n < N) {\n`;
      store += `          let o = z * P.cBatch + m * P.cRow + n;\n          var v = c${i}_${j} + bias;\n`;
      store += `          if ((P.flags & 1u) != 0u) { v += f32(C[o]); }\n          C[o] = ${cT}(v);\n        }\n      }\n`;
    }
    store += `    }\n  }\n`;
  }
  return /* wgsl */ `${enableFor(aT, bT, cT)}
${DYN}
struct Params {
  M: u32, K: u32, nFixed: u32, nPerBatch: u32,
  aBatch: u32, aRow: u32, bBatch: u32, bRow: u32,
  cBatch: u32, cRow: u32, flags: u32, nbiasStride: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> A: array<${aT}>;
@group(0) @binding(3) var<storage, read> B: array<${bT}>;
@group(0) @binding(4) var<storage, read_write> C: array<${cT}>;
@group(0) @binding(5) var<storage, read> nbias: array<f32>;

var<workgroup> As: array<f32, ${TK * TM}>; // [TK][TM]
var<workgroup> Bs: array<f32, ${TK * TN}>; // [TK][TN]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let tx = l.x;
  let ty = l.y;
  let tid = ty * 16u + tx;
  let z = wg.z;
  let m0 = wg.y * ${TM}u;
  let n0 = wg.x * ${TN}u;
  var N = P.nFixed;
  if (N == 0u) { N = dyn.n * P.nPerBatch; }
  let aBase = z * P.aBatch;
  let bBase = z * P.bBatch;
${decl}
  for (var k0 = 0u; k0 < P.K; k0 += ${TK}u) {
    for (var r = 0u; r < ${(TK * TM) / 256}u; r++) {
      let idx = tid + r * 256u;
      let k = k0 + idx / ${TM}u;
      let m = m0 + idx % ${TM}u;
      var a = 0.0;
      if (k < P.K && m < P.M) { a = f32(A[aBase + k * P.aRow + m]); }
      As[idx] = a;
    }
    for (var r = 0u; r < ${(TK * TN) / 256}u; r++) {
      let idx = tid + r * 256u;
      let k = k0 + idx / ${TN}u;
      let n = n0 + idx % ${TN}u;
      var b = 0.0;
      if (k < P.K && n < N) { b = f32(B[bBase + k * P.bRow + n]); }
      Bs[idx] = b;
    }
    workgroupBarrier();
${inner}    workgroupBarrier();
  }
${store}}
`;
};
