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
// C = output [oc][point]). Each workgroup of 16 × 16 threads computes a (16·rm) × (16·rn) tile of C, each thread an
// rm × rn block (rows ty + 16i, columns tx + 16j), reading tk of the reduction at a time through workgroup memory.
//
// The inner product is fully unrolled by this generator and the accumulators are scalars, so no index in it is a
// run-time value except the thread's own offsets into workgroup memory. (WebKit's WGSL compiler clamps every array
// index that is not a compile-time constant: Source/WebGPU/WGSL/BoundsCheck.cpp.)
//
// arith 'f16' (needs shader-f16): workgroup memory and the multiply-adds are f16; each tk-long chunk is summed in f16
// accumulators, which are then added into f32 accumulators, so rounding error grows with tk, not with the whole K.

/** GEMM v2's shape: rows and columns per thread, the reduction chunk, and the arithmetic type. */
export interface GemmConfig {
  rm: 4 | 8;
  rn: 4 | 8;
  tk: 8 | 16;
  /** 'f16' and 'f16x2' need shader-f16. 'f16x2' works on pairs of output columns as vec2<f16>. */
  arith: 'f32' | 'f16' | 'f16x2';
}

/** The default: 4 × 8 per thread, chunks of 16, f32 arithmetic (as measured in issue #33). */
export const GEMM_DEFAULT: GemmConfig = { rm: 4, rn: 8, tk: 16, arith: 'f32' };

/**
 * The GEMM v2 shape to use on a device, from measurements on b18 (fp16 storage, Winograd fused, batch 64):
 * - Apple GPUs (Safari on the owner's iPad, docs/spikes/results/2026-11-b18-webgpu-ipad.md): 4 × 4, chunk 8 was fastest in both
 *   arithmetic types, with f16 arithmetic ahead of f32 (124.8 against 116.6 evaluations/s); `GEMM_DEFAULT` gave 112.9.
 * - Everything else keeps `GEMM_DEFAULT`: on the owner's RTX 3080 Laptop GPU in Edge (issues #38 and #41) the four
 *   fastest shapes, the default among them, were within 1.4% of each other, and f16 arithmetic was slower.
 * f16 arithmetic is only chosen for fp16 storage: an f32 network keeps f32 arithmetic, so it stays within the f32
 * goldens tolerances (f16 arithmetic under f32 storage failed them on the iPad).
 * @param vendor The adapter's reported vendor (`GPUAdapterInfo.vendor`), if any.
 * @param precision The network's storage precision ('f16' needs the device feature `shader-f16`).
 */
export function defaultGemmConfig(vendor: string | undefined, precision: Elem): GemmConfig {
  if (vendor === 'apple') return { rm: 4, rn: 4, tk: 8, arith: precision };
  return GEMM_DEFAULT;
}

/** The tile of C a workgroup computes. */
export const gemmTile = (c: GemmConfig) => ({ tm: 16 * c.rm, tn: 16 * c.rn });

/**
 * GEMM v2. Params: M, K; N is `nFixed` when non-zero, else dyn.n · nPerBatch; z is the workgroup's z index (the
 * dispatch covers exactly the batches wanted). Strides are in elements. flags bit 0: add to C (the residual add);
 * bit 1: add `nbias[z · nbiasStride + m]` (per batch position and output channel, for 1×1 convolutions).
 */
export const GEMM_V2 = (aT: Elem, bT: Elem, cT: Elem, config: GemmConfig = GEMM_DEFAULT) => {
  const { rm, rn, tk: TK, arith } = config;
  const { tm: TM, tn: TN } = gemmTile(config);
  /** The element type of workgroup memory and of the multiply-adds. */
  const W: Elem = arith === 'f32' ? 'f32' : 'f16';
  let decl = '';
  for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) decl += `  var c${i}_${j} = 0.0;\n`;
  let inner = '';
  if (arith === 'f16') for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) inner += `    var h${i}_${j} = f16(0.0);\n`;
  // f16x2: one vec2<f16> accumulator per pair of columns (j = 2p, 2p + 1).
  if (arith === 'f16x2') for (let i = 0; i < rm; i++) for (let p = 0; p < rn / 2; p++) inner += `    var h${i}_${p} = vec2<f16>(0.0);\n`;
  // One block per reduction step, so only that step's operands are live at a time.
  for (let kk = 0; kk < TK; kk++) {
    inner += '    {\n';
    for (let i = 0; i < rm; i++) inner += `      let a${i} = As[${kk * TM + i * 16}u + ty];\n`;
    for (let j = 0; j < rn; j++) inner += `      let b${j} = Bs[${kk * TN + j * 16}u + tx];\n`;
    if (arith === 'f16x2') {
      for (let p = 0; p < rn / 2; p++) inner += `      let bp${p} = vec2<f16>(b${2 * p}, b${2 * p + 1});\n`;
      for (let i = 0; i < rm; i++) for (let p = 0; p < rn / 2; p++) inner += `      h${i}_${p} = fma(vec2<f16>(a${i}), bp${p}, h${i}_${p});\n`;
    } else {
      const acc = arith === 'f16' ? 'h' : 'c';
      for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) inner += `      ${acc}${i}_${j} = fma(a${i}, b${j}, ${acc}${i}_${j});\n`;
    }
    inner += '    }\n';
  }
  if (arith === 'f16') for (let i = 0; i < rm; i++) for (let j = 0; j < rn; j++) inner += `    c${i}_${j} += f32(h${i}_${j});\n`;
  if (arith === 'f16x2') {
    for (let i = 0; i < rm; i++)
      for (let p = 0; p < rn / 2; p++) inner += `    c${i}_${2 * p} += f32(h${i}_${p}.x);\n    c${i}_${2 * p + 1} += f32(h${i}_${p}.y);\n`;
  }
  let store = '';
  for (let i = 0; i < rm; i++) {
    store += `  { let m = m0 + ty + ${i * 16}u;\n    if (m < P.M) {\n`;
    store += `      var bias = 0.0;\n      if ((P.flags & 2u) != 0u) { bias = nbias[z * P.nbiasStride + m]; }\n`;
    for (let j = 0; j < rn; j++) {
      store += `      { let n = n0 + tx + ${j * 16}u;\n        if (n < N) {\n`;
      store += `          let o = z * P.cBatch + m * P.cRow + n;\n          var v = c${i}_${j} + bias;\n`;
      store += `          if ((P.flags & 1u) != 0u) { v += f32(C[o]); }\n          C[o] = ${cT}(v);\n        }\n      }\n`;
    }
    store += `    }\n  }\n`;
  }
  return /* wgsl */ `${enableFor(aT, bT, cT, W)}
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

var<workgroup> As: array<${W}, ${TK * TM}>; // [TK][TM]
var<workgroup> Bs: array<${W}, ${TK * TN}>; // [TK][TN]

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
      var a = ${W}(0.0);
      if (k < P.K && m < P.M) { a = ${W}(A[aBase + k * P.aRow + m]); }
      As[idx] = a;
    }
    for (var r = 0u; r < ${(TK * TN) / 256}u; r++) {
      let idx = tid + r * 256u;
      let k = k0 + idx / ${TN}u;
      let n = n0 + idx % ${TN}u;
      var b = ${W}(0.0);
      if (k < P.K && n < N) { b = ${W}(B[bBase + k * P.bRow + n]); }
      Bs[idx] = b;
    }
    workgroupBarrier();
${inner}    workgroupBarrier();
  }
${store}}
`;
};

// ------------------------------------------------------------------ GEMM v3
//
// GEMM v2's tiles, parameters and dispatch, with three changes aimed at the time spent in workgroup memory and waiting
// on loads:
// - Each thread owns runs of 4 adjacent rows and 4 adjacent columns (rows 64g + 4ty + i, columns 64h + 4tx + j), so
//   it reads its operands from workgroup memory as vec4s: rm / 4 + rn / 4 reads per reduction step, where v2 makes
//   rm + rn scalar reads. Accumulators are vec4s, updated with one vector fma per row.
// - Workgroup memory holds vec4<f32> (always f32 arithmetic), filled four values at a time.
// - The next chunk of A and B is read from global memory into registers while the current chunk is multiplied, and
//   only then stored to workgroup memory.
// Each output is the same sequence of fma steps over k as v2's f32 arithmetic, so the two give the same results.

/** GEMM v3's shape: rows and columns per thread (multiples of 4) and the reduction chunk. */
export interface GemmV3Config {
  rm: 4 | 8;
  rn: 4 | 8;
  tk: 8 | 16 | 32;
  /** Two buffers of workgroup memory, alternating, so each chunk needs one barrier (default false). */
  double?: boolean;
  /** Threads along M (default 16): the workgroup is 16 × wy threads and its tile wy·rm × 16·rn. */
  wy?: 8 | 16;
}

/** GEMM v3's tile: rows (M) and columns (N) of C per workgroup. */
export const gemmV3Tile = (c: GemmV3Config) => ({ tm: (c.wy ?? 16) * c.rm, tn: 16 * c.rn });

/**
 * The GEMM kernel to use on a device: 'v3' everywhere except Apple GPUs, which keep 'v2' with the shape measured
 * fastest there (`defaultGemmConfig`), since v3 hasn't been measured on them. On the owner's RTX 3080 Laptop GPU in
 * Edge, v3 (4 × 8, chunks of 16) took 15.9, 25.6 and 49.3 ms against v2's 23.8, 71.0 and 90.1 ms for b18's GEMMs at
 * batches 16, 32 and 64 (docs/spikes/results/2026-11-b18-gemm-v3.md).
 * @param vendor The adapter's reported vendor (`GPUAdapterInfo.vendor`), if any.
 */
export function defaultGemm(vendor: string | undefined): 'v2' | 'v3' {
  return vendor === 'apple' ? 'v2' : 'v3';
}

/** GEMM v3's default shape (the same tile as GEMM_DEFAULT). */
export const GEMM_V3_DEFAULT: GemmV3Config = { rm: 4, rn: 8, tk: 16 };

/** GEMM v3: GEMM v2's bindings, parameters and dispatch, with the tile 16·rm × 16·rn. */
export const GEMM_V3 = (aT: Elem, bT: Elem, cT: Elem, config: GemmV3Config = GEMM_V3_DEFAULT) => {
  const { rm, rn, tk: TK } = config;
  const DB = config.double === true;
  const WY = config.wy ?? 16;
  const THREADS = 16 * WY;
  const TM = WY * rm;
  const TN = 16 * rn;
  const GM = rm / 4;
  const GN = rn / 4;
  const NA = (TK * TM) / 4; // vec4s of A per chunk
  const NB = (TK * TN) / 4;
  const LA = Math.ceil(NA / THREADS);
  const LB = Math.ceil(NB / THREADS);
  const comps = ['x', 'y', 'z', 'w'];
  const guard = (r: number, count: number) => ((r + 1) * THREADS > count ? `if (tid + ${r * THREADS}u < ${count}u) ` : '');
  let decl = '';
  for (let r = 0; r < LA; r++) decl += `  var ra${r} = vec4<f32>(0.0);\n`;
  for (let r = 0; r < LB; r++) decl += `  var rb${r} = vec4<f32>(0.0);\n`;
  for (let i = 0; i < rm; i++) for (let h = 0; h < GN; h++) decl += `  var c${i}_${h} = vec4<f32>(0.0);\n`;
  // Read the chunk starting at k into registers: A is [K][M] (rows of M), B is [K][N].
  const loadOne = (reg: string, r: number, count: number, perRow: number, k: string, base: string, row: string, pos: string, limit: string, src: string) => {
    let s = `    ${guard(r, count)}{\n      let idx = tid + ${r * THREADS}u;\n      let k = ${k} + idx / ${perRow}u;\n`;
    s += `      let ${pos} = ${pos}0 + (idx % ${perRow}u) * 4u;\n      var v = vec4<f32>(0.0);\n`;
    s += `      if (k < P.K) {\n        let o = ${base} + k * ${row} + ${pos};\n`;
    for (let j = 0; j < 4; j++) s += `        if (${pos} + ${j}u < ${limit}) { v.${comps[j]} = f32(${src}[o + ${j}u]); }\n`;
    s += `      }\n      ${reg}${r} = v;\n    }\n`;
    return s;
  };
  const load = (k: string) => {
    let s = '';
    for (let r = 0; r < LA; r++) s += loadOne('ra', r, NA, TM / 4, k, 'aBase', 'P.aRow', 'm', 'P.M', 'A');
    for (let r = 0; r < LB; r++) s += loadOne('rb', r, NB, TN / 4, k, 'bBase', 'P.bRow', 'n', 'N', 'B');
    return s;
  };
  // Store the registers into workgroup memory, at offsets `ao` and `bo` (the buffer, when double-buffered).
  const putAt = (ao: string, bo: string) => {
    let s = '';
    for (let r = 0; r < LA; r++) s += `    ${guard(r, NA)}{ As[${ao}tid + ${r * THREADS}u] = ra${r}; }\n`;
    for (let r = 0; r < LB; r++) s += `    ${guard(r, NB)}{ Bs[${bo}tid + ${r * THREADS}u] = rb${r}; }\n`;
    return s;
  };
  const put = putAt('', '');
  let inner = '';
  for (let kk = 0; kk < TK; kk++) {
    inner += '    {\n';
    for (let g = 0; g < GM; g++) inner += `      let a${g} = As[${DB ? 'ab + ' : ''}${kk * (TM / 4) + g * WY}u + ty];\n`;
    for (let h = 0; h < GN; h++) inner += `      let b${h} = Bs[${DB ? 'bb + ' : ''}${kk * (TN / 4) + h * 16}u + tx];\n`;
    for (let g = 0; g < GM; g++)
      for (let i = 0; i < 4; i++)
        for (let h = 0; h < GN; h++) inner += `      c${g * 4 + i}_${h} = fma(vec4<f32>(a${g}.${comps[i]}), b${h}, c${g * 4 + i}_${h});\n`;
    inner += '    }\n';
  }
  let store = '';
  for (let g = 0; g < GM; g++)
    for (let i = 0; i < 4; i++) {
      store += `  { let m = m0 + ${g * 4 * WY + i}u + 4u * ty;\n    if (m < P.M) {\n`;
      store += `      var bias = 0.0;\n      if ((P.flags & 2u) != 0u) { bias = nbias[z * P.nbiasStride + m]; }\n`;
      for (let h = 0; h < GN; h++)
        for (let j = 0; j < 4; j++) {
          store += `      { let n = n0 + ${h * 64 + j}u + 4u * tx;\n        if (n < N) {\n`;
          store += `          let o = z * P.cBatch + m * P.cRow + n;\n          var v = c${g * 4 + i}_${h}.${comps[j]} + bias;\n`;
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

var<workgroup> As: array<vec4<f32>, ${DB ? 2 * NA : NA}>; // [${DB ? '2][' : ''}TK][TM / 4]
var<workgroup> Bs: array<vec4<f32>, ${DB ? 2 * NB : NB}>; // [${DB ? '2][' : ''}TK][TN / 4]

@compute @workgroup_size(16, ${WY})
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
${load('0u')}${put}  workgroupBarrier();
${DB ? `  var cur = 0u;
  for (var k0 = 0u; k0 < P.K; k0 += ${TK}u) {
    let more = k0 + ${TK}u < P.K;
    if (more) {
${load(`k0 + ${TK}u`)}    }
    let ab = cur * ${NA}u;
    let bb = cur * ${NB}u;
${inner}    if (more) {
      let an = (1u - cur) * ${NA}u;
      let bn = (1u - cur) * ${NB}u;
${putAt('an + ', 'bn + ')}    }
    workgroupBarrier();
    cur = 1u - cur;
  }` : `  for (var k0 = 0u; k0 < P.K; k0 += ${TK}u) {
    let more = k0 + ${TK}u < P.K;
    if (more) {
${load(`k0 + ${TK}u`)}    }
${inner}    workgroupBarrier();
    if (more) {
${put}    }
    workgroupBarrier();
  }`}
${store}}
`;
};

// ------------------------------------------------------------------ Winograd transforms v2
//
// The same transforms as WINO_IN / WINO_OUT, staged through workgroup memory. A workgroup handles `cpw` channels of
// one board position: the input transform first copies the padded board into workgroup memory with adjacent reads
// (applying a fused batch norm and activation once per point, where v1 recomputes it for every tile that overlaps the
// point), then each thread builds its tile from there; the output transform has each thread put its 4 × 4 block into
// workgroup memory, then the workgroup writes the board with adjacent writes (and the residual add). Board sizes are
// compile-time constants, and every tile index is unrolled.

/** Channels per workgroup for the v2 transforms: as many as fit in 256 threads and 16 KB of workgroup memory. */
export function winoTransformChannels(kind: 'in' | 'out', xSize: number, ySize: number): number {
  const tilesX = Math.ceil(xSize / 4);
  const tilesY = Math.ceil(ySize / 4);
  const perChannel = kind === 'in' ? (tilesX * 4 + 2) * (tilesY * 4 + 2) : tilesX * 4 * (tilesY * 4);
  return Math.max(1, Math.min(Math.floor(256 / (tilesX * tilesY)), Math.floor(4096 / perChannel)));
}

/** WINO_IN with workgroup staging; same bindings and parameters. Dispatch: [ceil(C / cpw), n]. */
export const WINO_IN_V2 = (inT: Elem, vT: Elem, xSize: number, ySize: number) => {
  const TX = Math.ceil(xSize / 4);
  const TPB = TX * Math.ceil(ySize / 4);
  const PX = TX * 4 + 2;
  const PY = Math.ceil(ySize / 4) * 4 + 2;
  const CPW = winoTransformChannels('in', xSize, ySize);
  const WG = CPW * TPB;
  let gather = '';
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) gather += `  d[${i * 6 + j}] = S[sbase + ${i * PX + j}u];\n`;
  let store = '';
  for (let e = 0; e < 36; e++) store += `  V[(${e}u * P.C + c) * P.tileStride + t] = ${vT}(o[${e}]);\n`;
  return /* wgsl */ `${enableFor(inT, vT)}
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

var<workgroup> S: array<f32, ${CPW * PY * PX}>; // [cpw][${PY}][${PX}], the board with a 1-point border of zeros

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c0 = wg.x * ${CPW}u;
  let b = wg.y;
  if (b >= dyn.n) { return; }
  let fused = (P.flags & 1u) != 0u;
  for (var idx = li; idx < ${CPW * PY * PX}u; idx += ${WG}u) {
    let cc = idx / ${PY * PX}u;
    let rem = idx % ${PY * PX}u;
    let y = i32(rem / ${PX}u) - 1;
    let x = i32(rem % ${PX}u) - 1;
    let c = c0 + cc;
    var v = 0.0;
    if (c < P.C && y >= 0 && y < ${ySize} && x >= 0 && x < ${xSize}) {
      let pos = u32(y) * ${xSize}u + u32(x);
      v = f32(inp[(b * P.C + c) * ${xSize * ySize}u + pos]);
      if (fused) {
        var preBias = 0.0;
        if ((P.flags & 2u) != 0u) { preBias = pre[b * P.C + c]; }
        v = select(0.0, activate((v + preBias) * sb[c] + sb[P.C + c], P.act), mask[b * P.maskStride + pos] == 1.0);
      }
    }
    S[idx] = v;
  }
  workgroupBarrier();
  let cc = li / ${TPB}u;
  let r = li % ${TPB}u;
  let c = c0 + cc;
  if (c >= P.C) { return; }
  let sbase = cc * ${PY * PX}u + (r / ${TX}u) * ${4 * PX}u + (r % ${TX}u) * 4u;
  var d: array<f32, 36>;
${gather}  var o: array<f32, 36>;
${sandwich(BT, 'd', 6, 'u', 'o')}
  let t = b * ${TPB}u + r;
${store}}
`;
};

/** WINO_OUT with workgroup staging; same bindings and parameters. Dispatch: [ceil(outC / cpw), n]. */
export const WINO_OUT_V2 = (mT: Elem, outT: Elem, xSize: number, ySize: number) => {
  const TX = Math.ceil(xSize / 4);
  const TPB = TX * Math.ceil(ySize / 4);
  const OX = TX * 4;
  const OY = Math.ceil(ySize / 4) * 4;
  const CPW = winoTransformChannels('out', xSize, ySize);
  const WG = CPW * TPB;
  let load = '';
  for (let e = 0; e < 36; e++) load += `    m[${e}] = f32(M[(${e}u * P.outC + c) * P.tileStride + t]);\n`;
  let put = '';
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) put += `    S[sbase + ${i * OX + j}u] = y[${i * 4 + j}];\n`;
  return /* wgsl */ `${enableFor(mT, outT)}
${DYN}
struct Params { outC: u32, xSize: u32, ySize: u32, tilesX: u32, tilesPerBoard: u32, tileStride: u32, flags: u32, _pad0: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> M: array<${mT}>;
@group(0) @binding(3) var<storage, read_write> outp: array<${outT}>;
@group(0) @binding(4) var<storage, read> nbias: array<f32>;

var<workgroup> S: array<f32, ${CPW * OY * OX}>; // [cpw][${OY}][${OX}], whole tiles

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c0 = wg.x * ${CPW}u;
  let b = wg.y;
  if (b >= dyn.n) { return; }
  let cc = li / ${TPB}u;
  let r = li % ${TPB}u;
  let c = c0 + cc;
  if (c < P.outC) {
    let t = b * ${TPB}u + r;
    var m: array<f32, 36>;
${load}    var y: array<f32, 16>;
${sandwich(AT, 'm', 6, 'u', 'y')}
    let sbase = cc * ${OY * OX}u + (r / ${TX}u) * ${4 * OX}u + (r % ${TX}u) * 4u;
${put}  }
  workgroupBarrier();
  for (var idx = li; idx < ${CPW * xSize * ySize}u; idx += ${WG}u) {
    let cc2 = idx / ${xSize * ySize}u;
    let pos = idx % ${xSize * ySize}u;
    let c2 = c0 + cc2;
    if (c2 >= P.outC) { continue; }
    var v = S[cc2 * ${OY * OX}u + (pos / ${xSize}u) * ${OX}u + pos % ${xSize}u];
    if ((P.flags & 2u) != 0u) { v += nbias[b * P.outC + c2]; }
    let o = (b * P.outC + c2) * ${xSize * ySize}u + pos;
    if ((P.flags & 1u) != 0u) { v += f32(outp[o]); }
    outp[o] = ${outT}(v);
  }
}
`;
};
