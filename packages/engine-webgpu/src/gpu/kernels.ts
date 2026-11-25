/**
 * WGSL kernels for the network, fp32. Tensors are [batch][channel][y][x], with the board's points contiguous.
 *
 * These are the first, correctness-first kernels: an implicit-GEMM convolution tiled in workgroup memory, and simple
 * element-wise, pooling and matrix-vector kernels. Winograd F(4×4,3×3), fused epilogues and fp16 come after them,
 * checked against these and against the reference forward pass.
 *
 * Every kernel reads the batch size from `dyn.n`, so one set of bind groups serves every batch size up to the plan's.
 */

/** Activation codes, shared with the plan builder. */
export const ACT = { identity: 0, relu: 1, mish: 2, silu: 3 } as const;

const ACTIVATE = /* wgsl */ `
fn activate(x: f32, act: u32) -> f32 {
  switch act {
    case 1u: { return max(x, 0.0); }
    case 2u: {
      // KataGo's mish: x * tanh(softplus(x)), with softplus = log1p(exp(min(x, 20))) + max(x, 20) - 20.
      // tanh is evaluated on min(sp, 20): tanh(20) is 1 in f32, and large arguments overflow on some drivers.
      let sp = log(1.0 + exp(min(x, 20.0))) + max(x, 20.0) - 20.0;
      return x * tanh(min(sp, 20.0));
    }
    case 3u: { return x / (exp(-x) + 1.0); }
    default: { return x; }
  }
}
`;

const DYN = /* wgsl */ `struct Dyn { n: u32 }`;

/** Convolution tile sizes: 64 output channels × 64 board points per workgroup, 16 of the reduction at a time. */
export const CONV_TILE_M = 64;
export const CONV_TILE_N = 64;

/**
 * Convolution as an implicit GEMM: out[b][oc][p] = Σ_k w[k][oc] · in[b][ic(k)][p + offset(k)], k = (ic, ky, kx).
 * Zero padding. flags bit 0: add to the output instead of overwriting it (the residual add); bit 1: add a per-(batch,
 * output channel) bias, `nbias[b][oc]`.
 */
export const CONV = /* wgsl */ `
${DYN}
struct Params {
  inC: u32, outC: u32, kh: u32, kw: u32,
  dilY: u32, dilX: u32, xSize: u32, ySize: u32,
  K: u32, flags: u32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
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
      if (k < P.K && m < P.outC) { a = w[k * P.outC + m]; }
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
          v = inp[(b * P.inC + ic) * area + u32(yy) * P.xSize + u32(xx)];
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
      if ((P.flags & 1u) != 0u) { v += outp[o]; }
      outp[o] = v;
    }
  }
}
`;

/**
 * Batch norm, activation, then the board mask: out = mask ? act(x · scale + bias) : 0. flags bit 0: first add a
 * per-(batch, channel) bias, `pre[b][c]` (the global-pooling bias, added before the batch norm as KataGo does).
 * `sb` holds the merged scales, then the merged biases. The mask is channel 0 of the network's input.
 */
export const BN_ACT = /* wgsl */ `
${DYN}
${ACTIVATE}
struct Params { C: u32, area: u32, flags: u32, act: u32, maskStride: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read> sb: array<f32>;
@group(0) @binding(4) var<storage, read_write> outp: array<f32>;
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
  var x = inp[idx];
  if ((P.flags & 1u) != 0u) { x += pre[b * P.C + c]; }
  let y = activate(x * sb[c] + sb[P.C + c], P.act);
  outp[idx] = select(0.0, y, mask[b * P.maskStride + pos] == 1.0);
}
`;

/**
 * Global pooling, one workgroup per (channel, batch). Writes out[b][3C]: [means, means · (√n − 14)/10, third], where
 * n counts the on-board points and the third is the max over them (mode 0) or the mean · ((√n − 14)²/100 − 0.1) the
 * value head uses (mode 1). Inputs are already masked to 0 off the board.
 */
export const GPOOL = /* wgsl */ `
${DYN}
struct Params { C: u32, area: u32, mode: u32, maskStride: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
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
    let x = inp[(b * P.C + c) * P.area + i];
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

/** Input transform: one thread per (tile, channel). V[e][c][t] = (Bᵀ d B)[e]. */
export const WINO_IN = /* wgsl */ `
${DYN}
struct Params { C: u32, xSize: u32, ySize: u32, tilesX: u32, tilesPerBoard: u32, tileStride: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;

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
  var d: array<f32, 36>;
  for (var i = 0; i < 6; i++) {
    for (var j = 0; j < 6; j++) {
      let y = y0 + i;
      let x = x0 + j;
      var v = 0.0;
      if (y >= 0 && y < i32(P.ySize) && x >= 0 && x < i32(P.xSize)) { v = inp[base + u32(y) * P.xSize + u32(x)]; }
      d[i * 6 + j] = v;
    }
  }
  var o: array<f32, 36>;
${sandwich(BT, 'd', 6, 't', 'o')}
  for (var e = 0u; e < 36u; e++) { V[(e * P.C + c) * P.tileStride + t] = o[e]; }
}
`;

/** 36 GEMMs, one per Winograd element: M[e][oc][t] = Σ_ic U[e][ic][oc] · V[e][ic][t]. Tiled like CONV. */
export const WINO_GEMM = /* wgsl */ `
${DYN}
struct Params { inC: u32, outC: u32, tilesPerBoard: u32, tileStride: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> M: array<f32>;

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
      if (k < P.inC && m0 + col < P.outC) { a = U[uBase + k * P.outC + m0 + col]; }
      As[idx] = a;
      var v = 0.0;
      if (k < P.inC && n0 + col < N) { v = V[vBase + k * P.tileStride + n0 + col]; }
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
      if (t < N) { M[(e * P.outC + m) * P.tileStride + t] = acc[i * 4u + j]; }
    }
  }
}
`;

/**
 * Output transform: one thread per (tile, output channel), Y = Aᵀ m A, written to the board with CONV's flags
 * (bit 0 accumulate, bit 1 per-(batch, channel) bias).
 */
export const WINO_OUT = /* wgsl */ `
${DYN}
struct Params { outC: u32, xSize: u32, ySize: u32, tilesX: u32, tilesPerBoard: u32, tileStride: u32, flags: u32, _pad0: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@group(0) @binding(2) var<storage, read> M: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@group(0) @binding(4) var<storage, read> nbias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let c = gid.y;
  if (t >= dyn.n * P.tilesPerBoard || c >= P.outC) { return; }
  var m: array<f32, 36>;
  for (var e = 0u; e < 36u; e++) { m[e] = M[(e * P.outC + c) * P.tileStride + t]; }
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
        if ((P.flags & 1u) != 0u) { v += outp[o]; }
        outp[o] = v;
      }
    }
  }
}
`;
