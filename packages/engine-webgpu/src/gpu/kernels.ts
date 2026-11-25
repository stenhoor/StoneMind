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
