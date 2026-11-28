/**
 * The reference forward pass: plain fp32 TypeScript, slow and simple, used to check the WebGPU kernels.
 *
 * Follows KataGo's Eigen (CPU) backend, `cpp/neuralnet/eigenbackend.cpp` (KataGo 1.18, MIT, © David J Wu):
 * - every batch norm is followed by its activation and then multiplied by the board mask (off-board points become 0),
 * - global pooling gives [mean, mean · (√area − 14) / 10, max over on-board points],
 * - the value head pools [mean, mean · (√area − 14) / 10, mean · ((√area − 14)² / 100 − 0.1)],
 * - convolutions pad with zeros,
 * - transformer blocks (`TransformerAttentionBlock`, `TransformerFFNBlock`): pre-norm RMSNorm (off-board points 0),
 *   then attention over the on-board points or a SwiGLU feed-forward, added to the trunk on the board.
 * Tensors are channel-major: [channel][y][x], one position at a time.
 */
import type {
  Activation,
  BatchNormLayer,
  Block,
  ConvLayer,
  MatBiasLayer,
  MatMulLayer,
  ModelDesc,
  TransformerAttentionBlock,
  TransformerFFNBlock,
  TransformerRMSNorm,
} from '../model/types.js';

/** The network's raw outputs for one position, before post-processing. */
export interface RawOutputs {
  xSize: number;
  ySize: number;
  /** [policyChannel][y * xSize + x]. Channel 0 is the policy; channel 1, where present, the optimistic policy. */
  policy: Float32Array[];
  /** One pass logit per policy channel. */
  pass: Float32Array;
  /** Win, loss, no-result logits. */
  value: Float32Array;
  /** Score values: scoreMean, scoreMeanSq (pre-softplus stdev), lead, varTimeLeft, shortterm errors. */
  scoreValue: Float32Array;
  /** Ownership, pre-tanh, [y * xSize + x]. */
  ownership: Float32Array;
}

export interface NetInput {
  xSize: number;
  ySize: number;
  /** [22][ySize][xSize], channel-major. Channel 0 is the on-board mask. */
  spatial: Float32Array;
  /** 19 global features. */
  global: Float32Array;
}

const f = Math.fround;

function activate(x: number, activation: Activation): number {
  switch (activation) {
    case 'identity':
      return x;
    case 'relu':
      return x > 0 ? x : 0;
    case 'mish': {
      // KataGo: x * tanh(log1p(exp(min(x, 20))) + max(x, 20) - 20)
      const sp = Math.log1p(Math.exp(Math.min(x, 20))) + (Math.max(x, 20) - 20);
      return f(x * Math.tanh(sp));
    }
    case 'silu':
      return f(x / (Math.exp(-x) + 1));
    case 'mish_scale8':
      // KataGo's OpenCL kernels (ACTIVATION == 12): a < 20/8 ? a * tanh(log1p(exp(8a))) : a.
      return x < 2.5 ? f(x * Math.tanh(Math.log1p(Math.exp(8 * x)))) : x;
  }
}

class Context {
  readonly area: number;
  readonly mask: Float32Array;
  readonly maskSum: number;

  constructor(
    readonly xSize: number,
    readonly ySize: number,
    mask: Float32Array,
  ) {
    this.area = xSize * ySize;
    this.mask = mask;
    let s = 0;
    for (const m of mask) s += m;
    this.maskSum = s;
  }

  conv(layer: ConvLayer, input: Float32Array): Float32Array {
    const { xSize, ySize, area } = this;
    const { inChannels, outChannels, convYSize, convXSize, dilationY, dilationX, weights } = layer;
    const out = new Float32Array(outChannels * area);
    const ry = (convYSize - 1) >> 1;
    const rx = (convXSize - 1) >> 1;
    const acc = new Float64Array(area);
    for (let oc = 0; oc < outChannels; oc++) {
      acc.fill(0);
      for (let ic = 0; ic < inChannels; ic++) {
        const src = ic * area;
        for (let ky = 0; ky < convYSize; ky++) {
          const dy = (ky - ry) * dilationY;
          for (let kx = 0; kx < convXSize; kx++) {
            const dx = (kx - rx) * dilationX;
            const w = weights[((oc * inChannels + ic) * convYSize + ky) * convXSize + kx]!;
            if (w === 0) continue;
            const y0 = Math.max(0, -dy);
            const y1 = Math.min(ySize, ySize - dy);
            const x0 = Math.max(0, -dx);
            const x1 = Math.min(xSize, xSize - dx);
            for (let y = y0; y < y1; y++) {
              const row = y * xSize;
              const srow = src + (y + dy) * xSize + dx;
              for (let x = x0; x < x1; x++) acc[row + x]! += w * input[srow + x]!;
            }
          }
        }
      }
      out.set(acc, oc * area);
    }
    return out;
  }

  /** Batch norm, then activation, then the mask. */
  bnAct(bn: BatchNormLayer, activation: Activation, input: Float32Array): Float32Array {
    const { area, mask } = this;
    const out = new Float32Array(input.length);
    for (let c = 0; c < bn.numChannels; c++) {
      const s = bn.mergedScale[c]!;
      const b = bn.mergedBias[c]!;
      for (let i = 0; i < area; i++) {
        out[c * area + i] = mask[i] === 1 ? activate(f(input[c * area + i]! * s + b), activation) : 0;
      }
    }
    return out;
  }

  gpool(input: Float32Array, channels: number): Float32Array {
    const { area, mask, maskSum } = this;
    const out = new Float32Array(channels * 3);
    const sqrtArea = Math.sqrt(maskSum);
    for (let c = 0; c < channels; c++) {
      let s = 0;
      let m = -1;
      for (let i = 0; i < area; i++) {
        const x = input[c * area + i]!;
        s += x;
        m = Math.max(m, x + (mask[i]! - 1));
      }
      const mean = s / maskSum;
      out[c] = mean;
      out[c + channels] = mean * (sqrtArea - 14) * 0.1;
      out[c + 2 * channels] = m;
    }
    return out;
  }

  valuePool(input: Float32Array, channels: number): Float32Array {
    const { area, maskSum } = this;
    const out = new Float32Array(channels * 3);
    const sqrtArea = Math.sqrt(maskSum);
    for (let c = 0; c < channels; c++) {
      let s = 0;
      for (let i = 0; i < area; i++) s += input[c * area + i]!;
      const mean = s / maskSum;
      out[c] = mean;
      out[c + channels] = mean * (sqrtArea - 14) * 0.1;
      out[c + 2 * channels] = mean * ((sqrtArea - 14) * (sqrtArea - 14) * 0.01 - 0.1);
    }
    return out;
  }

  /** Add a per-channel bias to a spatial tensor, in place. */
  addChannelBias(target: Float32Array, bias: Float32Array): void {
    const { area } = this;
    for (let c = 0; c < bias.length; c++) for (let i = 0; i < area; i++) target[c * area + i]! += bias[c]!;
  }

  addInPlace(target: Float32Array, add: Float32Array): void {
    for (let i = 0; i < target.length; i++) target[i]! += add[i]!;
  }

  block(block: Block, trunk: Float32Array): void {
    switch (block.kind) {
      case 'ordinary': {
        const a = this.bnAct(block.preBN, block.preActivation.activation, trunk);
        const mid = this.conv(block.regularConv, a);
        const b = this.bnAct(block.midBN, block.midActivation.activation, mid);
        this.addInPlace(trunk, this.conv(block.finalConv, b));
        return;
      }
      case 'gpool': {
        const a = this.bnAct(block.preBN, block.preActivation.activation, trunk);
        const regular = this.conv(block.regularConv, a);
        const g = this.bnAct(block.gpoolBN, block.gpoolActivation.activation, this.conv(block.gpoolConv, a));
        const pooled = this.gpool(g, block.gpoolConv.outChannels);
        this.addChannelBias(regular, matMul(block.gpoolToBiasMul, pooled));
        const b = this.bnAct(block.midBN, block.midActivation.activation, regular);
        this.addInPlace(trunk, this.conv(block.finalConv, b));
        return;
      }
      case 'nested_bottleneck': {
        const a = this.bnAct(block.preBN, block.preActivation.activation, trunk);
        const inner = this.conv(block.preConv, a);
        for (const b of block.blocks) this.block(b, inner);
        const b = this.bnAct(block.postBN, block.postActivation.activation, inner);
        this.addInPlace(trunk, this.conv(block.postConv, b));
        return;
      }
      case 'transformer_attention':
        this.addMasked(trunk, this.attention(block, this.rmsNorm(block.preLN, trunk)));
        return;
      case 'transformer_ffn':
        this.addMasked(trunk, this.ffn(block, this.rmsNorm(block.preLN, trunk)));
        return;
    }
  }

  /** KataGo `TransformerRMSNormLayer::apply`: x · weight / √(mean(x²) + ε) over the channels at each point; 0 off the board. */
  rmsNorm(layer: TransformerRMSNorm, input: Float32Array): Float32Array {
    const { area, mask } = this;
    const C = layer.numChannels;
    const out = new Float32Array(C * area);
    for (let i = 0; i < area; i++) {
      if (mask[i] === 0) continue;
      let sumSq = 0;
      for (let c = 0; c < C; c++) {
        const v = input[c * area + i]!;
        sumSq += v * v;
      }
      const rms = f(1 / Math.sqrt(f(f(sumSq) / C + layer.epsilon)));
      for (let c = 0; c < C; c++) out[c * area + i] = f(input[c * area + i]! * rms) * layer.weight[c]!;
    }
    return out;
  }

  /** A matmul at every point (KataGo's matmuls over the flattened board): [ic][area] → [oc][area]. */
  pointMatMul(layer: MatMulLayer, input: Float32Array): Float32Array {
    const { area } = this;
    const { inChannels: IC, outChannels: OC, weights } = layer;
    const acc = new Float64Array(OC * area);
    for (let ic = 0; ic < IC; ic++) {
      const src = ic * area;
      for (let oc = 0; oc < OC; oc++) {
        const w = weights[ic * OC + oc]!;
        if (w === 0) continue;
        const dst = oc * area;
        for (let i = 0; i < area; i++) acc[dst + i]! += w * input[src + i]!;
      }
    }
    return Float32Array.from(acc);
  }

  /** The residual add of a transformer block, on the board only. */
  addMasked(trunk: Float32Array, add: Float32Array): void {
    const { area, mask } = this;
    for (let i = 0; i < trunk.length; i++) trunk[i]! += add[i]! * mask[i % area]!;
  }

  /**
   * KataGo `TransformerAttentionDesc::computeRopeCosSin` for learnable RoPE: the rotation angle x · freqX + y · freqY of
   * each pair of each key/value head at each point, as [kvHead · pairs + pair][point].
   */
  ropeTables(block: TransformerAttentionBlock): { cos: Float32Array; sin: Float32Array } {
    const { xSize, ySize, area } = this;
    const tables = block.numKVHeads * (block.qHeadDim / 2);
    const cos = new Float32Array(tables * area);
    const sin = new Float32Array(tables * area);
    for (let t = 0; t < tables; t++) {
      const freqX = block.ropeFreqs[t * 2]!;
      const freqY = block.ropeFreqs[t * 2 + 1]!;
      for (let y = 0; y < ySize; y++) {
        for (let x = 0; x < xSize; x++) {
          const angle = f(f(x * freqX) + f(y * freqY));
          cos[t * area + y * xSize + x] = Math.cos(angle);
          sin[t * area + y * xSize + x] = Math.sin(angle);
        }
      }
    }
    return { cos, sin };
  }

  /** KataGo's RoPE on Q or K, [numBufHeads · qHeadDim][area], in place: each pair (2p, 2p + 1) of a head rotated. */
  applyRope(block: TransformerAttentionBlock, data: Float32Array, numBufHeads: number, tables: { cos: Float32Array; sin: Float32Array }): void {
    const { area } = this;
    const numPairs = block.qHeadDim / 2;
    for (let h = 0; h < numBufHeads; h++) {
      // A query head uses its key/value head's table.
      const table = Math.trunc((h * block.numKVHeads) / numBufHeads) * numPairs;
      for (let p = 0; p < numPairs; p++) {
        const c0 = (h * block.qHeadDim + 2 * p) * area;
        const c1 = c0 + area;
        const t = (table + p) * area;
        for (let i = 0; i < area; i++) {
          const cosV = tables.cos[t + i]!;
          const sinV = tables.sin[t + i]!;
          const x0 = data[c0 + i]!;
          const x1 = data[c1 + i]!;
          data[c0 + i] = f(x0 * cosV) - f(x1 * sinV);
          data[c1 + i] = f(x0 * sinV) + f(x1 * cosV);
        }
      }
    }
  }

  /**
   * KataGo `TransformerAttentionBlock::apply` from its normed input: Q, K and V projections, RoPE, then for each head
   * a softmax over the on-board keys of Q·K / √qHeadDim (a query off the board gets 0), weighting V, and the output
   * projection.
   */
  attention(block: TransformerAttentionBlock, normed: Float32Array): Float32Array {
    const unsupported = unsupportedAttention(block);
    if (unsupported) throw new Error(`${block.name}: ${unsupported}`);
    const { area, mask } = this;
    const { numHeads, numKVHeads, qHeadDim, vHeadDim } = block;
    const q = this.pointMatMul(block.qProj, normed);
    const k = this.pointMatMul(block.kProj, normed);
    const v = this.pointMatMul(block.vProj, normed);
    if (block.useRope) {
      const tables = this.ropeTables(block);
      this.applyRope(block, q, numHeads, tables);
      this.applyRope(block, k, numKVHeads, tables);
    }
    const scale = f(1 / Math.sqrt(qHeadDim));
    const kvGroupSize = numHeads / numKVHeads;
    const out = new Float32Array(numHeads * vHeadDim * area);
    const weights = new Float64Array(area);
    for (let h = 0; h < numHeads; h++) {
      const kvh = Math.trunc(h / kvGroupSize);
      for (let qi = 0; qi < area; qi++) {
        if (mask[qi] === 0) continue;
        let max = -Infinity;
        for (let ki = 0; ki < area; ki++) {
          if (mask[ki] === 0) continue;
          let s = 0;
          for (let d = 0; d < qHeadDim; d++) s += q[(h * qHeadDim + d) * area + qi]! * k[(kvh * qHeadDim + d) * area + ki]!;
          const score = f(f(s) * scale);
          weights[ki] = score;
          if (score > max) max = score;
        }
        let sumExp = 0;
        for (let ki = 0; ki < area; ki++) {
          if (mask[ki] === 0) {
            weights[ki] = 0;
            continue;
          }
          const e = f(Math.exp(weights[ki]! - max));
          weights[ki] = e;
          sumExp += e;
        }
        const invSum = f(1 / f(sumExp));
        for (let d = 0; d < vHeadDim; d++) {
          let s = 0;
          for (let ki = 0; ki < area; ki++) s += f(weights[ki]! * invSum) * v[(kvh * vHeadDim + d) * area + ki]!;
          out[(h * vHeadDim + d) * area + qi] = s;
        }
      }
    }
    return this.pointMatMul(block.outProj, out);
  }

  /** KataGo `TransformerFFNBlock::apply` from its normed input: linear2(silu(linear1(x)) × linearGate(x)). */
  ffn(block: TransformerFFNBlock, normed: Float32Array): Float32Array {
    if (!block.linearGate) throw new Error(`${block.name}: transformer feed-forward blocks without SwiGLU are not supported (as in KataGo's CPU backend)`);
    const a = this.pointMatMul(block.linear1, normed);
    const gate = this.pointMatMul(block.linearGate, normed);
    for (let i = 0; i < a.length; i++) {
      const x = a[i]!;
      a[i] = f(f(x / f(1 + f(Math.exp(-x)))) * gate[i]!);
    }
    return this.pointMatMul(block.linear2, a);
  }
}

/**
 * Why this reference can't run an attention block, or null: the variants KataGo's model format allows but no network
 * here has, so nothing has checked them against native (the tf3-b11c768 network has learnable RoPE and as many
 * key/value heads as query heads).
 */
export function unsupportedAttention(block: TransformerAttentionBlock): string | null {
  if (!block.useRope) return 'attention without RoPE is not supported yet';
  if (!block.learnableRope) return 'fixed RoPE (from a theta) is not supported yet, only learnable RoPE';
  if (block.numKVHeads !== block.numHeads) return 'grouped key/value heads (fewer than the query heads) are not supported yet';
  return null;
}

function matMul(layer: MatMulLayer, input: Float32Array): Float32Array {
  const out = new Float32Array(layer.outChannels);
  for (let oc = 0; oc < layer.outChannels; oc++) {
    let s = 0;
    for (let ic = 0; ic < layer.inChannels; ic++) s += input[ic]! * layer.weights[ic * layer.outChannels + oc]!;
    out[oc] = s;
  }
  return out;
}

function addBias(values: Float32Array, bias: MatBiasLayer): void {
  for (let i = 0; i < values.length; i++) values[i]! += bias.weights[i]!;
}

function activateVector(values: Float32Array, activation: Activation): void {
  for (let i = 0; i < values.length; i++) values[i] = activate(values[i]!, activation);
}

/**
 * Run the network on one position.
 * @throws Error for models this reference doesn't support (SGF metadata input).
 */
export function forward(model: ModelDesc, input: NetInput): RawOutputs {
  const { xSize, ySize } = input;
  const area = xSize * ySize;
  if (input.spatial.length !== model.numInputChannels * area) throw new Error('spatial input has the wrong size');
  if (input.global.length !== model.numInputGlobalChannels) throw new Error('global input has the wrong size');
  if (model.trunk.sgfMetadataEncoder) throw new Error('models with an SGF metadata encoder are not supported yet');
  const ctx = new Context(xSize, ySize, input.spatial.subarray(0, area));

  const t = model.trunk;
  const trunk = ctx.conv(t.initialConv, input.spatial);
  ctx.addChannelBias(trunk, matMul(t.initialMatMul, input.global));
  for (const b of t.blocks) ctx.block(b, trunk);
  const tip = ctx.bnAct(t.trunkTipBN, t.trunkTipActivation.activation, trunk);

  // Policy head.
  const p = model.policyHead;
  const p1 = ctx.conv(p.p1Conv, tip);
  const g1 = ctx.bnAct(p.g1BN, p.g1Activation.activation, ctx.conv(p.g1Conv, tip));
  const g1Pooled = ctx.gpool(g1, p.g1Conv.outChannels);
  ctx.addChannelBias(p1, matMul(p.gpoolToBiasMul, g1Pooled));
  const p1Out = ctx.bnAct(p.p1BN, p.p1Activation.activation, p1);
  const p2 = ctx.conv(p.p2Conv, p1Out);
  const policy: Float32Array[] = [];
  for (let c = 0; c < p.policyOutChannels; c++) policy.push(p2.slice(c * area, (c + 1) * area));
  let pass: Float32Array;
  if (p.gpoolToPassMul2 && p.gpoolToPassBias && p.passActivation) {
    const hidden = matMul(p.gpoolToPassMul, g1Pooled);
    addBias(hidden, p.gpoolToPassBias);
    activateVector(hidden, p.passActivation.activation);
    pass = matMul(p.gpoolToPassMul2, hidden);
  } else {
    pass = matMul(p.gpoolToPassMul, g1Pooled);
  }

  // Value head.
  const v = model.valueHead;
  const v1 = ctx.bnAct(v.v1BN, v.v1Activation.activation, ctx.conv(v.v1Conv, tip));
  const pooled = ctx.valuePool(v1, v.v1Conv.outChannels);
  const v2 = matMul(v.v2Mul, pooled);
  addBias(v2, v.v2Bias);
  activateVector(v2, v.v2Activation.activation);
  const value = matMul(v.v3Mul, v2);
  addBias(value, v.v3Bias);
  const scoreValue = matMul(v.sv3Mul, v2);
  addBias(scoreValue, v.sv3Bias);
  const ownership = ctx.conv(v.vOwnershipConv, v1);

  return { xSize, ySize, policy, pass, value, scoreValue, ownership };
}
