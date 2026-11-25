/**
 * KataGo's "scale8" transform for running a convolutional net in fp16, ported from
 * `ModelDesc::applyScale8ToReduceActivations` and the per-layer functions it calls in `cpp/neuralnet/desc.cpp`
 * (KataGo 1.18, MIT, © David J Wu).
 *
 * It divides every activation in the network by 8, keeping them well inside fp16's range:
 * - the initial convolution's and initial matmul's outputs are scaled by 1/8;
 * - every batch norm's merged bias, and every dense-layer bias, is scaled by 1/8 (a batch norm's merged scale is
 *   linear in its input and stays as it is);
 * - mish becomes mish_scale8(x) = x · tanh(softplus(8x)) = mish(8x)/8; ReLU and identity are unchanged;
 * - the network's raw outputs come out 1/8 as large, so `outputScaleMultiplier` is multiplied by 8.
 *
 * The result computes the same function as the original (exactly, in real arithmetic). KataGo refuses it for
 * RMSNorm trunks, SiLU and transformer blocks; so does this.
 */
import type {
  ActivationLayer,
  BatchNormLayer,
  Block,
  ConvLayer,
  MatBiasLayer,
  MatMulLayer,
  ModelDesc,
} from './types.js';

const S = 0.125;

const scaleArray = (a: Float32Array) => a.map((v) => Math.fround(v * S));

function bn(layer: BatchNormLayer): BatchNormLayer {
  return { ...layer, mergedBias: scaleArray(layer.mergedBias) };
}

function act(layer: ActivationLayer): ActivationLayer {
  switch (layer.activation) {
    case 'identity':
    case 'relu':
      return layer;
    case 'mish':
      return { ...layer, activation: 'mish_scale8' };
    case 'silu':
      throw new Error(`${layer.name}: scale8 is not defined for SiLU (as in KataGo)`);
    case 'mish_scale8':
      throw new Error(`${layer.name}: scale8 has already been applied`);
  }
}

function matBias(layer: MatBiasLayer): MatBiasLayer {
  return { ...layer, weights: scaleArray(layer.weights) };
}

/** KataGo `ConvLayerDesc::scaleOutputChannels` with every factor 1/8 (weights are [oc][ic][y][x]: all of them). */
function convOutputs(layer: ConvLayer): ConvLayer {
  return { ...layer, weights: scaleArray(layer.weights) };
}

/** KataGo `MatMulLayerDesc::scaleOutputChannels` with every factor 1/8. */
function matMulOutputs(layer: MatMulLayer): MatMulLayer {
  return { ...layer, weights: scaleArray(layer.weights) };
}

function block(b: Block): Block {
  switch (b.kind) {
    case 'ordinary':
      return {
        ...b,
        preBN: bn(b.preBN), preActivation: act(b.preActivation),
        midBN: bn(b.midBN), midActivation: act(b.midActivation),
      };
    case 'gpool':
      return {
        ...b,
        preBN: bn(b.preBN), preActivation: act(b.preActivation),
        gpoolBN: bn(b.gpoolBN), gpoolActivation: act(b.gpoolActivation),
        midBN: bn(b.midBN), midActivation: act(b.midActivation),
      };
    case 'nested_bottleneck':
      return {
        ...b,
        preBN: bn(b.preBN), preActivation: act(b.preActivation),
        blocks: b.blocks.map(block),
        postBN: bn(b.postBN), postActivation: act(b.postActivation),
      };
  }
}

/**
 * A copy of the model with KataGo's scale8 transform applied.
 * @throws Error for models KataGo would refuse it for (RMSNorm trunk, SiLU), or with an SGF metadata encoder
 * (not supported by this engine).
 */
export function applyScale8(model: ModelDesc): ModelDesc {
  const t = model.trunk;
  if (t.trunkNormKind !== 0) throw new Error('scale8 is not defined for a trunk without batch norm (as in KataGo)');
  if (t.sgfMetadataEncoder) throw new Error('models with an SGF metadata encoder are not supported yet');
  const p = model.policyHead;
  const v = model.valueHead;
  return {
    ...model,
    postProcessParams: { ...model.postProcessParams, outputScaleMultiplier: model.postProcessParams.outputScaleMultiplier * 8 },
    trunk: {
      ...t,
      initialConv: convOutputs(t.initialConv),
      initialMatMul: matMulOutputs(t.initialMatMul),
      blocks: t.blocks.map(block),
      trunkTipBN: bn(t.trunkTipBN),
      trunkTipActivation: act(t.trunkTipActivation),
    },
    policyHead: {
      ...p,
      g1BN: bn(p.g1BN), g1Activation: act(p.g1Activation),
      p1BN: bn(p.p1BN), p1Activation: act(p.p1Activation),
      gpoolToPassBias: p.gpoolToPassBias ? matBias(p.gpoolToPassBias) : null,
      passActivation: p.passActivation ? act(p.passActivation) : null,
    },
    valueHead: {
      ...v,
      v1BN: bn(v.v1BN), v1Activation: act(v.v1Activation),
      v2Bias: matBias(v.v2Bias), v2Activation: act(v.v2Activation),
      v3Bias: matBias(v.v3Bias),
      sv3Bias: matBias(v.sv3Bias),
    },
  };
}
