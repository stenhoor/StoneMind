/**
 * A KataGo network, as parsed from a `.bin.gz` / `.txt.gz` model file.
 *
 * Mirrors KataGo's `cpp/neuralnet/desc.h` (KataGo 1.18, MIT, © David J Wu): field names follow KataGo's so the two can
 * be read side by side. Weights are stored in the layouts the reference forward pass reads; see each layer.
 */

/** `mish_scale8` never comes from a model file: only `applyScale8` (`model/scale8.ts`) produces it. */
export type Activation = 'identity' | 'relu' | 'mish' | 'silu' | 'mish_scale8';

export interface ConvLayer {
  name: string;
  convYSize: number;
  convXSize: number;
  inChannels: number;
  outChannels: number;
  dilationY: number;
  dilationX: number;
  /** Layout [oc][ic][y][x], as KataGo stores it after loading (the file order is y, x, ic, oc). */
  weights: Float32Array;
}

export interface BatchNormLayer {
  name: string;
  numChannels: number;
  epsilon: number;
  hasScale: boolean;
  hasBias: boolean;
  mean: Float32Array;
  variance: Float32Array;
  scale: Float32Array;
  bias: Float32Array;
  /** y = x * mergedScale + mergedBias, computed in float32 as KataGo does. */
  mergedScale: Float32Array;
  mergedBias: Float32Array;
}

export interface ActivationLayer {
  name: string;
  activation: Activation;
}

export interface MatMulLayer {
  name: string;
  inChannels: number;
  outChannels: number;
  /** Layout [ic][oc], the file order. */
  weights: Float32Array;
}

export interface MatBiasLayer {
  name: string;
  numChannels: number;
  weights: Float32Array;
}

export interface OrdinaryBlock {
  kind: 'ordinary';
  name: string;
  preBN: BatchNormLayer;
  preActivation: ActivationLayer;
  regularConv: ConvLayer;
  midBN: BatchNormLayer;
  midActivation: ActivationLayer;
  finalConv: ConvLayer;
}

export interface GlobalPoolingBlock {
  kind: 'gpool';
  name: string;
  preBN: BatchNormLayer;
  preActivation: ActivationLayer;
  regularConv: ConvLayer;
  gpoolConv: ConvLayer;
  gpoolBN: BatchNormLayer;
  gpoolActivation: ActivationLayer;
  gpoolToBiasMul: MatMulLayer;
  midBN: BatchNormLayer;
  midActivation: ActivationLayer;
  finalConv: ConvLayer;
}

export interface NestedBottleneckBlock {
  kind: 'nested_bottleneck';
  name: string;
  numBlocks: number;
  preBN: BatchNormLayer;
  preActivation: ActivationLayer;
  preConv: ConvLayer;
  blocks: Block[];
  postBN: BatchNormLayer;
  postActivation: ActivationLayer;
  postConv: ConvLayer;
}

/** KataGo `TransformerRMSNormDesc`: RMSNorm over the channels at each point, scaled by `weight`, with no bias. */
export interface TransformerRMSNorm {
  name: string;
  numChannels: number;
  epsilon: number;
  weight: Float32Array;
}

/**
 * KataGo `TransformerAttentionDesc`: pre-norm multi-head attention over the board's points, with grouped key/value
 * heads (`numHeads` a multiple of `numKVHeads`) and optional 2D rotary position embedding (RoPE), fixed from a theta or
 * learned per key/value head.
 */
export interface TransformerAttentionBlock {
  kind: 'transformer_attention';
  name: string;
  numHeads: number;
  numKVHeads: number;
  qHeadDim: number;
  vHeadDim: number;
  useRope: boolean;
  learnableRope: boolean;
  preLN: TransformerRMSNorm;
  qProj: MatMulLayer;
  kProj: MatMulLayer;
  vProj: MatMulLayer;
  outProj: MatMulLayer;
  /** Learnable RoPE: [kvHead][pair][x, y] frequencies; empty otherwise. */
  ropeFreqs: Float32Array;
  /** Fixed RoPE: its theta; 0 otherwise. */
  ropeTheta: number;
}

/** KataGo `TransformerFFNDesc`: pre-norm feed-forward block, SwiGLU when `useSwiGLU` (silu(linear1) × linearGate). */
export interface TransformerFFNBlock {
  kind: 'transformer_ffn';
  name: string;
  numChannels: number;
  ffnChannels: number;
  useSwiGLU: boolean;
  preLN: TransformerRMSNorm;
  linear1: MatMulLayer;
  linearGate: MatMulLayer | null;
  linear2: MatMulLayer;
}

export type Block = OrdinaryBlock | GlobalPoolingBlock | NestedBottleneckBlock | TransformerAttentionBlock | TransformerFFNBlock;

export interface SGFMetadataEncoder {
  name: string;
  metaEncoderVersion: number;
  numInputMetaChannels: number;
  mul1: MatMulLayer;
  bias1: MatBiasLayer;
  act1: ActivationLayer;
  mul2: MatMulLayer;
  bias2: MatBiasLayer;
  act2: ActivationLayer;
  mul3: MatMulLayer;
}

export interface Trunk {
  name: string;
  numBlocks: number;
  trunkNumChannels: number;
  midNumChannels: number;
  regularNumChannels: number;
  gpoolNumChannels: number;
  /** 0: batch norm at the trunk tip. 1 (RMSNorm) is not supported yet. */
  trunkNormKind: number;
  initialConv: ConvLayer;
  initialMatMul: MatMulLayer;
  sgfMetadataEncoder: SGFMetadataEncoder | null;
  blocks: Block[];
  trunkTipBN: BatchNormLayer;
  trunkTipActivation: ActivationLayer;
}

export interface PolicyHead {
  name: string;
  policyOutChannels: number;
  p1Conv: ConvLayer;
  g1Conv: ConvLayer;
  g1BN: BatchNormLayer;
  g1Activation: ActivationLayer;
  gpoolToBiasMul: MatMulLayer;
  p1BN: BatchNormLayer;
  p1Activation: ActivationLayer;
  p2Conv: ConvLayer;
  gpoolToPassMul: MatMulLayer;
  /** Model version 15+: a hidden layer in the pass head. */
  gpoolToPassBias: MatBiasLayer | null;
  passActivation: ActivationLayer | null;
  gpoolToPassMul2: MatMulLayer | null;
}

export interface ValueHead {
  name: string;
  v1Conv: ConvLayer;
  v1BN: BatchNormLayer;
  v1Activation: ActivationLayer;
  v2Mul: MatMulLayer;
  v2Bias: MatBiasLayer;
  v2Activation: ActivationLayer;
  v3Mul: MatMulLayer;
  v3Bias: MatBiasLayer;
  sv3Mul: MatMulLayer;
  sv3Bias: MatBiasLayer;
  vOwnershipConv: ConvLayer;
}

/** Scales applied when turning raw outputs into predictions (KataGo `ModelPostProcessParams`). */
export interface PostProcessParams {
  tdScoreMultiplier: number;
  scoreMeanMultiplier: number;
  scoreStdevMultiplier: number;
  leadMultiplier: number;
  varianceTimeMultiplier: number;
  shorttermValueErrorMultiplier: number;
  shorttermScoreErrorMultiplier: number;
  outputScaleMultiplier: number;
}

export interface ModelDesc {
  name: string;
  modelVersion: number;
  numInputChannels: number;
  numInputGlobalChannels: number;
  numInputMetaChannels: number;
  metaEncoderVersion: number;
  postProcessParams: PostProcessParams;
  preferPassAliveUnderSuicideRules: boolean;
  preferExcludeTerritoryAdjacentToAtari: boolean;
  trunk: Trunk;
  policyHead: PolicyHead;
  valueHead: ValueHead;
  numPolicyChannels: number;
  numValueChannels: number;
  numScoreValueChannels: number;
  numOwnershipChannels: number;
}
