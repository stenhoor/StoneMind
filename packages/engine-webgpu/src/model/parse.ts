/**
 * The KataGo model file parser: a port of the reading order in KataGo's `cpp/neuralnet/desc.cpp` (KataGo 1.18, MIT,
 * © David J Wu). Written fresh in TypeScript; the structure follows KataGo's so the two can be compared line by line.
 *
 * A model file is a whitespace-separated token stream. In `.bin` files each block of floats is "@BIN@" followed by the
 * expected number of little-endian float32 values; in `.txt` files the floats are ASCII tokens.
 */
import type {
  Activation,
  ActivationLayer,
  BatchNormLayer,
  Block,
  ConvLayer,
  MatBiasLayer,
  MatMulLayer,
  ModelDesc,
  PolicyHead,
  PostProcessParams,
  SGFMetadataEncoder,
  Trunk,
  ValueHead,
} from './types.js';

export const OLDEST_MODEL_VERSION = 8;
export const LATEST_MODEL_VERSION = 17;

export class ModelFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelFormatError';
  }
}

const isSpace = (b: number) => b === 0x20 || b === 0x0a || b === 0x0d || b === 0x09 || b === 0x0b || b === 0x0c;
const latin1 = new TextDecoder('latin1');

class Reader {
  pos = 0;
  readonly #view: DataView;

  constructor(
    readonly bytes: Uint8Array,
    readonly binaryFloats: boolean,
  ) {
    this.#view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  token(what: string): string {
    const b = this.bytes;
    while (this.pos < b.length && isSpace(b[this.pos]!)) this.pos++;
    const start = this.pos;
    while (this.pos < b.length && !isSpace(b[this.pos]!)) this.pos++;
    if (start === this.pos) throw new ModelFormatError(`unexpected end of file while reading ${what}`);
    return latin1.decode(b.subarray(start, this.pos));
  }

  int(what: string): number {
    const t = this.token(what);
    if (!/^[+-]?\d+$/.test(t)) throw new ModelFormatError(`${what}: expected an integer, got "${t.slice(0, 40)}"`);
    return Number(t);
  }

  float(what: string): number {
    const t = this.token(what);
    const v = Number.parseFloat(t);
    if (Number.isNaN(v)) throw new ModelFormatError(`${what}: expected a number, got "${t.slice(0, 40)}"`);
    return Math.fround(v);
  }

  bool(what: string): boolean {
    const v = this.int(what);
    if (v !== 0 && v !== 1) throw new ModelFormatError(`${what}: expected 0 or 1, got ${v}`);
    return v === 1;
  }

  floats(n: number, what: string): Float32Array {
    const out = new Float32Array(n);
    if (!this.binaryFloats) {
      for (let i = 0; i < n; i++) out[i] = this.float(what);
    } else {
      const b = this.bytes;
      let skipped = 0;
      while (this.pos < b.length && b[this.pos] !== 0x40 /* @ */) {
        if (++skipped > 100) throw new ModelFormatError(`${what}: expected @BIN@ before float data`);
        this.pos++;
      }
      if (latin1.decode(b.subarray(this.pos, this.pos + 5)) !== '@BIN@') {
        throw new ModelFormatError(`${what}: expected @BIN@ before float data`);
      }
      this.pos += 5;
      if (this.pos + n * 4 > b.length) throw new ModelFormatError(`${what}: file ends inside its float data`);
      for (let i = 0; i < n; i++) out[i] = this.#view.getFloat32(this.pos + i * 4, true);
      this.pos += n * 4;
    }
    for (let i = 0; i < n; i++) {
      const v = out[i]!;
      if (!Number.isFinite(v)) throw new ModelFormatError(`${what}: weight is not finite`);
      // KataGo flushes float32 subnormals to zero on load.
      if (v !== 0 && Math.abs(v) < 1.1754943508222875e-38) out[i] = 0;
    }
    return out;
  }
}

function readConv(r: Reader): ConvLayer {
  const name = r.token('conv name');
  const convYSize = r.int(`${name}: convYSize`);
  const convXSize = r.int(`${name}: convXSize`);
  const inChannels = r.int(`${name}: inChannels`);
  const outChannels = r.int(`${name}: outChannels`);
  const dilationY = r.int(`${name}: dilationY`);
  const dilationX = r.int(`${name}: dilationX`);
  if (convXSize <= 0 || convYSize <= 0 || convXSize % 2 !== 1 || convYSize % 2 !== 1) {
    throw new ModelFormatError(`${name}: conv sizes must be positive and odd`);
  }
  if (inChannels <= 0 || outChannels <= 0 || dilationX <= 0 || dilationY <= 0) {
    throw new ModelFormatError(`${name}: channels and dilations must be positive`);
  }
  const raw = r.floats(convYSize * convXSize * inChannels * outChannels, name);
  // File order is y, x, ic, oc; store as oc, ic, y, x.
  const weights = new Float32Array(raw.length);
  const ocStride = convYSize * convXSize * inChannels;
  const icStride = convYSize * convXSize;
  let idx = 0;
  for (let y = 0; y < convYSize; y++)
    for (let x = 0; x < convXSize; x++)
      for (let ic = 0; ic < inChannels; ic++)
        for (let oc = 0; oc < outChannels; oc++) weights[oc * ocStride + ic * icStride + y * convXSize + x] = raw[idx++]!;
  return { name, convYSize, convXSize, inChannels, outChannels, dilationY, dilationX, weights };
}

function readBatchNorm(r: Reader): BatchNormLayer {
  const name = r.token('batch norm name');
  const numChannels = r.int(`${name}: numChannels`);
  const epsilon = r.float(`${name}: epsilon`);
  const hasScale = r.bool(`${name}: hasScale`);
  const hasBias = r.bool(`${name}: hasBias`);
  if (numChannels < 1) throw new ModelFormatError(`${name}: numChannels must be positive`);
  if (!(epsilon > 0)) throw new ModelFormatError(`${name}: epsilon must be positive`);
  const mean = r.floats(numChannels, name);
  const variance = r.floats(numChannels, name);
  const scale = hasScale ? r.floats(numChannels, name) : new Float32Array(numChannels).fill(1);
  const bias = hasBias ? r.floats(numChannels, name) : new Float32Array(numChannels);
  const mergedScale = new Float32Array(numChannels);
  const mergedBias = new Float32Array(numChannels);
  for (let c = 0; c < numChannels; c++) {
    mergedScale[c] = Math.fround(scale[c]! / Math.fround(Math.sqrt(Math.fround(variance[c]! + epsilon))));
    mergedBias[c] = Math.fround(bias[c]! - Math.fround(mergedScale[c]! * mean[c]!));
  }
  return { name, numChannels, epsilon, hasScale, hasBias, mean, variance, scale, bias, mergedScale, mergedBias };
}

const ACTIVATIONS: Record<string, Activation> = {
  ACTIVATION_IDENTITY: 'identity',
  ACTIVATION_RELU: 'relu',
  ACTIVATION_MISH: 'mish',
  ACTIVATION_SILU: 'silu',
};

function readActivation(r: Reader, modelVersion: number): ActivationLayer {
  const name = r.token('activation name');
  if (modelVersion < 11) return { name, activation: 'relu' };
  const kind = r.token(`${name}: activation kind`);
  const activation = ACTIVATIONS[kind];
  if (!activation) throw new ModelFormatError(`${name}: unknown activation ${kind}`);
  return { name, activation };
}

function readMatMul(r: Reader): MatMulLayer {
  const name = r.token('matmul name');
  const inChannels = r.int(`${name}: inChannels`);
  const outChannels = r.int(`${name}: outChannels`);
  if (inChannels <= 0 || outChannels <= 0) throw new ModelFormatError(`${name}: channels must be positive`);
  return { name, inChannels, outChannels, weights: r.floats(inChannels * outChannels, name) };
}

function readMatBias(r: Reader): MatBiasLayer {
  const name = r.token('matbias name');
  const numChannels = r.int(`${name}: numChannels`);
  if (numChannels <= 0) throw new ModelFormatError(`${name}: numChannels must be positive`);
  return { name, numChannels, weights: r.floats(numChannels, name) };
}

function expect(cond: boolean, message: string): void {
  if (!cond) throw new ModelFormatError(message);
}

function readBlockStack(r: Reader, modelVersion: number, parent: string, numBlocks: number, channels: number): Block[] {
  const blocks: Block[] = [];
  for (let i = 0; i < numBlocks; i++) {
    const kind = r.token(`${parent}: block kind`);
    let block: Block;
    if (kind === 'ordinary_block') {
      const name = r.token('block name');
      block = {
        kind: 'ordinary',
        name,
        preBN: readBatchNorm(r),
        preActivation: readActivation(r, modelVersion),
        regularConv: readConv(r),
        midBN: readBatchNorm(r),
        midActivation: readActivation(r, modelVersion),
        finalConv: readConv(r),
      };
      expect(block.preBN.numChannels === block.regularConv.inChannels, `${name}: preBN/regularConv channel mismatch`);
      expect(block.midBN.numChannels === block.regularConv.outChannels, `${name}: midBN/regularConv channel mismatch`);
      expect(block.midBN.numChannels === block.finalConv.inChannels, `${name}: midBN/finalConv channel mismatch`);
      expect(block.preBN.numChannels === channels && block.finalConv.outChannels === channels, `${name}: trunk channel mismatch`);
    } else if (kind === 'gpool_block') {
      const name = r.token('block name');
      block = {
        kind: 'gpool',
        name,
        preBN: readBatchNorm(r),
        preActivation: readActivation(r, modelVersion),
        regularConv: readConv(r),
        gpoolConv: readConv(r),
        gpoolBN: readBatchNorm(r),
        gpoolActivation: readActivation(r, modelVersion),
        gpoolToBiasMul: readMatMul(r),
        midBN: readBatchNorm(r),
        midActivation: readActivation(r, modelVersion),
        finalConv: readConv(r),
      };
      expect(block.gpoolBN.numChannels * 3 === block.gpoolToBiasMul.inChannels, `${name}: gpool matmul input mismatch`);
      expect(block.midBN.numChannels === block.gpoolToBiasMul.outChannels, `${name}: gpool matmul output mismatch`);
      expect(block.preBN.numChannels === channels && block.finalConv.outChannels === channels, `${name}: trunk channel mismatch`);
    } else if (kind === 'nested_bottleneck_block') {
      const name = r.token('block name');
      const numBlocks = r.int(`${name}: numBlocks`);
      expect(numBlocks >= 1, `${name}: nested bottleneck needs at least one block`);
      const preBN = readBatchNorm(r);
      const preActivation = readActivation(r, modelVersion);
      const preConv = readConv(r);
      const inner = readBlockStack(r, modelVersion, name, numBlocks, preConv.outChannels);
      block = {
        kind: 'nested_bottleneck',
        name,
        numBlocks,
        preBN,
        preActivation,
        preConv,
        blocks: inner,
        postBN: readBatchNorm(r),
        postActivation: readActivation(r, modelVersion),
        postConv: readConv(r),
      };
      expect(block.preBN.numChannels === channels && block.postConv.outChannels === channels, `${name}: trunk channel mismatch`);
    } else if (kind === 'transformer_attention_block' || kind === 'transformer_ffn_block') {
      throw new ModelFormatError(`${parent}: transformer blocks are not supported yet (spike 0001, phase 6)`);
    } else {
      throw new ModelFormatError(`${parent}: unknown block kind "${kind}"`);
    }
    blocks.push(block);
  }
  return blocks;
}

function readMetaEncoder(r: Reader, modelVersion: number, metaEncoderVersion: number): SGFMetadataEncoder {
  const name = r.token('metadata encoder name');
  const numInputMetaChannels = r.int(`${name}: numInputMetaChannels`);
  expect(numInputMetaChannels === 192, `${name}: expected 192 metadata channels`);
  return {
    name,
    metaEncoderVersion,
    numInputMetaChannels,
    mul1: readMatMul(r),
    bias1: readMatBias(r),
    act1: readActivation(r, modelVersion),
    mul2: readMatMul(r),
    bias2: readMatBias(r),
    act2: readActivation(r, modelVersion),
    mul3: readMatMul(r),
  };
}

function readTrunk(r: Reader, modelVersion: number, metaEncoderVersion: number): Trunk {
  const name = r.token('trunk name');
  const numBlocks = r.int(`${name}: numBlocks`);
  const trunkNumChannels = r.int(`${name}: trunkNumChannels`);
  const midNumChannels = r.int(`${name}: midNumChannels`);
  const regularNumChannels = r.int(`${name}: regularNumChannels`);
  r.int(`${name}: dilatedNumChannels (unused)`);
  const gpoolNumChannels = r.int(`${name}: gpoolNumChannels`);
  let trunkNormKind = 0;
  if (modelVersion >= 15) {
    trunkNormKind = r.int(`${name}: trunkNormKind`);
    for (let i = 0; i < 5; i++) r.int(`${name}: reserved`);
    if (trunkNormKind !== 0) throw new ModelFormatError(`${name}: RMSNorm trunks are not supported yet (spike 0001, phase 6)`);
  }
  expect(numBlocks >= 1, `${name}: numBlocks must be positive`);
  const initialConv = readConv(r);
  expect(initialConv.outChannels === trunkNumChannels, `${name}: initial conv channel mismatch`);
  const initialMatMul = readMatMul(r);
  expect(initialMatMul.outChannels === trunkNumChannels, `${name}: initial matmul channel mismatch`);
  const sgfMetadataEncoder = metaEncoderVersion > 0 ? readMetaEncoder(r, modelVersion, metaEncoderVersion) : null;
  const blocks = readBlockStack(r, modelVersion, name, numBlocks, trunkNumChannels);
  const trunkTipBN = readBatchNorm(r);
  expect(trunkTipBN.numChannels === trunkNumChannels, `${name}: trunk tip channel mismatch`);
  const trunkTipActivation = readActivation(r, modelVersion);
  return {
    name,
    numBlocks,
    trunkNumChannels,
    midNumChannels,
    regularNumChannels,
    gpoolNumChannels,
    trunkNormKind,
    initialConv,
    initialMatMul,
    sgfMetadataEncoder,
    blocks,
    trunkTipBN,
    trunkTipActivation,
  };
}

function readPolicyHead(r: Reader, modelVersion: number): PolicyHead {
  const name = r.token('policy head name');
  let policyOutChannels: number;
  if (modelVersion >= 17) {
    policyOutChannels = r.int(`${name}: policyOutChannels`);
    expect(policyOutChannels === 2 || policyOutChannels === 4, `${name}: policyOutChannels must be 2 or 4`);
    for (let i = 0; i < 3; i++) r.int(`${name}: reserved`);
  } else if (modelVersion === 16) policyOutChannels = 4;
  else if (modelVersion >= 12) policyOutChannels = 2;
  else policyOutChannels = 1;
  const p1Conv = readConv(r);
  const g1Conv = readConv(r);
  const g1BN = readBatchNorm(r);
  const g1Activation = readActivation(r, modelVersion);
  const gpoolToBiasMul = readMatMul(r);
  const p1BN = readBatchNorm(r);
  const p1Activation = readActivation(r, modelVersion);
  const p2Conv = readConv(r);
  const gpoolToPassMul = readMatMul(r);
  let gpoolToPassBias: MatBiasLayer | null = null;
  let passActivation: ActivationLayer | null = null;
  let gpoolToPassMul2: MatMulLayer | null = null;
  if (modelVersion >= 15) {
    gpoolToPassBias = readMatBias(r);
    passActivation = readActivation(r, modelVersion);
    gpoolToPassMul2 = readMatMul(r);
  }
  expect(gpoolToBiasMul.inChannels === g1BN.numChannels * 3, `${name}: gpool matmul input mismatch`);
  expect(gpoolToBiasMul.outChannels === p1BN.numChannels, `${name}: gpool matmul output mismatch`);
  expect(p2Conv.outChannels === policyOutChannels, `${name}: p2Conv must output ${policyOutChannels} channels`);
  return {
    name,
    policyOutChannels,
    p1Conv,
    g1Conv,
    g1BN,
    g1Activation,
    gpoolToBiasMul,
    p1BN,
    p1Activation,
    p2Conv,
    gpoolToPassMul,
    gpoolToPassBias,
    passActivation,
    gpoolToPassMul2,
  };
}

function readValueHead(r: Reader, modelVersion: number): ValueHead {
  const name = r.token('value head name');
  if (modelVersion >= 17) for (let i = 0; i < 3; i++) r.int(`${name}: reserved`);
  const head: ValueHead = {
    name,
    v1Conv: readConv(r),
    v1BN: readBatchNorm(r),
    v1Activation: readActivation(r, modelVersion),
    v2Mul: readMatMul(r),
    v2Bias: readMatBias(r),
    v2Activation: readActivation(r, modelVersion),
    v3Mul: readMatMul(r),
    v3Bias: readMatBias(r),
    sv3Mul: readMatMul(r),
    sv3Bias: readMatBias(r),
    vOwnershipConv: readConv(r),
  };
  expect(head.v2Mul.inChannels === head.v1BN.numChannels * 3, `${name}: v2Mul input mismatch`);
  expect(head.v3Mul.outChannels === 3, `${name}: value head must output 3 channels`);
  const expectedScore = modelVersion >= 9 ? 6 : 4;
  expect(head.sv3Mul.outChannels === expectedScore, `${name}: score value head must output ${expectedScore} channels`);
  return head;
}

const DEFAULT_POST_PROCESS: PostProcessParams = {
  tdScoreMultiplier: 20,
  scoreMeanMultiplier: 20,
  scoreStdevMultiplier: 20,
  leadMultiplier: 20,
  varianceTimeMultiplier: 40,
  shorttermValueErrorMultiplier: 0.25,
  shorttermScoreErrorMultiplier: 30,
  outputScaleMultiplier: 1,
};

/**
 * Parse a decompressed KataGo model file.
 * @param bytes The file's bytes after gunzip.
 * @param binaryFloats Whether float blocks are "@BIN@" binary (`.bin.gz`) rather than text (`.txt.gz`).
 * @throws ModelFormatError when the file isn't a supported KataGo model.
 */
export function parseModel(bytes: Uint8Array, binaryFloats: boolean): ModelDesc {
  const r = new Reader(bytes, binaryFloats);
  const name = r.token('model name');
  const modelVersion = r.int(`${name}: model version`);
  if (modelVersion < OLDEST_MODEL_VERSION || modelVersion > LATEST_MODEL_VERSION) {
    throw new ModelFormatError(`${name}: model version ${modelVersion} is not supported (${OLDEST_MODEL_VERSION}–${LATEST_MODEL_VERSION})`);
  }
  const numInputChannels = r.int(`${name}: numInputChannels`);
  const numInputGlobalChannels = r.int(`${name}: numInputGlobalChannels`);
  expect(numInputChannels === 22 && numInputGlobalChannels === 19, `${name}: expected input format V7 (22 spatial, 19 global features)`);

  const postProcessParams = { ...DEFAULT_POST_PROCESS };
  if (modelVersion >= 13) {
    postProcessParams.tdScoreMultiplier = r.float('tdScoreMultiplier');
    postProcessParams.scoreMeanMultiplier = r.float('scoreMeanMultiplier');
    postProcessParams.scoreStdevMultiplier = r.float('scoreStdevMultiplier');
    postProcessParams.leadMultiplier = r.float('leadMultiplier');
    postProcessParams.varianceTimeMultiplier = r.float('varianceTimeMultiplier');
    postProcessParams.shorttermValueErrorMultiplier = r.float('shorttermValueErrorMultiplier');
    postProcessParams.shorttermScoreErrorMultiplier = r.float('shorttermScoreErrorMultiplier');
  }
  let metaEncoderVersion = 0;
  let preferPassAliveUnderSuicideRules = false;
  let preferExcludeTerritoryAdjacentToAtari = false;
  if (modelVersion >= 15) {
    metaEncoderVersion = r.int('metaEncoderVersion');
    expect(metaEncoderVersion === 0 || metaEncoderVersion === 1, `${name}: unsupported metaEncoderVersion ${metaEncoderVersion}`);
    preferPassAliveUnderSuicideRules = r.bool('preferPassAliveUnderSuicideRules');
    preferExcludeTerritoryAdjacentToAtari = r.bool('preferExcludeTerritoryAdjacentToAtari');
    for (let i = 0; i < 5; i++) r.int('reserved');
  }

  const trunk = readTrunk(r, modelVersion, metaEncoderVersion);
  const policyHead = readPolicyHead(r, modelVersion);
  const valueHead = readValueHead(r, modelVersion);
  expect(trunk.initialConv.inChannels === numInputChannels, `${name}: initial conv input mismatch`);
  expect(trunk.initialMatMul.inChannels === numInputGlobalChannels, `${name}: initial matmul input mismatch`);
  expect(policyHead.p1Conv.inChannels === trunk.trunkNumChannels, `${name}: policy head input mismatch`);
  expect(valueHead.v1Conv.inChannels === trunk.trunkNumChannels, `${name}: value head input mismatch`);

  return {
    name,
    modelVersion,
    numInputChannels,
    numInputGlobalChannels,
    numInputMetaChannels: metaEncoderVersion > 0 ? 192 : 0,
    metaEncoderVersion,
    postProcessParams,
    preferPassAliveUnderSuicideRules,
    preferExcludeTerritoryAdjacentToAtari,
    trunk,
    policyHead,
    valueHead,
    numPolicyChannels: policyHead.policyOutChannels,
    numValueChannels: valueHead.v3Mul.outChannels,
    numScoreValueChannels: valueHead.sv3Mul.outChannels,
    numOwnershipChannels: valueHead.vOwnershipConv.outChannels,
  };
}

/**
 * Load a model from file bytes: gunzips when gzipped (with the platform's DecompressionStream) and detects the float
 * format from the content ("@BIN@" markers) rather than the file name.
 */
export async function loadModel(bytes: Uint8Array): Promise<ModelDesc> {
  let data = bytes;
  if (data.length >= 2 && data[0] === 0x1f && data[1] === 0x8b) {
    const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'));
    data = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const head = latin1.decode(data.subarray(0, Math.min(data.length, 1 << 16)));
  return parseModel(data, head.includes('@BIN@'));
}

/** The number of parameters, counted as KataGo counts them (convs, BN scale/bias when present, matmuls, biases). */
export function countParameters(model: ModelDesc): number {
  let n = 0;
  const conv = (c: ConvLayer) => (n += c.weights.length);
  const bn = (b: BatchNormLayer) => (n += (b.hasScale ? b.numChannels : 0) + (b.hasBias ? b.numChannels : 0));
  const mm = (m: MatMulLayer) => (n += m.weights.length);
  const mb = (m: MatBiasLayer) => (n += m.weights.length);
  const blocks = (list: readonly Block[]) => {
    for (const b of list) {
      if (b.kind === 'ordinary') {
        bn(b.preBN), conv(b.regularConv), bn(b.midBN), conv(b.finalConv);
      } else if (b.kind === 'gpool') {
        bn(b.preBN), conv(b.regularConv), conv(b.gpoolConv), bn(b.gpoolBN), mm(b.gpoolToBiasMul), bn(b.midBN), conv(b.finalConv);
      } else {
        bn(b.preBN), conv(b.preConv), blocks(b.blocks), bn(b.postBN), conv(b.postConv);
      }
    }
  };
  const t = model.trunk;
  conv(t.initialConv), mm(t.initialMatMul);
  if (t.sgfMetadataEncoder) {
    const e = t.sgfMetadataEncoder;
    mm(e.mul1), mb(e.bias1), mm(e.mul2), mb(e.bias2), mm(e.mul3);
  }
  blocks(t.blocks);
  bn(t.trunkTipBN);
  const p = model.policyHead;
  conv(p.p1Conv), conv(p.g1Conv), bn(p.g1BN), mm(p.gpoolToBiasMul), bn(p.p1BN), conv(p.p2Conv), mm(p.gpoolToPassMul);
  if (p.gpoolToPassBias) mb(p.gpoolToPassBias);
  if (p.gpoolToPassMul2) mm(p.gpoolToPassMul2);
  const v = model.valueHead;
  conv(v.v1Conv), bn(v.v1BN), mm(v.v2Mul), mb(v.v2Bias), mm(v.v3Mul), mb(v.v3Bias), mm(v.sv3Mul), mb(v.sv3Bias), conv(v.vOwnershipConv);
  return n;
}
