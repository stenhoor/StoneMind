/**
 * The network on WebGPU: weights uploaded once, a plan of dispatches built once for a board size and a largest batch,
 * then each evaluation is one write of the inputs, one submit and one readback.
 *
 * The layer order and maths follow the reference forward pass (`../reference/forward.ts`), which follows KataGo's
 * Eigen backend. Positions on smaller boards are placed in the top-left corner of the plan's board and masked, as
 * KataGo does with its fixed-size network buffers.
 */
import type { Activation, BatchNormLayer, Block, ConvLayer, MatMulLayer, MatBiasLayer, ModelDesc } from '../model/types.js';
import { applyScale8 } from '../model/scale8.js';
import type { NetInput, RawOutputs } from '../reference/forward.js';
import type { EngineDevice } from './device.js';
import { toFloat16Array } from './f16.js';
import { ACT, BN_ACT, CONV, CONV_TILE_M, CONV_TILE_N, GEMM_DEFAULT, GEMM_V2, gemmTile, type GemmConfig, GPOOL, MATVEC, WINO_GEMM, WINO_IN, WINO_OUT, winogradWeights, type Elem } from './kernels.js';
import { BufferUsage, MapMode, type GPUBindGroup, type GPUBuffer, type GPUComputePipeline, type GPUDevice, type GPUQuerySet } from './webgpu.js';

export interface GpuNetworkOptions {
  /** The plan's board size (default 19 × 19). Smaller boards fit in its top-left corner. */
  xSize?: number;
  ySize?: number;
  /** The largest batch the plan accepts (default 1). */
  maxBatch?: number;
  /** Run 3×3 convolutions with Winograd F(4×4,3×3) (default true); false uses the implicit GEMM for every size. */
  winograd?: boolean;
  /** With Winograd, apply a preceding batch norm and activation inside the input transform (default true). */
  fuse?: boolean;
  /**
   * Storage precision of activations and convolution weights (default 'f32'). 'f16' needs the device feature
   * `shader-f16`; arithmetic stays f32 either way.
   */
  precision?: 'f32' | 'f16';
  /**
   * Run KataGo's scale8 transform on the model first (`model/scale8.ts`), as KataGo's OpenCL backend does, keeping
   * activations inside fp16's range. Default: on for 'f16', off for 'f32'. Results are returned unscaled either way.
   */
  scale8?: boolean;
  /**
   * The matrix-multiply kernel for the Winograd GEMMs and 1×1 convolutions (default 'v2'): 'v2' is the generic
   * unrolled GEMM (`GEMM_V2`), with 1×1 convolutions as a plain GEMM per batch position; 'v1' keeps the first kernels
   * (the Winograd GEMM and the implicit-GEMM convolution). On Edge with an RTX 3080 Laptop GPU, v2 was 1.11–1.29× faster
   * at batches 8–64 in every variant (docs/spikes/results/2026-11-b18-webgpu-gemm-v2.md).
   */
  gemm?: 'v1' | 'v2';
  /** GEMM v2's shape (default `GEMM_DEFAULT`). arith 'f16' needs the device feature `shader-f16`. */
  gemmConfig?: GemmConfig;
}

/** An op as the plan builder records it, before its shader is compiled. */
interface OpSpec {
  kind: string;
  code: string;
  buffers: GPUBuffer[];
  workgroups: (n: number) => [number, number] | [number, number, number];
}

/** Timings from {@link GpuNetwork.profile}. GPU times come from timestamp queries, in nanoseconds as the browser reports them. */
export interface GpuProfile {
  batch: number;
  /** Wall-clock time of the whole evaluation in milliseconds: input upload, GPU work and readback. */
  wallMs: number;
  /** mode 'pass': GPU time of the network's single compute pass. */
  passNs?: number;
  /** mode 'ops': GPU time of each dispatch, each run in its own compute pass, in plan order. */
  ops?: { kind: string; ns: number }[];
}

interface Op {
  /** What the op is, for profiles (several kinds can share one pipeline). */
  kind: string;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroups: (n: number) => [number, number] | [number, number, number];
}

interface OutputRegion {
  name: 'policy' | 'pass' | 'value' | 'scoreValue' | 'ownership';
  buffer: GPUBuffer;
  /** Floats per position. */
  perBatch: number;
  stagingOffset: number;
}

const STORAGE = BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC;
const MAX_GROUPS = 65535;

function splitGroups(groups: number): [number, number] {
  if (groups <= MAX_GROUPS) return [Math.max(groups, 1), 1];
  const y = Math.ceil(groups / MAX_GROUPS);
  return [Math.ceil(groups / y), y];
}

/** Conv weights from [oc][ic][ky][kx] to the GEMM layout [k][oc], k = (ic, ky, kx). */
function convWeightsKxOc(layer: ConvLayer): Float32Array {
  const K = layer.inChannels * layer.convYSize * layer.convXSize;
  const oc = layer.outChannels;
  const out = new Float32Array(K * oc);
  for (let o = 0; o < oc; o++) for (let k = 0; k < K; k++) out[k * oc + o] = layer.weights[o * K + k]!;
  return out;
}

/** A batch norm's merged scales, then its merged biases. */
function mergedScaleBias(bn: BatchNormLayer): Float32Array {
  const sb = new Float32Array(2 * bn.numChannels);
  sb.set(bn.mergedScale, 0);
  sb.set(bn.mergedBias, bn.numChannels);
  return sb;
}

function maxChannels(model: ModelDesc): number {
  let m = 0;
  const conv = (c: ConvLayer) => (m = Math.max(m, c.inChannels, c.outChannels));
  const block = (b: Block): void => {
    if (b.kind === 'ordinary') [b.regularConv, b.finalConv].forEach(conv);
    else if (b.kind === 'gpool') [b.regularConv, b.gpoolConv, b.finalConv].forEach(conv);
    else {
      [b.preConv, b.postConv].forEach(conv);
      b.blocks.forEach(block);
    }
  };
  conv(model.trunk.initialConv);
  model.trunk.blocks.forEach(block);
  const p = model.policyHead;
  const v = model.valueHead;
  [p.p1Conv, p.g1Conv, p.p2Conv, v.v1Conv, v.vOwnershipConv].forEach(conv);
  return m;
}

function nestingDepth(blocks: readonly Block[]): number {
  let d = 0;
  for (const b of blocks) if (b.kind === 'nested_bottleneck') d = Math.max(d, 1 + nestingDepth(b.blocks));
  return d;
}

export class GpuNetwork {
  readonly model: ModelDesc;
  readonly xSize: number;
  readonly ySize: number;
  readonly maxBatch: number;
  readonly device: GPUDevice;
  readonly #area: number;
  readonly #ops: Op[] = [];
  readonly #owned: GPUBuffer[] = [];
  readonly #outputs: OutputRegion[] = [];
  #dyn!: GPUBuffer;
  #input!: GPUBuffer;
  #global!: GPUBuffer;
  #staging!: GPUBuffer;
  #stagingFloats = 0;
  readonly #specs: OpSpec[] = [];
  readonly #elems = new Map<GPUBuffer, Elem>();
  /** The model the plan runs: `model`, or its scale8 transform. */
  #desc!: ModelDesc;
  /** What raw outputs are multiplied by on readback: 8 with scale8, else 1. */
  #outputFactor = 1;
  #elem: Elem = 'f32';
  #gemm: 'v1' | 'v2' = 'v2';
  #gemmConfig: GemmConfig = GEMM_DEFAULT;
  #winograd = true;
  #timestamps = false;
  #fuse = true;
  #wino?: { V: GPUBuffer; M: GPUBuffer; tilesX: number; tilesPerBoard: number; tileStride: number };
  #dummy!: GPUBuffer;
  #busy: Promise<unknown> = Promise.resolve();
  #destroyed = false;

  private constructor(device: GPUDevice, model: ModelDesc, xSize: number, ySize: number, maxBatch: number) {
    this.device = device;
    this.model = model;
    this.xSize = xSize;
    this.ySize = ySize;
    this.maxBatch = maxBatch;
    this.#area = xSize * ySize;
  }

  /**
   * Upload a model and build its plan.
   * @throws Error for models the GPU path doesn't support yet (SGF metadata input), or if WebGPU reports an error.
   */
  static async create(engine: EngineDevice, model: ModelDesc, options: GpuNetworkOptions = {}): Promise<GpuNetwork> {
    if (model.trunk.sgfMetadataEncoder) throw new Error('models with an SGF metadata encoder are not supported yet');
    const net = new GpuNetwork(engine.device, model, options.xSize ?? 19, options.ySize ?? 19, options.maxBatch ?? 1);
    net.#winograd = options.winograd ?? true;
    net.#timestamps = engine.features.has('timestamp-query');
    net.#elem = options.precision ?? 'f32';
    net.#gemm = options.gemm ?? 'v2';
    net.#gemmConfig = options.gemmConfig ?? GEMM_DEFAULT;
    if (net.#gemmConfig.arith !== 'f32' && !engine.features.has('shader-f16')) throw new Error(`gemmConfig.arith '${net.#gemmConfig.arith}' needs the device feature shader-f16`);
    if (net.#elem === 'f16' && !engine.features.has('shader-f16')) throw new Error("precision 'f16' needs the device feature shader-f16");
    const scale8 = options.scale8 ?? net.#elem === 'f16';
    net.#desc = scale8 ? applyScale8(model) : model;
    net.#outputFactor = scale8 ? 8 : 1;
    net.#fuse = options.fuse ?? true;
    const { device } = net;
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    await net.#build();
    const oom = await device.popErrorScope();
    const invalid = await device.popErrorScope();
    if (oom || invalid) {
      net.destroy();
      throw new Error(`WebGPU error while building the network: ${(oom ?? invalid)!.message}`);
    }
    return net;
  }

  // ------------------------------------------------------------------ building

  #buffer(count: number, usage = STORAGE, elem: Elem = 'f32'): GPUBuffer {
    const bytes = count * (elem === 'f16' ? 2 : 4);
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
    this.#owned.push(b);
    this.#elems.set(b, elem);
    return b;
  }

  /** A large tensor, in the plan's storage precision. */
  #tensor(count: number): GPUBuffer {
    return this.#buffer(count, STORAGE, this.#elem);
  }

  #upload(data: Float32Array, elem: Elem = 'f32'): GPUBuffer {
    const b = this.#buffer(data.length, BufferUsage.STORAGE | BufferUsage.COPY_DST, elem);
    this.device.queue.writeBuffer(b, 0, elem === 'f16' ? toFloat16Array(data) : data);
    return b;
  }

  #elemOf(buffer: GPUBuffer): Elem {
    return this.#elems.get(buffer) ?? 'f32';
  }

  #uniform(values: readonly number[]): GPUBuffer {
    const u = new Uint32Array(Math.ceil(values.length / 4) * 4);
    u.set(values);
    const b = this.#buffer(u.length, BufferUsage.UNIFORM | BufferUsage.COPY_DST);
    this.device.queue.writeBuffer(b, 0, u);
    return b;
  }

  #bind(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  /** Compile every distinct shader the plan uses, then bind each op. */
  async #compile(): Promise<void> {
    const { device } = this;
    const pipelines = new Map<string, Promise<GPUComputePipeline>>();
    for (const spec of this.#specs) {
      if (!pipelines.has(spec.code)) {
        const module = device.createShaderModule({ code: spec.code });
        pipelines.set(
          spec.code,
          device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }).catch(async (e: unknown) => {
            const info = await module.getCompilationInfo();
            const messages = info.messages.map((m) => `${m.type} at ${m.lineNum}:${m.linePos}: ${m.message}`).join('; ');
            throw new Error(`${spec.kind} shader failed to compile: ${messages || String(e)}`);
          }),
        );
      }
    }
    for (const spec of this.#specs) {
      const pipeline = await pipelines.get(spec.code)!;
      this.#ops.push({ kind: spec.kind, pipeline, bindGroup: this.#bind(pipeline, spec.buffers), workgroups: spec.workgroups });
    }
  }

  async #build(): Promise<void> {
    const model = this.#desc;

    const B = this.maxBatch;
    const area = this.#area;
    const C = maxChannels(model);
    this.#dummy = this.#buffer(4);
    this.#dyn = this.#buffer(4, BufferUsage.UNIFORM | BufferUsage.COPY_DST);
    this.#input = this.#buffer(B * model.numInputChannels * area);
    this.#global = this.#buffer(B * model.numInputGlobalChannels);

    const t = model.trunk;
    const trunk = this.#tensor(B * t.trunkNumChannels * area);
    const inner = Array.from({ length: nestingDepth(t.blocks) }, () => this.#tensor(B * C * area));
    const s = {
      a: this.#tensor(B * C * area),
      m: this.#tensor(B * C * area),
      b: this.#tensor(B * C * area),
      g: this.#tensor(B * C * area),
      pooled: this.#buffer(B * 3 * C),
      vec: this.#buffer(B * C),
      vec2: this.#buffer(B * C),
    };
    if (this.#winograd) {
      const tilesX = Math.ceil(this.xSize / 4);
      const tilesPerBoard = tilesX * Math.ceil(this.ySize / 4);
      const tileStride = B * tilesPerBoard;
      this.#wino = { V: this.#tensor(36 * C * tileStride), M: this.#tensor(36 * C * tileStride), tilesX, tilesPerBoard, tileStride };
    }

    // Trunk.
    this.#matvec(t.initialMatMul, this.#global, model.numInputGlobalChannels, s.vec, t.trunkNumChannels, 0);
    this.#conv(t.initialConv, this.#input, trunk, { nbias: s.vec });
    const block = (blk: Block, target: GPUBuffer, depth: number): void => {
      switch (blk.kind) {
        case 'ordinary':
          this.#bnActConv(blk.preBN, blk.preActivation.activation, target, blk.regularConv, s.m, s.a);
          this.#bnActConv(blk.midBN, blk.midActivation.activation, s.m, blk.finalConv, target, s.b, { accumulate: true });
          return;
        case 'gpool':
          if (this.#fusesInto(blk.regularConv) && this.#fusesInto(blk.gpoolConv)) {
            this.#bnActConv(blk.preBN, blk.preActivation.activation, target, blk.regularConv, s.m, s.a);
            this.#bnActConv(blk.preBN, blk.preActivation.activation, target, blk.gpoolConv, s.g, s.a);
          } else {
            // Both convolutions read the same normalised input: compute it once.
            this.#bnAct(blk.preBN, blk.preActivation.activation, target, s.a);
            this.#conv(blk.regularConv, s.a, s.m);
            this.#conv(blk.gpoolConv, s.a, s.g);
          }
          this.#bnAct(blk.gpoolBN, blk.gpoolActivation.activation, s.g, s.b);
          this.#gpool(s.b, blk.gpoolConv.outChannels, s.pooled, 0);
          this.#matvec(blk.gpoolToBiasMul, s.pooled, 3 * blk.gpoolConv.outChannels, s.vec, blk.regularConv.outChannels, 0);
          this.#bnActConv(blk.midBN, blk.midActivation.activation, s.m, blk.finalConv, target, s.b, { accumulate: true, pre: s.vec });
          return;
        case 'nested_bottleneck': {
          const mid = inner[depth]!;
          this.#bnAct(blk.preBN, blk.preActivation.activation, target, s.a);
          this.#conv(blk.preConv, s.a, mid);
          for (const b of blk.blocks) block(b, mid, depth + 1);
          this.#bnAct(blk.postBN, blk.postActivation.activation, mid, s.b);
          this.#conv(blk.postConv, s.b, target, { accumulate: true });
          return;
        }
      }
    };
    for (const b of t.blocks) block(b, trunk, 0);
    const tip = s.a;
    this.#bnAct(t.trunkTipBN, t.trunkTipActivation.activation, trunk, tip);

    // Policy head.
    const p = model.policyHead;
    const g1C = p.g1Conv.outChannels;
    const policy = this.#buffer(B * p.policyOutChannels * area);
    const pass = this.#buffer(B * p.policyOutChannels);
    this.#conv(p.p1Conv, tip, s.m);
    this.#conv(p.g1Conv, tip, s.g);
    this.#bnAct(p.g1BN, p.g1Activation.activation, s.g, s.b);
    this.#gpool(s.b, g1C, s.pooled, 0);
    this.#matvec(p.gpoolToBiasMul, s.pooled, 3 * g1C, s.vec, p.p1Conv.outChannels, 0);
    this.#bnAct(p.p1BN, p.p1Activation.activation, s.m, s.b, s.vec);
    this.#conv(p.p2Conv, s.b, policy);
    if (p.gpoolToPassMul2 && p.gpoolToPassBias && p.passActivation) {
      const hidden = p.gpoolToPassMul.outChannels;
      this.#matvec(p.gpoolToPassMul, s.pooled, 3 * g1C, s.vec2, hidden, 0, p.gpoolToPassBias, p.passActivation.activation);
      this.#matvec(p.gpoolToPassMul2, s.vec2, hidden, pass, p.policyOutChannels, 0);
    } else {
      this.#matvec(p.gpoolToPassMul, s.pooled, 3 * g1C, pass, p.policyOutChannels, 0);
    }

    // Value head.
    const v = model.valueHead;
    const v1C = v.v1Conv.outChannels;
    const value = this.#buffer(B * v.v3Mul.outChannels);
    const scoreValue = this.#buffer(B * v.sv3Mul.outChannels);
    const ownership = this.#buffer(B * v.vOwnershipConv.outChannels * area);
    this.#conv(v.v1Conv, tip, s.g);
    this.#bnAct(v.v1BN, v.v1Activation.activation, s.g, s.b);
    this.#gpool(s.b, v1C, s.pooled, 1);
    this.#matvec(v.v2Mul, s.pooled, 3 * v1C, s.vec, v.v2Mul.outChannels, 0, v.v2Bias, v.v2Activation.activation);
    this.#matvec(v.v3Mul, s.vec, v.v2Mul.outChannels, value, v.v3Mul.outChannels, 0, v.v3Bias);
    this.#matvec(v.sv3Mul, s.vec, v.v2Mul.outChannels, scoreValue, v.sv3Mul.outChannels, 0, v.sv3Bias);
    this.#conv(v.vOwnershipConv, s.b, ownership);

    // One staging buffer for every output, read back in one map.
    const regions: [OutputRegion['name'], GPUBuffer, number][] = [
      ['policy', policy, p.policyOutChannels * area],
      ['pass', pass, p.policyOutChannels],
      ['value', value, v.v3Mul.outChannels],
      ['scoreValue', scoreValue, v.sv3Mul.outChannels],
      ['ownership', ownership, v.vOwnershipConv.outChannels * area],
    ];
    let offset = 0;
    for (const [name, buffer, perBatch] of regions) {
      this.#outputs.push({ name, buffer, perBatch, stagingOffset: offset });
      offset += B * perBatch;
    }
    this.#stagingFloats = offset;
    this.#staging = this.#buffer(offset, BufferUsage.MAP_READ | BufferUsage.COPY_DST);
    await this.#compile();
  }

  #conv(layer: ConvLayer, input: GPUBuffer, output: GPUBuffer, opts: { accumulate?: boolean; nbias?: GPUBuffer } = {}): void {
    if (this.#winogradEligible(layer)) {
      this.#winogradConv(layer, input, output, opts);
      return;
    }
    if (this.#gemm === 'v2' && layer.convYSize === 1 && layer.convXSize === 1) {
      this.#conv1x1(layer, input, output, opts);
      return;
    }
    const K = layer.inChannels * layer.convYSize * layer.convXSize;
    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
    const params = this.#uniform([
      layer.inChannels, layer.outChannels, layer.convYSize, layer.convXSize,
      layer.dilationY, layer.dilationX, this.xSize, this.ySize,
      K, flags, 0, 0,
    ]);
    const weights = this.#upload(convWeightsKxOc(layer), this.#elem);
    const area = this.#area;
    this.#specs.push({
      kind: 'conv',
      code: CONV(this.#elemOf(input), this.#elem, this.#elemOf(output)),
      buffers: [params, this.#dyn, input, weights, output, opts.nbias ?? this.#dummy],
      workgroups: (n) => [Math.ceil((n * area) / CONV_TILE_N), Math.ceil(layer.outChannels / CONV_TILE_M)],
    });
  }

  /** A 1×1 convolution as GEMM v2, one GEMM per batch position: out[oc][p] = Σ_ic w[ic][oc] · in[ic][p]. */
  #conv1x1(layer: ConvLayer, input: GPUBuffer, output: GPUBuffer, opts: { accumulate?: boolean; nbias?: GPUBuffer }): void {
    const { inChannels: ic, outChannels: oc } = layer;
    const area = this.#area;
    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
    const weights = this.#upload(convWeightsKxOc(layer), this.#elem);
    this.#specs.push({
      kind: 'conv1x1',
      code: GEMM_V2(this.#elemOf(weights), this.#elemOf(input), this.#elemOf(output), this.#gemmConfig),
      buffers: [
        this.#uniform([oc, ic, area, 0, 0, oc, ic * area, area, oc * area, area, flags, oc]),
        this.#dyn, weights, input, output, opts.nbias ?? this.#dummy,
      ],
      workgroups: (n) => [Math.ceil(area / gemmTile(this.#gemmConfig).tn), Math.ceil(oc / gemmTile(this.#gemmConfig).tm), n],
    });
  }

  #winogradEligible(layer: ConvLayer): boolean {
    return this.#wino !== undefined && layer.convYSize === 3 && layer.convXSize === 3 && layer.dilationY === 1 && layer.dilationX === 1;
  }

  #fusesInto(layer: ConvLayer): boolean {
    return this.#fuse && this.#winogradEligible(layer);
  }

  /**
   * Batch norm + activation + mask, then a convolution. A Winograd convolution applies the batch norm itself in its
   * input transform; otherwise the normalised tensor goes through `scratch`.
   */
  #bnActConv(
    bn: BatchNormLayer,
    activation: Activation,
    input: GPUBuffer,
    layer: ConvLayer,
    output: GPUBuffer,
    scratch: GPUBuffer,
    opts: { accumulate?: boolean; pre?: GPUBuffer } = {},
  ): void {
    const conv = { accumulate: opts.accumulate ?? false };
    if (this.#fusesInto(layer)) {
      this.#winogradConv(layer, input, output, conv, { bn, activation, ...(opts.pre ? { pre: opts.pre } : {}) });
      return;
    }
    this.#bnAct(bn, activation, input, scratch, opts.pre);
    this.#conv(layer, scratch, output, conv);
  }

  #winogradConv(
    layer: ConvLayer,
    input: GPUBuffer,
    output: GPUBuffer,
    opts: { accumulate?: boolean; nbias?: GPUBuffer },
    fused?: { bn: BatchNormLayer; activation: Activation; pre?: GPUBuffer },
  ): void {
    const { V, M, tilesX, tilesPerBoard, tileStride } = this.#wino!;
    const { inChannels: ic, outChannels: oc } = layer;
    const inFlags = fused ? 1 | (fused.pre ? 2 : 0) : 0;
    const act = fused ? ACT[fused.activation] : 0;
    this.#specs.push({
      kind: 'winoIn',
      code: WINO_IN(this.#elemOf(input), this.#elemOf(V)),
      buffers: [
        this.#uniform([ic, this.xSize, this.ySize, tilesX, tilesPerBoard, tileStride, inFlags, act, this.#desc.numInputChannels * this.#area, 0, 0, 0]),
        this.#dyn, input, V,
        fused ? this.#upload(mergedScaleBias(fused.bn)) : this.#dummy,
        fused?.pre ?? this.#dummy,
        this.#input,
      ],
      workgroups: (n) => [Math.ceil((n * tilesPerBoard) / 64), ic],
    });
    const U = this.#upload(winogradWeights(layer.weights, ic, oc), this.#elem);
    if (this.#gemm === 'v2') {
      // M[e][oc][t] = Σ_ic U[e][ic][oc] · V[e][ic][t], as GEMM v2 with z = e.
      this.#specs.push({
        kind: 'winoGemm',
        code: GEMM_V2(this.#elemOf(U), this.#elemOf(V), this.#elemOf(M), this.#gemmConfig),
        buffers: [
          this.#uniform([oc, ic, 0, tilesPerBoard, ic * oc, oc, ic * tileStride, tileStride, oc * tileStride, tileStride, 0, 0]),
          this.#dyn, U, V, M, this.#dummy,
        ],
        workgroups: (n) => [Math.ceil((n * tilesPerBoard) / gemmTile(this.#gemmConfig).tn), Math.ceil(oc / gemmTile(this.#gemmConfig).tm), 36],
      });
    } else {
      this.#specs.push({
        kind: 'winoGemm',
        code: WINO_GEMM(this.#elemOf(U), this.#elemOf(V), this.#elemOf(M)),
        buffers: [this.#uniform([ic, oc, tilesPerBoard, tileStride]), this.#dyn, U, V, M],
        workgroups: (n) => [Math.ceil((n * tilesPerBoard) / 64), Math.ceil(oc / 64), 36],
      });
    }
    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
    this.#specs.push({
      kind: 'winoOut',
      code: WINO_OUT(this.#elemOf(M), this.#elemOf(output)),
      buffers: [
        this.#uniform([oc, this.xSize, this.ySize, tilesX, tilesPerBoard, tileStride, flags, 0]),
        this.#dyn, M, output, opts.nbias ?? this.#dummy,
      ],
      workgroups: (n) => [Math.ceil((n * tilesPerBoard) / 64), oc],
    });
  }

  #bnAct(bn: BatchNormLayer, activation: Activation, input: GPUBuffer, output: GPUBuffer, pre?: GPUBuffer): void {
    const C = bn.numChannels;
    const area = this.#area;
    const params = this.#uniform([C, area, pre ? 1 : 0, ACT[activation], this.#desc.numInputChannels * area, 0, 0, 0]);
    const sb = mergedScaleBias(bn);
    this.#specs.push({
      kind: 'bnAct',
      code: BN_ACT(this.#elemOf(input), this.#elemOf(output)),
      buffers: [params, this.#dyn, input, this.#upload(sb), output, pre ?? this.#dummy, this.#input],
      workgroups: (n) => splitGroups(Math.ceil((n * C * area) / 256)),
    });
  }

  #gpool(input: GPUBuffer, channels: number, output: GPUBuffer, mode: 0 | 1): void {
    const area = this.#area;
    const params = this.#uniform([channels, area, mode, this.#desc.numInputChannels * area]);
    this.#specs.push({
      kind: 'gpool',
      code: GPOOL(this.#elemOf(input)),
      buffers: [params, this.#dyn, input, output, this.#input],
      workgroups: (n) => [channels, n],
    });
  }

  #matvec(
    layer: MatMulLayer,
    input: GPUBuffer,
    inStride: number,
    output: GPUBuffer,
    outStride: number,
    outOffset: number,
    bias?: MatBiasLayer,
    activation: Activation = 'identity',
  ): void {
    const params = this.#uniform([layer.inChannels, layer.outChannels, inStride, outStride, outOffset, bias ? 1 : 0, ACT[activation], 0]);
    this.#specs.push({
      kind: 'matvec',
      code: MATVEC,
      buffers: [params, this.#dyn, input, this.#upload(layer.weights), output, bias ? this.#upload(bias.weights) : this.#dummy],
      workgroups: (n) => [Math.ceil(layer.outChannels / 64), n],
    });
  }

  // ---------------------------------------------------------------- evaluating

  /** The number of compute dispatches in one evaluation. */
  get dispatchCount(): number {
    return this.#ops.length;
  }

  /**
   * Evaluate a batch of positions (at most `maxBatch`, each no larger than the plan's board). Calls are queued, so
   * one network evaluates one batch at a time.
   */
  evaluate(inputs: readonly NetInput[]): Promise<RawOutputs[]> {
    const run = this.#busy.then(() => this.#evaluate(inputs).then((r) => r.outputs));
    this.#busy = run.catch(() => undefined);
    return run;
  }

  /** Whether {@link profile} can run: the device has `timestamp-query`. */
  get canProfile(): boolean {
    return this.#timestamps;
  }

  /**
   * Evaluate a batch while timing it. Mode 'pass' times the normal plan (one compute pass) on the GPU. Mode 'ops' runs
   * every dispatch in its own compute pass to time each one; that adds pass overhead, so its wall time is not the
   * plan's.
   * @throws Error if the device has no `timestamp-query`.
   */
  profile(inputs: readonly NetInput[], mode: 'pass' | 'ops'): Promise<GpuProfile> {
    if (!this.#timestamps) throw new Error('this device has no timestamp-query');
    const run = this.#busy.then(async () => {
      const t0 = performance.now();
      const { timestamps } = await this.#evaluate(inputs, mode);
      const wallMs = performance.now() - t0;
      const ts = timestamps!;
      const delta = (i: number) => Number(ts[2 * i + 1]! - ts[2 * i]!);
      const result: GpuProfile = { batch: inputs.length, wallMs };
      if (mode === 'pass') result.passNs = delta(0);
      else result.ops = this.#ops.map((op, i) => ({ kind: op.kind, ns: delta(i) }));
      return result;
    });
    this.#busy = run.catch(() => undefined);
    return run;
  }

  async #evaluate(inputs: readonly NetInput[], profile?: 'pass' | 'ops'): Promise<{ outputs: RawOutputs[]; timestamps?: BigUint64Array }> {
    if (this.#destroyed) throw new Error('the network has been destroyed');
    const n = inputs.length;
    if (n === 0) return { outputs: [] };
    if (n > this.maxBatch) throw new Error(`batch of ${n} is larger than the plan's ${this.maxBatch}`);
    const { model, device, xSize, ySize } = this;
    const area = this.#area;
    const nIn = model.numInputChannels;
    const nGlobal = model.numInputGlobalChannels;

    const spatial = new Float32Array(n * nIn * area);
    const global = new Float32Array(n * nGlobal);
    inputs.forEach((input, b) => {
      if (input.xSize > xSize || input.ySize > ySize) throw new Error(`a ${input.xSize}×${input.ySize} board doesn't fit the plan's ${xSize}×${ySize}`);
      if (input.spatial.length !== nIn * input.xSize * input.ySize) throw new Error('spatial input has the wrong size');
      if (input.global.length !== nGlobal) throw new Error('global input has the wrong size');
      const inArea = input.xSize * input.ySize;
      for (let c = 0; c < nIn; c++) {
        for (let y = 0; y < input.ySize; y++) {
          const src = c * inArea + y * input.xSize;
          spatial.set(input.spatial.subarray(src, src + input.xSize), (b * nIn + c) * area + y * xSize);
        }
      }
      global.set(input.global, b * nGlobal);
    });
    device.queue.writeBuffer(this.#input, 0, spatial);
    device.queue.writeBuffer(this.#global, 0, global);
    device.queue.writeBuffer(this.#dyn, 0, new Uint32Array([n, 0, 0, 0]));

    const encoder = device.createCommandEncoder();
    let query: { set: GPUQuerySet; resolve: GPUBuffer; read: GPUBuffer; count: number } | undefined;
    if (profile) {
      const count = profile === 'pass' ? 2 : 2 * this.#ops.length;
      query = {
        set: device.createQuerySet({ type: 'timestamp', count }),
        resolve: device.createBuffer({ size: count * 8, usage: BufferUsage.QUERY_RESOLVE | BufferUsage.COPY_SRC }),
        read: device.createBuffer({ size: count * 8, usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST }),
        count,
      };
    }
    const dispatch = (pass: ReturnType<typeof encoder.beginComputePass>, op: Op) => {
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.bindGroup);
      const [x, y, z = 1] = op.workgroups(n);
      pass.dispatchWorkgroups(x, y, z);
    };
    if (profile === 'ops') {
      this.#ops.forEach((op, i) => {
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query!.set, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        dispatch(pass, op);
        pass.end();
      });
    } else {
      const pass = encoder.beginComputePass(
        query ? { timestampWrites: { querySet: query.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
      );
      for (const op of this.#ops) dispatch(pass, op);
      pass.end();
    }
    for (const out of this.#outputs) encoder.copyBufferToBuffer(out.buffer, 0, this.#staging, out.stagingOffset * 4, n * out.perBatch * 4);
    if (query) {
      encoder.resolveQuerySet(query.set, 0, query.count, query.resolve, 0);
      encoder.copyBufferToBuffer(query.resolve, 0, query.read, 0, query.count * 8);
    }
    device.queue.submit([encoder.finish()]);

    await this.#staging.mapAsync(MapMode.READ, 0, this.#stagingFloats * 4);
    const data = new Float32Array(this.#staging.getMappedRange(0, this.#stagingFloats * 4).slice(0));
    this.#staging.unmap();
    // Undo scale8: every raw output came out 1/8 as large. Multiplying by 8 is exact in f32.
    if (this.#outputFactor !== 1) for (let i = 0; i < data.length; i++) data[i]! *= this.#outputFactor;
    let timestamps: BigUint64Array | undefined;
    if (query) {
      await query.read.mapAsync(MapMode.READ);
      timestamps = new BigUint64Array(query.read.getMappedRange().slice(0));
      query.read.unmap();
      query.set.destroy();
      query.resolve.destroy();
      query.read.destroy();
    }

    const region = (name: OutputRegion['name'], b: number) => {
      const r = this.#outputs.find((o) => o.name === name)!;
      return data.subarray(r.stagingOffset + b * r.perBatch, r.stagingOffset + (b + 1) * r.perBatch);
    };
    const crop = (plane: Float32Array, input: NetInput) => {
      const out = new Float32Array(input.xSize * input.ySize);
      for (let y = 0; y < input.ySize; y++) out.set(plane.subarray(y * xSize, y * xSize + input.xSize), y * input.xSize);
      return out;
    };
    const outputs = inputs.map((input, b) => {
      const policyAll = region('policy', b);
      const policy: Float32Array[] = [];
      for (let c = 0; c < model.policyHead.policyOutChannels; c++) policy.push(crop(policyAll.subarray(c * area, (c + 1) * area), input));
      return {
        xSize: input.xSize,
        ySize: input.ySize,
        policy,
        pass: region('pass', b).slice(),
        value: region('value', b).slice(),
        scoreValue: region('scoreValue', b).slice(),
        ownership: crop(region('ownership', b), input),
      };
    });
    return { outputs, ...(timestamps ? { timestamps } : {}) };
  }

  /** Free the network's GPU buffers. The device is left alone. */
  destroy(): void {
    this.#destroyed = true;
    for (const b of this.#owned) b.destroy();
    this.#owned.length = 0;
  }
}
