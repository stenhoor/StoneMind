/**
 * The network on WebGPU: weights uploaded once, a plan of dispatches built once for a board size and a largest batch,
 * then each evaluation is one write of the inputs, one submit and one readback.
 *
 * The layer order and maths follow the reference forward pass (`../reference/forward.ts`), which follows KataGo's
 * Eigen backend. Positions on smaller boards are placed in the top-left corner of the plan's board and masked, as
 * KataGo does with its fixed-size network buffers.
 */
import type { Activation, BatchNormLayer, Block, ConvLayer, MatMulLayer, MatBiasLayer, ModelDesc } from '../model/types.js';
import type { NetInput, RawOutputs } from '../reference/forward.js';
import type { EngineDevice } from './device.js';
import { ACT, BN_ACT, CONV, CONV_TILE_M, CONV_TILE_N, GPOOL, MATVEC, WINO_GEMM, WINO_IN, WINO_OUT, winogradWeights } from './kernels.js';
import { BufferUsage, MapMode, type GPUBindGroup, type GPUBuffer, type GPUComputePipeline, type GPUDevice } from './webgpu.js';

export interface GpuNetworkOptions {
  /** The plan's board size (default 19 × 19). Smaller boards fit in its top-left corner. */
  xSize?: number;
  ySize?: number;
  /** The largest batch the plan accepts (default 1). */
  maxBatch?: number;
  /** Run 3×3 convolutions with Winograd F(4×4,3×3) (default true); false uses the implicit GEMM for every size. */
  winograd?: boolean;
}

interface Op {
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
  #pipelines!: Record<'conv' | 'bnAct' | 'gpool' | 'matvec' | 'winoIn' | 'winoGemm' | 'winoOut', GPUComputePipeline>;
  #winograd = true;
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

  #buffer(floats: number, usage = STORAGE): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil((floats * 4) / 16) * 16), usage });
    this.#owned.push(b);
    return b;
  }

  #upload(data: Float32Array): GPUBuffer {
    const b = this.#buffer(data.length, BufferUsage.STORAGE | BufferUsage.COPY_DST);
    this.device.queue.writeBuffer(b, 0, data);
    return b;
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

  async #build(): Promise<void> {
    const { device, model } = this;
    const compile = async (code: string) =>
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    const [conv, bnAct, gpool, matvec, winoIn, winoGemm, winoOut] = await Promise.all(
      [CONV, BN_ACT, GPOOL, MATVEC, WINO_IN, WINO_GEMM, WINO_OUT].map(compile),
    );
    this.#pipelines = { conv: conv!, bnAct: bnAct!, gpool: gpool!, matvec: matvec!, winoIn: winoIn!, winoGemm: winoGemm!, winoOut: winoOut! };

    const B = this.maxBatch;
    const area = this.#area;
    const C = maxChannels(model);
    this.#dummy = this.#buffer(4);
    this.#dyn = this.#buffer(4, BufferUsage.UNIFORM | BufferUsage.COPY_DST);
    this.#input = this.#buffer(B * model.numInputChannels * area);
    this.#global = this.#buffer(B * model.numInputGlobalChannels);

    const t = model.trunk;
    const trunk = this.#buffer(B * t.trunkNumChannels * area);
    const inner = Array.from({ length: nestingDepth(t.blocks) }, () => this.#buffer(B * C * area));
    const s = {
      a: this.#buffer(B * C * area),
      m: this.#buffer(B * C * area),
      b: this.#buffer(B * C * area),
      g: this.#buffer(B * C * area),
      pooled: this.#buffer(B * 3 * C),
      vec: this.#buffer(B * C),
      vec2: this.#buffer(B * C),
    };
    if (this.#winograd) {
      const tilesX = Math.ceil(this.xSize / 4);
      const tilesPerBoard = tilesX * Math.ceil(this.ySize / 4);
      const tileStride = B * tilesPerBoard;
      this.#wino = { V: this.#buffer(36 * C * tileStride), M: this.#buffer(36 * C * tileStride), tilesX, tilesPerBoard, tileStride };
    }

    // Trunk.
    this.#matvec(t.initialMatMul, this.#global, model.numInputGlobalChannels, s.vec, t.trunkNumChannels, 0);
    this.#conv(t.initialConv, this.#input, trunk, { nbias: s.vec });
    const block = (blk: Block, target: GPUBuffer, depth: number): void => {
      switch (blk.kind) {
        case 'ordinary':
          this.#bnAct(blk.preBN, blk.preActivation.activation, target, s.a);
          this.#conv(blk.regularConv, s.a, s.m);
          this.#bnAct(blk.midBN, blk.midActivation.activation, s.m, s.b);
          this.#conv(blk.finalConv, s.b, target, { accumulate: true });
          return;
        case 'gpool':
          this.#bnAct(blk.preBN, blk.preActivation.activation, target, s.a);
          this.#conv(blk.regularConv, s.a, s.m);
          this.#conv(blk.gpoolConv, s.a, s.g);
          this.#bnAct(blk.gpoolBN, blk.gpoolActivation.activation, s.g, s.b);
          this.#gpool(s.b, blk.gpoolConv.outChannels, s.pooled, 0);
          this.#matvec(blk.gpoolToBiasMul, s.pooled, 3 * blk.gpoolConv.outChannels, s.vec, blk.regularConv.outChannels, 0);
          this.#bnAct(blk.midBN, blk.midActivation.activation, s.m, s.b, s.vec);
          this.#conv(blk.finalConv, s.b, target, { accumulate: true });
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
  }

  #conv(layer: ConvLayer, input: GPUBuffer, output: GPUBuffer, opts: { accumulate?: boolean; nbias?: GPUBuffer } = {}): void {
    if (this.#wino && layer.convYSize === 3 && layer.convXSize === 3 && layer.dilationY === 1 && layer.dilationX === 1) {
      this.#winogradConv(layer, input, output, opts);
      return;
    }
    const K = layer.inChannels * layer.convYSize * layer.convXSize;
    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
    const params = this.#uniform([
      layer.inChannels, layer.outChannels, layer.convYSize, layer.convXSize,
      layer.dilationY, layer.dilationX, this.xSize, this.ySize,
      K, flags, 0, 0,
    ]);
    const weights = this.#upload(convWeightsKxOc(layer));
    const pipeline = this.#pipelines.conv;
    const area = this.#area;
    this.#ops.push({
      pipeline,
      bindGroup: this.#bind(pipeline, [params, this.#dyn, input, weights, output, opts.nbias ?? this.#dummy]),
      workgroups: (n) => [Math.ceil((n * area) / CONV_TILE_N), Math.ceil(layer.outChannels / CONV_TILE_M)],
    });
  }

  #winogradConv(layer: ConvLayer, input: GPUBuffer, output: GPUBuffer, opts: { accumulate?: boolean; nbias?: GPUBuffer }): void {
    const { V, M, tilesX, tilesPerBoard, tileStride } = this.#wino!;
    const { inChannels: ic, outChannels: oc } = layer;
    const { winoIn, winoGemm, winoOut } = this.#pipelines;
    this.#ops.push({
      pipeline: winoIn,
      bindGroup: this.#bind(winoIn, [this.#uniform([ic, this.xSize, this.ySize, tilesX, tilesPerBoard, tileStride, 0, 0]), this.#dyn, input, V]),
      workgroups: (n) => [Math.ceil((n * tilesPerBoard) / 64), ic],
    });
    this.#ops.push({
      pipeline: winoGemm,
      bindGroup: this.#bind(winoGemm, [this.#uniform([ic, oc, tilesPerBoard, tileStride]), this.#dyn, this.#upload(winogradWeights(layer.weights, ic, oc)), V, M]),
      workgroups: (n) => [Math.ceil((n * tilesPerBoard) / 64), Math.ceil(oc / 64), 36],
    });
    const flags = (opts.accumulate ? 1 : 0) | (opts.nbias ? 2 : 0);
    this.#ops.push({
      pipeline: winoOut,
      bindGroup: this.#bind(winoOut, [
        this.#uniform([oc, this.xSize, this.ySize, tilesX, tilesPerBoard, tileStride, flags, 0]),
        this.#dyn, M, output, opts.nbias ?? this.#dummy,
      ]),
      workgroups: (n) => [Math.ceil((n * tilesPerBoard) / 64), oc],
    });
  }

  #bnAct(bn: BatchNormLayer, activation: Activation, input: GPUBuffer, output: GPUBuffer, pre?: GPUBuffer): void {
    const C = bn.numChannels;
    const area = this.#area;
    const params = this.#uniform([C, area, pre ? 1 : 0, ACT[activation], this.model.numInputChannels * area, 0, 0, 0]);
    const sb = new Float32Array(2 * C);
    sb.set(bn.mergedScale, 0);
    sb.set(bn.mergedBias, C);
    const pipeline = this.#pipelines.bnAct;
    this.#ops.push({
      pipeline,
      bindGroup: this.#bind(pipeline, [params, this.#dyn, input, this.#upload(sb), output, pre ?? this.#dummy, this.#input]),
      workgroups: (n) => splitGroups(Math.ceil((n * C * area) / 256)),
    });
  }

  #gpool(input: GPUBuffer, channels: number, output: GPUBuffer, mode: 0 | 1): void {
    const area = this.#area;
    const params = this.#uniform([channels, area, mode, this.model.numInputChannels * area]);
    const pipeline = this.#pipelines.gpool;
    this.#ops.push({
      pipeline,
      bindGroup: this.#bind(pipeline, [params, this.#dyn, input, output, this.#input]),
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
    const pipeline = this.#pipelines.matvec;
    this.#ops.push({
      pipeline,
      bindGroup: this.#bind(pipeline, [params, this.#dyn, input, this.#upload(layer.weights), output, bias ? this.#upload(bias.weights) : this.#dummy]),
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
    const run = this.#busy.then(() => this.#evaluate(inputs));
    this.#busy = run.catch(() => undefined);
    return run;
  }

  async #evaluate(inputs: readonly NetInput[]): Promise<RawOutputs[]> {
    if (this.#destroyed) throw new Error('the network has been destroyed');
    const n = inputs.length;
    if (n === 0) return [];
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
    const pass = encoder.beginComputePass();
    for (const op of this.#ops) {
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.bindGroup);
      const [x, y, z = 1] = op.workgroups(n);
      pass.dispatchWorkgroups(x, y, z);
    }
    pass.end();
    for (const out of this.#outputs) encoder.copyBufferToBuffer(out.buffer, 0, this.#staging, out.stagingOffset * 4, n * out.perBatch * 4);
    device.queue.submit([encoder.finish()]);

    await this.#staging.mapAsync(MapMode.READ, 0, this.#stagingFloats * 4);
    const data = new Float32Array(this.#staging.getMappedRange(0, this.#stagingFloats * 4).slice(0));
    this.#staging.unmap();

    const region = (name: OutputRegion['name'], b: number) => {
      const r = this.#outputs.find((o) => o.name === name)!;
      return data.subarray(r.stagingOffset + b * r.perBatch, r.stagingOffset + (b + 1) * r.perBatch);
    };
    const crop = (plane: Float32Array, input: NetInput) => {
      const out = new Float32Array(input.xSize * input.ySize);
      for (let y = 0; y < input.ySize; y++) out.set(plane.subarray(y * xSize, y * xSize + input.xSize), y * input.xSize);
      return out;
    };
    return inputs.map((input, b) => {
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
  }

  /** Free the network's GPU buffers. The device is left alone. */
  destroy(): void {
    this.#destroyed = true;
    for (const b of this.#owned) b.destroy();
    this.#owned.length = 0;
  }
}
