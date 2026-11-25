/**
 * The small part of the WebGPU API this engine uses, declared here because TypeScript's DOM library doesn't include
 * WebGPU yet. Names and shapes follow the WebGPU specification (https://www.w3.org/TR/webgpu/).
 */

export interface GPU {
  requestAdapter(options?: { powerPreference?: 'low-power' | 'high-performance' }): Promise<GPUAdapter | null>;
}

export interface GPUAdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

export interface GPUAdapter {
  readonly features: ReadonlySet<string>;
  readonly limits: Readonly<Record<string, number>>;
  readonly info?: GPUAdapterInfo;
  requestDevice(descriptor?: { requiredFeatures?: string[]; requiredLimits?: Record<string, number> }): Promise<GPUDevice>;
}

export interface GPUDevice {
  readonly features: ReadonlySet<string>;
  readonly limits: Readonly<Record<string, number>>;
  readonly queue: GPUQueue;
  readonly lost: Promise<{ reason: string; message: string }>;
  createBuffer(descriptor: { size: number; usage: number; mappedAtCreation?: boolean; label?: string }): GPUBuffer;
  createShaderModule(descriptor: { code: string; label?: string }): GPUShaderModule;
  createComputePipelineAsync(descriptor: {
    layout: 'auto';
    compute: { module: GPUShaderModule; entryPoint: string; constants?: Record<string, number> };
    label?: string;
  }): Promise<GPUComputePipeline>;
  createBindGroup(descriptor: {
    layout: GPUBindGroupLayout;
    entries: { binding: number; resource: { buffer: GPUBuffer; offset?: number; size?: number } }[];
    label?: string;
  }): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  createQuerySet(descriptor: { type: 'timestamp'; count: number }): GPUQuerySet;
  pushErrorScope(filter: 'validation' | 'out-of-memory' | 'internal'): void;
  popErrorScope(): Promise<{ message: string } | null>;
  destroy(): void;
}

export interface GPUBuffer {
  readonly size: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

export interface GPUQueue {
  writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBufferView | ArrayBuffer, dataOffset?: number, size?: number): void;
  submit(commandBuffers: GPUCommandBuffer[]): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface GPUCommandEncoder {
  beginComputePass(descriptor?: { label?: string; timestampWrites?: GPUComputePassTimestampWrites }): GPUComputePassEncoder;
  resolveQuerySet(querySet: GPUQuerySet, firstQuery: number, queryCount: number, destination: GPUBuffer, destinationOffset: number): void;
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  finish(): GPUCommandBuffer;
}

export interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, group: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface GPUShaderModule {
  getCompilationInfo(): Promise<{ messages: readonly { type: string; message: string; lineNum: number; linePos: number }[] }>;
}

export interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

export interface GPUComputePassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

export interface GPUQuerySet {
  readonly count: number;
  destroy(): void;
}

// Opaque handles.
export interface GPUBindGroupLayout {
  readonly __brand?: 'GPUBindGroupLayout';
}
export interface GPUBindGroup {
  readonly __brand?: 'GPUBindGroup';
}
export interface GPUCommandBuffer {
  readonly __brand?: 'GPUCommandBuffer';
}

/** GPUBufferUsage flags. */
export const BufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  QUERY_RESOLVE: 0x0200,
} as const;

/** GPUMapMode flags. */
export const MapMode = { READ: 0x0001, WRITE: 0x0002 } as const;
