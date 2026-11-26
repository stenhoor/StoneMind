/**
 * Device setup. Same shape as smaltkit/gpu's shared device, but it asks for the optional features the network can use
 * (`shader-f16`, `subgroups`, `timestamp-query`) when the adapter has them, and for the adapter's largest buffers.
 */
import type { GPU, GPUAdapterInfo, GPUDevice } from './webgpu.js';

export const OPTIONAL_FEATURES = ['shader-f16', 'subgroups', 'timestamp-query'] as const;
export type OptionalFeature = (typeof OPTIONAL_FEATURES)[number];

export interface EngineDevice {
  device: GPUDevice;
  /** The optional features the device was created with. */
  features: ReadonlySet<OptionalFeature>;
  /**
   * Every feature the adapter offers, sorted, including ones the engine doesn't use yet (such as
   * `chromium-experimental-subgroup-matrix`, the route to matrix hardware). For reports only.
   */
  adapterFeatures: readonly string[];
  /** What the adapter says about itself, where the browser exposes it. */
  adapterInfo: Partial<GPUAdapterInfo>;
}

export class WebGpuUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGpuUnavailableError';
  }
}

/** The browser's WebGPU entry point, or undefined (no WebGPU, or not a secure context). */
export function getGpu(): GPU | undefined {
  return (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu;
}

/**
 * Create a device for the engine.
 * @param want Optional features to ask for (all of them by default); each is requested only if the adapter has it.
 * @throws WebGpuUnavailableError when there is no WebGPU or no adapter.
 */
export async function requestEngineDevice(want: readonly OptionalFeature[] = OPTIONAL_FEATURES): Promise<EngineDevice> {
  const gpu = getGpu();
  if (!gpu) throw new WebGpuUnavailableError('this browser has no WebGPU (or the page is not a secure context)');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGpuUnavailableError('no WebGPU adapter is available');
  const requiredFeatures = want.filter((f) => adapter.features.has(f));
  const requiredLimits: Record<string, number> = {};
  for (const limit of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize']) {
    const v = adapter.limits[limit];
    if (v !== undefined) requiredLimits[limit] = v;
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  const info = adapter.info;
  return {
    device,
    features: new Set(requiredFeatures),
    adapterFeatures: [...adapter.features].sort(),
    adapterInfo: info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : {},
  };
}
