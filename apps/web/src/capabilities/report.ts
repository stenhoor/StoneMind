// The capability report (build plan, Milestone 1, step 2): what one browser on one device supports, as plain JSON so it
// can be copied into an issue. `summarize` turns it into the rows the page shows; it has no DOM, so node --test covers it.

export interface GpuProbe {
  api: boolean;
  adapter: boolean;
  vendor?: string;
  architecture?: string;
  description?: string;
  fallbackAdapter?: boolean;
  features?: string[];
  limits?: Record<string, number>;
  error?: string;
}

export interface WorkerProbe {
  started: boolean;
  webgpu?: { api: boolean; adapter: boolean; vendor?: string; error?: string };
  opfs?: { api: boolean; syncAccessHandle: boolean; error?: string };
  error?: string;
}

export interface CapabilityReport {
  kind: 'stonemind-capabilities';
  version: 1;
  date: string;
  userAgent: string;
  hardwareConcurrency: number;
  deviceMemory?: number;
  devicePixelRatio: number;
  screen: { width: number; height: number };
  webgpu: GpuProbe;
  dedicatedWorker: WorkerProbe;
  sharedWorker: WorkerProbe & { api: boolean };
  opfs: { api: boolean; createWritable: boolean; error?: string };
  storage: { estimate: boolean; quota?: number; usage?: number; persistApi: boolean; persisted?: boolean };
  webLocks: { api: boolean; works: boolean };
  broadcastChannel: { api: boolean; works: boolean };
  css: { backdropFilter: boolean; backdropFilterSvg: boolean; containerQueries: boolean };
}

/** One line of the page: `ok` is true (supported), false (not supported) or null (not applicable or not tested). */
export interface Row {
  group: string;
  feature: string;
  ok: boolean | null;
  detail: string;
}

const MB = 1024 * 1024;

function bytes(n: number | undefined): string {
  if (n === undefined) return '';
  if (n >= 1024 * MB) return `${(n / (1024 * MB)).toFixed(1)} GB`;
  return `${Math.round(n / MB)} MB`;
}

function workerRows(where: string, w: WorkerProbe): Row[] {
  if (!w.started) return [{ group: 'WebGPU', feature: `In a ${where}`, ok: false, detail: w.error ?? `The ${where} did not start` }];
  return [
    {
      group: 'WebGPU',
      feature: `In a ${where}`,
      ok: !!w.webgpu?.adapter,
      detail: w.webgpu?.error ?? (w.webgpu?.adapter ? w.webgpu.vendor || 'Adapter found' : w.webgpu?.api ? 'navigator.gpu, but no adapter' : 'No navigator.gpu'),
    },
  ];
}

export function summarize(r: CapabilityReport): Row[] {
  const rows: Row[] = [];
  const g = r.webgpu;
  rows.push({
    group: 'WebGPU',
    feature: 'In the page',
    ok: g.adapter,
    detail: g.error ?? (g.adapter ? [g.vendor, g.architecture, g.description].filter(Boolean).join(' · ') || 'Adapter found' : g.api ? 'navigator.gpu, but no adapter' : 'No navigator.gpu'),
  });
  if (g.adapter) {
    rows.push({ group: 'WebGPU', feature: 'Half precision (shader-f16)', ok: !!g.features?.includes('shader-f16'), detail: '' });
    rows.push({ group: 'WebGPU', feature: 'Software adapter', ok: null, detail: g.fallbackAdapter ? 'Yes: a fallback adapter, not the GPU' : 'No' });
    const max = g.limits?.['maxStorageBufferBindingSize'];
    if (max !== undefined) rows.push({ group: 'WebGPU', feature: 'Largest storage buffer', ok: null, detail: bytes(max) });
  }
  rows.push(...workerRows('dedicated worker', r.dedicatedWorker));
  if (!r.sharedWorker.api) rows.push({ group: 'WebGPU', feature: 'In a shared worker', ok: false, detail: 'No SharedWorker in this browser' });
  else rows.push(...workerRows('shared worker', r.sharedWorker));

  rows.push({ group: 'Storage', feature: 'Origin private file system (OPFS)', ok: r.opfs.api, detail: r.opfs.error ?? '' });
  rows.push({ group: 'Storage', feature: 'createWritable in the page', ok: r.opfs.createWritable, detail: '' });
  rows.push({
    group: 'Storage',
    feature: 'createSyncAccessHandle in a worker',
    ok: r.dedicatedWorker.opfs?.syncAccessHandle ?? false,
    detail: r.dedicatedWorker.opfs?.error ?? '',
  });
  rows.push({
    group: 'Storage',
    feature: 'Quota',
    ok: r.storage.estimate,
    detail: r.storage.estimate ? `${bytes(r.storage.usage)} used of ${bytes(r.storage.quota)}` : '',
  });
  rows.push({
    group: 'Storage',
    feature: 'Persistent storage',
    ok: r.storage.persistApi,
    detail: r.storage.persistApi ? (r.storage.persisted ? 'Files are kept' : 'Not granted yet') : '',
  });

  rows.push({ group: 'Between tabs', feature: 'Web Locks', ok: r.webLocks.works, detail: r.webLocks.api && !r.webLocks.works ? 'API present, but a lock was not granted' : '' });
  rows.push({ group: 'Between tabs', feature: 'BroadcastChannel', ok: r.broadcastChannel.works, detail: r.broadcastChannel.api && !r.broadcastChannel.works ? 'API present, but no message arrived' : '' });

  rows.push({ group: 'Glass', feature: 'backdrop-filter', ok: r.css.backdropFilter, detail: '' });
  rows.push({
    group: 'Glass',
    feature: 'SVG filters in backdrop-filter',
    ok: r.css.backdropFilterSvg,
    detail: 'What CSS.supports reports; whether the edge distortion draws is checked by eye',
  });
  rows.push({ group: 'Glass', feature: 'Container queries', ok: r.css.containerQueries, detail: '' });
  return rows;
}

/** The report as the page shows and downloads it: stable key order, two-space indent. */
export function reportJson(r: CapabilityReport): string {
  return JSON.stringify(r, null, 2) + '\n';
}
