// The capability page (build plan, Milestone 1, step 2): runs every probe, shows the results, and offers the report as
// JSON to copy or download. No probe asks the player for anything; "Ask to keep files" is a button they choose to press.
import { type CapabilityReport, type GpuProbe, type WorkerProbe, reportJson, summarize } from './report.js';

// The few WebGPU shapes the probe reads. The DOM library has no WebGPU types, and outside type packages need approval.
interface GpuAdapterLike {
  readonly features: ReadonlySet<string>;
  readonly limits: Record<string, number>;
  readonly info?: { vendor?: string; architecture?: string; description?: string; isFallbackAdapter?: boolean };
  readonly isFallbackAdapter?: boolean;
}
interface GpuLike {
  requestAdapter(): Promise<GpuAdapterLike | null>;
}

const TIMEOUT = 8000;

function withTimeout<T>(promise: Promise<T>, what: string, ms = TIMEOUT): Promise<T> {
  return Promise.race([promise, new Promise<never>((_, reject) => setTimeout(() => reject(new Error(`${what} did not answer within ${ms} ms`)), ms))]);
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

const LIMITS = ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX'];

async function probeGpu(): Promise<GpuProbe> {
  const gpu = (navigator as Navigator & { gpu?: GpuLike }).gpu;
  const out: GpuProbe = { api: !!gpu, adapter: false };
  if (!gpu) return out;
  try {
    const adapter = await withTimeout(gpu.requestAdapter(), 'requestAdapter');
    if (!adapter) return out;
    out.adapter = true;
    const info = adapter.info ?? {};
    out.vendor = info.vendor ?? '';
    out.architecture = info.architecture ?? '';
    out.description = info.description ?? '';
    out.fallbackAdapter = info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false;
    out.features = [...adapter.features].sort();
    out.limits = Object.fromEntries(LIMITS.filter((k) => typeof adapter.limits[k] === 'number').map((k) => [k, adapter.limits[k] as number]));
  } catch (e) {
    out.error = message(e);
  }
  return out;
}

type Port = Pick<MessagePort, 'postMessage'> & { onmessage: ((e: MessageEvent) => void) | null };

function askWorker(port: Port, onerror: (handler: (e: Event) => void) => void): Promise<WorkerProbe> {
  return withTimeout(
    new Promise<WorkerProbe>((resolve) => {
      port.onmessage = (e) => resolve({ started: true, ...(e.data as Omit<WorkerProbe, 'started'>) });
      onerror((e) => resolve({ started: false, error: (e as ErrorEvent).message || 'The worker failed to load' }));
      port.postMessage({ probe: true });
    }),
    'The worker',
    TIMEOUT * 2,
  ).catch((e) => ({ started: false, error: message(e) }));
}

const WORKER_URL = new URL('worker-probe.js', import.meta.url);

async function probeDedicated(): Promise<WorkerProbe> {
  try {
    const worker = new Worker(WORKER_URL);
    const result = await askWorker(worker, (h) => worker.addEventListener('error', h));
    worker.terminate();
    return result;
  } catch (e) {
    return { started: false, error: message(e) };
  }
}

async function probeShared(): Promise<WorkerProbe & { api: boolean }> {
  if (typeof SharedWorker === 'undefined') return { api: false, started: false };
  try {
    const worker = new SharedWorker(WORKER_URL, { name: 'stonemind-capability-probe' });
    worker.port.start();
    const result = await askWorker(worker.port, (h) => worker.addEventListener('error', h));
    worker.port.close();
    return { api: true, ...result };
  } catch (e) {
    return { api: true, started: false, error: message(e) };
  }
}

async function probeOpfs(): Promise<CapabilityReport['opfs']> {
  const out: CapabilityReport['opfs'] = { api: typeof navigator.storage?.getDirectory === 'function', createWritable: false };
  if (!out.api) return out;
  try {
    const root = await withTimeout(navigator.storage.getDirectory(), 'getDirectory');
    const name = `capability-probe-${Math.random().toString(36).slice(2)}`;
    const file = await root.getFileHandle(name, { create: true });
    const writable = (file as FileSystemFileHandle & { createWritable?: () => Promise<FileSystemWritableFileStream> }).createWritable;
    if (typeof writable === 'function') {
      const stream = await writable.call(file);
      await stream.write(new Uint8Array([1, 2, 3, 4]));
      await stream.close();
      out.createWritable = (await file.getFile()).size === 4;
    }
    await root.removeEntry(name);
  } catch (e) {
    out.error = message(e);
  }
  return out;
}

async function probeStorage(): Promise<CapabilityReport['storage']> {
  const s = navigator.storage;
  const out: CapabilityReport['storage'] = { estimate: typeof s?.estimate === 'function', persistApi: typeof s?.persist === 'function' };
  try {
    if (out.estimate) {
      const e = await s.estimate();
      if (e.quota !== undefined) out.quota = e.quota;
      if (e.usage !== undefined) out.usage = e.usage;
    }
    if (typeof s?.persisted === 'function') out.persisted = await s.persisted();
  } catch {
    // Reported as missing above.
  }
  return out;
}

async function probeLocks(): Promise<CapabilityReport['webLocks']> {
  const locks = (navigator as Navigator & { locks?: LockManager }).locks;
  if (!locks) return { api: false, works: false };
  try {
    const works = await withTimeout(locks.request('stonemind-capability-probe', async (lock) => lock !== null), 'navigator.locks.request');
    return { api: true, works };
  } catch {
    return { api: true, works: false };
  }
}

async function probeBroadcast(): Promise<CapabilityReport['broadcastChannel']> {
  if (typeof BroadcastChannel === 'undefined') return { api: false, works: false };
  const name = `stonemind-capability-probe-${Math.random().toString(36).slice(2)}`;
  const a = new BroadcastChannel(name);
  const b = new BroadcastChannel(name);
  try {
    const works = await withTimeout(
      new Promise<boolean>((resolve) => {
        b.onmessage = (e) => resolve(e.data === 'ping');
        a.postMessage('ping');
      }),
      'BroadcastChannel',
      3000,
    );
    return { api: true, works };
  } catch {
    return { api: true, works: false };
  } finally {
    a.close();
    b.close();
  }
}

function probeCss(): CapabilityReport['css'] {
  const supports = (p: string, v: string) => typeof CSS !== 'undefined' && CSS.supports(p, v);
  return {
    backdropFilter: supports('backdrop-filter', 'blur(2px)') || supports('-webkit-backdrop-filter', 'blur(2px)'),
    backdropFilterSvg: supports('backdrop-filter', 'url(#a)'),
    containerQueries: supports('container-type', 'inline-size'),
  };
}

export async function runProbes(): Promise<CapabilityReport> {
  const [webgpu, dedicatedWorker, sharedWorker, opfs, storage, webLocks, broadcastChannel] = await Promise.all([
    probeGpu(), probeDedicated(), probeShared(), probeOpfs(), probeStorage(), probeLocks(), probeBroadcast(),
  ]);
  const report: CapabilityReport = {
    kind: 'stonemind-capabilities',
    version: 1,
    date: new Date().toISOString(),
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency ?? 0,
    devicePixelRatio: window.devicePixelRatio,
    screen: { width: screen.width, height: screen.height },
    webgpu, dedicatedWorker, sharedWorker, opfs, storage, webLocks, broadcastChannel,
    css: probeCss(),
  };
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (memory !== undefined) report.deviceMemory = memory;
  return report;
}

// ----------------------------------------------------------------- page

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

function render(report: CapabilityReport): void {
  const body = $<HTMLTableSectionElement>('results');
  body.replaceChildren();
  let group = '';
  for (const row of summarize(report)) {
    if (row.group !== group) {
      group = row.group;
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.colSpan = 3;
      th.scope = 'colgroup';
      th.className = 'cap-group';
      th.textContent = group;
      tr.append(th);
      body.append(tr);
    }
    const tr = document.createElement('tr');
    const feature = document.createElement('th');
    feature.scope = 'row';
    feature.textContent = row.feature;
    const result = document.createElement('td');
    result.className = `cap-result cap-${row.ok === null ? 'info' : row.ok ? 'yes' : 'no'}`;
    // Status is never colour alone: an icon and a word.
    result.textContent = row.ok === null ? 'ℹ Info' : row.ok ? '✓ Yes' : '✗ No';
    const detail = document.createElement('td');
    detail.className = 'cap-detail';
    detail.textContent = row.detail;
    tr.append(feature, result, detail);
    body.append(tr);
  }
  const json = reportJson(report);
  $<HTMLPreElement>('json').textContent = json;
  const download = $<HTMLAnchorElement>('download');
  URL.revokeObjectURL(download.href);
  download.href = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
  download.download = `stonemind-capabilities-${report.date.slice(0, 10)}.json`;
  document.body.dataset['ready'] = 'true';
}

let current: CapabilityReport | null = null;

async function run(): Promise<void> {
  $<HTMLElement>('status').textContent = 'Checking this browser…';
  current = await runProbes();
  render(current);
  $<HTMLElement>('status').textContent = 'Done. Copy or download the report to share it.';
}

$<HTMLButtonElement>('copy').addEventListener('click', async () => {
  if (!current) return;
  try {
    await navigator.clipboard.writeText(reportJson(current));
    $<HTMLElement>('status').textContent = 'Report copied.';
  } catch {
    $<HTMLElement>('status').textContent = 'Copying is not allowed here; select the report below and copy it instead.';
  }
});

$<HTMLButtonElement>('persist').addEventListener('click', async () => {
  try {
    const granted = await navigator.storage.persist();
    $<HTMLElement>('status').textContent = granted ? 'This browser will keep StoneMind’s files.' : 'This browser did not agree to keep files.';
  } catch {
    $<HTMLElement>('status').textContent = 'This browser cannot be asked to keep files.';
  }
  await run();
});

$<HTMLButtonElement>('rerun').addEventListener('click', () => void run());

void run();
