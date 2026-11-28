// A tab's side of the engine host (build plan, Milestone 1, step 5): connects to the host worker, follows its status,
// starts and stops the engine, and gives the tab an `AnalysisEngine` for it, the same contract as native KataGo
// through the bridge.
//
// The host runs as a shared worker, so every tab of the app in this browser uses the one engine. Where a shared worker
// can't run WebGPU but this page can, the host runs as a dedicated worker for this tab only, and says so in its status.
import { type AnalysisEngine, type EngineDescriptor, JsonLineSession, type LineTransport } from '@stonemind/engine';
import type { EngineStatus, FromHost, HostStatus, StartRequest, ToHost } from './protocol.js';

export type HostMode = 'auto' | 'shared' | 'dedicated';

interface Channel {
  post(message: ToHost): void;
  onMessage(receiver: (message: FromHost) => void): void;
  close(): void;
}

const WORKER_URL = new URL('./host-worker.js', import.meta.url);
const NAME = 'stonemind-engines';

function sharedChannel(): Channel {
  const worker = new SharedWorker(WORKER_URL, { type: 'module', name: NAME });
  const port = worker.port;
  port.start();
  return {
    post: (m) => port.postMessage(m),
    onMessage: (r) => port.addEventListener('message', (e) => r(e.data as FromHost)),
    close: () => port.close(),
  };
}

function dedicatedChannel(): Channel {
  const worker = new Worker(WORKER_URL, { type: 'module', name: NAME });
  return {
    post: (m) => worker.postMessage(m),
    onMessage: (r) => worker.addEventListener('message', (e) => r(e.data as FromHost)),
    close: () => worker.terminate(),
  };
}

async function pageHasWebGpu(): Promise<boolean> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    return Boolean(await gpu?.requestAdapter());
  } catch {
    return false;
  }
}

export class EngineHostClient extends EventTarget {
  #channel: Channel;
  #status: HostStatus | null = null;
  #lines = new Map<string, Set<(line: string) => void>>();
  #closers = new Map<string, Set<(reason: string) => void>>();
  #sessions = new Map<string, JsonLineSession>();

  private constructor(channel: Channel) {
    super();
    this.#channel = channel;
    this.#listen(channel);
  }

  /**
   * Connects to the engine host. `auto` uses the shared host, unless it has no WebGPU while this page has: then a host
   * for this tab alone, which can.
   */
  static async connect(mode: HostMode = 'auto'): Promise<EngineHostClient> {
    const canShare = typeof SharedWorker === 'function';
    if (mode === 'dedicated' || (mode === 'auto' && !canShare)) return EngineHostClient.#open(dedicatedChannel());
    const client = await EngineHostClient.#open(sharedChannel());
    if (mode === 'auto' && !client.status.webgpu && (await pageHasWebGpu())) {
      client.#disconnect();
      return EngineHostClient.#open(dedicatedChannel());
    }
    return client;
  }

  static async #open(channel: Channel): Promise<EngineHostClient> {
    const client = new EngineHostClient(channel);
    const first = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('The engine host did not answer.')), 10_000);
      client.addEventListener('status', () => {
        clearTimeout(timer);
        resolve();
      }, { once: true });
    });
    channel.post({ type: 'hello' });
    await first;
    addEventListener('pagehide', () => client.#disconnect(), { once: true });
    return client;
  }

  #listen(channel: Channel): void {
    channel.onMessage((m) => {
      if (m.type === 'status') {
        this.#status = m.status;
        this.dispatchEvent(new CustomEvent('status', { detail: m.status }));
      } else if (m.type === 'line') {
        for (const r of this.#lines.get(m.engineId) ?? []) r(m.line);
      } else if (m.type === 'closed') {
        for (const r of this.#closers.get(m.engineId) ?? []) r(m.reason);
        this.#sessions.delete(m.engineId);
      }
    });
  }

  #disconnect(): void {
    this.#channel.post({ type: 'bye' });
    this.#channel.close();
  }

  /** The host's latest status. */
  get status(): HostStatus {
    if (!this.#status) throw new Error('not connected');
    return this.#status;
  }

  engineStatus(engineId: string): EngineStatus | undefined {
    return this.#status?.engines.find((e) => e.id === engineId);
  }

  /** Starts an engine (or restarts it with new settings). Resolves with its status once it is running or has failed. */
  start(request: StartRequest): Promise<EngineStatus> {
    const token = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    return new Promise((resolve) => {
      const onStatus = () => {
        const s = this.engineStatus(request.engineId);
        if (s?.token === token && (s.state === 'running' || s.state === 'failed')) {
          this.removeEventListener('status', onStatus);
          resolve(s);
        }
      };
      this.addEventListener('status', onStatus);
      this.#channel.post({ type: 'start', request: { ...request, token } });
    });
  }

  /** Stops an engine. Every tab's queries to it end. */
  stop(engineId: string): void {
    this.#channel.post({ type: 'stop', engineId });
  }

  /**
   * The engine as an `AnalysisEngine` for this tab: KataGo's protocol over the host. Queries from other tabs never
   * reach this tab's results.
   */
  engine(engineId: string, descriptor: EngineDescriptor = { id: engineId, kind: 'browser', displayName: 'StoneMind browser engine' }): AnalysisEngine {
    const existing = this.#sessions.get(engineId);
    if (existing) return existing;
    const lines = new Set<(line: string) => void>();
    const closers = new Set<(reason: string) => void>();
    this.#lines.set(engineId, lines);
    this.#closers.set(engineId, closers);
    const transport: LineTransport = {
      send: (line) => this.#channel.post({ type: 'line', engineId, line }),
      onLine: (r) => void lines.add(r),
      onClose: (r) => void closers.add(r),
      close: () => {
        lines.clear();
        closers.clear();
        this.#sessions.delete(engineId);
      },
    };
    const session = new JsonLineSession(descriptor, transport);
    this.#sessions.set(engineId, session);
    return session;
  }
}
