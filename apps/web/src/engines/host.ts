// The engine host's logic (build plan, Milestone 1, step 5): the engines it runs and the tabs connected to it. Free of
// workers and of the DOM, so it is tested under node with fake ports and engines; engines/host-worker.ts puts it in a
// shared worker (or, where a shared worker can't run WebGPU, a dedicated one for one tab).
//
// One engine runs for now. The host keeps a list so that more can be added later (one per GPU, a CPU engine, native
// KataGo through the bridge) without changing the tabs: each tab names the engine it talks to.
import { type EngineKind, type EngineStatus, type FromHost, type HostStatus, type StartRequest, type ToHost, engineConfig, fromEngine, toEngine } from './protocol.js';

/** A connection that speaks KataGo's protocol in lines (the browser engine's `KataGoAnalysisLines`, or a fake). */
export interface EngineLines {
  send(line: string): void;
  onLine(receiver: (line: string) => void): void;
  onClose(receiver: (reason: string) => void): void;
  close(): void;
}

export interface StartedEngine {
  readonly lines: EngineLines;
  /** What the status shows about it: the model, the GPU. */
  readonly info?: Pick<EngineStatus, 'model' | 'adapter'>;
}

/** Starts an engine of one kind, with KataGo's config. Rejects with a message the player can read. */
export type EngineFactory = (request: StartRequest, config: Readonly<Record<string, string>>) => Promise<StartedEngine>;

export interface TabPort {
  post(message: FromHost): void;
}

interface Hosted {
  status: EngineStatus;
  lines?: EngineLines;
}

/** How many engines may run at once. One for now. */
export const MAX_ENGINES = 1;

export class EngineHost {
  readonly #tabs = new Map<string, TabPort>();
  readonly #engines = new Map<string, Hosted>();
  // The queries each tab has sent, so its work can be stopped when it goes.
  readonly #sent = new Map<string, Set<string>>();
  #next = 0;
  #byeId = 0;

  constructor(
    readonly shared: boolean,
    readonly webgpu: boolean,
    readonly factories: Partial<Record<EngineKind, EngineFactory>>,
  ) {}

  status(): HostStatus {
    return { shared: this.shared, webgpu: this.webgpu, tabs: this.#tabs.size, engines: [...this.#engines.values()].map((e) => e.status) };
  }

  /** A tab connects. Returns what the tab's messages go to. */
  connect(port: TabPort): { receive(message: ToHost): void; disconnect(): void } {
    const prefix = `t${++this.#next}`;
    this.#tabs.set(prefix, port);
    this.#sent.set(prefix, new Set());
    this.#broadcast();
    return {
      receive: (message) => void this.#receive(prefix, message),
      disconnect: () => this.#disconnect(prefix),
    };
  }

  #broadcast(): void {
    const status = this.status();
    for (const port of this.#tabs.values()) port.post({ type: 'status', status });
  }

  async #receive(prefix: string, message: ToHost): Promise<void> {
    switch (message.type) {
      case 'hello':
        this.#tabs.get(prefix)?.post({ type: 'status', status: this.status() });
        return;
      case 'start':
        return this.#start(message.request);
      case 'stop':
        return this.#stop(message.engineId, 'stopped');
      case 'bye':
        return this.#disconnect(prefix);
      case 'line': {
        const hosted = this.#engines.get(message.engineId);
        const port = this.#tabs.get(prefix);
        if (!hosted?.lines || hosted.status.state !== 'running') {
          port?.post({ type: 'closed', engineId: message.engineId, reason: 'the engine is not running' });
          return;
        }
        const { line, refused } = toEngine(prefix, message.line);
        if (refused) {
          port?.post({ type: 'line', engineId: message.engineId, line: refused });
          return;
        }
        const id = idOf(message.line);
        if (id !== undefined && !isAction(message.line)) this.#sent.get(prefix)?.add(id);
        hosted.lines.send(line);
        return;
      }
    }
  }

  async #start(request: StartRequest): Promise<void> {
    const running = [...this.#engines.values()].filter((e) => e.status.id !== request.engineId && (e.status.state === 'running' || e.status.state === 'starting'));
    const config = engineConfig(request.config);
    const base = { id: request.engineId, kind: request.kind, weightsPath: request.weightsPath, config, ...(request.token ? { token: request.token } : {}) };
    if (running.length >= MAX_ENGINES) {
      this.#engines.set(request.engineId, { status: { ...base, state: 'failed', error: `Only ${MAX_ENGINES} engine runs at a time for now; stop ${running[0]?.status.id} first.` } });
      this.#broadcast();
      return;
    }
    await this.#stop(request.engineId, 'restarted');
    const factory = this.factories[request.kind];
    if (!factory) {
      this.#engines.set(request.engineId, { status: { ...base, state: 'failed', error: `This host can't run a ${request.kind} engine.` } });
      this.#broadcast();
      return;
    }
    const hosted: Hosted = { status: { ...base, state: 'starting' } };
    this.#engines.set(request.engineId, hosted);
    this.#broadcast();
    try {
      const started = await factory(request, config);
      if (this.#engines.get(request.engineId) !== hosted) {
        started.lines.close();
        return;
      }
      hosted.lines = started.lines;
      hosted.status = { ...hosted.status, ...started.info, state: 'running' };
      started.lines.onLine((line) => this.#fromEngine(request.engineId, line));
      started.lines.onClose((reason) => {
        if (this.#engines.get(request.engineId) !== hosted || hosted.status.state !== 'running') return;
        hosted.status = { ...hosted.status, state: 'failed', error: `The engine stopped: ${reason}` };
        delete hosted.lines;
        for (const port of this.#tabs.values()) port.post({ type: 'closed', engineId: request.engineId, reason });
        this.#broadcast();
      });
    } catch (e) {
      if (this.#engines.get(request.engineId) !== hosted) return;
      hosted.status = { ...hosted.status, state: 'failed', error: e instanceof Error ? e.message : String(e) };
    }
    this.#broadcast();
  }

  async #stop(engineId: string, reason: string): Promise<void> {
    const hosted = this.#engines.get(engineId);
    if (!hosted) return;
    this.#engines.delete(engineId);
    if (hosted.lines) {
      hosted.lines.close();
      for (const port of this.#tabs.values()) port.post({ type: 'closed', engineId, reason });
    }
    if (reason === 'stopped') this.#engines.set(engineId, { status: { ...hosted.status, state: 'stopped' } });
    this.#broadcast();
  }

  #fromEngine(engineId: string, raw: string): void {
    const { prefix, line } = fromEngine(raw);
    if (prefix === null) {
      for (const port of this.#tabs.values()) port.post({ type: 'line', engineId, line });
      return;
    }
    // Replies to a tab that has gone, and to the host's own terminates, go nowhere.
    this.#tabs.get(prefix)?.post({ type: 'line', engineId, line });
  }

  #disconnect(prefix: string): void {
    if (!this.#tabs.has(prefix)) return;
    this.#tabs.delete(prefix);
    // Stop the work the tab left behind: a terminate for each query it sent (one already finished just replies).
    const sent = this.#sent.get(prefix) ?? new Set();
    this.#sent.delete(prefix);
    for (const hosted of this.#engines.values()) {
      if (!hosted.lines || hosted.status.state !== 'running') continue;
      for (const id of sent) hosted.lines.send(JSON.stringify({ id: `host\u001fbye-${++this.#byeId}`, action: 'terminate', terminateId: `${prefix}\u001f${id}` }));
    }
    // With no tab left, nothing can use the engines.
    if (!this.#tabs.size) for (const id of [...this.#engines.keys()]) void this.#stop(id, 'every tab closed');
    this.#broadcast();
  }
}

function parse(line: string): Record<string, unknown> | null {
  try {
    const m: unknown = JSON.parse(line);
    return m && typeof m === 'object' && !Array.isArray(m) ? (m as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function idOf(line: string): string | undefined {
  const id = parse(line)?.['id'];
  return typeof id === 'string' ? id : undefined;
}

function isAction(line: string): boolean {
  return typeof parse(line)?.['action'] === 'string';
}
