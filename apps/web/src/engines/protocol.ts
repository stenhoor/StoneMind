// The messages between the engine host (a shared worker, engines/host-worker.ts) and the app's tabs, and the query-id
// rewriting that lets every tab use one engine (build plan, Milestone 1, step 5). Free of the DOM and of workers, so it
// is tested under node.
//
// The engine speaks KataGo's Analysis Engine protocol: JSON lines, each query and reply carrying an `id`. Two tabs may
// both send a query "q1", so the host gives each tab a prefix and rewrites ids on the way in (`id`, and `terminateId`
// in a terminate action) and back on the way out, sending each reply only to the tab whose query it answers.

/** The engines the host can run. One for now; more (one per GPU, a CPU engine, native KataGo) come later. */
export type EngineKind = 'webgpu' | 'fake';

export type EngineState = 'stopped' | 'starting' | 'running' | 'failed';

export interface EngineStatus {
  readonly id: string;
  readonly kind: EngineKind;
  readonly state: EngineState;
  /** Why it failed, in plain words. */
  readonly error?: string;
  /** The weights file it runs, as a path in this browser's storage. */
  readonly weightsPath?: string;
  /** The model's name and version, from the file. */
  readonly model?: { readonly name: string; readonly version: number };
  /** The GPU it runs on, as the browser names it (Firefox names none). */
  readonly adapter?: { readonly vendor?: string; readonly architecture?: string; readonly description?: string };
  /** KataGo's config it runs with. */
  readonly config?: Readonly<Record<string, string>>;
  /** The token of the start request this status answers, so the tab that asked can tell it apart. */
  readonly token?: string;
}

export interface HostStatus {
  /** Whether the host is a shared worker (every tab uses it) or a dedicated one (this tab only). */
  readonly shared: boolean;
  /** Whether WebGPU gave the host an adapter. */
  readonly webgpu: boolean;
  /** How many tabs are connected. */
  readonly tabs: number;
  readonly engines: readonly EngineStatus[];
}

export interface StartRequest {
  readonly engineId: string;
  readonly kind: EngineKind;
  /** The weights file, as a path in this browser's storage (the Files app's store). */
  readonly weightsPath: string;
  /** KataGo config keys to set over the defaults (KATAGO_DEFAULTS). */
  readonly config?: Readonly<Record<string, string>>;
  /** Echoed in the engine's status, so the asking tab knows which start it answers. */
  readonly token?: string;
}

/** Tab → host. */
export type ToHost =
  | { readonly type: 'hello' }
  | { readonly type: 'start'; readonly request: StartRequest }
  | { readonly type: 'stop'; readonly engineId: string }
  | { readonly type: 'line'; readonly engineId: string; readonly line: string }
  | { readonly type: 'bye' };

/** Host → tab. */
export type FromHost =
  | { readonly type: 'status'; readonly status: HostStatus }
  | { readonly type: 'line'; readonly engineId: string; readonly line: string }
  | { readonly type: 'closed'; readonly engineId: string; readonly reason: string };

/**
 * KataGo's own defaults for the analysis engine, from its `analysis_example.cfg` (cpp/configs): the settings an engine
 * starts with unless the player changes them. `logDir` and the native cache sizes are accepted by the browser engine
 * and mean nothing there.
 */
export const KATAGO_DEFAULTS: Readonly<Record<string, string>> = {
  reportAnalysisWinratesAs: 'BLACK',
  maxVisits: '500',
  numAnalysisThreads: '2',
  numSearchThreadsPerAnalysisThread: '16',
  nnMaxBatchSize: '64',
  nnCacheSizePowerOfTwo: '23',
  nnMutexPoolSizePowerOfTwo: '17',
  nnRandomize: 'true',
};

const SEPARATOR = '\u001f';

/** A tab's query, as the engine sees it: ids prefixed with the tab's prefix. Lines that aren't JSON objects pass as they are. */
export function toEngine(prefix: string, line: string): { line: string; refused?: string } {
  let message: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(line);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return { line };
    message = parsed as Record<string, unknown>;
  } catch {
    return { line };
  }
  // terminate_all would stop every tab's queries; a tab stops its own with terminate.
  if (message['action'] === 'terminate_all') {
    const id = typeof message['id'] === 'string' ? message['id'] : undefined;
    return {
      line,
      refused: JSON.stringify({ ...(id !== undefined ? { id } : {}), error: 'terminate_all would stop every tab’s queries on the shared engine: terminate each query instead' }),
    };
  }
  const out = { ...message };
  if (typeof out['id'] === 'string') out['id'] = prefix + SEPARATOR + out['id'];
  if (typeof out['terminateId'] === 'string') out['terminateId'] = prefix + SEPARATOR + out['terminateId'];
  return { line: JSON.stringify(out) };
}

/**
 * Whom an engine's reply is for: the tab whose prefix its id carries, with the id (and a terminate reply's
 * `terminateId`) given back as the tab sent it; or every tab, for a line with no id (a general error).
 */
export function fromEngine(line: string): { prefix: string | null; line: string } {
  let message: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(line);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return { prefix: null, line };
    message = parsed as Record<string, unknown>;
  } catch {
    return { prefix: null, line };
  }
  const id = message['id'];
  if (typeof id !== 'string' || !id.includes(SEPARATOR)) return { prefix: null, line };
  const at = id.indexOf(SEPARATOR);
  const prefix = id.slice(0, at);
  const out: Record<string, unknown> = { ...message, id: id.slice(at + 1) };
  const t = out['terminateId'];
  if (typeof t === 'string' && t.startsWith(prefix + SEPARATOR)) out['terminateId'] = t.slice(prefix.length + 1);
  return { prefix, line: JSON.stringify(out) };
}

/** The config an engine starts with: KataGo's defaults, then the player's keys over them. */
export function engineConfig(overrides: Readonly<Record<string, string>> = {}): Record<string, string> {
  return { ...KATAGO_DEFAULTS, ...overrides };
}
