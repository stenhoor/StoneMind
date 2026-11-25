import { AnalysisError, type ActionRequest, type AnalysisEngine, type AnalyzeOptions, type EngineDescriptor } from './engine.js';
import {
  expectedFinals,
  isActionQuery,
  isAnalysisResponse,
  isErrorResponse,
  isWarningResponse,
  type ActionQuery,
  type ActionReply,
  type AnalysisQuery,
  type AnalysisResponse,
  type TerminateAction,
  type WarningResponse,
} from './protocol.js';

/** A line-based connection to an engine: one JSON object per line, each way. */
export interface LineTransport {
  send(line: string): void;
  /** Register the receiver for lines from the engine. Called once, before any send. */
  onLine(receiver: (line: string) => void): void;
  /** Register the receiver for the connection closing, with a reason. */
  onClose(receiver: (reason: string) => void): void;
  close(): void;
}

interface OpenQuery {
  remaining: number;
  queue: AnalysisResponse[];
  wake: (() => void) | undefined;
  failure: AnalysisError | undefined;
  onWarning: ((w: WarningResponse) => void) | undefined;
}

interface PendingAction {
  resolve: (reply: unknown) => void;
  reject: (error: AnalysisError) => void;
}

/**
 * An `AnalysisEngine` over any line transport that speaks KataGo's Analysis Engine protocol: a WebSocket to the local
 * bridge, a worker running the browser engine, or a test double. Routes every reply to its query by id.
 */
export class JsonLineSession implements AnalysisEngine {
  readonly #transport: LineTransport;
  readonly #queries = new Map<string, OpenQuery>();
  readonly #actions = new Map<string, PendingAction>();
  #nextActionId = 0;
  #closed: AnalysisError | undefined;

  /** Called for general errors (an error with no query id). */
  onGeneralError: ((error: AnalysisError) => void) | undefined;

  constructor(
    readonly descriptor: EngineDescriptor,
    transport: LineTransport,
  ) {
    this.#transport = transport;
    transport.onLine((line) => this.#receive(line));
    transport.onClose((reason) => this.#fail(new AnalysisError(`engine connection closed: ${reason}`)));
  }

  analyze(query: AnalysisQuery, options: AnalyzeOptions = {}): AsyncIterable<AnalysisResponse> {
    if (this.#closed) throw this.#closed;
    if (this.#queries.has(query.id) || this.#actions.has(query.id)) {
      throw new AnalysisError(`a query with id "${query.id}" is already open`, query.id);
    }
    const open: OpenQuery = {
      remaining: expectedFinals(query),
      queue: [],
      wake: undefined,
      failure: undefined,
      onWarning: options.onWarning,
    };
    this.#queries.set(query.id, open);
    this.#transport.send(JSON.stringify(query));

    const { signal } = options;
    const onAbort = () => {
      if (this.#queries.has(query.id)) {
        void this.action<TerminateAction>({ action: 'terminate', terminateId: query.id }).catch(() => {});
      }
    };
    if (signal?.aborted) onAbort();
    else signal?.addEventListener('abort', onAbort, { once: true });

    const queries = this.#queries;
    return {
      async *[Symbol.asyncIterator]() {
        try {
          for (;;) {
            const next = open.queue.shift();
            if (next) {
              yield next;
              continue;
            }
            if (open.failure) throw open.failure;
            if (open.remaining === 0) return;
            await new Promise<void>((resolve) => (open.wake = resolve));
            open.wake = undefined;
          }
        } finally {
          signal?.removeEventListener('abort', onAbort);
          queries.delete(query.id);
        }
      },
    };
  }

  action<A extends ActionQuery>(action: ActionRequest<A>): Promise<ActionReply<A>> {
    if (this.#closed) return Promise.reject(this.#closed);
    const id = `action-${++this.#nextActionId}`;
    return new Promise((resolve, reject) => {
      this.#actions.set(id, { resolve: resolve as (r: unknown) => void, reject });
      this.#transport.send(JSON.stringify({ ...action, id }));
    });
  }

  async close(): Promise<void> {
    if (!this.#closed) {
      this.#transport.close();
      this.#fail(new AnalysisError('engine closed'));
    }
  }

  #receive(line: string): void {
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch {
      this.onGeneralError?.(new AnalysisError(`unparseable line from engine: ${line.slice(0, 200)}`));
      return;
    }

    if (isErrorResponse(message)) {
      const error = new AnalysisError(message.error, message.id, message.field);
      const id = message.id;
      if (id !== undefined && this.#actions.has(id)) {
        this.#actions.get(id)!.reject(error);
        this.#actions.delete(id);
      } else if (id !== undefined && this.#queries.has(id)) {
        const open = this.#queries.get(id)!;
        open.failure = error;
        open.wake?.();
      } else {
        this.onGeneralError?.(error);
      }
      return;
    }

    if (isWarningResponse(message)) {
      this.#queries.get(message.id)?.onWarning?.(message);
      return;
    }

    if (isActionQuery(message)) {
      const pending = this.#actions.get(message.id);
      if (pending) {
        this.#actions.delete(message.id);
        pending.resolve(message);
      }
      return;
    }

    if (isAnalysisResponse(message)) {
      const open = this.#queries.get(message.id);
      if (!open) return; // a late result for a query the caller stopped reading
      open.queue.push(message);
      if (!message.isDuringSearch) open.remaining = Math.max(0, open.remaining - 1);
      open.wake?.();
    }
  }

  #fail(error: AnalysisError): void {
    if (this.#closed) return;
    this.#closed = error;
    for (const open of this.#queries.values()) {
      if (open.remaining > 0) open.failure = error;
      open.wake?.();
    }
    for (const pending of this.#actions.values()) pending.reject(error);
    this.#actions.clear();
  }
}
