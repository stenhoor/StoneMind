import type { ActionQuery, ActionReply, AnalysisQuery, AnalysisResponse, WarningResponse } from './protocol.js';

/** An action without its id, kept per member of the union. */
export type ActionRequest<A extends ActionQuery = ActionQuery> = A extends unknown ? Omit<A, 'id'> : never;

/** Where an engine runs. */
export type EngineKind = 'local' | 'browser';

export interface EngineDescriptor {
  /** Stable id, e.g. "katago-local" or "stonemind-webgpu". */
  id: string;
  kind: EngineKind;
  displayName: string;
}

export interface AnalyzeOptions {
  /**
   * Stops the search early. The engine terminates the query and the stream still ends normally, after the final
   * result of every turn (which may carry `noResults: true`), so partial analysis isn't lost.
   */
  signal?: AbortSignal;
  /** Called for each parse warning the engine reports for this query. */
  onWarning?: (warning: WarningResponse) => void;
}

/**
 * The contract every StoneMind analysis engine implements: KataGo's Analysis Engine protocol, as an async API.
 * A native KataGo (through the local bridge) and the browser engine are interchangeable behind it.
 */
export interface AnalysisEngine {
  readonly descriptor: EngineDescriptor;

  /**
   * Analyse a query. Yields every result for it in arrival order: partial results (`isDuringSearch: true`) and
   * exactly one final result per analysed turn, then ends.
   * @throws AnalysisError when the engine rejects the query.
   */
  analyze(query: AnalysisQuery, options?: AnalyzeOptions): AsyncIterable<AnalysisResponse>;

  /** Send an action query and wait for its reply. The `id` is supplied by the engine. */
  action<A extends ActionQuery>(action: ActionRequest<A>): Promise<ActionReply<A>>;

  /** Stop every query and release the engine. */
  close(): Promise<void>;
}

/** An engine rejected a query or a field of it, or the connection to it failed. */
export class AnalysisError extends Error {
  constructor(
    message: string,
    readonly queryId?: string,
    readonly field?: string,
  ) {
    super(message);
    this.name = 'AnalysisError';
  }
}
