/**
 * Storage ports.
 *
 * The architecture places all persistence on the Smaltworks platform (SQL + Cypher graph +
 * DataFrames, persisted to OPFS). Smaltworks is not yet available as a dependency of this
 * repository, so core code depends only on these narrow ports. A Smaltworks adapter will
 * implement them; tests can use in-memory adapters.
 *
 * Keep these minimal until the Smaltworks API is known: they define what StoneMind needs,
 * not what Smaltworks offers.
 */

/** Relational data: players, games, goals, study plans, reviews, journals, resources. */
export interface RelationalStore {
  query<Row = Record<string, unknown>>(sql: string, params?: readonly unknown[]): Promise<Row[]>;
  execute(sql: string, params?: readonly unknown[]): Promise<void>;
}

/** Graph data: Go ontology, player graph, style graph, resource graph. */
export interface GraphStore {
  cypher<Row = Record<string, unknown>>(
    query: string,
    params?: Record<string, unknown>,
  ): Promise<Row[]>;
}

/** Analytical data: performance trends, tsumego success rates, style evolution. */
export interface AnalyticsStore {
  /** Placeholder until the Smaltworks DataFrame API is specified. */
  readonly kind: "analytics";
}

export interface StoneMindStorage {
  relational: RelationalStore;
  graph: GraphStore;
  analytics: AnalyticsStore;
}

/**
 * Files the player keeps: game records, KataGo weights, configs. A tree of folders and files, addressed by absolute
 * '/'-separated paths such as "/weights/kata1-b18c384nbt.bin.gz". `@stonemind/files` implements it on the browser's
 * origin private file system (OPFS).
 */
export interface FileEntry {
  /** The absolute path, e.g. "/weights/kata1-b18c384nbt.bin.gz". */
  readonly path: string;
  readonly name: string;
  readonly kind: "file" | "directory";
  /** Bytes; files only. */
  readonly size?: number;
  /** Milliseconds since the epoch; files only. */
  readonly lastModified?: number;
}

export interface FileWriteOptions {
  /** Called as bytes are written, with the total so far. */
  onProgress?: (written: number) => void;
  /** Stops the write; a file that was being created is removed. */
  signal?: AbortSignal;
}

export type FileData = Blob | ArrayBuffer | ArrayBufferView | string | ReadableStream<Uint8Array>;

export interface FileStore {
  /** The folder's entries, folders first, then by name. */
  list(dir: string): Promise<FileEntry[]>;
  /** The entry at `path`, or null if there is none. */
  stat(path: string): Promise<FileEntry | null>;
  read(path: string): Promise<Blob>;
  /** Creates or replaces the file, creating missing folders on the way. */
  write(path: string, data: FileData, options?: FileWriteOptions): Promise<void>;
  /** Creates the folder and any missing parents. */
  mkdir(path: string): Promise<void>;
  /** Moves or renames a file or folder. The destination must not exist. */
  move(from: string, to: string): Promise<void>;
  /** Removes a file, or a folder (which must be empty unless `recursive`). */
  remove(path: string, options?: { recursive?: boolean }): Promise<void>;
}
