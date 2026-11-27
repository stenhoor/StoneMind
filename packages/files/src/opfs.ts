// OpfsFileStore: the `FileStore` port (`@stonemind/storage`) on the browser's origin private file system (OPFS).
//
// - Writes use `createWritable` from the page where the browser has it, and otherwise a dedicated worker with
//   `createSyncAccessHandle` (src/sync-writer.ts). Both stream: a 100 MB weights file never sits whole in memory.
// - Every write, move and remove holds a Web Lock on the paths it touches, so two tabs never write the same file at
//   once. Without Web Locks it goes ahead unlocked.
// - After a change it fires a `change` event ({ paths }) here and, through a BroadcastChannel, in every other tab's
//   store, so open views can refresh.
import type { FileData, FileEntry, FileStore, FileWriteOptions } from '@stonemind/storage';
import { PathError, isWithin, join, nameOf, normalize, parentOf, segments } from './paths.js';

export type WriteMethod = 'auto' | 'writable' | 'worker';

export interface OpfsFileStoreOptions {
  /** Which way to write; 'auto' (the default) uses createWritable where the browser has it. */
  writeMethod?: WriteMethod;
  /** The BroadcastChannel name shared by every tab's store. */
  channel?: string;
}

/** Detail of a `change` event: the paths that changed (written, moved from or to, or removed). */
export interface ChangeDetail {
  readonly paths: readonly string[];
  /** True when another tab made the change. */
  readonly remote: boolean;
}

/** A path that does not exist, or exists as the other kind (a file where a folder was expected, or the reverse). */
export class NotFoundError extends Error {
  override readonly name = 'NotFoundError';
}

const LOCK_PREFIX = 'stonemind-files:';
const CHUNK = 1 << 20;

type Writable = { write(data: BufferSource | Blob): Promise<void>; close(): Promise<void>; abort(): Promise<void> };
type MovableHandle = FileSystemHandle & { move?: (dir: FileSystemDirectoryHandle, name: string) => Promise<void> };

function hasCreateWritable(): boolean {
  return typeof FileSystemFileHandle !== 'undefined' && 'createWritable' in FileSystemFileHandle.prototype;
}

async function withLocks<T>(paths: string[], fn: () => Promise<T>): Promise<T> {
  const locks = (globalThis.navigator as Navigator & { locks?: LockManager } | undefined)?.locks;
  if (!locks) return fn();
  // Always take locks in the same order, so two tabs can't wait on each other.
  const names = [...new Set(paths)].sort().map((p) => LOCK_PREFIX + p);
  const take = (i: number): Promise<T> => (i === names.length ? fn() : (locks.request(names[i] as string, () => take(i + 1)) as Promise<T>));
  return take(0);
}

/** The data as a stream of byte chunks, whatever form it came in. */
function toStream(data: FileData): ReadableStream<Uint8Array> {
  if (data instanceof ReadableStream) return data;
  if (data instanceof Blob) return data.stream();
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Blob([bytes as Uint8Array<ArrayBuffer>]).stream();
}

/** Re-chunks a stream into pieces of at most CHUNK bytes, so progress and the worker's messages stay small. */
async function* chunks(stream: ReadableStream<Uint8Array>, signal?: AbortSignal): AsyncGenerator<Uint8Array> {
  const reader = stream.getReader();
  try {
    for (;;) {
      if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError');
      const { done, value } = await reader.read();
      if (done) return;
      for (let i = 0; i < value.byteLength; i += CHUNK) yield value.subarray(i, Math.min(i + CHUNK, value.byteLength));
    }
  } finally {
    reader.releaseLock();
  }
}

export class OpfsFileStore extends EventTarget implements FileStore {
  readonly #root: FileSystemDirectoryHandle;
  readonly #method: 'writable' | 'worker';
  readonly #channel: BroadcastChannel | null;

  private constructor(root: FileSystemDirectoryHandle, method: 'writable' | 'worker', channel: string) {
    super();
    this.#root = root;
    this.#method = method;
    this.#channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(channel);
    this.#channel?.addEventListener('message', (e: MessageEvent) => {
      const paths = (e.data as { paths?: unknown })?.paths;
      if (Array.isArray(paths)) this.dispatchEvent(new CustomEvent<ChangeDetail>('change', { detail: { paths: paths.map(String), remote: true } }));
    });
  }

  /** Opens the origin's file system. Rejects if the browser has no OPFS. */
  static async open(options: OpfsFileStoreOptions = {}): Promise<OpfsFileStore> {
    if (typeof navigator === 'undefined' || typeof navigator.storage?.getDirectory !== 'function') throw new Error('This browser has no origin private file system');
    const root = await navigator.storage.getDirectory();
    const wanted = options.writeMethod ?? 'auto';
    const method = wanted === 'auto' ? (hasCreateWritable() ? 'writable' : 'worker') : wanted;
    return new OpfsFileStore(root, method, options.channel ?? 'stonemind-files');
  }

  /** How this store writes: from the page ('writable') or through a worker ('worker'). */
  get writeMethod(): 'writable' | 'worker' {
    return this.#method;
  }

  /** Stops listening to other tabs. */
  close(): void {
    this.#channel?.close();
  }

  // ------------------------------------------------------------- handles

  async #dir(path: string, create = false): Promise<FileSystemDirectoryHandle> {
    let d = this.#root;
    for (const s of segments(path)) {
      try {
        d = await d.getDirectoryHandle(s, { create });
      } catch (e) {
        throw this.#notFound(e, path, 'folder');
      }
    }
    return d;
  }

  async #file(path: string): Promise<FileSystemFileHandle> {
    const d = await this.#dir(parentOf(path));
    try {
      return await d.getFileHandle(nameOf(path));
    } catch (e) {
      throw this.#notFound(e, path, 'file');
    }
  }

  #notFound(e: unknown, path: string, kind: string): Error {
    const name = (e as { name?: string })?.name;
    if (name === 'NotFoundError' || name === 'TypeMismatchError' || name === 'TypeError') return new NotFoundError(`No ${kind} at ${path}`);
    return e instanceof Error ? e : new Error(String(e));
  }

  #changed(paths: string[]): void {
    this.dispatchEvent(new CustomEvent<ChangeDetail>('change', { detail: { paths, remote: false } }));
    this.#channel?.postMessage({ paths });
  }

  // ------------------------------------------------------------- reading

  async list(dir: string): Promise<FileEntry[]> {
    const base = normalize(dir);
    const d = await this.#dir(base);
    const out: FileEntry[] = [];
    for await (const [name, handle] of (d as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> }).entries()) {
      const path = join(base, name);
      if (handle.kind === 'file') {
        const f = await (handle as FileSystemFileHandle).getFile();
        out.push({ path, name, kind: 'file', size: f.size, lastModified: f.lastModified });
      } else out.push({ path, name, kind: 'directory' });
    }
    return out.sort((a, b) => (a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === 'directory' ? -1 : 1));
  }

  async stat(path: string): Promise<FileEntry | null> {
    const p = normalize(path);
    if (p === '/') return { path: '/', name: '', kind: 'directory' };
    let parent: FileSystemDirectoryHandle;
    try {
      parent = await this.#dir(parentOf(p));
    } catch (e) {
      if (e instanceof NotFoundError) return null;
      throw e;
    }
    const name = nameOf(p);
    try {
      const f = await (await parent.getFileHandle(name)).getFile();
      return { path: p, name, kind: 'file', size: f.size, lastModified: f.lastModified };
    } catch {
      try {
        await parent.getDirectoryHandle(name);
        return { path: p, name, kind: 'directory' };
      } catch {
        return null;
      }
    }
  }

  async read(path: string): Promise<Blob> {
    return (await this.#file(normalize(path))).getFile();
  }

  // ------------------------------------------------------------- writing

  async write(path: string, data: FileData, options: FileWriteOptions = {}): Promise<void> {
    const p = normalize(path);
    if (p === '/') throw new PathError('Cannot write to the root');
    await withLocks([p], async () => {
      const current = await this.stat(p);
      if (current?.kind === 'directory') throw new PathError(`${p} is a folder`);
      const existed = current?.kind === 'file';
      const stream = toStream(data);
      if (this.#method === 'writable') await this.#writeWritable(p, stream, options, existed);
      else await this.#writeWorker(p, stream, options);
    });
    this.#changed([p]);
  }

  async #writeWritable(path: string, stream: ReadableStream<Uint8Array>, options: FileWriteOptions, existed: boolean): Promise<void> {
    const dir = await this.#dir(parentOf(path), true);
    const handle = await dir.getFileHandle(nameOf(path), { create: true });
    // createWritable writes to a temporary file that replaces the old one only on close, so an abort keeps it.
    const w = (await (handle as FileSystemFileHandle & { createWritable(): Promise<Writable> }).createWritable()) as Writable;
    let written = 0;
    try {
      for await (const chunk of chunks(stream, options.signal)) {
        await w.write(chunk as Uint8Array<ArrayBuffer>);
        written += chunk.byteLength;
        options.onProgress?.(written);
      }
      await w.close();
    } catch (e) {
      await w.abort().catch(() => {});
      if (!existed) await dir.removeEntry(nameOf(path)).catch(() => {});
      throw e;
    }
  }

  async #writeWorker(path: string, stream: ReadableStream<Uint8Array>, options: FileWriteOptions): Promise<void> {
    const worker = new Worker(new URL('./sync-writer.js', import.meta.url), { type: 'module' });
    const ask = (message: unknown, transfer: Transferable[] = []) =>
      new Promise<Record<string, unknown>>((resolve, reject) => {
        worker.onmessage = (e) => ((e.data as { error?: string }).error ? reject(new Error((e.data as { error: string }).error)) : resolve(e.data));
        worker.onerror = (e) => reject(new Error(e.message || 'The file writer worker failed'));
        worker.postMessage(message, transfer);
      });
    try {
      await ask({ open: segments(path) });
      for await (const chunk of chunks(stream, options.signal)) {
        const copy = chunk.slice().buffer;
        const { written } = await ask({ chunk: copy }, [copy]);
        options.onProgress?.(written as number);
      }
      await ask({ close: true });
    } catch (e) {
      await ask({ abort: true }).catch(() => {});
      throw e;
    } finally {
      worker.terminate();
    }
  }

  async mkdir(path: string): Promise<void> {
    const p = normalize(path);
    await withLocks([p], () => this.#dir(p, true));
    this.#changed([p]);
  }

  async move(from: string, to: string): Promise<void> {
    const a = normalize(from);
    const b = normalize(to);
    if (a === '/' || b === '/') throw new PathError('The root cannot be moved');
    if (isWithin(b, a)) throw new PathError(`Cannot move ${a} into itself`);
    await withLocks([a, b], async () => {
      const source = await this.stat(a);
      if (!source) throw new NotFoundError(`Nothing at ${a}`);
      if (await this.stat(b)) throw new PathError(`${b} already exists`);
      await this.#dir(parentOf(b), true);
      if (source.kind === 'file' && (await this.#moveNative(a, b))) return;
      await this.#copy(a, b, source.kind);
      await (await this.#dir(parentOf(a))).removeEntry(nameOf(a), { recursive: true });
    });
    this.#changed([a, b]);
  }

  /** A file's own move(), where the browser has one. Returns false to fall back to copying. */
  async #moveNative(a: string, b: string): Promise<boolean> {
    const handle = (await this.#file(a)) as MovableHandle;
    if (typeof handle.move !== 'function') return false;
    try {
      await handle.move(await this.#dir(parentOf(b)), nameOf(b));
      return true;
    } catch {
      return false;
    }
  }

  async #copy(a: string, b: string, kind: 'file' | 'directory'): Promise<void> {
    if (kind === 'file') {
      const blob = await this.read(a);
      if (this.#method === 'writable') await this.#writeWritable(b, blob.stream(), {}, false);
      else await this.#writeWorker(b, blob.stream(), {});
      return;
    }
    await this.#dir(b, true);
    for (const entry of await this.list(a)) await this.#copy(entry.path, join(b, entry.name), entry.kind);
  }

  async remove(path: string, options: { recursive?: boolean } = {}): Promise<void> {
    const p = normalize(path);
    if (p === '/') throw new PathError('The root cannot be removed');
    await withLocks([p], async () => {
      if (!(await this.stat(p))) throw new NotFoundError(`Nothing at ${p}`);
      await (await this.#dir(parentOf(p))).removeEntry(nameOf(p), { recursive: options.recursive ?? false });
    });
    this.#changed([p]);
  }
}
