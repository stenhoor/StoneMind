// Browser helpers around a FileStore: bringing files in (file picker, drag and drop), sending them out (download), and
// the origin's storage quota and persistence.
import type { FileStore } from '@stonemind/storage';
import { join } from './paths.js';

/** Opens the browser's file picker. Resolves with the chosen files, or none if the player cancels. */
export function pickFiles(options: { accept?: string; multiple?: boolean } = {}): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    if (options.accept) input.accept = options.accept;
    input.multiple = options.multiple ?? true;
    input.addEventListener('change', () => resolve([...(input.files ?? [])]), { once: true });
    input.addEventListener('cancel', () => resolve([]), { once: true });
    input.click();
  });
}

/** The files in a drop (folders and other items are left out). */
export function filesFromDrop(data: DataTransfer | null): File[] {
  return data ? [...data.files] : [];
}

export interface ImportProgress {
  /** Which file, counting from 0, and how many there are. */
  readonly index: number;
  readonly count: number;
  readonly name: string;
  /** Bytes of this file written so far, and its size. */
  readonly written: number;
  readonly size: number;
}

/**
 * Streams files into `dir`, one after another, keeping their names. Resolves with the paths written. A file that fails
 * stops the import; the files already written stay.
 */
export async function importFiles(store: FileStore, files: readonly File[], dir: string, options: { onProgress?: (p: ImportProgress) => void; signal?: AbortSignal } = {}): Promise<string[]> {
  const written: string[] = [];
  for (const [index, file] of files.entries()) {
    const path = join(dir, file.name);
    const report = (n: number) => options.onProgress?.({ index, count: files.length, name: file.name, written: n, size: file.size });
    report(0);
    await store.write(path, file.stream(), { onProgress: report, ...(options.signal ? { signal: options.signal } : {}) });
    written.push(path);
  }
  return written;
}

/** Saves a blob to the player's computer through the browser's download. */
export function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.style.display = 'none';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export interface StorageInfo {
  /** Bytes used and available to this origin, as the browser estimates them; absent if it can't say. */
  readonly usage?: number;
  readonly quota?: number;
  /** Whether the browser has agreed to keep this origin's files when space runs low. */
  readonly persisted: boolean;
}

export async function storageInfo(): Promise<StorageInfo> {
  const s = navigator.storage;
  const estimate = typeof s?.estimate === 'function' ? await s.estimate() : {};
  const persisted = typeof s?.persisted === 'function' ? await s.persisted() : false;
  return { ...(estimate.usage !== undefined ? { usage: estimate.usage } : {}), ...(estimate.quota !== undefined ? { quota: estimate.quota } : {}), persisted };
}

/** Asks the browser to keep this origin's files. Some browsers ask the player; resolves with the answer. */
export async function requestPersistence(): Promise<boolean> {
  return typeof navigator.storage?.persist === 'function' ? navigator.storage.persist() : false;
}
