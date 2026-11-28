// The Files app's catalog: the kind of each file, kept beside the files in this browser's storage.
//
// A file's kind is worked out from its name and content when it is imported (kinds.ts), and the player can change it
// (a game that is really a problem). The catalog keeps both, as JSON at CATALOG_PATH, so a kind is not worked out
// again on every visit and the player's choice lasts. The `.stonemind` folder is the app's own and is not listed.
// Files the catalog doesn't know (written by an older version, say) are worked out when they are first seen.
import type { FileEntry, FileStore } from '@stonemind/files';
import type { FileKind } from './kinds.js';

export const APP_FOLDER = '/.stonemind';
export const CATALOG_PATH = `${APP_FOLDER}/catalog.json`;

export interface CatalogEntry {
  readonly kind: FileKind;
  /** Who set the kind: worked out from the file, or chosen by the player. */
  readonly by: 'evaluated' | 'player';
}

export interface CatalogData {
  readonly version: 1;
  readonly files: Readonly<Record<string, CatalogEntry>>;
}

export const emptyCatalog = (): CatalogData => ({ version: 1, files: {} });

/** Whether a path is the app's own (hidden from the player). */
export const isAppPath = (path: string): boolean => path === APP_FOLDER || path.startsWith(APP_FOLDER + '/');

export async function loadCatalog(store: FileStore): Promise<CatalogData> {
  if (!(await store.stat(CATALOG_PATH))) return emptyCatalog();
  try {
    const data = JSON.parse(await (await store.read(CATALOG_PATH)).text()) as CatalogData;
    return data?.version === 1 && data.files ? data : emptyCatalog();
  } catch {
    return emptyCatalog();
  }
}

export async function saveCatalog(store: FileStore, data: CatalogData): Promise<void> {
  await store.write(CATALOG_PATH, new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
}

/** The catalog with `path` set to `entry`. */
export function withEntry(data: CatalogData, path: string, entry: CatalogEntry): CatalogData {
  return { version: 1, files: { ...data.files, [path]: entry } };
}

/** The catalog after moving `from` (a file, or a folder and everything in it) to `to`. */
export function withMove(data: CatalogData, from: string, to: string): CatalogData {
  const files: Record<string, CatalogEntry> = {};
  for (const [path, entry] of Object.entries(data.files)) {
    if (path === from) files[to] = entry;
    else if (path.startsWith(from + '/')) files[to + path.slice(from.length)] = entry;
    else files[path] = entry;
  }
  return { version: 1, files };
}

/** The catalog without `path` (a file, or a folder and everything in it). */
export function without(data: CatalogData, path: string): CatalogData {
  const files: Record<string, CatalogEntry> = {};
  for (const [p, entry] of Object.entries(data.files)) if (p !== path && !p.startsWith(path + '/')) files[p] = entry;
  return { version: 1, files };
}

/**
 * Every file and folder under `dir`, depth first, leaving out the app's own folder. A folder removed while it is being
 * walked (by another tab, say) is skipped.
 */
export async function walk(store: FileStore, dir = '/'): Promise<FileEntry[]> {
  const out: FileEntry[] = [];
  let entries: FileEntry[];
  try {
    entries = await store.list(dir);
  } catch (e) {
    if ((e as { name?: string }).name === 'NotFoundError') return out;
    throw e;
  }
  for (const e of entries) {
    if (isAppPath(e.path)) continue;
    out.push(e);
    if (e.kind === 'directory') out.push(...(await walk(store, e.path)));
  }
  return out;
}
