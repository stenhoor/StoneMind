// The Files app (build plan, Milestone 1, step 4): a file browser in the style of macOS Finder over the file kit
// (`@stonemind/files`). A singleton app: one instance, belonging to no workspace.
//
// Folders hold files; filters narrow what is shown:
// - Folders are where files are kept, starting at This browser. The player always browses a folder, and its folders
//   always show.
// - Filters (Games, Problems, Weights, Configs) are kinds of file. A file's kind is worked out from its name and
//   content when it is imported (kinds.ts), and the player can change a game to a problem and back. Any number of
//   filters can be on; the open folder then shows only files of those kinds. With none on, it shows every file.
// A file of no kind Files knows is refused. Beside them: this browser's storage; a sortable list; an inspector (for
// KataGo networks: the network from its name, the model version from its header, a SHA-256 on request); import from the
// file picker or by dropping files, streamed with progress and a Cancel button; new folder, move, rename, download and
// delete (with a confirmation).
import { type FileEntry, NotFoundError, OpfsFileStore, type StorageInfo, downloadBlob, filesFromDrop, importFiles, join, nameOf, normalize, parentOf, requestPersistence, storageInfo } from '@stonemind/files';
import type { Menu } from '../frame/menu.js';
import { type CatalogData, emptyCatalog, isAppPath, loadCatalog, saveCatalog, walk, withEntry, withMove, without } from './catalog.js';
import { ACCEPTED, type FileKind, KINDS, KIND_NAMES, evaluateKind } from './kinds.js';
import { type WeightsHeader, describeNetwork, formatBytes, grouped, parseWeightsName, readWeightsHeader, sha256 } from './weights.js';

type SortKey = 'name' | 'size' | 'modified';

interface ImportTask {
  name: string;
  index: number;
  count: number;
  written: number;
  size: number;
  abort: AbortController;
}

interface Column {
  readonly label: string;
  readonly key?: SortKey;
  readonly numeric?: boolean;
  readonly cell: (e: FileEntry) => string;
}

// The empty folders an earlier version of this app made at the top of this browser's storage, when it took kinds for
// folders. They are removed once if still empty; anything the player put in them stays.
const OLD_KIND_FOLDERS = ['/games', '/problems', '/weights', '/configs'];

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & { dataset?: Record<string, string> } = {}, ...children: (Node | string | null)[]): HTMLElementTagNameMap[K] {
  const { dataset, ...rest } = props;
  const node = Object.assign(document.createElement(tag), rest);
  if (dataset) Object.assign(node.dataset, dataset);
  node.append(...children.filter((c): c is Node | string => c !== null));
  return node;
}

function button(text: string, onClick: () => void, props: Partial<HTMLButtonElement> & { dataset?: Record<string, string> } = {}): HTMLButtonElement {
  const b = el('button', { type: 'button', className: 'sm-btn files-btn', ...props }, text);
  b.addEventListener('click', onClick);
  return b;
}

const dateFormat = new Intl.DateTimeFormat(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });

// Every change to the catalog, and the file operations it follows (move, rename, delete), runs under this Web Lock, so
// two refreshes, an import and a move, or two tabs, never write it from different snapshots.
const CATALOG_LOCK = 'stonemind-files-catalog';
const withCatalogLock = <T>(fn: () => Promise<T>): Promise<T> => navigator.locks.request(CATALOG_LOCK, fn) as Promise<T>;

/** A folder's name for the player: "This browser" for the top. */
export const folderName = (path: string): string => (path === '/' ? 'This browser' : path);

export class FilesView {
  readonly element = el('div', { className: 'files' });
  #store: OpfsFileStore | null = null;
  #error = '';
  #messages: { text: string; ok: boolean }[] = [];
  #folder = '/';
  #filters = new Set<FileKind>();
  #all: FileEntry[] = [];
  #catalog: CatalogData = emptyCatalog();
  #why = new Map<string, string>();
  #entries: FileEntry[] = [];
  #selected: string | null = null;
  #sort: { key: SortKey; descending: boolean } = { key: 'name', descending: false };
  #storage: StorageInfo | null = null;
  #task: ImportTask | null = null;
  #header = new Map<string, WeightsHeader | string>();
  #checksums = new Map<string, string>();
  #onChange: () => void;
  #shown = '';

  constructor(onMenusChanged: () => void = () => {}) {
    this.#onChange = onMenusChanged;
    void this.#open();
  }

  async #open(): Promise<void> {
    try {
      this.#store = await OpfsFileStore.open();
      for (const path of OLD_KIND_FOLDERS) {
        const e = await this.#store.stat(path);
        if (e?.kind === 'directory' && !(await this.#store.list(path)).length) await this.#store.remove(path);
      }
      this.#store.addEventListener('change', () => void this.refresh().catch((e: unknown) => {
        this.#error = e instanceof Error ? e.message : String(e);
        this.render();
      }));
      await this.refresh();
    } catch (e) {
      this.#error = e instanceof Error ? e.message : String(e);
      this.render();
    }
  }

  /** The kind of the file at `path`, if it has one. */
  kindOf(path: string): FileKind | undefined {
    return this.#catalog.files[path]?.kind;
  }

  async refresh(): Promise<void> {
    const store = this.#store;
    if (!store) return;
    await withCatalogLock(async () => {
      const all = await walk(store);
      const loaded = await loadCatalog(store);
      // Work out the kind of any file the catalog doesn't know yet.
      let catalog = loaded;
      for (const e of all) {
        if (e.kind !== 'file' || catalog.files[e.path]) continue;
        // A file that is being written or removed meanwhile is worked out on a later refresh.
        const result = await store.read(e.path).then((blob) => evaluateKind(e.name, blob)).catch(() => null);
        if (result && 'kind' in result) {
          catalog = withEntry(catalog, e.path, { kind: result.kind, by: 'evaluated' });
          this.#why.set(e.path, result.why);
        }
      }
      // Forget entries for files that are gone.
      for (const path of Object.keys(catalog.files)) if (!all.some((e) => e.path === path)) catalog = without(catalog, path);
      if (catalog !== loaded) await saveCatalog(store, catalog);
      this.#all = all;
      this.#catalog = catalog;
    });
    try {
      this.#entries = (await store.list(this.#folder)).filter((e) => !isAppPath(e.path));
    } catch (e) {
      if (!(e instanceof NotFoundError)) throw e;
      this.#folder = '/';
      this.#entries = (await store.list('/')).filter((e) => !isAppPath(e.path));
    }
    if (this.#selected && !this.#visible().some((e) => e.path === this.#selected)) this.#selected = null;
    this.#storage = await storageInfo().catch(() => null);
    // A refresh that finds nothing new (the change notice that follows the player's own action, say) leaves the page
    // alone: re-rendering would replace the controls under the player's hand, and a press in flight could be lost.
    const shown = JSON.stringify([this.#folder, [...this.#filters], this.#selected, this.#all, this.#catalog, this.#entries, this.#storage, this.#messages, this.#error]);
    if (shown === this.#shown) return;
    this.#shown = shown;
    this.render();
    this.#onChange();
  }

  // ---------------------------------------------------------- actions

  /** Turns a filter on or off. */
  toggleFilter(kind: FileKind): void {
    if (this.#filters.has(kind)) this.#filters.delete(kind);
    else this.#filters.add(kind);
    if (this.#selected && !this.#visible().some((e) => e.path === this.#selected)) this.#selected = null;
    this.render();
    this.#onChange();
  }

  /** Turns every filter off, so every file in the folder shows. */
  clearFilters(): void {
    this.#filters.clear();
    this.render();
    this.#onChange();
  }

  /** The open folder's contents that the filters let through: every folder, and the files of the kinds that are on. */
  #visible(): FileEntry[] {
    if (!this.#filters.size) return this.#entries;
    return this.#entries.filter((e) => {
      if (e.kind === 'directory') return true;
      const k = this.kindOf(e.path);
      return k !== undefined && this.#filters.has(k);
    });
  }

  /** Shows a folder's contents. */
  openFolder(path: string): void {
    this.#folder = normalize(path);
    this.#selected = null;
    this.#messages = [];
    void this.refresh();
  }

  get selected(): FileEntry | null {
    return this.#visible().find((e) => e.path === this.#selected) ?? null;
  }

  /** Where imported files and new folders go: the open folder. */
  get #target(): string {
    return this.#folder;
  }

  // One file input for the app's life. Whatever files it is given are imported when it fires `change`; `cancel` is
  // ignored. (Firefox under Playwright fires `cancel` and then `change` for the same pick, so a picker that settles
  // on the first of them sees nothing chosen.)
  #picker = (() => {
    const input = el('input', { type: 'file', multiple: true });
    input.addEventListener('change', () => {
      const files = [...(input.files ?? [])];
      input.value = '';
      if (files.length) void this.import(files);
    });
    return input;
  })();

  /** Opens the browser's file picker; the chosen files are imported. */
  importPicked(): void {
    this.#picker.click();
  }

  /** Imports files: each one's kind is worked out first, and files of no kind Files takes are refused. */
  async import(files: File[]): Promise<void> {
    const store = this.#store;
    if (!store || this.#task) return;
    this.#error = '';
    this.#messages = [];
    const accepted: { file: File; kind: FileKind; why: string }[] = [];
    for (const file of files) {
      const result = await evaluateKind(file.name, file);
      if ('kind' in result) accepted.push({ file, kind: result.kind, why: result.why });
      else this.#messages.push({ text: `Not imported: ${result.refused}`, ok: false });
    }
    if (!accepted.length) {
      this.render();
      return;
    }
    const dir = this.#target;
    const abort = new AbortController();
    this.#task = { name: '', index: 0, count: accepted.length, written: 0, size: 0, abort };
    this.render();
    let written: string[] = [];
    try {
      written = await importFiles(store, accepted.map((a) => a.file), dir, {
        signal: abort.signal,
        onProgress: (p) => {
          Object.assign(this.#task ?? {}, p);
          this.#renderTask();
        },
      });
    } catch (e) {
      if ((e as { name?: string }).name !== 'AbortError') this.#error = `Import failed: ${e instanceof Error ? e.message : String(e)}`;
    } finally {
      this.#task = null;
    }
    await withCatalogLock(async () => {
      let catalog = await loadCatalog(store);
      for (const [i, path] of written.entries()) {
        const a = accepted[i];
        if (!a) continue;
        catalog = withEntry(catalog, path, { kind: a.kind, by: 'evaluated' });
        this.#why.set(path, a.why);
        this.#messages.push({ text: `Imported ${a.file.name} as ${KIND_NAMES[a.kind].a} into ${folderName(dir)}. ${a.why}`, ok: true });
      }
      await saveCatalog(store, catalog);
    });
    await this.refresh();
  }

  /** Changes the kind of the selected file (a game to a problem, or back). */
  async setKind(path: string, kind: FileKind): Promise<void> {
    const store = this.#store;
    if (!store) return;
    await this.#run(() => withCatalogLock(async () => saveCatalog(store, withEntry(await loadCatalog(store), path, { kind, by: 'player' }))));
  }

  async newFolder(): Promise<void> {
    const name = await this.#ask('New folder', 'Name', '', 'Create');
    if (!name || !this.#store) return;
    await this.#run(() => this.#store?.mkdir(join(this.#target, name)));
  }

  async #moveTo(from: string, to: string): Promise<void> {
    const store = this.#store;
    if (!store) return;
    await withCatalogLock(async () => {
      await store.move(from, to);
      await saveCatalog(store, withMove(await loadCatalog(store), from, to));
    });
    if (this.#selected === from) this.#selected = to;
  }

  async rename(): Promise<void> {
    const entry = this.selected;
    if (!entry) return;
    const name = await this.#ask(`Rename ${entry.name}`, 'New name', entry.name, 'Rename');
    if (!name || name === entry.name) return;
    await this.#run(() => this.#moveTo(entry.path, join(parentOf(entry.path), name)));
  }

  /** Moves the selected file or folder into another folder. */
  async move(): Promise<void> {
    const entry = this.selected;
    if (!entry) return;
    const folders = ['/', ...this.#all.filter((e) => e.kind === 'directory').map((e) => e.path)]
      .filter((p) => p !== parentOf(entry.path) && p !== entry.path && !p.startsWith(entry.path + '/'));
    if (!folders.length) {
      this.#messages = [{ text: 'There is no other folder to move it to. Make one with New folder.', ok: false }];
      this.render();
      return;
    }
    const select = el('select', { className: 'files-input', id: 'files-folder' }, ...folders.map((p) => el('option', { value: p }, folderName(p))));
    if (!(await this.#dialog(`Move ${entry.name}`, [el('label', { htmlFor: 'files-folder', className: 'sm-label' }, 'To folder'), select], 'Move'))) return;
    await this.#run(() => this.#moveTo(entry.path, join(select.value, entry.name)));
  }

  async remove(): Promise<void> {
    const entry = this.selected;
    const store = this.#store;
    if (!entry || !store) return;
    const what = entry.kind === 'directory' ? `the folder ${entry.name} and everything in it` : entry.name;
    if (!(await this.#confirm(`Delete ${what}?`, 'It is removed from this browser’s storage. This cannot be undone.', 'Delete'))) return;
    await this.#run(() => withCatalogLock(async () => {
      await store.remove(entry.path, { recursive: true });
      await saveCatalog(store, without(await loadCatalog(store), entry.path));
    }));
  }

  async download(): Promise<void> {
    const entry = this.selected;
    if (!entry || entry.kind !== 'file' || !this.#store) return;
    downloadBlob(await this.#store.read(entry.path), entry.name);
  }

  async keepFiles(): Promise<void> {
    await requestPersistence().catch(() => false);
    await this.refresh();
  }

  async computeChecksum(path: string): Promise<void> {
    if (!this.#store) return;
    const redraw = () => this.element.querySelector('.files-inspector')?.replaceWith(this.#inspector());
    this.#checksums.set(path, 'Computing…');
    redraw();
    try {
      this.#checksums.set(path, await sha256(await this.#store.read(path)));
    } catch (e) {
      this.#checksums.set(path, `Could not compute it: ${e instanceof Error ? e.message : String(e)}`);
    }
    redraw();
  }

  async #run(action: () => Promise<unknown> | undefined): Promise<void> {
    try {
      this.#error = '';
      this.#messages = [];
      await action();
    } catch (e) {
      this.#error = e instanceof Error ? e.message : String(e);
    }
    await this.refresh();
  }

  /** The Files app's menus, for the menu bar while it is focused. */
  menus(): Menu[] {
    const s = this.selected;
    const sortItem = (key: SortKey, label: string) => ({ label: `Sort by ${label}`, checked: () => this.#sort.key === key, run: () => this.sortBy(key) });
    return [
      {
        label: 'File',
        items: [
          { label: 'Import…', run: () => this.importPicked() },
          { label: 'New folder…', run: () => void this.newFolder() },
          { label: 'Move to…', disabled: !s, run: () => void this.move() },
          { label: 'Download', disabled: s?.kind !== 'file', run: () => void this.download() },
          { label: 'Rename…', disabled: !s, run: () => void this.rename() },
          { label: 'Delete…', disabled: !s, run: () => void this.remove() },
        ],
      },
      {
        label: 'View',
        items: [
          ...KINDS.map((k) => ({ label: `Show ${KIND_NAMES[k].plural}`, checked: () => this.#filters.has(k), run: () => this.toggleFilter(k) })),
          { label: 'Show all files', disabled: !this.#filters.size, run: () => this.clearFilters() },
          sortItem('name', 'name'),
          sortItem('size', 'size'),
          sortItem('modified', 'date modified'),
        ],
      },
      {
        label: 'Go',
        items: this.#topFolders().map((path) => ({ label: folderName(path).replace(/^\//, ''), checked: () => this.#folder === path, run: () => this.openFolder(path) })),
      },
    ];
  }

  sortBy(key: SortKey): void {
    this.#sort = { key, descending: this.#sort.key === key ? !this.#sort.descending : key !== 'name' };
    this.render();
  }

  // ---------------------------------------------------------- dialogs

  #dialog(title: string, body: Node[], ok: string): Promise<boolean> {
    return new Promise((resolve) => {
      const dialog = el('dialog', { className: 'files-dialog' });
      const form = el('form', { method: 'dialog' }, el('h3', { className: 'files-dialog-title' }, title), ...body,
        // The default button comes first, so Enter in the field chooses it; the row is reversed to show it on the right.
        el('div', { className: 'files-dialog-actions' },
          el('button', { type: 'submit', value: 'ok', className: 'sm-btn sm-btn--primary' }, ok),
          el('button', { type: 'submit', value: 'cancel', className: 'sm-btn', formNoValidate: true }, 'Cancel')));
      dialog.append(form);
      dialog.addEventListener('close', () => {
        resolve(dialog.returnValue === 'ok');
        dialog.remove();
      });
      this.element.append(dialog);
      dialog.showModal();
    });
  }

  async #ask(title: string, label: string, value: string, ok: string): Promise<string | null> {
    const input = el('input', { type: 'text', value, required: true, className: 'files-input', id: 'files-name' });
    input.setAttribute('autocomplete', 'off');
    const yes = await this.#dialog(title, [el('label', { htmlFor: 'files-name', className: 'sm-label' }, label), input], ok);
    const name = input.value.trim();
    return yes && name ? name : null;
  }

  #confirm(title: string, text: string, ok: string): Promise<boolean> {
    return this.#dialog(title, [el('p', {}, text)], ok);
  }

  // ---------------------------------------------------------- rendering

  render(): void {
    // Keep focus on the same control (by its data-key) across a re-render.
    const focused = document.activeElement;
    const active = focused instanceof HTMLElement && this.element.contains(focused) ? focused.dataset['key'] : undefined;
    if (!this.#store) {
      this.element.replaceChildren(el('p', { className: 'files-note' }, this.#error ? `Files can’t open this browser’s storage: ${this.#error}` : 'Opening this browser’s storage…'));
      return;
    }
    // Keep where each pane is scrolled: a re-render would otherwise jump a long list back to the top under the player's
    // pointer (and a double-click's second click would land on whatever moved there).
    const panes = ['.files-body', '.files-side', '.files-main', '.files-inspector'];
    const scroll = panes.map((sel) => this.element.querySelector(sel)?.scrollTop ?? 0);
    this.element.replaceChildren(this.#toolbar(), el('div', { className: 'files-body' }, this.#sidebar(), this.#main(), this.#inspector()));
    panes.forEach((sel, i) => {
      const pane = this.element.querySelector(sel);
      if (pane) pane.scrollTop = scroll[i] ?? 0;
    });
    if (active) this.element.querySelector<HTMLElement>(`[data-key="${CSS.escape(active)}"]`)?.focus({ preventScroll: true });
  }

  /** This browser and the folders at its top. */
  #topFolders(): string[] {
    return ['/', ...this.#all.filter((e) => e.kind === 'directory' && parentOf(e.path) === '/').map((e) => e.path).sort()];
  }

  #toolbar(): HTMLElement {
    const where = el('nav', { className: 'files-path' });
    where.setAttribute('aria-label', 'Folder');
    const parts = this.#folder.split('/').filter(Boolean);
    const crumb = (label: string, path: string, last: boolean) => {
      const b = button(label, () => this.openFolder(path), { className: 'files-crumb', dataset: { key: `crumb:${path}` } });
      if (last) b.setAttribute('aria-current', 'location');
      return b;
    };
    where.append(crumb('This browser', '/', parts.length === 0));
    parts.forEach((part, i) => where.append(el('span', { className: 'files-crumb-sep' }, '›'), crumb(part, '/' + parts.slice(0, i + 1).join('/'), i === parts.length - 1)));
    const s = this.selected;
    const bar = el('div', { className: 'files-toolbar' }, where, el('span', { className: 'files-spacer' }),
      button('Import…', () => this.importPicked(), { className: 'sm-btn sm-btn--primary files-btn', dataset: { key: 'import' } }),
      button('New folder…', () => void this.newFolder(), { dataset: { key: 'mkdir' } }),
      button('Move to…', () => void this.move(), { disabled: !s, dataset: { key: 'move' } }),
      button('Download', () => void this.download(), { disabled: s?.kind !== 'file', dataset: { key: 'download' } }),
      button('Rename…', () => void this.rename(), { disabled: !s, dataset: { key: 'rename' } }),
      button('Delete…', () => void this.remove(), { disabled: !s, dataset: { key: 'delete' } }));
    bar.setAttribute('role', 'toolbar');
    bar.setAttribute('aria-label', 'Files');
    return bar;
  }

  #sidebar(): HTMLElement {
    const side = el('div', { className: 'files-side' });
    const folder = this.#folder;

    const folders = el('nav', { className: 'files-side-group' }, el('p', { className: 'sm-label files-side-label' }, 'Folders'));
    folders.setAttribute('aria-label', 'Folders');
    for (const path of this.#topFolders()) {
      const b = button(`${path === '/' ? '' : '📁︎ '}${path === '/' ? 'This browser' : nameOf(path)}`, () => this.openFolder(path), { className: 'files-place', dataset: { key: `folder:${path}` } });
      const inside = path === '/' ? folder === '/' : folder === path || folder.startsWith(path + '/');
      if (inside) b.setAttribute('aria-current', 'true');
      folders.append(b);
    }

    // Filters: toggles, any number on. Each count is the files of that kind in the open folder.
    const kinds = el('div', { className: 'files-side-group' }, el('p', { className: 'sm-label files-side-label', id: 'files-filters-label' }, 'Filters'));
    kinds.setAttribute('role', 'group');
    kinds.setAttribute('aria-labelledby', 'files-filters-label');
    for (const k of KINDS) {
      const count = this.#entries.filter((e) => e.kind === 'file' && this.kindOf(e.path) === k).length;
      const b = button('', () => this.toggleFilter(k), { className: 'files-place files-filter', dataset: { key: `filter:${k}` } });
      b.append(el('span', { className: 'files-check' }, this.#filters.has(k) ? '✓' : ''), el('span', { className: 'files-filter-name' }, KIND_NAMES[k].plural), el('span', { className: 'files-count' }, String(count)));
      b.setAttribute('aria-pressed', String(this.#filters.has(k)));
      b.setAttribute('aria-label', `${KIND_NAMES[k].plural}, ${count} here`);
      kinds.append(b);
    }
    kinds.append(button('Show all files', () => this.clearFilters(), { className: 'files-place', disabled: !this.#filters.size, dataset: { key: 'filter:none' } }));

    const info = this.#storage;
    const used = info?.usage ?? 0;
    const quota = info?.quota ?? 0;
    const meter = el('div', { className: 'files-meter' }, el('div', { className: 'files-meter-fill' }));
    meter.setAttribute('role', 'meter');
    meter.setAttribute('aria-label', 'Storage used');
    meter.setAttribute('aria-valuemin', '0');
    meter.setAttribute('aria-valuemax', String(quota));
    meter.setAttribute('aria-valuenow', String(used));
    (meter.firstElementChild as HTMLElement).style.width = quota ? `${Math.min(100, (used / quota) * 100)}%` : '0';
    const storage = el('div', { className: 'files-storage' },
      el('p', { className: 'sm-label files-side-label' }, 'Storage'),
      meter,
      el('p', { className: 'files-storage-text' }, info?.quota !== undefined ? `${formatBytes(used)} used of ${formatBytes(quota)}` : 'Space used is not known'),
      el('p', { className: 'files-storage-text' }, info?.persisted ? '✓ Files are kept' : 'The browser may clear these files when space runs low.'),
      info?.persisted ? null : button('Keep my files', () => void this.keepFiles(), { dataset: { key: 'persist' } }));

    side.append(folders, kinds, storage);
    return side;
  }

  #columns(): Column[] {
    const name: Column = { label: 'Name', key: 'name', cell: (e) => e.name };
    const size: Column = { label: 'Size', key: 'size', numeric: true, cell: (e) => (e.kind === 'file' ? formatBytes(e.size) : '—') };
    const modified: Column = { label: 'Modified', key: 'modified', cell: (e) => (e.lastModified ? dateFormat.format(e.lastModified) : '—') };
    const kind: Column = { label: 'Kind', cell: (e) => (e.kind === 'directory' ? 'Folder' : (() => { const k = this.kindOf(e.path); return k ? KIND_NAMES[k].one : 'Not recognised'; })()) };
    // A Network column whenever networks are shown.
    if (this.#visible().some((e) => this.kindOf(e.path) === 'weights')) {
      const network: Column = { label: 'Network', cell: (e) => { if (this.kindOf(e.path) !== 'weights') return ''; const w = parseWeightsName(e.name); return w ? `b${w.blocks}c${w.channels}${w.kind}` : '—'; } };
      return [name, kind, network, size, modified];
    }
    return [name, kind, size, modified];
  }

  #sorted(): FileEntry[] {
    const { key, descending } = this.#sort;
    const value = (e: FileEntry) => (key === 'size' ? e.size ?? -1 : key === 'modified' ? e.lastModified ?? 0 : e.name.toLowerCase());
    return [...this.#visible()].sort((a, b) => {
      if (a.kind !== b.kind) return a.kind === 'directory' ? -1 : 1;
      const x = value(a);
      const y = value(b);
      const c = x < y ? -1 : x > y ? 1 : 0;
      return descending ? -c : c;
    });
  }

  #main(): HTMLElement {
    const main = el('div', { className: 'files-main' });
    main.addEventListener('dragover', (e) => {
      e.preventDefault();
      main.classList.add('files-drop');
    });
    main.addEventListener('dragleave', () => main.classList.remove('files-drop'));
    main.addEventListener('drop', (e) => {
      e.preventDefault();
      main.classList.remove('files-drop');
      const files = filesFromDrop(e.dataTransfer);
      if (files.length) void this.import(files);
    });
    if (this.#error) {
      const error = el('p', { className: 'files-error' }, `✗ ${this.#error}`);
      error.setAttribute('role', 'alert');
      main.append(error);
    }
    if (this.#messages.length) {
      const list = el('ul', { className: 'files-messages' }, ...this.#messages.map((m) => el('li', { className: m.ok ? 'files-ok' : 'files-refused' }, `${m.ok ? '✓' : '✗'} ${m.text}`)));
      list.setAttribute('role', 'status');
      main.append(list);
    }
    main.append(el('div', { className: 'files-task-host' }));
    const cols = this.#columns();
    const head = el('tr', {}, ...cols.map((c) => {
      const th = el('th', { scope: 'col', className: c.numeric ? 'files-num' : '' });
      if (c.key) {
        const on = this.#sort.key === c.key;
        th.setAttribute('aria-sort', on ? (this.#sort.descending ? 'descending' : 'ascending') : 'none');
        th.append(button(`${c.label}${on ? (this.#sort.descending ? ' ↓' : ' ↑') : ''}`, () => this.sortBy(c.key as SortKey), { className: 'files-sort', dataset: { key: `sort:${c.key}` } }));
      } else th.textContent = c.label;
      return th;
    }));
    const body = el('tbody');
    for (const e of this.#sorted()) {
      const selected = e.path === this.#selected;
      const tr = el('tr', { className: selected ? 'files-row files-selected' : 'files-row', dataset: { path: e.path } });
      tr.setAttribute('aria-selected', String(selected));
      cols.forEach((c, i) => {
        const td = el('td', { className: c.numeric ? 'files-num' : '' });
        if (i === 0) {
          const name = button(`${e.kind === 'directory' ? '📁︎ ' : ''}${e.name}`, () => {
            if (e.kind === 'directory' && this.#selected === e.path) this.openFolder(e.path);
            else {
              this.#select(e.path);
              void this.#loadHeader(e);
            }
          }, { className: 'files-name', dataset: { key: `row:${e.path}` } });
          name.setAttribute('aria-label', `${e.kind === 'directory' ? 'Folder' : 'File'} ${e.name}`);
          name.addEventListener('dblclick', () => e.kind === 'directory' && this.openFolder(e.path));
          name.addEventListener('keydown', (k) => {
            if (k.key === 'Enter' && e.kind === 'directory') {
              k.preventDefault();
              this.openFolder(e.path);
            }
          });
          td.append(name);
        } else td.textContent = c.cell(e);
        tr.append(td);
      });
      body.append(tr);
    }
    const on = KINDS.filter((k) => this.#filters.has(k)).map((k) => KIND_NAMES[k].plural.toLowerCase());
    const caption = `Contents of ${folderName(this.#folder)}${on.length ? `, showing ${on.join(' and ')}` : ''}`;
    const table = el('table', { className: 'files-table' }, el('caption', { className: 'files-caption' }, caption), el('thead', {}, head), body);
    main.append(el('div', { className: 'files-table-wrap' }, table));
    if (!this.#entries.length) main.append(el('p', { className: 'files-note' }, 'This folder is empty.'));
    else if (!this.#visible().length) main.append(el('p', { className: 'files-note' }, `Nothing here matches the filters (${on.join(', ')}).`));
    main.append(el('p', { className: 'files-note' }, `Import files, or drop them here, to put them in this folder. Each file’s kind is worked out from the file. ${ACCEPTED}`));
    queueMicrotask(() => this.#renderTask());
    return main;
  }

  /**
   * Selects a row without rebuilding the list: only the highlight, the toolbar and the inspector change, so the rows
   * stay exactly where they are under the pointer (a double-click's second click lands on the same row).
   */
  #select(path: string): void {
    this.#selected = path;
    for (const tr of this.element.querySelectorAll<HTMLTableRowElement>('tbody tr')) {
      const on = tr.dataset['path'] === path;
      tr.classList.toggle('files-selected', on);
      tr.setAttribute('aria-selected', String(on));
    }
    this.element.querySelector('.files-toolbar')?.replaceWith(this.#toolbar());
    this.element.querySelector('.files-inspector')?.replaceWith(this.#inspector());
    this.#onChange();
  }

  #renderTask(): void {
    const host = this.element.querySelector<HTMLElement>('.files-task-host');
    if (!host) return;
    const t = this.#task;
    if (!t) {
      host.replaceChildren();
      return;
    }
    const pct = t.size ? Math.round((t.written / t.size) * 100) : 0;
    const bar = el('div', { className: 'files-meter' }, el('div', { className: 'files-meter-fill' }));
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-label', `Importing ${t.name}`);
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    bar.setAttribute('aria-valuenow', String(pct));
    (bar.firstElementChild as HTMLElement).style.width = `${pct}%`;
    const status = el('div', { className: 'files-task' },
      el('p', {}, el('strong', {}, 'Importing '), `${t.name || '…'}${t.count > 1 ? ` (${t.index + 1} of ${t.count})` : ''} · ${formatBytes(t.written)} of ${formatBytes(t.size)}`),
      bar,
      button('Cancel', () => t.abort.abort(), { dataset: { key: 'cancel-import' } }));
    status.setAttribute('role', 'status');
    host.replaceChildren(status);
  }

  async #loadHeader(e: FileEntry): Promise<void> {
    if (e.kind !== 'file' || this.kindOf(e.path) !== 'weights' || this.#header.has(e.path) || !this.#store) return;
    try {
      this.#header.set(e.path, await readWeightsHeader(await this.#store.read(e.path)));
    } catch (err) {
      this.#header.set(e.path, err instanceof Error ? err.message : String(err));
    }
    if (this.#selected === e.path) this.element.querySelector('.files-inspector')?.replaceWith(this.#inspector());
  }

  #kindControl(e: FileEntry): Node {
    const kind = this.kindOf(e.path);
    if (!kind) return document.createTextNode('Not recognised');
    const by = this.#catalog.files[e.path]?.by;
    const note = by === 'player' ? 'Chosen by you.' : `Worked out from the file.${this.#why.get(e.path) ? ` ${this.#why.get(e.path)}` : ''}`;
    if (kind !== 'game' && kind !== 'problem') return el('span', {}, KIND_NAMES[kind].one, el('span', { className: 'files-kind-note' }, note));
    const group = el('div', { className: 'sm-segmented files-kind' });
    group.setAttribute('role', 'group');
    group.setAttribute('aria-label', 'Kind');
    for (const k of ['game', 'problem'] as const) {
      const b = el('button', { type: 'button', textContent: KIND_NAMES[k].one, dataset: { key: `set-kind:${k}` } });
      b.setAttribute('aria-pressed', String(kind === k));
      b.addEventListener('click', () => kind !== k && void this.setKind(e.path, k));
      group.append(b);
    }
    return el('div', {}, group, el('span', { className: 'files-kind-note' }, note));
  }

  #inspector(): HTMLElement {
    const aside = el('aside', { className: 'files-inspector' });
    aside.setAttribute('aria-label', 'Details');
    const e = this.selected;
    if (!e) {
      aside.append(el('p', { className: 'files-note' }, 'Select a file to see its details.'));
      return aside;
    }
    const rows: [string, string | Node][] = [];
    if (e.kind === 'directory') rows.push(['Kind', 'Folder']);
    else {
      rows.push(['Kind', this.#kindControl(e)]);
      rows.push(['Size', (e.size ?? 0) < 1024 ? formatBytes(e.size) : `${formatBytes(e.size)} (${grouped(e.size ?? 0)} bytes)`]);
      if (e.lastModified) rows.push(['Modified', dateFormat.format(e.lastModified)]);
      if (this.kindOf(e.path) === 'weights') {
        const w = parseWeightsName(e.name);
        if (w) {
          rows.push(['Training run', w.run]);
          rows.push(['Network', describeNetwork(w)]);
          rows.push(['Training samples (s)', grouped(w.samples)]);
          rows.push(['Training data rows (d)', grouped(w.dataRows)]);
          rows.push(['Format', w.format === 'binary' ? 'Binary, gzip' : 'Text, gzip']);
        }
        const h = this.#header.get(e.path);
        if (typeof h === 'object') rows.push(['Model version', String(h.modelVersion)]);
        else if (typeof h === 'string') rows.push(['Model version', `Not read: ${h}`]);
      }
      const sum = this.#checksums.get(e.path);
      rows.push(['SHA-256', sum ? el('code', { className: 'files-hash' }, sum) : button('Compute', () => void this.computeChecksum(e.path), { dataset: { key: 'checksum' } })]);
    }
    rows.push(['Folder', folderName(parentOf(e.path))]);
    const dl = el('dl', { className: 'files-details' });
    for (const [k, v] of rows) dl.append(el('dt', {}, k), el('dd', {}, v));
    aside.append(el('h3', { className: 'files-inspector-title' }, nameOf(e.path)), dl);
    return aside;
  }
}
