// The Files app (build plan, Milestone 1, step 4): a file browser in the style of macOS Finder over the file kit
// (`@stonemind/files`). A singleton app: one instance, belonging to no workspace.
//
// Library folders (Games, Problems, Weights, Configs) in a sidebar with this browser's storage; a path; sortable
// columns that depend on the folder; an inspector for the selected file (for KataGo weights: the network from its
// name, the model version from its header, a SHA-256 on request); import from the file picker or by dropping files,
// streamed with progress and a Cancel button; new folder, rename, download and delete (with a confirmation).
import { type FileEntry, NotFoundError, OpfsFileStore, type StorageInfo, downloadBlob, filesFromDrop, importFiles, join, nameOf, normalize, parentOf, pickFiles, requestPersistence, storageInfo } from '@stonemind/files';
import type { Menu } from '../frame/menu.js';
import { type WeightsHeader, describeNetwork, formatBytes, grouped, parseWeightsName, readWeightsHeader, sha256 } from './weights.js';

export const LIBRARY: readonly { readonly path: string; readonly name: string }[] = [
  { path: '/games', name: 'Games' },
  { path: '/problems', name: 'Problems' },
  { path: '/weights', name: 'Weights' },
  { path: '/configs', name: 'Configs' },
];

type SortKey = 'name' | 'size' | 'modified';

interface ImportTask {
  name: string;
  index: number;
  count: number;
  written: number;
  size: number;
  abort: AbortController;
}

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

export class FilesView {
  readonly element = el('div', { className: 'files' });
  #store: OpfsFileStore | null = null;
  #error = '';
  #folder = '/weights';
  #entries: FileEntry[] = [];
  #selected: string | null = null;
  #sort: { key: SortKey; descending: boolean } = { key: 'name', descending: false };
  #storage: StorageInfo | null = null;
  #task: ImportTask | null = null;
  #header = new Map<string, WeightsHeader | string>();
  #checksums = new Map<string, string>();
  #onChange: () => void;

  constructor(onMenusChanged: () => void = () => {}) {
    this.#onChange = onMenusChanged;
    void this.#open();
  }

  async #open(): Promise<void> {
    try {
      this.#store = await OpfsFileStore.open();
      for (const f of LIBRARY) if (!(await this.#store.stat(f.path))) await this.#store.mkdir(f.path);
      this.#store.addEventListener('change', () => void this.refresh());
      await this.refresh();
    } catch (e) {
      this.#error = e instanceof Error ? e.message : String(e);
      this.render();
    }
  }

  async refresh(): Promise<void> {
    if (!this.#store) return;
    try {
      this.#entries = await this.#store.list(this.#folder);
    } catch (e) {
      if (!(e instanceof NotFoundError)) throw e;
      this.#folder = '/weights';
      this.#entries = await this.#store.list(this.#folder);
    }
    if (this.#selected && !this.#entries.some((e) => e.path === this.#selected)) this.#selected = null;
    this.#storage = await storageInfo().catch(() => null);
    this.render();
    this.#onChange();
  }

  // ---------------------------------------------------------- actions

  open(folder: string): void {
    this.#folder = normalize(folder);
    this.#selected = null;
    void this.refresh();
  }

  get selected(): FileEntry | null {
    return this.#entries.find((e) => e.path === this.#selected) ?? null;
  }

  async importPicked(): Promise<void> {
    const files = await pickFiles({ multiple: true });
    if (files.length) await this.import(files);
  }

  async import(files: File[]): Promise<void> {
    if (!this.#store || this.#task) return;
    const abort = new AbortController();
    this.#task = { name: '', index: 0, count: files.length, written: 0, size: 0, abort };
    this.render();
    try {
      await importFiles(this.#store, files, this.#folder, {
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
      await this.refresh();
    }
  }

  async newFolder(): Promise<void> {
    const name = await this.#ask('New folder', 'Name', '', 'Create');
    if (!name || !this.#store) return;
    await this.#run(() => this.#store?.mkdir(join(this.#folder, name)));
  }

  async rename(): Promise<void> {
    const entry = this.selected;
    if (!entry || !this.#store) return;
    const name = await this.#ask(`Rename ${entry.name}`, 'New name', entry.name, 'Rename');
    if (!name || name === entry.name) return;
    const to = join(parentOf(entry.path), name);
    await this.#run(async () => {
      await this.#store?.move(entry.path, to);
      this.#selected = to;
    });
  }

  async remove(): Promise<void> {
    const entry = this.selected;
    if (!entry || !this.#store) return;
    const what = entry.kind === 'directory' ? `the folder ${entry.name} and everything in it` : entry.name;
    if (!(await this.#confirm(`Delete ${what}?`, 'It is removed from this browser’s storage. This cannot be undone.', 'Delete'))) return;
    await this.#run(() => this.#store?.remove(entry.path, { recursive: true }));
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
    this.#checksums.set(path, 'Computing…');
    this.render();
    try {
      this.#checksums.set(path, await sha256(await this.#store.read(path)));
    } catch (e) {
      this.#checksums.set(path, `Could not compute it: ${e instanceof Error ? e.message : String(e)}`);
    }
    this.render();
  }

  async #run(action: () => Promise<unknown> | undefined): Promise<void> {
    try {
      this.#error = '';
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
          { label: 'Import…', run: () => void this.importPicked() },
          { label: 'New folder…', run: () => void this.newFolder() },
          { label: 'Download', disabled: s?.kind !== 'file', run: () => void this.download() },
          { label: 'Rename…', disabled: !s, run: () => void this.rename() },
          { label: 'Delete…', disabled: !s, run: () => void this.remove() },
        ],
      },
      { label: 'View', items: [sortItem('name', 'name'), sortItem('size', 'size'), sortItem('modified', 'date modified')] },
      { label: 'Go', items: LIBRARY.map((f) => ({ label: f.name, checked: () => this.#folder === f.path, run: () => this.open(f.path) })) },
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
    this.element.replaceChildren(this.#toolbar(), el('div', { className: 'files-body' }, this.#sidebar(), this.#main(), this.#inspector()));
    if (active) this.element.querySelector<HTMLElement>(`[data-key="${CSS.escape(active)}"]`)?.focus();
  }

  #toolbar(): HTMLElement {
    const crumbs = el('nav', { className: 'files-path' });
    crumbs.setAttribute('aria-label', 'Folder');
    const parts = this.#folder.split('/').filter(Boolean);
    crumbs.append(el('span', { className: 'files-crumb-root' }, 'This browser'));
    parts.forEach((part, i) => {
      const path = '/' + parts.slice(0, i + 1).join('/');
      const last = i === parts.length - 1;
      crumbs.append(el('span', { className: 'files-crumb-sep' }, '›'));
      const b = button(LIBRARY.find((f) => f.path === path)?.name ?? part, () => this.open(path), { className: 'files-crumb', dataset: { key: `crumb:${path}` } });
      if (last) b.setAttribute('aria-current', 'location');
      crumbs.append(b);
    });
    const s = this.selected;
    const bar = el('div', { className: 'files-toolbar' }, crumbs, el('span', { className: 'files-spacer' }),
      button('Import…', () => void this.importPicked(), { className: 'sm-btn sm-btn--primary files-btn', dataset: { key: 'import' } }),
      button('New folder…', () => void this.newFolder(), { dataset: { key: 'mkdir' } }),
      button('Download', () => void this.download(), { disabled: s?.kind !== 'file', dataset: { key: 'download' } }),
      button('Rename…', () => void this.rename(), { disabled: !s, dataset: { key: 'rename' } }),
      button('Delete…', () => void this.remove(), { disabled: !s, dataset: { key: 'delete' } }));
    bar.setAttribute('role', 'toolbar');
    bar.setAttribute('aria-label', 'Files');
    return bar;
  }

  #sidebar(): HTMLElement {
    const nav = el('nav', { className: 'files-side' });
    nav.setAttribute('aria-label', 'Library');
    nav.append(el('p', { className: 'sm-label files-side-label' }, 'Library'));
    for (const f of LIBRARY) {
      const b = button(f.name, () => this.open(f.path), { className: 'files-place', dataset: { key: `place:${f.path}` } });
      if (this.#folder === f.path || this.#folder.startsWith(f.path + '/')) b.setAttribute('aria-current', 'true');
      nav.append(b);
    }
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
      el('p', { className: 'sm-label files-side-label' }, 'This browser'),
      meter,
      el('p', { className: 'files-storage-text' }, info?.quota !== undefined ? `${formatBytes(used)} used of ${formatBytes(quota)}` : 'Space used is not known'),
      el('p', { className: 'files-storage-text' }, info?.persisted ? '✓ Files are kept' : 'The browser may clear these files when space runs low.'),
      info?.persisted ? null : button('Keep my files', () => void this.keepFiles(), { dataset: { key: 'persist' } }));
    nav.append(storage);
    return nav;
  }

  #columns(): { label: string; key?: SortKey; cell: (e: FileEntry) => string; numeric?: boolean }[] {
    const size = { label: 'Size', key: 'size' as const, numeric: true, cell: (e: FileEntry) => (e.kind === 'file' ? formatBytes(e.size) : '—') };
    const modified = { label: 'Modified', key: 'modified' as const, cell: (e: FileEntry) => (e.lastModified ? dateFormat.format(e.lastModified) : '—') };
    if (this.#folder === '/weights' || this.#folder.startsWith('/weights/')) {
      return [
        { label: 'Name', key: 'name', cell: (e) => e.name },
        { label: 'Network', cell: (e) => { const w = parseWeightsName(e.name); return w ? `b${w.blocks}c${w.channels}${w.kind}` : '—'; } },
        size,
        modified,
      ];
    }
    return [{ label: 'Name', key: 'name', cell: (e) => e.name }, size, modified];
  }

  #sorted(): FileEntry[] {
    const { key, descending } = this.#sort;
    const value = (e: FileEntry) => (key === 'size' ? e.size ?? -1 : key === 'modified' ? e.lastModified ?? 0 : e.name.toLowerCase());
    return [...this.#entries].sort((a, b) => {
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
      const tr = el('tr', { className: selected ? 'files-row files-selected' : 'files-row' });
      tr.setAttribute('aria-selected', String(selected));
      cols.forEach((c, i) => {
        const td = el('td', { className: c.numeric ? 'files-num' : '' });
        if (i === 0) {
          const name = button(`${e.kind === 'directory' ? '📁︎ ' : ''}${e.name}`, () => {
            if (e.kind === 'directory' && this.#selected === e.path) this.open(e.path);
            else {
              this.#selected = e.path;
              this.render();
              this.#onChange();
              void this.#loadHeader(e);
            }
          }, { className: 'files-name', dataset: { key: `row:${e.path}` } });
          name.setAttribute('aria-label', `${e.kind === 'directory' ? 'Folder' : 'File'} ${e.name}`);
          name.addEventListener('dblclick', () => e.kind === 'directory' && this.open(e.path));
          name.addEventListener('keydown', (k) => {
            if (k.key === 'Enter' && e.kind === 'directory') {
              k.preventDefault();
              this.open(e.path);
            }
          });
          td.append(name);
        } else td.textContent = c.cell(e);
        tr.append(td);
      });
      body.append(tr);
    }
    const table = el('table', { className: 'files-table' }, el('caption', { className: 'files-caption' }, `Contents of ${this.#folder}`), el('thead', {}, head), body);
    main.append(el('div', { className: 'files-table-wrap' }, table));
    if (!this.#entries.length) main.append(el('p', { className: 'files-note' }, 'This folder is empty. Import files, or drop them here.'));
    else main.append(el('p', { className: 'files-note' }, 'Drop files here to import them into this folder.'));
    queueMicrotask(() => this.#renderTask());
    return main;
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
    if (e.kind !== 'file' || !e.name.endsWith('.gz') || this.#header.has(e.path) || !this.#store) return;
    try {
      this.#header.set(e.path, await readWeightsHeader(await this.#store.read(e.path)));
    } catch (err) {
      this.#header.set(e.path, err instanceof Error ? err.message : String(err));
    }
    if (this.#selected === e.path) this.render();
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
      rows.push(['Size', (e.size ?? 0) < 1024 ? formatBytes(e.size) : `${formatBytes(e.size)} (${grouped(e.size ?? 0)} bytes)`]);
      if (e.lastModified) rows.push(['Modified', dateFormat.format(e.lastModified)]);
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
      const sum = this.#checksums.get(e.path);
      rows.push(['SHA-256', sum ? el('code', { className: 'files-hash' }, sum) : button('Compute', () => void this.computeChecksum(e.path), { dataset: { key: 'checksum' } })]);
    }
    rows.push(['Location', parentOf(e.path)]);
    const dl = el('dl', { className: 'files-details' });
    for (const [k, v] of rows) dl.append(el('dt', {}, k), el('dd', {}, v));
    aside.append(el('h3', { className: 'files-inspector-title' }, nameOf(e.path)), dl);
    return aside;
  }
}
