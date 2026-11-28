// The engine console, the first part of the Engines app (build plan, Milestone 1, step 6): start the browser engine
// with a weights file from Files and KataGo's settings, then send it KataGo's Analysis Engine protocol directly, one
// JSON line per query, and read every reply, timed, for running benchmarks by hand.
import { OpfsFileStore } from '@stonemind/files';
import { loadCatalog, walk } from '../files/catalog.js';
import { formatBytes } from '../files/weights.js';
import { EngineHostClient } from './client.js';
import { KATAGO_DEFAULTS, type EngineStatus } from './protocol.js';

export const ENGINE_ID = 'browser';
const MAX_LOG = 2000;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & { dataset?: Record<string, string> } = {}, ...children: (Node | string | null)[]): HTMLElementTagNameMap[K] {
  const { dataset, ...rest } = props;
  const node = Object.assign(document.createElement(tag), rest);
  if (dataset) Object.assign(node.dataset, dataset);
  node.append(...children.filter((c): c is Node | string => c !== null));
  return node;
}

function button(text: string, onClick: () => void, props: Partial<HTMLButtonElement> & { dataset?: Record<string, string> } = {}): HTMLButtonElement {
  const b = el('button', { type: 'button', className: 'sm-btn', ...props }, text);
  b.addEventListener('click', onClick);
  return b;
}

/** KataGo's config as text, one `key = value` per line. */
export function configText(config: Readonly<Record<string, string>>): string {
  return Object.entries(config).map(([k, v]) => `${k} = ${v}`).join('\n');
}

/** Reads `key = value` lines; `#` starts a comment. Throws on a line that is neither. */
export function parseConfigText(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  text.split(/\r?\n/).forEach((raw, i) => {
    const line = raw.replace(/#.*$/, '').trim();
    if (!line) return;
    const m = /^([A-Za-z0-9_.]+)\s*=\s*(.*)$/.exec(line);
    if (!m) throw new Error(`Line ${i + 1} of the settings is not "key = value": ${raw.trim()}`);
    out[m[1] as string] = (m[2] as string).trim();
  });
  return out;
}

/** Example queries for the console. Each gets a fresh id when it is inserted. */
export const PRESETS: readonly { readonly label: string; readonly query: Record<string, unknown> }[] = [
  { label: 'query_version', query: { action: 'query_version' } },
  { label: 'query_models', query: { action: 'query_models' } },
  { label: 'Empty board, 500 visits', query: { moves: [], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, maxVisits: 500 } },
  { label: 'Empty board, 5,000 visits', query: { moves: [], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, maxVisits: 5000 } },
  { label: 'Opening, every turn, 100 visits', query: { moves: [['B', 'Q16'], ['W', 'D4'], ['B', 'Q3'], ['W', 'D16']], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, analyzeTurns: [0, 1, 2, 3, 4], maxVisits: 100 } },
];

interface Sent {
  at: number;
  turns: number;
}

export class EngineConsole {
  readonly element = el('div', { className: 'console' });
  #client: EngineHostClient | null = null;
  #error = '';
  #weights: string[] = [];
  #sent = new Map<string, Sent>();
  #next = 0;
  #log: HTMLOListElement = el('ol', { className: 'console-log' });
  #entries: string[] = [];
  #showDuring = false;
  #during = 0;
  #statusBox = el('div', { className: 'console-status' });
  #weightsSelect = el('select', { className: 'console-input', id: 'console-weights' });
  #config = el('textarea', { className: 'console-input console-config', id: 'console-config', rows: 9, spellcheck: false, value: configText(KATAGO_DEFAULTS) });
  #input = el('textarea', { className: 'console-input console-command', id: 'console-command', rows: 4, spellcheck: false, placeholder: '{"id":"q1","moves":[],"rules":"chinese","komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":500}' });

  constructor() {
    this.#render();
    void this.#open();
  }

  async #open(): Promise<void> {
    try {
      const [client, store] = await Promise.all([EngineHostClient.connect('auto'), OpfsFileStore.open()]);
      this.#client = client;
      client.addEventListener('status', () => this.#renderStatus());
      client.onLine(ENGINE_ID, (line) => this.#received(line));
      const catalog = await loadCatalog(store);
      const files = (await walk(store)).filter((e) => e.kind === 'file');
      this.#weights = files.filter((e) => catalog.files[e.path]?.kind === 'weights' || (!catalog.files[e.path] && /\.(bin|txt)\.gz$/i.test(e.name))).map((e) => e.path).sort();
      this.#weightsSelect.replaceChildren(...(this.#weights.length
        ? this.#weights.map((p) => {
            const f = files.find((e) => e.path === p);
            return el('option', { value: p }, `${p}${f?.size ? ` (${formatBytes(f.size)})` : ''}`);
          })
        : [el('option', { value: '' }, 'No weights in Files yet')]));
      const running = client.engineStatus(ENGINE_ID)?.weightsPath;
      if (running && this.#weights.includes(running)) this.#weightsSelect.value = running;
    } catch (e) {
      this.#error = e instanceof Error ? e.message : String(e);
    }
    this.#renderStatus();
  }

  // ---------------------------------------------------------- actions

  async start(): Promise<void> {
    const client = this.#client;
    if (!client) return;
    let config: Record<string, string>;
    try {
      config = parseConfigText(this.#config.value);
    } catch (e) {
      this.#error = e instanceof Error ? e.message : String(e);
      return this.#renderStatus();
    }
    const weightsPath = this.#weightsSelect.value;
    if (!weightsPath) {
      this.#error = 'Import a KataGo network in Files first, then choose it here.';
      return this.#renderStatus();
    }
    this.#error = '';
    this.#note(`starting with ${weightsPath}`);
    const s = await client.start({ engineId: ENGINE_ID, kind: 'webgpu', weightsPath, config });
    this.#note(s.state === 'running' ? `running: ${s.model?.name ?? ''}` : `did not start: ${s.error ?? ''}`);
  }

  stop(): void {
    this.#client?.stop(ENGINE_ID);
    this.#note('stopped');
  }

  /** Sends what is in the command box: one JSON query per line. A query without an id gets one. */
  send(): void {
    const client = this.#client;
    if (!client) return;
    const lines = this.#input.value.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    for (const raw of lines) {
      let line = raw;
      try {
        const q = JSON.parse(raw) as Record<string, unknown>;
        if (q && typeof q === 'object' && !Array.isArray(q)) {
          if (typeof q['id'] !== 'string') q['id'] = `c${++this.#next}`;
          const id = q['id'] as string;
          if (!q['action']) this.#sent.set(id, { at: performance.now(), turns: Array.isArray(q['analyzeTurns']) ? q['analyzeTurns'].length : 1 });
          line = JSON.stringify(q);
        }
      } catch {
        // Sent as typed: the engine answers what it can't read with an error.
      }
      this.#append('out', line);
      client.sendLine(ENGINE_ID, line);
    }
  }

  insertPreset(index: number): void {
    const p = PRESETS[index];
    if (!p) return;
    const line = JSON.stringify({ id: `c${++this.#next}`, ...p.query });
    this.#input.value = this.#input.value.trim() ? `${this.#input.value.trimEnd()}\n${line}` : line;
    this.#input.focus();
  }

  clear(): void {
    this.#entries = [];
    this.#during = 0;
    this.#log.replaceChildren();
    this.#renderCounts();
  }

  /** The log as JSON lines: what was sent and received, in order. */
  logText(): string {
    return this.#entries.join('\n') + (this.#entries.length ? '\n' : '');
  }

  #received(line: string): void {
    let note = '';
    let during = false;
    try {
      const r = JSON.parse(line) as { id?: string; isDuringSearch?: boolean; rootInfo?: { visits?: number }; turnNumber?: number };
      during = r.isDuringSearch === true;
      const sent = r.id !== undefined ? this.#sent.get(r.id) : undefined;
      if (sent && r.isDuringSearch === false) {
        const ms = performance.now() - sent.at;
        const visits = r.rootInfo?.visits;
        note = `${Math.round(ms)} ms${visits !== undefined ? ` · ${visits} visits · ${Math.round((visits / ms) * 1000)} visits/s` : ''}`;
        if (--sent.turns <= 0) this.#sent.delete(r.id as string);
      }
    } catch {
      // Not JSON: logged as it came.
    }
    if (during) {
      this.#during++;
      if (!this.#showDuring) return this.#renderCounts();
    }
    this.#append('in', line, note);
  }

  #note(text: string): void {
    this.#append('note', text);
  }

  #append(dir: 'in' | 'out' | 'note', line: string, note = ''): void {
    if (dir !== 'note') this.#entries.push(line);
    const time = new Date().toLocaleTimeString(undefined, { hour12: false });
    // A long reply (a final result runs to many kilobytes) shows its start, with the timing first; the whole line opens
    // below it.
    const LONG = 160;
    const body: Node = line.length > LONG
      ? el('details', { className: 'console-details' }, el('summary', { className: 'console-line' }, `${line.slice(0, LONG)}…`), el('code', { className: 'console-line console-full' }, line))
      : el('code', { className: 'console-line' }, line);
    const item = el('li', { className: `console-${dir}` },
      el('span', { className: 'console-time' }, time),
      el('span', { className: 'console-dir' }, dir === 'out' ? '→' : dir === 'in' ? '←' : '•'),
      el('div', { className: 'console-body' }, note ? el('span', { className: 'console-timing' }, note) : null, body));
    this.#log.append(item);
    while (this.#log.childElementCount > MAX_LOG) this.#log.firstElementChild?.remove();
    const pane = this.#log.parentElement;
    if (pane) pane.scrollTop = pane.scrollHeight;
    this.#renderCounts();
  }

  // ---------------------------------------------------------- rendering

  #counts = el('span', { className: 'console-counts' });

  #renderCounts(): void {
    this.#counts.textContent = `${this.#entries.length} lines${this.#during ? ` · ${this.#during} reports during search${this.#showDuring ? '' : ' (hidden)'}` : ''}`;
  }

  #renderStatus(): void {
    const s: EngineStatus | undefined = this.#client?.engineStatus(ENGINE_ID);
    const host = this.#client ? this.#client.status : null;
    const parts: (string | Node)[] = [];
    if (this.#error) {
      const error = el('p', { className: 'console-error' }, `✗ ${this.#error}`);
      error.setAttribute('role', 'alert');
      parts.push(error);
    }
    if (!host) parts.push(el('p', {}, 'Connecting to the engine host…'));
    else {
      const state = s ? { stopped: 'Stopped', starting: 'Starting…', running: 'Running', failed: 'Did not start' }[s.state] : 'Not started';
      const gpu = s?.adapter ? [s.adapter.vendor, s.adapter.architecture, s.adapter.description].filter(Boolean).join(' · ') : '';
      parts.push(
        el('p', { className: 'console-state' }, el('strong', {}, state), s?.model ? ` · ${s.model.name} (model version ${s.model.version})` : ''),
        el('p', {}, `GPU: ${gpu || (s?.state === 'running' ? 'the browser does not name it' : '—')} · ${host.shared ? `shared by ${host.tabs} tab${host.tabs === 1 ? '' : 's'}` : 'this tab only'}${host.webgpu ? '' : ' · no WebGPU in the engine host'}`),
      );
      if (s?.state === 'failed' && s.error) parts.push(el('p', { className: 'console-error' }, s.error));
    }
    this.#statusBox.replaceChildren(...parts);
    const running = s?.state === 'running';
    for (const b of this.element.querySelectorAll<HTMLButtonElement>('[data-needs-engine]')) b.disabled = !running;
  }

  #render(): void {
    const presets = el('select', { className: 'console-input', id: 'console-presets' }, el('option', { value: '' }, 'Insert an example…'), ...PRESETS.map((p, i) => el('option', { value: String(i) }, p.label)));
    presets.setAttribute('aria-label', 'Insert an example query');
    presets.addEventListener('change', () => {
      if (presets.value) this.insertPreset(Number(presets.value));
      presets.value = '';
    });
    this.#input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        this.send();
      }
    });
    const during = el('input', { type: 'checkbox', id: 'console-during' });
    during.addEventListener('change', () => {
      this.#showDuring = during.checked;
      this.#renderCounts();
    });
    const sendButton = button('Send', () => this.send(), { className: 'sm-btn sm-btn--primary' });
    sendButton.dataset['needsEngine'] = '';
    const engine = el('section', { className: 'console-engine' },
      el('h3', { className: 'console-title' }, 'Engine'),
      el('label', { htmlFor: 'console-weights', className: 'sm-label' }, 'Weights'),
      this.#weightsSelect,
      el('label', { htmlFor: 'console-config', className: 'sm-label' }, 'KataGo settings (key = value)'),
      this.#config,
      el('div', { className: 'console-row' }, button('Start', () => void this.start(), { className: 'sm-btn sm-btn--primary' }), button('Stop', () => this.stop())),
      this.#statusBox);
    const log = el('div', { className: 'console-log-pane' }, this.#log);
    log.setAttribute('role', 'log');
    log.setAttribute('aria-label', 'Engine log');
    const copy = button('Copy log', () => void navigator.clipboard?.writeText(this.logText()));
    const download = button('Download log', () => {
      const url = URL.createObjectURL(new Blob([this.logText()], { type: 'application/x-ndjson' }));
      const a = el('a', { href: url, download: `engine-log-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl` });
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    });
    const consoleSection = el('section', { className: 'console-main' },
      el('h3', { className: 'console-title' }, 'Console'),
      el('label', { htmlFor: 'console-command', className: 'sm-label' }, 'Queries: one JSON line each (KataGo’s Analysis Engine protocol). Ctrl+Enter sends.'),
      this.#input,
      el('div', { className: 'console-row' }, sendButton, presets, el('label', { className: 'console-check' }, during, 'Show reports during search')),
      el('div', { className: 'console-row' }, this.#counts, el('span', { className: 'console-spacer' }), copy, download, button('Clear', () => this.clear())),
      log);
    this.element.replaceChildren(engine, consoleSection);
    this.#renderCounts();
  }
}
