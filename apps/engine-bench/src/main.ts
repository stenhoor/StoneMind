import {
  AnalysisError,
  connectBridge,
  parsePairingFragment,
  type AnalysisEngine,
  type AnalysisQuery,
  type Move,
  type MoveInfo,
  type QueryModelsAction,
  type QueryVersionAction,
} from '@stonemind/engine';

const PAIRING_KEY = 'stonemind.bench.pairing';

const POSITIONS: Record<string, Move[]> = {
  empty: [],
  opening: [
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16'],
  ],
  fight: [
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q3'],
    ['W', 'D16'],
    ['B', 'C3'],
    ['W', 'D3'],
    ['B', 'C4'],
    ['W', 'D5'],
    ['B', 'C5'],
    ['W', 'D6'],
  ],
};

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const ui = {
  status: $<HTMLOutputElement>('status'),
  pair: $<HTMLFormElement>('pair'),
  url: $<HTMLInputElement>('bridge-url'),
  token: $<HTMLInputElement>('bridge-token'),
  connect: $<HTMLButtonElement>('connect'),
  disconnect: $<HTMLButtonElement>('disconnect'),
  info: $<HTMLElement>('engine-info'),
  version: $<HTMLElement>('info-version'),
  model: $<HTMLElement>('info-model'),
  batch: $<HTMLElement>('info-batch'),
  fp16: $<HTMLElement>('info-fp16'),
  analyse: $<HTMLFormElement>('analyse'),
  position: $<HTMLSelectElement>('position'),
  maxVisits: $<HTMLInputElement>('max-visits'),
  run: $<HTMLButtonElement>('run'),
  stop: $<HTMLButtonElement>('stop'),
  visits: $<HTMLElement>('visits'),
  elapsed: $<HTMLElement>('elapsed'),
  rate: $<HTMLElement>('rate'),
  winrate: $<HTMLElement>('winrate'),
  moves: $<HTMLTableSectionElement>('moves'),
};

let engine: AnalysisEngine | null = null;
let running: AbortController | null = null;
let queryCount = 0;

function setStatus(text: string, state: 'idle' | 'connected' | 'busy' | 'error'): void {
  ui.status.textContent = text;
  ui.status.dataset.state = state;
}

function setConnected(connected: boolean): void {
  ui.connect.disabled = connected;
  ui.disconnect.disabled = !connected;
  ui.url.disabled = connected;
  ui.token.disabled = connected;
  ui.run.disabled = !connected || running !== null;
  ui.info.hidden = !connected;
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const signed = (x: number) => `${x >= 0 ? '+' : ''}${x.toFixed(1)}`;

function renderMoves(infos: readonly MoveInfo[]): void {
  const rows = [...infos]
    .sort((a, b) => a.order - b.order)
    .slice(0, 10)
    .map((m) => {
      const tr = document.createElement('tr');
      for (const text of [m.move, String(m.visits), pct(m.winrate), signed(m.scoreLead), pct(m.prior), m.pv.slice(0, 8).join(' ')]) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    });
  ui.moves.replaceChildren(...rows);
}

async function connect(url: string, token: string): Promise<void> {
  setStatus('Connecting…', 'busy');
  try {
    const session = await connectBridge({ url, token });
    session.onGeneralError = (e) => setStatus(`Engine error: ${e.message}`, 'error');
    engine = session;
    try {
      sessionStorage.setItem(PAIRING_KEY, JSON.stringify({ url, token }));
    } catch {}
    const version = await session.action<QueryVersionAction>({ action: 'query_version' });
    ui.version.textContent = version.version;
    try {
      const { models } = await session.action<QueryModelsAction>({ action: 'query_models' });
      const first = models[0];
      ui.model.textContent = first ? first.internalName : '–';
      ui.batch.textContent = first ? String(first.maxBatchSize) : '–';
      ui.fp16.textContent = first ? first.usingFP16 : '–';
    } catch {
      ui.model.textContent = 'not reported by this KataGo version';
    }
    setConnected(true);
    setStatus('Connected to KataGo on this computer', 'connected');
  } catch (e) {
    engine = null;
    setConnected(false);
    setStatus(e instanceof AnalysisError ? e.message : `Could not connect: ${(e as Error).message}`, 'error');
  }
}

async function disconnect(): Promise<void> {
  running?.abort();
  await engine?.close();
  engine = null;
  try {
    sessionStorage.removeItem(PAIRING_KEY);
  } catch {}
  setConnected(false);
  setStatus('Not connected', 'idle');
}

async function analyse(): Promise<void> {
  if (!engine || running) return;
  const maxVisits = Math.max(1, Math.min(1_000_000, Math.round(Number(ui.maxVisits.value) || 1000)));
  const query: AnalysisQuery = {
    id: `bench-${++queryCount}`,
    moves: POSITIONS[ui.position.value] ?? [],
    rules: 'chinese',
    komi: 7.5,
    boardXSize: 19,
    boardYSize: 19,
    maxVisits,
    reportDuringSearchEvery: 0.25,
  };
  running = new AbortController();
  ui.run.disabled = true;
  ui.stop.disabled = false;
  setStatus('Analysing…', 'busy');
  const started = performance.now();
  try {
    for await (const result of engine.analyze(query, { signal: running.signal })) {
      const seconds = (performance.now() - started) / 1000;
      if (result.rootInfo) {
        ui.visits.textContent = result.rootInfo.visits.toLocaleString();
        ui.winrate.textContent = pct(result.rootInfo.winrate);
        ui.rate.textContent = seconds > 0 ? Math.round(result.rootInfo.visits / seconds).toLocaleString() : '–';
      }
      ui.elapsed.textContent = `${seconds.toFixed(2)} s`;
      if (result.moveInfos) renderMoves(result.moveInfos);
    }
    setStatus('Connected to KataGo on this computer', 'connected');
  } catch (e) {
    setStatus(`Analysis failed: ${(e as Error).message}`, 'error');
  } finally {
    running = null;
    ui.stop.disabled = true;
    ui.run.disabled = engine === null;
  }
}

ui.pair.addEventListener('submit', (e) => {
  e.preventDefault();
  void connect(ui.url.value.trim(), ui.token.value.trim());
});
ui.disconnect.addEventListener('click', () => void disconnect());
ui.analyse.addEventListener('submit', (e) => {
  e.preventDefault();
  void analyse();
});
ui.stop.addEventListener('click', () => running?.abort());

// Pair from the link the bridge printed, then take the token out of the address bar and history.
const fromLink = parsePairingFragment(location.hash);
if (location.hash) history.replaceState(null, '', location.pathname + location.search);
let saved: { url: string; token: string } | null = null;
try {
  saved = JSON.parse(sessionStorage.getItem(PAIRING_KEY) ?? 'null') as { url: string; token: string } | null;
} catch {}
const pairing = fromLink ?? saved;
if (pairing) {
  ui.url.value = pairing.url;
  ui.token.value = pairing.token;
  void connect(pairing.url, pairing.token);
}
