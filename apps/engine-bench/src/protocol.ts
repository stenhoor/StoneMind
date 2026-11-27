// The protocol check (spike 0001, parity plan step 12): the same queries go to native KataGo through the bridge and to
// the browser engine on this GPU, both through @stonemind/engine's AnalysisEngine. It passes when every field native
// returns is present in the browser engine's results, with values within the network goldens' tolerances (the
// engine-webgpu search check's, test/search.test.mjs), and errors, warnings and action replies are the same.
//
// Native must run deterministically for values to be comparable: one analysis thread, one search thread, no random
// symmetries or root noise, fp32 (scripts/protocol-config.mjs writes that config). The browser engine gets the same
// config keys.
import {
  AnalysisError,
  connectBridge,
  parsePairingFragment,
  type AnalysisEngine,
  type AnalysisQuery,
  type AnalysisResponse,
  type ClearCacheAction,
  type QueryModelsAction,
  type QueryVersionAction,
  type WarningResponse,
} from '@stonemind/engine';
import { GpuNetwork, loadModel, requestEngineDevice } from '@stonemind/engine-webgpu';
import { createAnalysisEngine } from '@stonemind/engine-webgpu/session';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const ui = {
  status: $<HTMLOutputElement>('status'),
  form: $<HTMLFormElement>('check'),
  model: $<HTMLInputElement>('model-file'),
  config: $<HTMLTextAreaElement>('config'),
  run: $<HTMLButtonElement>('run-check'),
  results: $<HTMLOListElement>('results'),
  download: $<HTMLAnchorElement>('download'),
};

function setStatus(text: string, state: 'idle' | 'connected' | 'busy' | 'error'): void {
  ui.status.textContent = text;
  ui.status.dataset.state = state;
}

/** A KataGo config file's settings: `key = value` lines, `#` comments. */
function parseConfig(text: string): Record<string, string> {
  const config: Record<string, string> = {};
  for (const raw of text.split('\n')) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) throw new Error(`Could not read config line: ${raw}`);
    config[line.slice(0, eq).trim()] = line.slice(eq + 1).trim();
  }
  return config;
}

// ------------------------------------------------------------------ the queries

/** One case: a query, or an action, run on both engines. */
interface Case {
  name: string;
  query?: AnalysisQuery;
  /** Stop the query at its first report during the search, as a page does when the player moves on. */
  abortAtFirstReport?: boolean;
  action?: 'query_version' | 'query_models' | 'clear_cache';
}

const ALL = {
  includeOwnership: true,
  includeOwnershipStdev: true,
  includeMovesOwnership: true,
  includeMovesOwnershipStdev: true,
  includePolicy: true,
  includePVVisits: true,
  includeNoResultValue: true,
};
const OPENING_19: AnalysisQuery['moves'] = [['B', 'Q16'], ['W', 'D4'], ['B', 'Q4']];

const CASES: Case[] = [
  { name: 'query_version', action: 'query_version' },
  { name: 'query_models', action: 'query_models' },
  // Values for Black (the config's side): Black to move at turn 2, White at turn 3.
  { name: 'opening-19', query: { id: 'opening-19', moves: OPENING_19, rules: 'tromp-taylor', komi: 7.5, boardXSize: 19, boardYSize: 19, analyzeTurns: [2, 3], maxVisits: 30, ...ALL } },
  // Values for White, with Black to move (turn 0) and White to move (turn 1); situational superko.
  { name: 'aga-white-19', query: { id: 'aga-white-19', moves: OPENING_19.slice(0, 2), rules: 'aga', komi: 7.5, boardXSize: 19, boardYSize: 19, analyzeTurns: [0, 1], maxVisits: 30, includeOwnership: true, overrideSettings: { reportAnalysisWinratesAs: 'WHITE' } } },
  { name: 'side-to-move-19', query: { id: 'side-to-move-19', moves: OPENING_19, rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, maxVisits: 30, analysisPVLen: 4, overrideSettings: { reportAnalysisWinratesAs: 'SIDETOMOVE' } } },
  // An integer komi: the game can end in a draw.
  { name: 'draw-9', query: { id: 'draw-9', moves: [['B', 'E5'], ['W', 'C3'], ['B', 'G3']], rules: 'chinese', komi: 7, boardXSize: 9, boardYSize: 9, maxVisits: 60, includeOwnership: true, includePolicy: true } },
  { name: 'rectangle-7x9', query: { id: 'rectangle-7x9', moves: [['B', 'D5']], rules: 'tromp-taylor', komi: 7.5, boardXSize: 7, boardYSize: 9, maxVisits: 40, includeOwnership: true } },
  { name: 'setup-9', query: { id: 'setup-9', moves: [], initialStones: [['B', 'C3'], ['B', 'G7'], ['B', 'D6'], ['W', 'G3'], ['W', 'C7'], ['W', 'F5']], initialPlayer: 'W', rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, maxVisits: 60 } },
  { name: 'handicap-9', query: { id: 'handicap-9', moves: [], initialStones: [['B', 'C3'], ['B', 'G7'], ['B', 'G3']], whiteHandicapBonus: 'N-1', rules: 'chinese', komi: 0.5, boardXSize: 9, boardYSize: 9, maxVisits: 60 } },
  // KataGo takes allowMoves or avoidMoves in a query, not both.
  { name: 'avoid-9', query: { id: 'avoid-9', moves: [['B', 'E5']], rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, maxVisits: 60, avoidMoves: [{ player: 'W', moves: ['C3', 'G7', 'C7', 'G3'], untilDepth: 1 }] } },
  { name: 'allow-9', query: { id: 'allow-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, maxVisits: 60, allowMoves: [{ player: 'B', moves: ['G7', 'C7', 'G3', 'D4', 'F6'], untilDepth: 2 }] } },
  { name: 'error-avoid-and-allow', query: { id: 'error-avoid-and-allow', moves: [], rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, avoidMoves: [{ player: 'W', moves: ['C3'], untilDepth: 1 }], allowMoves: [{ player: 'B', moves: ['G7'], untilDepth: 1 }] } },
  {
    name: 'settings-9',
    query: {
      id: 'settings-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, maxVisits: 60,
      rootPolicyTemperature: 1.5, rootFpuReductionMax: 0.5, overrideSettings: { cpuctExploration: 1.1, staticScoreUtilityFactor: 0.2 },
    },
  },
  { name: 'clear_cache', action: 'clear_cache' },
  { name: 'during-search-9', query: { id: 'during-search-9', moves: [['B', 'E5']], rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, maxVisits: 100, reportDuringSearchEvery: 0.1 } },
  { name: 'stopped-19', abortAtFirstReport: true, query: { id: 'stopped-19', moves: OPENING_19, rules: 'tromp-taylor', komi: 7.5, boardXSize: 19, boardYSize: 19, analyzeTurns: [2, 3], maxVisits: 100000, reportDuringSearchEvery: 0.2 } },
  // Errors and warnings.
  { name: 'error-komi', query: { id: 'error-komi', moves: [], rules: 'chinese', komi: 7.25, boardXSize: 19, boardYSize: 19 } },
  { name: 'error-board-size', query: { id: 'error-board-size', moves: [], rules: 'chinese', komi: 7.5, boardXSize: 25, boardYSize: 19 } },
  { name: 'error-illegal-move', query: { id: 'error-illegal-move', moves: [['B', 'D4'], ['W', 'D4']], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19 } },
  { name: 'error-turns', query: { id: 'error-turns', moves: [['B', 'D4']], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, analyzeTurns: [5] } },
  { name: 'warning-unused', query: { id: 'warning-unused', moves: [], rules: 'chinese', komi: 7.5, boardXSize: 9, boardYSize: 9, maxVisits: 5, ...({ zzz: 1, aaa: true } as object) } as AnalysisQuery },
];

// ------------------------------------------------------------------ running a case

interface Outcome {
  results?: AnalysisResponse[];
  warnings?: WarningResponse[];
  error?: { message: string; field?: string };
  reply?: Record<string, unknown>;
  ms: number;
}

async function runCase(engine: AnalysisEngine, c: Case): Promise<Outcome> {
  const started = performance.now();
  const ms = () => Math.round(performance.now() - started);
  try {
    if (c.action === 'query_version') return { reply: { ...(await engine.action<QueryVersionAction>({ action: 'query_version' })) }, ms: ms() };
    if (c.action === 'query_models') return { reply: { ...(await engine.action<QueryModelsAction>({ action: 'query_models' })) }, ms: ms() };
    if (c.action === 'clear_cache') return { reply: { ...(await engine.action<ClearCacheAction>({ action: 'clear_cache' })) }, ms: ms() };
    const results: AnalysisResponse[] = [];
    const warnings: WarningResponse[] = [];
    const controller = new AbortController();
    for await (const r of engine.analyze(c.query!, { signal: controller.signal, onWarning: (w) => warnings.push(w) })) {
      results.push(r);
      if (c.abortAtFirstReport && r.isDuringSearch) controller.abort();
    }
    return { results, warnings, ms: ms() };
  } catch (e) {
    if (!(e instanceof AnalysisError)) throw e;
    return { error: e.field === undefined ? { message: e.message } : { message: e.message, field: e.field }, ms: ms() };
  }
}

// ------------------------------------------------------------------ comparing

// The engine-webgpu search check's tolerances (test/search.test.mjs): the network goldens' 2e-4 for probabilities,
// policy and ownership and 2e-3 for scores, utility wider, relative 1e-3 for weights and play selection values, and
// ownership standard deviations compared as variances.
const TOL = { winrate: 2e-4, score: 2e-3, utility: 1e-3, relative: 1e-3, ownership: 2e-4 + 1e-6, ownershipVariance: 1e-3 };
type Check = (got: unknown, want: unknown, what: string, diffs: string[]) => void;

const num = (v: unknown): number => (typeof v === 'number' ? v : NaN);
const within = (tol: (want: number) => number): Check => (got, want, what, diffs) => {
  const g = num(got);
  const w = num(want);
  if (!(Math.abs(g - w) <= tol(w))) diffs.push(`${what}: ${g} vs native ${w}`);
};
const absolute = (t: number) => within(() => t);
const scaled = (t: number) => within((w) => t * Math.max(1, Math.abs(w)));
const exact: Check = (got, want, what, diffs) => {
  if (JSON.stringify(got) !== JSON.stringify(want)) diffs.push(`${what}: ${JSON.stringify(got)} vs native ${JSON.stringify(want)}`);
};
const each = (check: Check): Check => (got, want, what, diffs) => {
  if (!Array.isArray(got) || !Array.isArray(want) || got.length !== want.length) {
    diffs.push(`${what}: ${Array.isArray(got) ? got.length : typeof got} values vs native ${Array.isArray(want) ? want.length : typeof want}`);
    return;
  }
  const before = diffs.length;
  want.forEach((w, i) => check(got[i], w, `${what}[${i}]`, diffs));
  // One line per array: the first value out of tolerance and how many.
  if (diffs.length - before > 1) diffs.splice(before + 1, diffs.length, `${what}: ${diffs.length - before} values out of tolerance in all`);
};
const variance: Check = (got, want, what, diffs) => {
  const g = num(got);
  const w = num(want);
  if (!(Math.abs(g * g - w * w) <= TOL.ownershipVariance)) diffs.push(`${what}: ${g} vs native ${w} (as variances)`);
};
const policy: Check = (got, want, what, diffs) => (num(want) === -1 ? exact(got, want, what, diffs) : absolute(TOL.winrate)(got, want, what, diffs));

const MOVE: Record<string, Check> = {
  move: exact, visits: exact, edgeVisits: exact, order: exact, isSymmetryOf: exact, pv: exact, pvVisits: exact, pvEdgeVisits: exact,
  winrate: absolute(TOL.winrate), lcb: absolute(TOL.winrate), prior: absolute(TOL.winrate), noResultValue: absolute(TOL.winrate),
  scoreLead: absolute(TOL.score), scoreMean: absolute(TOL.score), scoreSelfplay: absolute(TOL.score), scoreStdev: scaled(TOL.score),
  utility: absolute(TOL.utility), utilityLcb: absolute(TOL.utility),
  weight: scaled(TOL.relative), edgeWeight: scaled(TOL.relative), playSelectionValue: scaled(TOL.relative),
  ownership: each(absolute(TOL.ownership)), ownershipStdev: each(variance),
};
const ROOT: Record<string, Check> = {
  visits: exact, currentPlayer: exact, thisHash: exact, symHash: exact, weight: scaled(TOL.relative),
  winrate: absolute(TOL.winrate), rawWinrate: absolute(TOL.winrate), rawNoResultProb: absolute(TOL.winrate), rawStWrError: absolute(TOL.winrate),
  scoreSelfplay: absolute(TOL.score), scoreLead: absolute(TOL.score), rawLead: absolute(TOL.score), rawScoreSelfplay: absolute(TOL.score),
  scoreStdev: scaled(TOL.score), rawScoreSelfplayStdev: scaled(TOL.score), rawStScoreError: scaled(TOL.score), rawVarTimeLeft: scaled(TOL.score),
  utility: absolute(TOL.utility),
};
const RESULT: Record<string, Check> = {
  id: exact, turnNumber: exact, isDuringSearch: exact, noResults: exact,
  policy: each(policy), ownership: MOVE['ownership']!, ownershipStdev: MOVE['ownershipStdev']!,
};

const keysOf = (o: object): string[] => Object.keys(o).sort();
function sameKeys(got: object, want: object, what: string, diffs: string[], ignore: readonly string[] = []): void {
  const g = keysOf(got).filter((k) => !ignore.includes(k));
  const w = keysOf(want).filter((k) => !ignore.includes(k));
  const missing = w.filter((k) => !g.includes(k));
  const extra = g.filter((k) => !w.includes(k));
  if (missing.length) diffs.push(`${what}: missing ${missing.join(', ')}`);
  if (extra.length) diffs.push(`${what}: not in native's: ${extra.join(', ')}`);
}

function compareFields(table: Record<string, Check>, got: Record<string, unknown>, want: Record<string, unknown>, what: string, diffs: string[]): void {
  for (const [k, w] of Object.entries(want)) {
    if (!(k in got)) continue;
    const check = table[k];
    if (!check) diffs.push(`${what} ${k}: a field this check doesn't know how to compare`);
    else check(got[k], w, `${what} ${k}`, diffs);
  }
}

/** A final result against native's: the same fields, each value within its tolerance. */
function compareFinal(got: AnalysisResponse, want: AnalysisResponse, what: string, diffs: string[]): void {
  sameKeys(got, want, what, diffs);
  const { moveInfos: gm, rootInfo: gr, ...gRest } = got;
  const { moveInfos: wm, rootInfo: wr, ...wRest } = want;
  compareFields(RESULT, gRest, wRest, what, diffs);
  if (gr && wr) {
    sameKeys(gr, wr, `${what} rootInfo`, diffs);
    compareFields(ROOT, gr as unknown as Record<string, unknown>, wr as unknown as Record<string, unknown>, `${what} rootInfo`, diffs);
  }
  if (gm && wm) {
    const ours = new Map(gm.map((m) => [m.move, m]));
    const moves = (list: { move: string }[]) => list.map((m) => m.move).sort().join(' ');
    if (moves(gm) !== moves(wm)) diffs.push(`${what} moves searched: ${moves(gm)} vs native ${moves(wm)}`);
    for (const w of wm) {
      const g = ours.get(w.move);
      if (!g) continue;
      sameKeys(g, w, `${what} ${w.move}`, diffs);
      compareFields(MOVE, g as unknown as Record<string, unknown>, w as unknown as Record<string, unknown>, `${what} ${w.move}`, diffs);
    }
  }
}

/** A result from a search stopped early, or a report during the search: the same fields as native's, no values. */
function compareForm(got: AnalysisResponse, want: AnalysisResponse, what: string, diffs: string[]): void {
  sameKeys(got, want, what, diffs);
  if (got.rootInfo && want.rootInfo) sameKeys(got.rootInfo, want.rootInfo, `${what} rootInfo`, diffs);
  const [g] = got.moveInfos ?? [];
  const [w] = want.moveInfos ?? [];
  if (g && w) sameKeys(g, w, `${what} moveInfos[0]`, diffs, ['isSymmetryOf']);
}

function compareCase(c: Case, browser: Outcome, native: Outcome): string[] {
  const diffs: string[] = [];
  if (c.action) {
    const g = browser.reply ?? {};
    const w = native.reply ?? {};
    if (!native.reply || !browser.reply) diffs.push(`reply: ${JSON.stringify(browser.error ?? g)} vs native ${JSON.stringify(native.error ?? w)}`);
    else if (c.action === 'query_version') exact(g['version'], w['version'], 'version', diffs);
    else if (c.action === 'query_models') {
      const gm = (g['models'] as Record<string, unknown>[] | undefined)?.[0] ?? {};
      const wm = (w['models'] as Record<string, unknown>[] | undefined)?.[0] ?? {};
      sameKeys(gm, wm, 'models[0]', diffs);
      // The model's own name and version must agree; name (the file name given) and maxBatchSize are each engine's own.
      for (const k of ['internalName', 'version', 'usesHumanSLProfile', 'usingFP16']) exact(gm[k], wm[k], `models[0].${k}`, diffs);
    } else exact(g['action'], w['action'], 'action', diffs);
    return diffs;
  }
  if (native.error || browser.error) {
    exact(browser.error ?? null, native.error ?? null, 'error', diffs);
    return diffs;
  }
  const strip = (ws: WarningResponse[] | undefined) => (ws ?? []).map((w) => ({ field: w.field, warning: w.warning }));
  exact(strip(browser.warnings), strip(native.warnings), 'warnings', diffs);
  const finals = (rs: AnalysisResponse[] | undefined) => (rs ?? []).filter((r) => !r.isDuringSearch);
  const during = (rs: AnalysisResponse[] | undefined) => (rs ?? []).filter((r) => r.isDuringSearch);
  const gf = finals(browser.results);
  const wf = finals(native.results);
  exact(gf.map((r) => r.turnNumber), wf.map((r) => r.turnNumber), 'final results by turn', diffs);
  const wantDuring = c.query?.reportDuringSearchEvery !== undefined;
  for (const [label, rs] of [['browser', browser.results], ['native', native.results]] as const) {
    const n = during(rs).length;
    if (wantDuring ? n === 0 : n > 0) diffs.push(`${label}: ${n} reports during the search`);
  }
  const [firstDuring] = during(native.results);
  if (firstDuring) during(browser.results).forEach((r, i) => compareForm(r, firstDuring, `during[${i}]`, diffs));
  wf.forEach((w, i) => {
    const g = gf[i];
    if (!g || g.turnNumber !== w.turnNumber) return;
    const what = `turn ${w.turnNumber}`;
    if (c.abortAtFirstReport) {
      // Stopped early: each engine stops where it is, so only the fields are compared.
      exact(!!g.noResults, !!w.noResults, `${what} noResults`, diffs);
      compareForm(g, w, what, diffs);
    } else compareFinal(g, w, what, diffs);
  });
  return diffs;
}

// ------------------------------------------------------------------ the run

interface CaseReport {
  name: string;
  ok: boolean;
  differences: string[];
  native: Outcome & { reportsDuringSearch?: number };
  browser: Outcome & { reportsDuringSearch?: number };
}

/** An outcome for the report: the final results, and of the reports during the search (compared for form), the first and a count. */
function forReport(o: Outcome): CaseReport['native'] {
  if (!o.results) return o;
  const during = o.results.filter((r) => r.isDuringSearch);
  const [first] = during;
  return { ...o, results: o.results.filter((r) => !r.isDuringSearch || r === first), reportsDuringSearch: during.length };
}

async function runCheck(): Promise<void> {
  const pairing = parsePairingFragment(location.hash) ?? savedPairing;
  const file = ui.model.files?.[0];
  if (!pairing) return setStatus('No bridge pairing: open this page from the link the bridge prints.', 'error');
  if (!file) return setStatus('Pick the model file the bridge runs.', 'error');
  ui.run.disabled = true;
  ui.results.replaceChildren();
  const report: Record<string, unknown> = { startedAt: new Date().toISOString(), userAgent: navigator.userAgent, modelFile: file.name };
  let native: AnalysisEngine | null = null;
  let browser: AnalysisEngine | null = null;
  try {
    const config = parseConfig(ui.config.value);
    report['config'] = config;
    setStatus('Connecting to KataGo on this computer…', 'busy');
    native = await connectBridge(pairing);
    setStatus('Loading the model in the browser…', 'busy');
    const model = await loadModel(new Uint8Array(await file.arrayBuffer()));
    const device = await requestEngineDevice();
    const threads = Number(config['numAnalysisThreads'] ?? 1) * Number(config['numSearchThreadsPerAnalysisThread'] ?? config['numSearchThreads'] ?? 1);
    const net = await GpuNetwork.create(device, model, { maxBatch: Math.max(1, Math.min(64, threads)) });
    const info = device.adapterInfo;
    report['adapter'] = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
    browser = await createAnalysisEngine({ model, modelName: file.name, net, config });
    const cases: CaseReport[] = [];
    report['cases'] = cases;
    for (const c of CASES) {
      setStatus(`Running ${c.name} (${cases.length + 1} of ${CASES.length})…`, 'busy');
      // One engine at a time, so neither slows the other.
      const n = await runCase(native, c);
      const b = await runCase(browser, c);
      const differences = compareCase(c, b, n);
      cases.push({ name: c.name, ok: differences.length === 0, differences, native: forReport(n), browser: forReport(b) });
      const li = document.createElement('li');
      li.dataset.ok = String(differences.length === 0);
      li.textContent = `${c.name}: ${differences.length === 0 ? 'pass' : `fail (${differences.length})`} · native ${n.ms} ms, browser ${b.ms} ms`;
      if (differences.length) {
        const ul = document.createElement('ul');
        for (const d of differences.slice(0, 20)) {
          const item = document.createElement('li');
          item.textContent = d;
          ul.append(item);
        }
        li.append(ul);
      }
      ui.results.append(li);
    }
    report['ok'] = cases.every((c) => c.ok);
    setStatus(report['ok'] ? `Pass: all ${cases.length} cases match native.` : `Fail: ${cases.filter((c) => !c.ok).length} of ${cases.length} cases differ from native.`, report['ok'] ? 'connected' : 'error');
  } catch (e) {
    report['ok'] = false;
    report['error'] = String(e instanceof Error ? (e.stack ?? e.message) : e);
    setStatus(`The check stopped: ${(e as Error).message}`, 'error');
  } finally {
    await native?.close();
    await browser?.close();
    ui.run.disabled = false;
  }
  const json = JSON.stringify(report, null, 2) + '\n';
  ui.download.href = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
  ui.download.hidden = false;
  // Served by scripts/serve.mjs, the report is also saved on this computer; elsewhere there is nothing to post to.
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
    await fetch('/report', { method: 'POST', headers: { 'content-type': 'application/json' }, body: json }).catch(() => undefined);
  }
}

// The pairing from the link the bridge printed, taken out of the address bar; the config from `?config=`, if given.
const savedPairing = parsePairingFragment(location.hash);
if (location.hash) history.replaceState(null, '', location.pathname + location.search);
const given = new URLSearchParams(location.search).get('config');
if (given) ui.config.value = given;
ui.form.addEventListener('submit', (e) => {
  e.preventDefault();
  void runCheck();
});
setStatus(savedPairing ? 'Paired with the bridge. Pick the model file, then run the check.' : 'Not paired: open this page from the link the bridge prints.', savedPairing ? 'idle' : 'error');
