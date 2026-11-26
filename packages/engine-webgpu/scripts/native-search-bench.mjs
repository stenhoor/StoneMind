#!/usr/bin/env node
// Time native KataGo's analysis engine on the same work as the GPU check page's search benchmark, from that
// benchmark's report: the live position (visits per second) and the reviewed game (every turn at the same visits).
//
//   node packages/engine-webgpu/scripts/native-search-bench.mjs --katago <katago> --config <analysis cfg> \
//     --model <model .bin.gz> --report <search benchmark report .json> [--override "k=v,..."] [--out <file>]
//
// KataGo runs with the given analysis config as it is (plus --override, if given). Times are wall-clock, from sending a
// query to its final response, after a warm-up query. KataGo's evaluation cache is cleared before each timed query
// (its `clear_cache` action), as the browser benchmark starts each search with an empty cache. The live query runs
// once per --repeat (default 3).
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';

const { values } = parseArgs({
  options: {
    katago: { type: 'string' },
    config: { type: 'string' },
    model: { type: 'string' },
    report: { type: 'string' },
    override: { type: 'string' },
    repeat: { type: 'string' },
    out: { type: 'string' },
  },
});
if (!values.katago || !values.config || !values.model || !values.report) {
  console.error('usage: native-search-bench.mjs --katago <katago> --config <analysis cfg> --model <model> --report <report.json> [--override "..."] [--out <file>]');
  process.exit(2);
}
const report = JSON.parse(readFileSync(values.report, 'utf8'));
if (report.kind !== 'search' || !report.live || !report.review) throw new Error('--report must be a search benchmark report');
const positions = JSON.parse(readFileSync(new URL('../test/goldens/positions.json', import.meta.url), 'utf8')).positions;
const live = positions.find((p) => p.id === report.live.position);

const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
const overrides = [`logDir=${logDir}`, ...(values.override ? [values.override] : [])].join(',');
const proc = spawn(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', overrides], { stdio: ['pipe', 'pipe', 'pipe'] });
let stderr = '';
proc.stderr.on('data', (d) => (stderr += d));
const waiting = new Map();
createInterface({ input: proc.stdout }).on('line', (line) => {
  const r = JSON.parse(line);
  if (r.error) throw new Error(`KataGo: ${r.error}`);
  if (r.isDuringSearch) return;
  const w = waiting.get(r.id);
  if (!w) return;
  w.responses.push(r);
  if (w.responses.length === w.expected) {
    waiting.delete(r.id);
    w.resolve(w.responses);
  }
});
proc.on('exit', (code) => {
  if (waiting.size) {
    console.error(stderr);
    console.error(`KataGo exited with code ${code}`);
    process.exit(1);
  }
});

/** Send a query; resolves with all its final responses (one per analysed turn) and the wall time. */
function query(q, expected = 1) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    waiting.set(q.id, { expected, responses: [], resolve: (responses) => resolve({ responses, ms: performance.now() - t0 }) });
    proc.stdin.write(JSON.stringify(q) + '\n');
  });
}

const base = { rules: 'tromp-taylor', komi: 7.5, boardXSize: 19, boardYSize: 19 };
await query({ id: 'warmup', ...base, moves: [], maxVisits: 200 });

const repeat = Number(values.repeat ?? 3);
const liveRuns = [];
for (let i = 0; i < repeat; i++) {
  await query({ id: `clear-live-${i}`, action: 'clear_cache' });
  const { responses, ms } = await query({ id: `live-${i}`, ...base, rules: live.rules, komi: live.komi, moves: live.moves, maxVisits: report.live.maxVisits });
  const visits = responses[0].rootInfo.visits;
  liveRuns.push({ visits, ms: +ms.toFixed(1), visitsPerSecond: +((visits * 1000) / ms).toFixed(1), bestMove: responses[0].moveInfos.find((m) => m.order === 0)?.move });
}

const game = report.review.game;
const turns = [...Array(game.length).keys()];
await query({ id: 'clear-review', action: 'clear_cache' });
const { responses: reviewResponses, ms: reviewMs } = await query({ id: 'review', ...base, moves: game, analyzeTurns: turns, maxVisits: report.review.visitsPerMove }, turns.length);
const reviewVisits = reviewResponses.reduce((s, r) => s + r.rootInfo.visits, 0);
// Native's recommended move at each turn, and how often each browser review mode recommended the same.
const nativeBest = new Map(reviewResponses.map((r) => [r.turnNumber, r.moveInfos.find((m) => m.order === 0)?.move ?? null]));
const modes = (report.review.modes ?? []).map((m) => ({
  batchSize: m.batchSize,
  concurrency: m.concurrency,
  minutesFor250Moves: m.minutesFor250Moves,
  agreesWithNative: m.bestMoves.filter((move, t) => move === nativeBest.get(t)).length,
  turns: m.bestMoves.length,
}));
proc.stdin.end();

const version = spawnSync(values.katago, ['version'], { encoding: 'utf8' }).stdout.split('\n')[0];
const out = {
  kind: 'native-search',
  katagoVersion: version,
  config: values.config,
  override: values.override ?? null,
  model: values.model,
  browserReport: { date: report.date, model: report.model, userAgent: report.userAgent },
  live: { position: live.id, maxVisits: report.live.maxVisits, runs: liveRuns },
  review: {
    moves: game.length,
    visitsPerMove: report.review.visitsPerMove,
    visits: reviewVisits,
    totalMs: +reviewMs.toFixed(1),
    visitsPerSecond: +((reviewVisits * 1000) / reviewMs).toFixed(1),
    minutesFor250Moves: +((reviewMs / game.length) * 250 / 60000).toFixed(2),
    bestMoves: turns.map((t) => nativeBest.get(t)),
  },
  browserReviewModes: modes,
};
const text = JSON.stringify(out, null, 2) + '\n';
if (values.out) writeFileSync(values.out, text);
console.log(text);
