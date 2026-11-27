#!/usr/bin/env node
// Top-move agreement between the browser engine's search and native KataGo (spike 0001, phase 5's exit criterion),
// from an agreement report of the GPU check page (its "Agreement positions" button, or gpu-check.mjs --agreement).
//
//   node packages/engine-webgpu/scripts/native-agreement.mjs --katago <katago> --config <analysis cfg> \
//     --model <model .bin.gz> --report <agreement report .json> [--reference-visits 4000] [--out <file>]
//
// For each visit count in the report, native KataGo analyses the same positions twice (its evaluation cache cleared
// before each run), so the agreement of native with itself is the control: KataGo's own search isn't deterministic with
// several threads. One more native run at --reference-visits is the stronger reference. KataGo runs with the given
// analysis config as it is. Each engine's recommended move is compared: each engine's move of order 0.
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
    'reference-visits': { type: 'string' },
    out: { type: 'string' },
  },
});
if (!values.katago || !values.config || !values.model || !values.report) {
  console.error('usage: native-agreement.mjs --katago <katago> --config <analysis cfg> --model <model> --report <report.json> [--reference-visits 4000] [--out <file>]');
  process.exit(2);
}
const report = JSON.parse(readFileSync(values.report, 'utf8'));
if (report.kind !== 'agreement') throw new Error('--report must be an agreement report');
const game = JSON.parse(readFileSync(new URL('../test/goldens/agreement-game.json', import.meta.url), 'utf8'));
const turns = report.positions.map((p) => p.turn);
const referenceVisits = Number(values['reference-visits'] ?? 4000);

const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
const proc = spawn(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', `logDir=${logDir}`], { stdio: ['pipe', 'pipe', 'pipe'] });
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
const query = (q, expected = 1) =>
  new Promise((resolve) => {
    waiting.set(q.id, { expected, responses: [], resolve });
    proc.stdin.write(JSON.stringify(q) + '\n');
  });

/** Native's recommended move at each sampled turn, as a map turn → move. */
async function nativeRun(id, maxVisits) {
  await query({ id: `clear-${id}`, action: 'clear_cache' });
  const responses = await query(
    { id, rules: game.rules, komi: game.komi, boardXSize: 19, boardYSize: 19, moves: game.moves, analyzeTurns: turns, maxVisits },
    turns.length,
  );
  const best = new Map();
  for (const r of responses) best.set(r.turnNumber, r.moveInfos.find((m) => m.order === 0)?.move ?? null);
  return best;
}

const agree = (a, b) => turns.filter((t) => a.get(t) === b.get(t)).length;
const results = [];
const reference = await nativeRun('reference', referenceVisits);
for (const v of report.visits) {
  const nativeA = await nativeRun(`native-a-${v}`, v);
  const nativeB = await nativeRun(`native-b-${v}`, v);
  for (const batch of report.batches) {
    const browser = new Map(report.positions.map((p) => [p.turn, p.byVisits[`${v}@${batch}`].best]));
    results.push({
      visits: v,
      browserBatch: batch,
      positions: turns.length,
      browserVsNative: { a: agree(browser, nativeA), b: agree(browser, nativeB) },
      nativeVsNative: agree(nativeA, nativeB),
      browserVsReference: agree(browser, reference),
      nativeVsReference: { a: agree(nativeA, reference), b: agree(nativeB, reference) },
      perTurn: turns.map((t) => ({ turn: t, browser: browser.get(t), nativeA: nativeA.get(t), nativeB: nativeB.get(t), reference: reference.get(t) })),
    });
  }
}
proc.stdin.end();

const version = spawnSync(values.katago, ['version'], { encoding: 'utf8' }).stdout.split('\n')[0];
const out = {
  kind: 'native-agreement',
  katagoVersion: version,
  config: values.config,
  model: values.model,
  referenceVisits,
  browserReport: { date: report.date, model: report.model, userAgent: report.userAgent, variant: report.variant },
  results,
};
const text = JSON.stringify(out, null, 2) + '\n';
if (values.out) writeFileSync(values.out, text);
console.log(JSON.stringify(results.map(({ perTurn, ...r }) => r), null, 2));
