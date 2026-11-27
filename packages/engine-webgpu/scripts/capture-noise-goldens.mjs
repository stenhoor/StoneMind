#!/usr/bin/env node
// Capture native KataGo's visits with wide root noise on, over many runs of a few positions, for the statistical check
// of the browser's wide root noise (test/noise.test.mjs). The noise is random and the two engines' random numbers
// differ, so the check compares the spread of visits over many runs rather than single runs.
//
//   node packages/engine-webgpu/scripts/capture-noise-goldens.mjs --katago <katago> --config <analysis_example.cfg> \
//     --model packages/engine-webgpu/test/fixtures/g170-b6c96-s175395328-d26788732.bin.gz [--runs 100] [--out <file>]
//
// Native runs with the exact check's overrides (scripts/capture-search-goldens.mjs) except wideRootNoise, which is
// left at native's analysis default (0.04). All runs of a position go to one KataGo process, as separate queries: each
// query's search gets its own random seed, and they share the evaluation cache, which is safe as every query has the
// same history.
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({ options: { katago: { type: 'string' }, config: { type: 'string' }, model: { type: 'string' }, runs: { type: 'string', default: '100' }, out: { type: 'string' } } });
if (!values.katago || !values.config || !values.model) {
  console.error('usage: capture-noise-goldens.mjs --katago <katago> --config <analysis cfg> --model <model .bin.gz> [--runs 100] [--out <file>]');
  process.exit(2);
}
const runs = Number(values.runs);

// The exact check's overrides, read from its script so the two can't drift apart, less the one that switches the noise off.
const searchScript = readFileSync(join(here, 'capture-search-goldens.mjs'), 'utf8');
const OVERRIDES = [...searchScript.matchAll(/^  '([^']+=[^']+)',$/gm)].map((m) => m[1]).filter((o) => !o.startsWith('wideRootNoise='));

const LATE_9 = JSON.parse(readFileSync(join(here, '..', 'test', 'goldens', 'search-late-9.json'), 'utf8')).moves;
const POSITIONS = [
  { id: 'opening-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'turn-33-9', moves: LATE_9.slice(0, 33), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
];

const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
const out = {
  capturedWith: {
    katagoVersion: spawnSync(values.katago, ['version'], { encoding: 'utf8' }).stdout.split('\n')[0],
    overrides: OVERRIDES,
    modelFile: basename(values.model),
    modelSha256: createHash('sha256').update(readFileSync(values.model)).digest('hex'),
    runs,
    date: new Date().toISOString(),
  },
  note: 'KataGo analysis engine, one search thread, symmetry 0, wide root noise at its default; each run lists the visits of every move searched.',
  positions: [],
};
for (const pos of POSITIONS) {
  const input = Array.from({ length: runs }, (_, i) =>
    JSON.stringify({ id: `${pos.id}-${i}`, moves: pos.moves, rules: pos.rules, komi: 7.5, boardXSize: pos.size, boardYSize: pos.size, maxVisits: pos.maxVisits }),
  ).join('\n');
  const run = spawnSync(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', [`logDir=${logDir}`, ...OVERRIDES].join(',')], { input: input + '\n', encoding: 'utf8', maxBuffer: 1 << 28 });
  if (run.status !== 0) {
    console.error(run.stderr);
    process.exit(1);
  }
  const byId = new Map(run.stdout.trim().split('\n').map((l) => JSON.parse(l)).map((r) => [r.id, r]));
  const visits = [];
  for (let i = 0; i < runs; i++) {
    const r = byId.get(`${pos.id}-${i}`);
    if (!r || r.error) throw new Error(`${pos.id}-${i}: ${r?.error ?? 'no response'}`);
    visits.push(Object.fromEntries(r.moveInfos.filter((m) => !m.isSymmetryOf).map((m) => [m.move, m.visits])));
  }
  out.positions.push({ ...pos, komi: 7.5, visits });
  console.error(`captured ${pos.id}: ${runs} runs`);
}
const outFile = values.out ?? join(here, '..', 'test', 'goldens', `noise-${basename(values.model).replace(/\.bin\.gz$/, '')}.json`);
writeFileSync(outFile, JSON.stringify(out) + '\n');
console.error(`wrote ${outFile}`);
