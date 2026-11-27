#!/usr/bin/env node
// Capture native KataGo's analysis of a few positions as goldens for the search (test/search.test.mjs).
//
// Native KataGo runs with one search thread, symmetry 0 for every evaluation (nnRandomize=false), and the search
// features our search doesn't have yet switched off, so the two searches should visit the same moves the same number
// of times. Use a KataGo built with the CPU (Eigen) backend, as for the network goldens (scripts/capture-goldens.mjs).
//
//   node packages/engine-webgpu/scripts/capture-search-goldens.mjs \
//     --katago <katago> --config <analysis_example.cfg> \
//     --model packages/engine-webgpu/test/fixtures/g170-b6c96-s175395328-d26788732.bin.gz [--set full|short] [--out <file>]
//
// --set short captures three 9x9 positions at 25 visits, for networks too slow to test at more with the reference
// forward pass (b18c384nbt, whose model version, 14, has the short-term error outputs uncertainty weighting uses).
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({ options: { katago: { type: 'string' }, config: { type: 'string' }, model: { type: 'string' }, set: { type: 'string', default: 'full' }, out: { type: 'string' } } });
if (!values.katago || !values.config || !values.model) {
  console.error('usage: capture-search-goldens.mjs --katago <katago> --config <analysis cfg> --model <model .bin.gz> [--out <file>]');
  process.exit(2);
}

/** The search features not ported yet, switched off, and the analysis engine's other defaults pinned for the test. */
const OVERRIDES = [
  'numAnalysisThreads=1',
  'numSearchThreadsPerAnalysisThread=1',
  'nnRandomize=false',
  'useGraphSearch=false',
  'wideRootNoise=0',
  'rootNumSymmetriesToSample=1',
  'useFP16=false',
  'reportAnalysisWinratesAs=SIDETOMOVE',
];

// A 9x9 game played by the search itself (b6, 30 visits a move), for positions near the end with passes in reach.
const LATE_9 = JSON.parse(readFileSync(join(here, '..', 'test', 'goldens', 'search-late-9.json'), 'utf8')).moves;
const FULL = [
  { id: 'opening-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', size: 9, maxVisits: 120 },
  { id: 'late-9', moves: LATE_9.slice(0, 48), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'late-9-after-pass', moves: LATE_9.slice(0, 51), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'late-9-game-over', moves: LATE_9, rules: 'tromp-taylor', size: 9, maxVisits: 60 },
  { id: 'late-9-chinese-after-pass', moves: LATE_9.slice(0, 51), rules: 'chinese', size: 9, maxVisits: 100 },
  // Positions where LCB chooses a move other than the most visited one (found by scanning the game with native).
  { id: 'lcb-9-turn-3', moves: LATE_9.slice(0, 3), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'lcb-9-turn-33', moves: LATE_9.slice(0, 33), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  // Positions where noise pruning changes native's visits (found by scanning the game with it on and off).
  { id: 'noise-9-turn-5', moves: LATE_9.slice(0, 5), rules: 'tromp-taylor', size: 9, maxVisits: 300 },
  { id: 'noise-9-turn-16', moves: LATE_9.slice(0, 16), rules: 'tromp-taylor', size: 9, maxVisits: 300 },
  // Positions where enableMorePassingHacks changes native's visits (found by scanning the game with it on and off).
  { id: 'pass-9-turn-36', moves: LATE_9.slice(0, 36), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'pass-9-turn-42', moves: LATE_9.slice(0, 42), rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  // White's last four moves passes, Black filling its own territory in between: rootPruneUselessMoves applies.
  { id: 'useless-9', moves: [...LATE_9.slice(0, 48), ['B', 'C3'], ['W', 'pass'], ['B', 'C2'], ['W', 'pass'], ['B', 'B3'], ['W', 'pass'], ['B', 'B2'], ['W', 'pass']], rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  // Symmetric positions, for rootSymmetryPruning: the empty board (Black to move) and after tengen (White to move).
  { id: 'sym-9-empty', moves: [], rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'sym-9-tengen', moves: [['B', 'E5']], rules: 'tromp-taylor', size: 9, maxVisits: 100 },
  { id: 'opening-19', moves: [['B', 'Q16'], ['W', 'D4'], ['B', 'Q4']], rules: 'tromp-taylor', size: 19, maxVisits: 30 },
  { id: 'opening-19-aga', moves: [['B', 'Q16'], ['W', 'D4']], rules: 'aga', size: 19, maxVisits: 30 },
];
const SHORT = [
  { id: 'opening-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', size: 9, maxVisits: 25 },
  { id: 'turn-33-9', moves: LATE_9.slice(0, 33), rules: 'tromp-taylor', size: 9, maxVisits: 25 },
  { id: 'late-9-after-pass', moves: LATE_9.slice(0, 51), rules: 'tromp-taylor', size: 9, maxVisits: 25 },
];
const queries = { full: FULL, short: SHORT }[values.set];
if (!queries) throw new Error(`--set must be full or short, not ${values.set}`);

const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
// One KataGo process per query: its evaluation cache, like ours, ignores move history, so a cache shared across
// queries would hand one query evaluations made for another query's history.
const byId = new Map();
for (const q of queries) {
  const query = { id: q.id, moves: q.moves, rules: q.rules, komi: 7.5, boardXSize: q.size, boardYSize: q.size, maxVisits: q.maxVisits, analysisPVLen: 6 };
  const run = spawnSync(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', [`logDir=${logDir}`, ...OVERRIDES].join(',')], { input: JSON.stringify(query) + '\n', encoding: 'utf8', maxBuffer: 1 << 28 });
  if (run.status !== 0) {
    console.error(run.stderr);
    process.exit(1);
  }
  const r = JSON.parse(run.stdout.trim().split('\n')[0]);
  byId.set(r.id, r);
  console.error(`captured ${q.id}`);
}
const version = spawnSync(values.katago, ['version'], { encoding: 'utf8' }).stdout.split('\n')[0];
const out = {
  capturedWith: {
    katagoVersion: version,
    overrides: OVERRIDES,
    modelFile: basename(values.model),
    modelSha256: createHash('sha256').update(readFileSync(values.model)).digest('hex'),
    set: values.set,
    date: new Date().toISOString(),
  },
  note: 'KataGo analysis engine, one search thread, symmetry 0, the features listed in overrides switched off; winrate and scoreLead for the side to move.',
  queries: queries.map((q) => {
    const r = byId.get(q.id);
    if (!r || r.error) throw new Error(`${q.id}: ${r?.error ?? 'no response'}`);
    return {
      ...q,
      komi: 7.5,
      rootVisits: r.rootInfo.visits,
      moveInfos: r.moveInfos.map((m) => ({ move: m.move, order: m.order, visits: m.visits, edgeVisits: m.edgeVisits, winrate: m.winrate, scoreLead: m.scoreLead, lcb: m.lcb, utilityLcb: m.utilityLcb, playSelectionValue: m.playSelectionValue, pv: m.pv, ...(m.isSymmetryOf ? { isSymmetryOf: m.isSymmetryOf } : {}) })),
    };
  }),
};
const outFile = values.out ?? join(here, '..', 'test', 'goldens', `search-${basename(values.model).replace(/\.bin\.gz$/, '')}.json`);
writeFileSync(outFile, JSON.stringify(out, null, 1) + '\n');
console.error(`wrote ${outFile}`);
