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
  // The query fields that change the search (part B, step 11c), each sent to native as given in `extra`.
  // A set-up position with White to move, and three handicap stones with White's bonus N-1.
  { id: 'setup-9-white', moves: [], rules: 'tromp-taylor', size: 9, maxVisits: 60, extra: { initialStones: [['B', 'C3'], ['B', 'G7'], ['B', 'D6'], ['W', 'G3'], ['W', 'C7'], ['W', 'F5']], initialPlayer: 'W' } },
  { id: 'handicap-9-n-1', moves: [], rules: 'chinese', size: 9, maxVisits: 60, extra: { initialStones: [['B', 'C3'], ['B', 'G7'], ['B', 'G3']], whiteHandicapBonus: 'N-1' } },
  // Black moving three times in a row: KataGo starts the history again at each, and counts them as handicap. The turn
  // number then counts from the last restart, which the early root policy temperature shows.
  { id: 'handicap-moves-9', moves: [['B', 'C3'], ['B', 'G7'], ['B', 'G3'], ['W', 'C7'], ['B', 'E5']], rules: 'chinese', size: 9, maxVisits: 60, extra: { overrideSettings: { rootPolicyTemperatureEarly: 3.0 } } },
  // White retakes a ko at once, which the rules forbid: KataGo plays it anyway. Searched with the history before the root
  // shown (the analysis engine otherwise leaves it out), so the network sees the illegal move.
  { id: 'tolerant-ko-9', moves: [['B', 'E6'], ['W', 'F6'], ['B', 'D5'], ['W', 'G5'], ['B', 'E4'], ['W', 'F4'], ['B', 'A1'], ['W', 'E5'], ['B', 'F5'], ['W', 'E5']], rules: 'tromp-taylor', size: 9, maxVisits: 60, extra: { overrideSettings: { ignorePreRootHistory: false } } },
  // Three kos under simple ko (KataGo's own rules test "Triple ko simple", cpp/tests/testrules.cpp, with an empty row
  // added): the cycle has been played once and five moves more, so White's F5 repeats the position a third time and
  // ends the game with no result, a terminal node of the search.
  { id: 'no-result-7', moves: [['B', 'D6'], ['W', 'B5'], ['B', 'F6'], ['W', 'D5'], ['B', 'B6'], ['W', 'F5'], ['B', 'D6'], ['W', 'B5'], ['B', 'F6'], ['W', 'D5'], ['B', 'B6']], rules: 'chinese', size: 7, maxVisits: 60, extra: { initialStones: [
    ...['A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'A6', 'C6', 'E6', 'G6', 'D5', 'F5', 'A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3'].map((v) => ['W', v]),
    ...['B6', 'A5', 'C5', 'E5', 'G5', 'A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4'].map((v) => ['B', v]),
  ] } },
  // Territory scoring and the group taxes (part C, step 13c). Native plays the query's moves with the encore prevented
  // (preventCleanupPhase, true by default), so after two passes play goes on past the phase end; inside the search,
  // two passes start the encore, and lines are cut where the root's phase would end.
  { id: 'late-9-japanese', moves: LATE_9.slice(0, 48), rules: 'japanese', size: 9, maxVisits: 100 },
  { id: 'late-9-japanese-after-pass', moves: LATE_9.slice(0, 51), rules: 'japanese', size: 9, maxVisits: 100 },
  { id: 'late-9-japanese-past-phase-end', moves: LATE_9, rules: 'japanese', size: 9, maxVisits: 60 },
  { id: 'late-9-territory-no-tax-after-pass', moves: LATE_9.slice(0, 51), rules: 'koSIMPLEscoreTERRITORYtaxNONEsui0', size: 9, maxVisits: 100 },
  { id: 'late-9-stone-scoring-after-pass', moves: LATE_9.slice(0, 51), rules: 'stone-scoring', size: 9, maxVisits: 100 },
  { id: 'late-9-ancient-territory-after-pass', moves: LATE_9.slice(0, 51), rules: 'ancient-territory', size: 9, maxVisits: 100 },
  { id: 'opening-19-japanese', moves: [['B', 'Q16'], ['W', 'D4'], ['B', 'Q4']], rules: 'japanese', size: 19, maxVisits: 30 },
  // The root in the encore: native run with preventCleanupPhase=false (`preventCleanupPhase: false` here; ours the
  // same), so two passes start the encore and two more its second phase. The empty board in the encore is symmetric;
  // Black moving twice in a row in the encore starts the history again in that phase.
  { id: 'encore-1-japanese', moves: LATE_9, rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false },
  { id: 'encore-2-japanese', moves: [...LATE_9, ['B', 'pass'], ['W', 'pass']], rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false },
  { id: 'encore-1-sym-9', moves: [['B', 'pass'], ['W', 'pass']], rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false },
  { id: 'encore-2-sym-9', moves: [['B', 'pass'], ['W', 'pass'], ['B', 'pass'], ['W', 'pass']], rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false },
  { id: 'encore-1-restart', moves: [...LATE_9, ['B', 'H4'], ['B', 'J4']], rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false },
  // A symmetric board whose second encore started from stones that aren't (C3 only): no symmetry at the root.
  { id: 'encore-2-asym-start', moves: [['B', 'C3'], ['W', 'pass'], ['B', 'pass'], ['W', 'pass'], ['B', 'pass'], ['W', 'pass'], ['B', 'G7']], rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false },
  // Two kos, mirror images through the centre. In the encore Black takes both, and White's pass for ko at B9 lifts the
  // block on A9 only: the stones are symmetric, the recapture blocks aren't.
  { id: 'encore-1-asym-recapture', moves: [['B', 'pass'], ['W', 'pass'], ['B', 'A9'], ['W', 'pass'], ['B', 'J1'], ['W', 'B9']], rules: 'japanese', size: 9, maxVisits: 60, preventCleanupPhase: false, extra: { initialStones: [
    ['W', 'B9'], ['W', 'A8'], ['B', 'C9'], ['B', 'B8'], ['W', 'H1'], ['W', 'J2'], ['B', 'G1'], ['B', 'H2'],
  ] } },
  // The button (AGA's button rules, area scoring): before the first pass, which takes it and the half point (the root
  // pass slightly discouraged), after it, and at the end of the game. Komi 7.5 with the button makes a draw possible,
  // so the draw value counts; komi 7 makes it impossible.
  { id: 'late-9-button', moves: LATE_9.slice(0, 48), rules: 'aga-button', size: 9, maxVisits: 100 },
  { id: 'late-9-button-after-pass', moves: LATE_9.slice(0, 49), rules: 'aga-button', size: 9, maxVisits: 100 },
  { id: 'late-9-button-after-passes', moves: LATE_9.slice(0, 51), rules: 'aga-button', size: 9, maxVisits: 100 },
  { id: 'late-9-button-game-over', moves: LATE_9, rules: 'aga-button', size: 9, maxVisits: 60 },
  { id: 'late-9-button-draw', moves: LATE_9.slice(0, 48), rules: 'aga-button', size: 9, maxVisits: 100, extra: { overrideSettings: { drawEquivalentWinsForWhite: 0.25 } } },
  { id: 'late-9-button-komi-7-draw', moves: LATE_9.slice(0, 48), rules: 'aga-button', komi: 7, size: 9, maxVisits: 100, extra: { overrideSettings: { drawEquivalentWinsForWhite: 0.25 } } },
  { id: 'opening-19-button', moves: [['B', 'Q16'], ['W', 'D4'], ['B', 'Q4']], rules: 'aga-button', size: 19, maxVisits: 30 },
  // Black avoids moves at the root, White its usual replies until depth 4.
  { id: 'avoid-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', size: 9, maxVisits: 80, extra: { avoidMoves: [{ player: 'B', moves: ['D3', 'C4', 'D4'], untilDepth: 1 }, { player: 'W', moves: ['C6', 'F3', 'D3', 'pass'], untilDepth: 4 }] } },
  // An avoid list turns off enableMorePassingHacks for that player at every depth: here both players avoid an empty
  // corner they don't play, in pass-9-turn-36's position, where the hack changes native's visits.
  { id: 'avoid-passing-hack-9', moves: LATE_9.slice(0, 36), rules: 'tromp-taylor', size: 9, maxVisits: 100, extra: { avoidMoves: [{ player: 'B', moves: ['A1'], untilDepth: 1 }, { player: 'W', moves: ['J9'], untilDepth: 1 }] } },
  { id: 'allow-9', moves: LATE_9.slice(0, 33), rules: 'tromp-taylor', size: 9, maxVisits: 80, extra: { allowMoves: [{ player: LATE_9[33][0], moves: ['B2', 'H8', 'E1', 'J5'], untilDepth: 1 }] } },
  { id: 'avoid-sym-9', moves: [], rules: 'tromp-taylor', size: 9, maxVisits: 80, extra: { avoidMoves: [{ player: 'B', moves: ['E5', 'C3', 'G7'], untilDepth: 1 }] } },
  { id: 'temperature-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', size: 9, maxVisits: 80, extra: { rootPolicyTemperature: 1.5 } },
  { id: 'fpu-9', moves: LATE_9.slice(0, 33), rules: 'tromp-taylor', size: 9, maxVisits: 80, extra: { rootFpuReductionMax: 0.4 } },
  { id: 'override-9', moves: LATE_9.slice(0, 16), rules: 'tromp-taylor', size: 9, maxVisits: 80, extra: { overrideSettings: { cpuctExploration: 1.3, staticScoreUtilityFactor: 0.2, useLcbForSelection: false, fpuLossProp: 0.4 } } },
  // The early root policy temperature, which fades with the turn number.
  { id: 'override-temperature-19', moves: [['B', 'Q16'], ['W', 'D4'], ['B', 'Q4']], rules: 'tromp-taylor', size: 19, maxVisits: 30, extra: { overrideSettings: { rootPolicyTemperatureEarly: 2.0, rootPolicyTemperature: 0.7 } } },
];
const SHORT = [
  { id: 'opening-9', moves: [['B', 'E5'], ['W', 'C3']], rules: 'tromp-taylor', size: 9, maxVisits: 25 },
  { id: 'turn-33-9', moves: LATE_9.slice(0, 33), rules: 'tromp-taylor', size: 9, maxVisits: 25 },
  { id: 'late-9-after-pass', moves: LATE_9.slice(0, 51), rules: 'tromp-taylor', size: 9, maxVisits: 25 },
  { id: 'late-9-japanese-after-pass', moves: LATE_9.slice(0, 51), rules: 'japanese', size: 9, maxVisits: 25 },
  { id: 'late-9-button', moves: LATE_9.slice(0, 48), rules: 'aga-button', size: 9, maxVisits: 25 },
];
const queries = { full: FULL, short: SHORT }[values.set];
if (!queries) throw new Error(`--set must be full or short, not ${values.set}`);

// Every optional output, so the goldens hold every field native reports.
const INCLUDE = { includeOwnership: true, includeOwnershipStdev: true, includeMovesOwnership: true, includeMovesOwnershipStdev: true, includePolicy: true, includePVVisits: true, includeNoResultValue: true };

const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
// One KataGo process per query: its evaluation cache, like ours, ignores move history, so a cache shared across
// queries would hand one query evaluations made for another query's history.
const byId = new Map();
for (const q of queries) {
  const query = { id: q.id, moves: q.moves, rules: q.rules, komi: q.komi ?? 7.5, boardXSize: q.size, boardYSize: q.size, maxVisits: q.maxVisits, analysisPVLen: 6, ...INCLUDE, ...q.extra };
  const overrides = [`logDir=${logDir}`, ...OVERRIDES, ...(q.preventCleanupPhase === false ? ['preventCleanupPhase=false'] : [])];
  const run = spawnSync(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', overrides.join(',')], { input: JSON.stringify(query) + '\n', encoding: 'utf8', maxBuffer: 1 << 28 });
  if (run.status !== 0) {
    console.error(run.stderr);
    process.exit(1);
  }
  // The response, after any warnings (a warning line has a "warning" field).
  const lines = run.stdout.trim().split('\n').map((l) => JSON.parse(l));
  for (const w of lines.filter((l) => 'warning' in l)) console.error(`${q.id}: warning: ${JSON.stringify(w)}`);
  const r = lines.find((l) => !('warning' in l));
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
  note: 'KataGo analysis engine, one search thread, symmetry 0, the features listed in overrides switched off, every optional output included; values for the side to move. Each query holds native\'s response as it is, less id, turnNumber and isDuringSearch.',
  queries: queries.map((q) => {
    const r = byId.get(q.id);
    if (!r || r.error) throw new Error(`${q.id}: ${r?.error ?? 'no response'}`);
    const { id, turnNumber, isDuringSearch, ...response } = r;
    return { ...q, komi: q.komi ?? 7.5, response };
  }),
};
const outFile = values.out ?? join(here, '..', 'test', 'goldens', `search-${basename(values.model).replace(/\.bin\.gz$/, '')}.json`);
// One query per line: the ownership maps make the file long otherwise.
const { queries: qs, ...head } = out;
writeFileSync(outFile, JSON.stringify(head, null, 1).replace(/\n}$/, ',\n "queries": [\n') + qs.map((q) => JSON.stringify(q)).join(',\n') + '\n ]\n}\n');
console.error(`wrote ${outFile}`);
