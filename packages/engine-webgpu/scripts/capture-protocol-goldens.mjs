#!/usr/bin/env node
// Capture native KataGo's analysis engine answering scripted sessions, as goldens for the browser engine's protocol
// (test/protocol.test.mjs): every line native writes, for query lines sent in steps with pauses between them.
//
// Native runs with one analysis thread and one search thread, symmetry 0 for every evaluation (nnRandomize=false), and
// the search features our search doesn't have switched off, as for the search goldens (capture-search-goldens.mjs), so
// the searches should match. Winrates are reported for Black, as in KataGo's analysis_example.cfg; the queries set the
// other sides through overrideSettings. Use a KataGo built with the CPU (Eigen) backend.
//
//   node packages/engine-webgpu/scripts/capture-protocol-goldens.mjs \
//     --katago <katago> --config <analysis_example.cfg> \
//     --model packages/engine-webgpu/test/fixtures/g170-b6c96-s175395328-d26788732.bin.gz [--out <file>]
import { spawn, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({ options: { katago: { type: 'string' }, config: { type: 'string' }, model: { type: 'string' }, out: { type: 'string' } } });
if (!values.katago || !values.config || !values.model) {
  console.error('usage: capture-protocol-goldens.mjs --katago <katago> --config <analysis cfg> --model <model .bin.gz> [--out <file>]');
  process.exit(2);
}

/** As the search goldens, less the reporting side (Black, from the config). */
const OVERRIDES = {
  numAnalysisThreads: '1',
  numSearchThreadsPerAnalysisThread: '1',
  nnRandomize: 'false',
  wideRootNoise: '0',
  rootNumSymmetriesToSample: '1',
  useFP16: 'false',
};

// The config both engines run with: native's config file, less its log directory, with the overrides.
const config = {};
for (const raw of readFileSync(values.config, 'utf8').split('\n')) {
  const line = raw.replace(/#.*$/, '').trim();
  if (!line) continue;
  const eq = line.indexOf('=');
  config[line.slice(0, eq).trim()] = line.slice(eq + 1).trim();
}
delete config.logDir;
Object.assign(config, OVERRIDES);

const q = (fields) => JSON.stringify(fields);
const base = { rules: 'tromp-taylor', komi: 7.5, boardXSize: 9, boardYSize: 9, moves: [['B', 'E5']], maxVisits: 2 };
const p = (id, fields = {}, drop = []) => {
  const o = { id, ...base, ...fields };
  for (const k of drop) delete o[k];
  return q(o);
};
const ALL = { includeOwnership: true, includeOwnershipStdev: true, includeMovesOwnership: true, includeMovesOwnershipStdev: true, includePolicy: true, includePVVisits: true, includeNoResultValue: true };
const MOVES_4 = [['B', 'E5'], ['W', 'C3'], ['B', 'G7'], ['W', 'G3']];
// The 9x9 game of the search goldens (capture-search-goldens.mjs), which ends in two passes.
const LATE_9 = JSON.parse(readFileSync(join(here, '..', 'test', 'goldens', 'search-late-9.json'), 'utf8')).moves;

/**
 * Each scenario runs in a fresh engine. `partial` lists ids whose final results come from a search stopped early, so
 * only their form is compared; `duringCount` says how many reports during the search each id must have ('some' for one
 * or more).
 */
const SCENARIOS = [
  {
    name: 'parse',
    note: 'Requests KataGo refuses, field by field, with its messages; and accepted ones, searched at 2 visits.',
    steps: [{
      lines: [
        'nonsense',
        '[1,2.0]',
        '{"id":5}',
        '{"action":"query_version"}',
        '{"id":"p-version","action":"query_version","extra":[1,2.50,1e-7,-0.0,1E21,123456789012,0.0001,1e15,1e16,true,null,"\\u0001"]}',
        q({ id: 'p-models', action: 'query_models' }),
        q({ id: 'p-clear', action: 'clear_cache' }),
        q({ id: 'p-bogus', action: 'bogus' }),
        q({ id: 'p-action-number', action: 5 }),
        q({ id: 'p-term-noid', action: 'terminate' }),
        q({ id: 'p-term-badturns', action: 'terminate', terminateId: 'x', turnNumbers: 'x' }),
        q({ id: 'p-termall', action: 'terminate_all', turnNumbers: [true, 2.7] }),
        q({ id: 'p-term-unknown', action: 'terminate', terminateId: 'nobody' }),
        p('p-nox', {}, ['boardXSize']),
        p('p-x20', { boardXSize: 20 }),
        '{"id":"p-y-float","rules":"tromp-taylor","komi":7.5,"boardXSize":9,"boardYSize":9.0,"moves":[["B","E5"]],"maxVisits":2}',
        p('p-stones-short', { initialStones: [['B']] }),
        p('p-stones-player', { initialStones: [['X', 'C3']] }),
        p('p-stones-loc', { initialStones: [['B', 'Z9']] }),
        p('p-stones-pass', { initialStones: [['B', 'pass']] }),
        p('p-nomoves', {}, ['moves']),
        p('p-moves-string', { moves: 'x' }),
        p('p-moves-forms', { moves: [['black', '(4,4)'], ['w', ' c3 '], ['B', 'PASS']] }),
        p('p-initial-player', { initialPlayer: 'q' }),
        p('p-turns-string', { analyzeTurns: 'x' }),
        p('p-turns-range', { analyzeTurns: [5] }),
        p('p-priorities-alone', { priorities: [1] }),
        p('p-priorities-length', { analyzeTurns: [0, 1], priorities: [1] }),
        p('p-priorities-range', { analyzeTurns: [0], priorities: [2 ** 62] }),
        p('p-priority-both', { analyzeTurns: [0], priorities: [1], priority: 1 }),
        '{"id":"p-priority-float","rules":"tromp-taylor","komi":7.5,"boardXSize":9,"boardYSize":9,"moves":[["B","E5"]],"maxVisits":2,"priority":1.5}',
        p('p-norules', {}, ['rules']),
        p('p-rules-number', { rules: 5 }),
        p('p-rules-bogus', { rules: 'bogus' }),
        p('p-rules-ko', { rules: { ko: 'BOGUS' } }),
        p('p-rules-komi', { rules: { komi: 7.25, ko: 'SIMPLE' } }),
        p('p-rules-compact', { rules: 'koSIMPLEscoreAREAtaxNONEsui1komi6.5' }),
        p('p-rules-json-string', { rules: '{"ko":"SIMPLE","friendlyPassOk":true}' }),
        p('p-komi-quarter', { komi: 7.25 }),
        p('p-komi-range', { komi: 500 }),
        p('p-komi-string', { komi: '7' }),
        p('p-whb-number', { whiteHandicapBonus: 5 }),
        p('p-whb-bogus', { whiteHandicapBonus: 'X' }),
        p('p-override-string', { overrideSettings: 'x' }),
        p('p-override-value', { overrideSettings: { cpuctExploration: 'abc' } }),
        p('p-override-unknown', { overrideSettings: { foo: 1, bar: 'x', cpuctExploration: 1.1 } }),
        p('p-visits-zero', { maxVisits: 0 }),
        p('p-pvlen', { analysisPVLen: 1001 }),
        p('p-fpu', { rootFpuReductionMax: 3 }),
        p('p-temperature', { rootPolicyTemperature: 0 }),
        p('p-include', { includePolicy: 'yes' }),
        p('p-every', { reportDuringSearchEvery: 0 }),
        p('p-first', { firstReportDuringSearchAfter: -1 }),
        p('p-allow-avoid', { allowMoves: [], avoidMoves: [] }),
        p('p-avoid-string', { avoidMoves: 'x' }),
        p('p-allow-three', { allowMoves: [{}, {}, {}] }),
        p('p-avoid-fields', { avoidMoves: [{ player: 'B' }] }),
        p('p-avoid-player', { avoidMoves: [{ player: 'X', moves: [], untilDepth: 1 }] }),
        p('p-avoid-loc', { avoidMoves: [{ player: 'B', moves: ['Z1'], untilDepth: 1 }] }),
        p('p-avoid-depth', { avoidMoves: [{ player: 'B', moves: ['C3'], untilDepth: 0 }] }),
        p('p-allow-twice', { allowMoves: [{ player: 'B', moves: ['C3'], untilDepth: 1 }, { player: 'b', moves: ['D4'], untilDepth: 1 }] }),
        // Out of order both ways, so that warnings in the fields' order, or the reverse, differ from KataGo's sorted order.
        p('p-unused-fields', { mmm: 1, zzz: 2, aaa: 3 }),
        p('p-spight', { rules: { ko: 'SPIGHT' } }),
        p('p-illegal', { moves: [['B', 'E5'], ['W', 'E5']], zz: 1 }),
      ],
    }],
  },
  {
    name: 'results',
    note: 'Results compared with the search check\'s tolerances: several turns, each side the values are for, every optional output, a rectangular board, and komi and draw value changes that the shared evaluation cache must keep apart.',
    steps: [{
      lines: [
        // Values for Black (the config's side) with Black to move and, at turn 1, White to move.
        q({ id: 'r-turns', ...base, moves: MOVES_4, analyzeTurns: [0, 1, 4], maxVisits: 40, ...ALL }),
        // Values for White with Black to move (turn 0) and White to move (turn 3).
        q({ id: 'r-white', ...base, moves: MOVES_4, analyzeTurns: [0, 3], maxVisits: 40, includeMovesOwnershipStdev: true, includeOwnershipStdev: true, overrideSettings: { reportAnalysisWinratesAs: 'WHITE' } }),
        q({ id: 'r-side-to-move', ...base, moves: MOVES_4, analyzeTurns: [3], maxVisits: 40, analysisPVLen: 3, includeOwnership: true, overrideSettings: { reportAnalysisWinratesAs: 'SIDETOMOVE' } }),
        q({ id: 'r-rectangle', ...base, boardXSize: 7, boardYSize: 9, moves: [['B', 'D5'], ['W', 'C3']], maxVisits: 40, includeOwnership: true, includePolicy: true }),
        q({ id: 'r-komi', ...base, komi: 0.5, moves: MOVES_4.slice(0, 2), maxVisits: 40 }),
        q({ id: 'r-draw', ...base, komi: 7, moves: MOVES_4.slice(0, 2), maxVisits: 40, overrideSettings: { drawEquivalentWinsForWhite: 0.9 } }),
      ],
    }],
  },
  {
    name: 'priority',
    note: 'With one analysis thread busy, queued queries start highest priority first. The 19x19 search is stopped once the others are queued.',
    partial: ['a-busy'],
    steps: [
      { lines: [q({ id: 'a-busy', ...base, boardXSize: 19, boardYSize: 19, moves: [['B', 'Q16']], maxVisits: 1000000 })], waitMs: 500 },
      {
        lines: [
          q({ id: 'b-zero', ...base, moves: MOVES_4.slice(0, 1), maxVisits: 5 }),
          q({ id: 'c-five', ...base, moves: MOVES_4.slice(0, 2), maxVisits: 5, priority: 5 }),
          q({ id: 'd-turns', ...base, moves: MOVES_4.slice(0, 2), analyzeTurns: [0, 1, 2], priorities: [1, 3, 2], maxVisits: 5 }),
          // The same priority as b-zero, sent later: it runs after b-zero.
          q({ id: 'e-zero', ...base, moves: MOVES_4.slice(0, 3), maxVisits: 5 }),
          q({ id: 'stop', action: 'terminate', terminateId: 'a-busy' }),
        ],
      },
    ],
  },
  {
    name: 'terminate',
    note: 'A queued turn terminated by turn number, then everything; a query after that runs normally.',
    partial: ['t-long'],
    steps: [
      { lines: [q({ id: 't-long', ...base, analyzeTurns: [0, 1], maxVisits: 1000000 }), q({ id: 'u-queued', ...base, moves: MOVES_4, maxVisits: 5 })], waitMs: 300 },
      { lines: [q({ id: 'stop-turn', action: 'terminate', terminateId: 't-long', turnNumbers: [1] })], waitMs: 300 },
      { lines: [q({ id: 'stop-all', action: 'terminate_all' })], waitMs: 300 },
      { lines: [q({ id: 'v-after', ...base, boardXSize: 7, boardYSize: 7, moves: [['B', 'D4']], maxVisits: 20 })] },
    ],
  },
  {
    name: 'during',
    note: 'Reports during the search: every 0.25 seconds, and once only after 0.02 seconds. Only their form is compared; the final results are compared in full.',
    duringCount: { 'r-every': 'some', 's-once': 1 },
    steps: [{
      lines: [
        q({ id: 'r-every', ...base, moves: MOVES_4, maxVisits: 100, reportDuringSearchEvery: 0.25 }),
        q({ id: 's-once', ...base, boardXSize: 7, boardYSize: 7, moves: [['B', 'D4']], maxVisits: 100, firstReportDuringSearchAfter: 0.02 }),
      ],
    }],
  },
  {
    name: 'territory',
    note: 'Territory scoring and the group taxes (part C, step 13c), late in a 9x9 game: the query\'s moves are played with the encore prevented (preventCleanupPhase), and lines with a pass are cut where the root\'s phase would end. Several turns of one query, before and after a pass and past the prevented phase end, share the evaluation cache.',
    steps: [{
      lines: [
        q({ id: 'j-turns', ...base, rules: 'japanese', moves: LATE_9, analyzeTurns: [48, 51, 52], maxVisits: 40, includePVVisits: true }),
        q({ id: 'j-ancient', ...base, rules: 'ancient-territory', moves: LATE_9.slice(0, 51), maxVisits: 40, includePVVisits: true }),
        q({ id: 'j-stone', ...base, rules: 'stone-scoring', moves: LATE_9.slice(0, 51), maxVisits: 40, includePVVisits: true }),
        q({ id: 'j-korean-19', ...base, rules: 'korean', boardXSize: 19, boardYSize: 19, moves: [['B', 'Q16'], ['W', 'D4']], maxVisits: 20 }),
      ],
    }],
  },
];

const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
const overrideArg = [`logDir=${logDir}`, ...Object.entries(OVERRIDES).map(([k, v]) => `${k}=${v}`)].join(',');

/** Runs one scenario in a fresh native engine: the steps' lines, with their pauses, then closes its input. */
function runScenario(s) {
  return new Promise((resolve, reject) => {
    const child = spawn(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', overrideArg], { stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (err += d));
    child.on('error', reject);
    child.on('close', (code) => (code === 0 ? resolve(out.split('\n').filter((l) => l.length > 0)) : reject(new Error(`${s.name}: exit ${code}\n${err}`))));
    // Wait for native to load the network before the first step, so pauses are measured from a running engine.
    const started = () => err.includes('Started, ready to begin handling requests');
    const send = async () => {
      while (!started()) await new Promise((r) => setTimeout(r, 50));
      for (const step of s.steps) {
        for (const line of step.lines) child.stdin.write(line + '\n');
        if (step.waitMs) await new Promise((r) => setTimeout(r, step.waitMs));
      }
      child.stdin.end();
    };
    send().catch(reject);
  });
}

// Native's messages name the config and model files by the paths given; the goldens keep only the file names.
const shortenPaths = (line) => line.split(values.config).join(basename(values.config)).split(values.model).join(basename(values.model));
const scenarios = [];
for (const s of SCENARIOS) {
  const output = (await runScenario(s)).map(shortenPaths);
  scenarios.push({ ...s, output });
  console.error(`captured ${s.name}: ${output.length} lines`);
}
const version = spawnSync(values.katago, ['version'], { encoding: 'utf8' }).stdout.split('\n').slice(0, 2).join(', ');
const golden = {
  capturedWith: {
    katagoVersion: version,
    configFile: basename(values.config),
    overrides: OVERRIDES,
    modelFile: basename(values.model),
    modelSha256: createHash('sha256').update(readFileSync(values.model)).digest('hex'),
    date: new Date().toISOString(),
  },
  note: 'Native KataGo analysis engine sessions: the lines sent, in steps with pauses, and every line native wrote (with the config and model file paths shortened to their names). config is the config both engines run with.',
  config,
  scenarios,
};
const outFile = values.out ?? join(here, '..', 'test', 'goldens', `protocol-${basename(values.model).replace(/\.bin\.gz$/, '')}.json`);
// One line of output per line: results with ownership maps are long.
const { scenarios: ss, ...head } = golden;
const body = ss.map(({ output, ...rest }) => `  ${JSON.stringify(rest).replace(/}$/, ',')}"output": [\n${output.map((l) => `   ${JSON.stringify(l)}`).join(',\n')}\n  ]}`);
writeFileSync(outFile, JSON.stringify(head, null, 1).replace(/\n}$/, ',\n "scenarios": [\n') + body.join(',\n') + '\n ]\n}\n');
console.error(`wrote ${outFile}`);
