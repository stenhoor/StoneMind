#!/usr/bin/env node
// Capture native KataGo's position hashes (rootInfo's thisHash and symHash) on positions the search goldens don't
// cover: a simple ko just taken (under simple ko and under positional superko rules), and rectangular boards, where
// symmetries that swap the sides change the board's size hash. Used by test/hash.test.mjs.
//
//   node packages/engine-webgpu/scripts/capture-hash-goldens.mjs --katago <katago> --config <analysis_example.cfg> \
//     --model packages/engine-webgpu/test/fixtures/g170-b6c96-s175395328-d26788732.bin.gz [--out <file>]
//
// The hashes depend only on the position, so each query searches 1 visit with the config as it is.
import { spawnSync } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({ options: { katago: { type: 'string' }, config: { type: 'string' }, model: { type: 'string' }, out: { type: 'string' } } });
if (!values.katago || !values.config || !values.model) {
  console.error('usage: capture-hash-goldens.mjs --katago <katago> --config <analysis cfg> --model <model .bin.gz> [--out <file>]');
  process.exit(2);
}
// Black takes a ko at F5, capturing E5: White may not retake at E5 at once.
const KO = [['B', 'E6'], ['W', 'F6'], ['B', 'D5'], ['W', 'G5'], ['B', 'E4'], ['W', 'F4'], ['B', 'A1'], ['W', 'E5'], ['B', 'F5']];
const POSITIONS = [
  { id: 'ko-9-chinese', moves: KO, rules: 'chinese', xSize: 9, ySize: 9 },
  { id: 'ko-9-tromp-taylor', moves: KO, rules: 'tromp-taylor', xSize: 9, ySize: 9 },
  { id: 'ko-19-chinese', moves: KO, rules: 'chinese', xSize: 19, ySize: 19 },
  { id: 'ko-9x7-chinese', moves: KO, rules: 'chinese', xSize: 9, ySize: 7 },
  { id: 'rect-7x9', moves: [['B', 'C3'], ['W', 'E7']], rules: 'tromp-taylor', xSize: 7, ySize: 9 },
];
const input = POSITIONS.map((p) => JSON.stringify({ id: p.id, moves: p.moves, rules: p.rules, komi: 7.5, boardXSize: p.xSize, boardYSize: p.ySize, maxVisits: 1 })).join('\n') + '\n';
const logDir = mkdtempSync(join(tmpdir(), 'katago-logs-'));
const run = spawnSync(values.katago, ['analysis', '-config', values.config, '-model', values.model, '-override-config', `logDir=${logDir}`], { input, encoding: 'utf8' });
if (run.status !== 0) {
  console.error(run.stderr);
  process.exit(1);
}
const byId = new Map(run.stdout.trim().split('\n').map((l) => JSON.parse(l)).map((r) => [r.id, r]));
const out = {
  capturedWith: { katagoVersion: spawnSync(values.katago, ['version'], { encoding: 'utf8' }).stdout.split('\n')[0], date: new Date().toISOString() },
  positions: POSITIONS.map((p) => {
    const r = byId.get(p.id);
    if (!r || r.error) throw new Error(`${p.id}: ${r?.error ?? 'no response'}`);
    return { ...p, komi: 7.5, thisHash: r.rootInfo.thisHash, symHash: r.rootInfo.symHash, currentPlayer: r.rootInfo.currentPlayer };
  }),
};
const outFile = values.out ?? join(here, '..', 'test', 'goldens', 'hashes.json');
writeFileSync(outFile, JSON.stringify(out, null, 1) + '\n');
console.error(`wrote ${outFile}`);
