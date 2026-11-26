#!/usr/bin/env node
// The search's own JavaScript time, without a GPU: the review's searches (test/goldens/agreement-game.json, one turn at a
// time) with a stand-in network whose outputs are a fixed function of its inputs, so every run, and every build that
// gives the same results, grows the same trees. Reports the searches' time outside the network (Search.stats.cpuMs),
// per network call, as the median of several runs.
//
// Usage: node scripts/search-cpu-bench.mjs --model <b18 .bin.gz> [--dist <built dist folder>] [--turns 100]
//          [--visits 100] [--batch 16] [--runs 5] [--profile <out.cpuprofile>]
// --profile writes a CPU profile of the timed searches only (not loading the model or building the positions).
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values: args } = parseArgs({
  options: {
    model: { type: 'string' },
    dist: { type: 'string', default: fileURLToPath(new URL('../dist', import.meta.url)) },
    turns: { type: 'string', default: '100' },
    visits: { type: 'string', default: '100' },
    batch: { type: 'string', default: '16' },
    runs: { type: 'string', default: '5' },
    profile: { type: 'string' },
  },
});
if (!args.model) throw new Error('--model is required');
const lib = await import(pathToFileURL(resolve(args.dist, 'index.js')).href);
const { loadModel, buildHistory, gtpMove, Search } = lib;
const model = await loadModel(new Uint8Array(readFileSync(args.model)));
const game = JSON.parse(readFileSync(fileURLToPath(new URL('../test/goldens/agreement-game.json', import.meta.url)), 'utf8'));
const turns = Number(args.turns);
const histories = [];
for (let t = 0; t < turns; t++) {
  histories.push(buildHistory({ xSize: 19, ySize: 19, rules: game.rules, komi: game.komi, moves: game.moves.slice(0, t).map(([c, l]) => gtpMove(c, l, 19)) }));
}

// Outputs from a hash of the input planes: a fixed function of the inputs, cheap next to the search.
const standIn = {
  maxBatch: 64,
  evaluate: async (inputs) =>
    inputs.map((input) => {
      const area = input.xSize * input.ySize;
      let h = 2166136261;
      for (let i = 0; i < 3 * area; i++) if (input.spatial[area + i] === 1) h = Math.imul(h ^ i, 16777619);
      const rand = () => {
        h = Math.imul(h ^ (h >>> 15), 2246822507) ^ Math.imul(h ^ (h >>> 13), 3266489909);
        return ((h >>> 0) / 4294967296);
      };
      const plane = () => Float32Array.from({ length: area }, () => 4 * rand() - 2);
      return {
        xSize: input.xSize,
        ySize: input.ySize,
        policy: Array.from({ length: model.policyHead.policyOutChannels }, plane),
        pass: Float32Array.from({ length: model.policyHead.policyOutChannels }, () => -3),
        value: Float32Array.of(rand() - 0.5, rand() - 0.5, -6),
        scoreValue: Float32Array.from({ length: model.numScoreValueChannels }, () => rand() - 0.5),
        ownership: plane(),
      };
    }),
};

let session;
if (args.profile) {
  const { Session } = await import('node:inspector/promises');
  session = new Session();
  session.connect();
  await session.post('Profiler.enable');
  await session.post('Profiler.setSamplingInterval', { interval: 200 });
  await session.post('Profiler.start');
}
const runs = [];
let evals = 0;
let calls = 0;
for (let r = 0; r < Number(args.runs); r++) {
  let cpuMs = 0;
  evals = 0;
  calls = 0;
  const cache = new Map();
  for (const hist of histories) {
    const search = new Search(model, standIn, hist, { batchSize: Number(args.batch) }, cache);
    await search.run({ maxVisits: Number(args.visits) });
    cpuMs += search.stats.cpuMs;
    evals += search.stats.nnEvals;
    calls += search.stats.batches;
  }
  runs.push(cpuMs);
}
if (session) {
  const { profile } = await session.post('Profiler.stop');
  (await import('node:fs')).writeFileSync(args.profile, JSON.stringify(profile));
}
runs.sort((a, b) => a - b);
const median = runs[runs.length >> 1];
console.log(JSON.stringify({
  dist: args.dist, turns, visits: Number(args.visits), batch: Number(args.batch), evals, calls,
  cpuMsRuns: runs.map((x) => +x.toFixed(1)), cpuMsMedian: +median.toFixed(1), msPerCall: +(median / calls).toFixed(3), msPerEval: +(median / evals).toFixed(4),
}));
