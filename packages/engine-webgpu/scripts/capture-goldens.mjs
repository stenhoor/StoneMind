#!/usr/bin/env node
// Capture golden outputs from a native KataGo for the positions in test/goldens/positions.json.
//
// For each position it sets the board size, rules and komi, plays the moves, and runs `kata-raw-nn <s> 0` for each
// symmetry s = 0..7: the raw network evaluation at policy optimism 0, straight from KataGo (no search, no cache).
// One symmetry per command, because KataGo's `kata-raw-nn all` puts blank lines between symmetries inside one reply,
// and GTP uses a blank line to end a reply.
// FP16 is turned off with -override-config so the outputs are an fp32 reference.
//
// Use a KataGo built with the CPU (Eigen) backend: the reference forward pass follows it. KataGo's own CUDA and Eigen
// backends differ on b18c384nbt by up to ~2e-3 in ownership and ~9e-4 in policy (both fp32; measured 2026-11-25, see
// docs/spikes/results/2026-11-b18-goldens-backends.md), more than the goldens test's tolerances.
//
// Usage (Node 22.18+, no dependencies):
//   node packages/engine-webgpu/scripts/capture-goldens.mjs \
//     --katago ~/katago/katago --config ~/katago/gtp.cfg \
//     --model ~/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
//     --backend EIGEN [--override "..."] [--out <file>]
//
// Writes test/goldens/<model name>.json (or --out). Commands used, all from KataGo's GTP (cpp/command/gtp.cpp, 1.18):
// boardsize, clear_board, kata-set-rules, komi, play, kata-raw-nn, version, quit.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({
  options: {
    katago: { type: 'string' },
    config: { type: 'string' },
    model: { type: 'string' },
    override: { type: 'string', default: '' },
    backend: { type: 'string' },
    out: { type: 'string' },
  },
});
if (!values.katago || !values.config || !values.model || !values.backend) {
  console.error('needs --katago, --config, --model and --backend, the backend KataGo was built with (see the comment at the top of this file)');
  process.exit(2);
}

const positions = JSON.parse(readFileSync(join(here, '..', 'test', 'goldens', 'positions.json'), 'utf8')).positions;
const modelBytes = readFileSync(values.model);
const modelSha256 = createHash('sha256').update(modelBytes).digest('hex');
const modelName = basename(values.model).replace(/\.(bin|txt)\.gz$/, '');
const override = ['useFP16=false', values.override].filter(Boolean).join(', ');

const katago = spawn(values.katago, ['gtp', '-config', values.config, '-model', values.model, '-override-config', override], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

// GTP replies start with "=" (success) or "?" (error) and end with a blank line.
let buffer = '';
const waiting = [];
katago.stdout.setEncoding('utf8');
katago.stdout.on('data', (chunk) => {
  buffer += chunk;
  let end;
  while ((end = buffer.indexOf('\n\n')) >= 0) {
    const reply = buffer.slice(0, end).replace(/\r/g, '');
    buffer = buffer.slice(end + 2);
    if (!reply.trim()) continue;
    const next = waiting.shift();
    if (!next || !/^[=?]/.test(reply)) {
      console.error(`unexpected GTP output (the reply framing is out of step): ${reply.slice(0, 200)}`);
      process.exit(1);
    }
    next(reply);
  }
});
katago.on('exit', (code) => {
  if (waiting.length) {
    console.error(`katago exited (code ${code}) with commands unanswered`);
    process.exit(1);
  }
});

function gtp(command) {
  return new Promise((resolve, reject) => {
    waiting.push((reply) => {
      if (reply.startsWith('=')) resolve(reply.slice(1).trim());
      else reject(new Error(`"${command}" failed: ${reply}`));
    });
    katago.stdin.write(command + '\n');
  });
}

/** Parse kata-raw-nn's "key value…" text (whitespace-insensitive, any key order, as KataGo's docs require). */
function parseRawNN(text, size) {
  const tokens = text.split(/\s+/).filter(Boolean);
  const results = [];
  let current = null;
  const takeNumbers = (n, i) => tokens.slice(i, i + n).map((t) => (t === 'NAN' ? null : Number(t)));
  for (let i = 0; i < tokens.length; ) {
    const key = tokens[i++];
    if (key === 'symmetry') {
      current = { symmetry: Number(tokens[i++]) };
      results.push(current);
    } else if (key === 'policy' || key === 'whiteOwnership') {
      current[key] = takeNumbers(size * size, i);
      i += size * size;
    } else {
      current[key] = Number(tokens[i++]);
    }
  }
  return results;
}

const version = await gtp('version');
const out = {
  capturedWith: { katagoVersion: version, override, modelFile: basename(values.model), modelSha256, date: new Date().toISOString(), backend: values.backend },
  note: 'kata-raw-nn <symmetry> 0 for symmetries 0-7: raw evaluation under each symmetry, policy optimism 0, fp32. Values from White’s perspective; policy null marks an illegal move; rows run from the top (A19 on 19x19).',
  positions: [],
};
for (const p of positions) {
  await gtp(`boardsize ${p.boardSize}`);
  await gtp('clear_board');
  await gtp(`kata-set-rules ${p.rules}`);
  await gtp(`komi ${p.komi}`);
  for (const [color, loc] of p.moves) await gtp(`play ${color} ${loc}`);
  const evaluations = [];
  for (let s = 0; s < 8; s++) {
    const parsed = parseRawNN(await gtp(`kata-raw-nn ${s} 0`), p.boardSize);
    if (parsed.length !== 1 || parsed[0].symmetry !== s) throw new Error(`${p.id}: expected one evaluation for symmetry ${s}`);
    evaluations.push(parsed[0]);
  }
  out.positions.push({ id: p.id, evaluations });
  console.error(`captured ${p.id}`);
}
await gtp('quit').catch(() => {});

const outFile = values.out ?? join(here, '..', 'test', 'goldens', `${modelName}.json`);
writeFileSync(outFile, JSON.stringify(out) + '\n');
console.error(`wrote ${outFile}`);
