#!/usr/bin/env node
// Capture goldens for the rules engine and the network's inputs (test/history.test.mjs) from KataGo's own: random games
// played on KataGo's BoardHistory, with its state, and its network inputs (NNInputs::fillRowV7), recorded after every
// move.
//
// It builds scripts/native/history-dump.cpp against a KataGo source tree, generates game specs, runs them through it, and
// writes each game's moves with a digest of each state and of its inputs (test/history-state.mjs). The games are random games on small
// boards, and games from the boards of KataGo's own rules tests (cpp/tests/testrules.cpp), which are built around kos;
// in some, the players prefer ko captures and passes for ko, to make ko fights and long cycles. Rules, the starting
// encore phase and KataGo's history modes vary between games. Up to 10 more games come from a pool of 1500 random games:
// those in which the encore repeats a position three times, but not with the same player to move (repeatsWithOtherPlayer).
// Up to 5 pairs of games more come from a pool of 2000 on tiny boards: pairs that reach the same position in the second
// encore from different starting stones (tinySpecs). Each game's inputs are made with its own options (the draw value, conservativePassAndIsRoot, enablePassingHacks,
// maxHistory), drawn from a generator of their own so the games don't change with them.
//
// The source tree must be KataGo's v1.18.2 tag, the version ported. For example, from a clone of KataGo:
//   git -C <katago clone> archive v1.18.2 cpp | tar -x -C <dir>
//   node packages/engine-webgpu/scripts/capture-history-goldens.mjs --katago-src <dir> [--cxx g++] [--out <file>]
//     [--full <file>]
//
// --full also writes every game with its full states (JSON lines), to find what differs when the test fails. It needs a
// C++17 compiler; nothing is installed.
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { digestInputs, digestState } from '../test/history-state.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const { values } = parseArgs({
  options: {
    'katago-src': { type: 'string' },
    cxx: { type: 'string', default: 'g++' },
    out: { type: 'string', default: join(here, '../test/goldens/history-katago.json') },
    full: { type: 'string' },
  },
});
const src = values['katago-src'];
if (!src) throw new Error('--katago-src <KataGo source tree, v1.18.2> is required');
const cpp = join(src, 'cpp');

const version = /getKataGoVersion\(\) \{\s*return string\("([^"]+)"\)/.exec(readFileSync(join(cpp, 'main.cpp'), 'utf8'))?.[1];
if (version !== '1.18.2') throw new Error(`the source tree is KataGo ${version}; the port follows 1.18.2`);

// Build the harness.
const work = mkdtempSync(join(tmpdir(), 'history-dump-'));
const exe = join(work, 'history-dump');
const sources = [
  join(here, 'native/history-dump.cpp'),
  ...['neuralnet/nninputs.cpp', 'game/graphhash.cpp', 'game/board.cpp', 'game/boardhistory.cpp', 'game/rules.cpp', 'core/global.cpp', 'core/hash.cpp', 'core/rand.cpp',
    'core/md5.cpp', 'core/sha2.cpp', 'core/rand_helpers.cpp', 'core/test.cpp', 'core/bsearch.cpp', 'core/timer.cpp'].map((f) => join(cpp, f)),
];
const build = spawnSync(values.cxx, ['-std=c++17', '-O2', `-I${cpp}`, '-o', exe, ...sources], { encoding: 'utf8' });
if (build.status !== 0) throw new Error(`building history-dump failed:\n${build.stderr}`);

// The boards of KataGo's rules tests: x black, o white.
const boards = [];
for (const m of readFileSync(join(cpp, 'tests/testrules.cpp'), 'utf8').matchAll(/Board::parseBoard\((\d+),(\d+),R"%%\(([\s\S]*?)\)%%"\)/g)) {
  const [xSize, ySize] = [Number(m[1]), Number(m[2])];
  const rows = m[3].split('\n').map((r) => r.trim()).filter((r) => r !== '');
  if (rows.length !== ySize || rows.some((r) => r.length !== xSize)) continue;
  boards.push({ xSize, ySize, initial: rows.join('').replace(/[xX]/g, '1').replace(/[oO]/g, '2').replace(/\./g, '0') });
}

/**
 * [game, state, first game, first state] for each state whose value (by `valueOf`) equals an earlier state's on a board
 * of the same size, with the first state that has it.
 */
function repeats(games, valueOf) {
  const first = new Map();
  const out = [];
  games.forEach((g, gi) => {
    g.states.forEach((st, si) => {
      const v = `${g.xSize}x${g.ySize}|${valueOf(st)}`;
      const f = first.get(v);
      if (f) out.push([gi, si, ...f]);
      else first.set(v, [gi, si]);
    });
  });
  return out;
}

/** A seeded generator (mulberry32), so a capture is reproducible. */
function generator(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Game specs. */
function specs(count, seed, fromBoards) {
  const rnd = generator(seed);
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const inputRnd = generator(seed + 1000);
  const inputPick = (a) => a[Math.floor(inputRnd() * a.length)];
  const out = [];
  for (let i = 0; i < count; i++) {
    let xSize = pick([2, 3, 3, 4, 4, 5, 5, 6, 7]);
    let ySize = rnd() < 0.7 ? xSize : pick([2, 3, 4, 5, 6]);
    const scoringRule = rnd() < 0.8 ? 1 : 0;
    const fill = pick([0, 0, 0.2, 0.4, 0.6]);
    let initial = '';
    for (let k = 0; k < xSize * ySize; k++) initial += rnd() < fill ? pick(['1', '2']) : '0';
    if (fromBoards) ({ xSize, ySize, initial } = pick(boards));
    const encorePhase = scoringRule === 1 && rnd() < 0.2 ? pick([1, 2]) : 0;
    out.push({
      xSize, ySize, koRule: pick([0, 1, 2]), scoringRule, taxRule: pick([0, 1, 2]), multiStoneSuicideLegal: rnd() < 0.5,
      whiteHandicapBonusRule: pick([0, 1, 2]), komi: pick([0, 0.5, 1, 6.5, 7, -3.5]), initial, nextPla: pick([1, 2]),
      encorePhase, alwaysPassAlive: rnd() < 0.3, excludeAdjAtari: rnd() < 0.5, assumeHandicap: rnd() < 0.7,
      seed: `g${i}-${seed}`, passProb: pick([0.05, 0.15, 0.3, 0.5]), preventEncoreProb: pick([0, 0, 0.2, 0.5]),
      tolerantProb: pick([0, 0.1, 0.3]), numMoves: pick([30, 60, 120]), movesAfterEnd: pick([0, 2, 6]),
      koBias: pick([0, 0.5, 0.8, 0.95]),
      inputParams: {
        drawEquivalentWinsForWhite: inputPick([0.5, 0.5, 0.25, 1]), conservativePassAndIsRoot: inputRnd() < 0.2,
        enablePassingHacks: inputRnd() < 0.3, maxHistory: inputPick([1000, 1000, 1000, 3, 1, 0]),
      },
    });
  }
  return out;
}

/** Runs specs through history-dump: the games, and their JSON lines. */
function play(list) {
  const input = list.map((g) => JSON.stringify(g)).join('\n') + '\n';
  const run = spawnSync(exe, [], { input, encoding: 'utf8', maxBuffer: 1 << 30 });
  if (run.status !== 0) throw new Error(`history-dump failed:\n${run.stderr}`);
  const lines = run.stdout.trim().split('\n');
  return { games: lines.map((l) => JSON.parse(l)), lines };
}

/**
 * True if, after a move in the encore other than a pass, its stones and ko recapture blocks have occurred three or more
 * times since the last pass, counting both players to move, and KataGo plays on. The encore's ko key includes the player
 * to move, so these aren't three repeats for its no-result rule. Random games rarely reach it, so the capture keeps the
 * games of a larger pool that do.
 */
function repeatsWithOtherPlayer(g) {
  let since = [];
  for (let i = 1; i < g.states.length; i++) {
    const s = g.states[i];
    const key = `${s.colors}|${s.koRecapBlocked}`;
    // KataGo clears its ko history on a pass in the encore (and at a phase change, also a pass), then adds the position.
    if (g.moves[i - 1].loc === 'pass') {
      since = [key];
      continue;
    }
    since.push(key);
    if (s.encorePhase > 0 && !s.isNoResult && since.filter((k) => k === key).length >= 3) return true;
  }
  return false;
}

const main = play([...specs(100, 21, false), ...specs(300, 22, true)]);
const pool = play(specs(1500, 23, false));
const picked = pool.games.map((g, i) => [g, i]).filter(([g]) => repeatsWithOtherPlayer(g)).slice(0, 10);
console.log(`${picked.length} games picked from the pool for three repeats with both players to move`);

/**
 * Games on tiny boards with the same rules and komi and many passes, from which to pick pairs that reach the same
 * position in the second encore, the same in everything KataGo's situation hash holds (the stones, the player to move,
 * the bans, the recapture blocks and the komi with its bonuses) except the stones the second encore started with. Random
 * games rarely give such pairs, and only they show that the search's keys hold those stones.
 */
function tinySpecs(count, seed) {
  const rnd = generator(seed);
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  return Array.from({ length: count }, (_, i) => {
    const [xSize, ySize] = pick([[2, 2], [2, 3], [3, 3]]);
    return {
      xSize, ySize, koRule: 0, scoringRule: 1, taxRule: 0, multiStoneSuicideLegal: false, whiteHandicapBonusRule: 0, komi: 0,
      initial: '0'.repeat(xSize * ySize), nextPla: 1, encorePhase: pick([0, 1, 2]), alwaysPassAlive: false, excludeAdjAtari: false,
      assumeHandicap: true, seed: `t${i}-${seed}`, passProb: pick([0.3, 0.5]), preventEncoreProb: 0, tolerantProb: 0, numMoves: 40,
      movesAfterEnd: 0, koBias: 0,
      inputParams: { drawEquivalentWinsForWhite: 0.5, conservativePassAndIsRoot: false, enablePassingHacks: false, maxHistory: 1000 },
    };
  });
}
const tinyPool = play(tinySpecs(2000, 24));
const startColorsBySituation = new Map();
const pairs = [];
const inPairs = new Set();
tinyPool.games.forEach((g, gi) => {
  for (const st of g.states) {
    if (st.encorePhase !== 2 || pairs.length >= 5 || inPairs.has(gi)) continue;
    const key = JSON.stringify([g.xSize, g.ySize, st.colors, st.nextPla, st.superKoBanned, st.koRecapBlocked, st.whiteBonusScore, st.whiteHandicapBonusScore]);
    const seen = startColorsBySituation.get(key) ?? new Map();
    startColorsBySituation.set(key, seen);
    const other = [...seen].find(([colors, i]) => colors !== st.secondEncoreStartColors && !inPairs.has(i));
    if (other) {
      pairs.push([other[1], gi]);
      inPairs.add(other[1]).add(gi);
    } else if (!seen.has(st.secondEncoreStartColors)) seen.set(st.secondEncoreStartColors, gi);
  }
});
const tinyPicked = pairs.flat();
console.log(`${pairs.length} pairs of games picked from the tiny-board pool for the second encore's starting stones`);

const games = [...main.games, ...picked.map(([g]) => g), ...tinyPicked.map((i) => tinyPool.games[i])];
if (values.full) {
  writeFileSync(values.full, [...main.lines, ...picked.map(([, i]) => pool.lines[i]), ...tinyPicked.map((i) => tinyPool.lines[i])].join('\n') + '\n');
}

const RULE_FIELDS = ['xSize', 'ySize', 'koRule', 'scoringRule', 'taxRule', 'multiStoneSuicideLegal', 'whiteHandicapBonusRule',
  'komi', 'initial', 'nextPla', 'encorePhase', 'alwaysPassAlive', 'excludeAdjAtari', 'assumeHandicap', 'inputParams'];
const out = {
  source: `KataGo ${version}'s BoardHistory, through scripts/native/history-dump.cpp (scripts/capture-history-goldens.mjs)`,
  format: {
    rules: 'koRule 0 simple, 1 positional, 2 situational; scoringRule 0 area, 1 territory; taxRule 0 none, 1 seki, 2 all; whiteHandicapBonusRule 0 zero, 1 N, 2 N-1',
    initial: 'one character per point, row by row from the top: 0 empty, 1 black, 2 white (stones that would have no liberties are left off, as KataGo setStoneFailIfNoLibs)',
    moves: '[x, y, player (1 black, 2 white), flags]: x and y -1 for a pass; flags bit 0 preventEncore, bit 1 played with makeBoardMoveTolerant',
    digests: 'one 16-character digest per state (test/history-state.mjs), the start and after each move, concatenated',
    inputDigests: "the same for the network's inputs for the player to move (NNInputs::fillRowV7), made with the game's inputParams",
    situationRepeats: "[game, state, first game, first state] for each state whose KataGo situation hash (getSituationRulesAndKoHash, with the game's draw value) equals that of an earlier state on a board of the same size, in game order, with the first such state",
    graphStateRepeats: 'the same for the graph-search state hash (GraphHash::getStateHash)',
  },
  games: games.map((g) => ({
    ...Object.fromEntries(RULE_FIELDS.map((k) => [k, g[k]])),
    moves: g.moves.map((m) => [...(m.loc === 'pass' ? [-1, -1] : m.loc), m.pla, (m.preventEncore ? 1 : 0) | (m.tolerant ? 2 : 0)]),
    digests: g.states.map(digestState).join(''),
    inputDigests: g.states.map((st) => digestInputs(st.inputs)).join(''),
  })),
  situationRepeats: repeats(games, (st) => st.situationHash),
  graphStateRepeats: repeats(games, (st) => st.graphStateHash),
};
writeFileSync(values.out, JSON.stringify(out) + '\n');
const states = games.reduce((n, g) => n + g.states.length, 0);
console.log(`${games.length} games, ${states} states -> ${values.out}`);
