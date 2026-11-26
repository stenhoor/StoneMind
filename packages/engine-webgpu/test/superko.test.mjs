// Superko bans (GameHistory.superKoBanned) against a brute-force check: play each candidate move on a copy of the board
// and look its position up among the game's earlier positions.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildHistory, Board, EMPTY, PASS_LOC, BLACK, WHITE } from '../dist/index.js';

const opp = (p) => (p === BLACK ? WHITE : BLACK);
const key = (board, pla, situational) => {
  let s = '';
  for (let y = 0; y < board.ySize; y++) for (let x = 0; x < board.xSize; x++) s += board.colors[board.loc(x, y)];
  return situational ? `${s}|${pla}` : s;
};

function expectedBans(hist) {
  const b = hist.board;
  const situational = hist.rules.koRule === 'SITUATIONAL';
  const seen = new Set(hist.boardsBefore.map((bb, i) => key(bb, hist.moves[i].pla, situational)));
  seen.add(key(b, hist.nextPla, situational));
  const bans = new Uint8Array(hist.superKoBanned.length);
  if (hist.rules.koRule === 'SIMPLE') return bans; // simple ko: no superko bans
  for (let y = 0; y < b.ySize; y++) for (let x = 0; x < b.xSize; x++) {
    const loc = b.loc(x, y);
    if (b.colors[loc] !== EMPTY || loc === b.koLoc || b.isIllegalSuicide(loc, hist.nextPla, hist.rules.multiStoneSuicideLegal)) continue;
    const after = b.clone();
    after.playMoveAssumeLegal(loc, hist.nextPla);
    if (seen.has(key(after, opp(hist.nextPla), situational))) bans[loc] = 1;
  }
  return bans;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('superko bans match playing each move on a copy, in random games with captures, suicide and repeats', () => {
  let bansSeen = 0;
  for (const rules of ['tromp-taylor', 'aga', 'chinese', 'new-zealand']) {
    for (const size of [4, 5, 6]) {
      for (let seed = 1; seed <= 6; seed++) {
        const rand = mulberry32(seed * 1000 + size);
        const hist = buildHistory({ xSize: size, ySize: size, rules, komi: 7.5 });
        for (let move = 0; move < 120 && !hist.isGameFinished; move++) {
          assert.deepEqual(hist.superKoBanned, expectedBans(hist), `${rules} ${size}x${size} seed ${seed} move ${move}`);
          bansSeen += hist.superKoBanned.reduce((a, v) => a + v, 0);
          const b = hist.board;
          const legal = [];
          for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) if (hist.isLegal(b.loc(x, y), hist.nextPla)) legal.push(b.loc(x, y));
          const loc = legal.length === 0 || rand() < 0.03 ? PASS_LOC : legal[Math.floor(rand() * legal.length)];
          hist.play(loc, hist.nextPla);
        }
      }
    }
  }
  assert.ok(bansSeen > 0, 'some positions had superko bans');
});
