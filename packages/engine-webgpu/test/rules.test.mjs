// Unit tests for the rules engine: captures, simple ko, superko, suicide, ladders and pass-alive area.
// Hand-built positions whose answers follow from the rules of Go; the native-KataGo goldens (goldens.test.mjs) are
// the end-to-end check.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Board, BLACK, WHITE, EMPTY, calculateArea, buildHistory, IllegalMoveError } from '../dist/index.js';

/** A board from rows of text: X black, O white, anything else empty. Row 0 is the top. */
function boardFrom(rows) {
  const b = new Board(rows[0].length, rows.length);
  rows.forEach((row, y) =>
    [...row].forEach((ch, x) => {
      if (ch === 'X') assert.ok(b.setStone(b.loc(x, y), BLACK));
      if (ch === 'O') assert.ok(b.setStone(b.loc(x, y), WHITE));
    }),
  );
  return b;
}

/** Setup stones for buildHistory from rows of text. */
function stonesFrom(rows) {
  const out = [];
  rows.forEach((row, y) => [...row].forEach((ch, x) => ch === 'X' ? out.push(['B', x, y]) : ch === 'O' ? out.push(['W', x, y]) : 0));
  return out;
}

test('a move that takes the last liberty captures the chain', () => {
  const b = boardFrom(['.XX..', 'XOO..', '.XX..', '.....', '.....']);
  b.playMoveAssumeLegal(b.loc(3, 1), BLACK);
  assert.equal(b.colors[b.loc(1, 1)], EMPTY);
  assert.equal(b.colors[b.loc(2, 1)], EMPTY);
  assert.equal(b.colors[b.loc(3, 1)], BLACK);
});

test('single-stone suicide is illegal under every rule set; multi-stone suicide only where the rules allow it', () => {
  const single = boardFrom(['.O...', 'O....', '.....']);
  assert.equal(single.isIllegalSuicide(single.loc(0, 0), BLACK, false), true);
  assert.equal(single.isIllegalSuicide(single.loc(0, 0), BLACK, true), true);

  const multi = boardFrom(['X.O..', 'OO...', '.....']);
  assert.equal(multi.isIllegalSuicide(multi.loc(1, 0), BLACK, false), true);
  assert.equal(multi.isIllegalSuicide(multi.loc(1, 0), BLACK, true), false);
  multi.playMoveAssumeLegal(multi.loc(1, 0), BLACK);
  assert.equal(multi.colors[multi.loc(0, 0)], EMPTY, 'the suicided stones are removed');
  assert.equal(multi.colors[multi.loc(1, 0)], EMPTY);
});

test('simple ko: the immediate recapture is banned, and allowed again after an exchange elsewhere', () => {
  const rows = ['.XO..', 'XO.O.', '.XO..', '.....', '.....'];
  const hist = buildHistory({ xSize: 5, ySize: 5, rules: 'japanese', komi: 6.5, initialStones: stonesFrom(rows), moves: [['B', 2, 1]] });
  const b = hist.board;
  assert.equal(b.colors[b.loc(1, 1)], EMPTY, 'black captured the white stone');
  assert.equal(b.koLoc, b.loc(1, 1));
  assert.equal(hist.isLegal(b.loc(1, 1), WHITE), false);
  assert.throws(() => hist.play(b.loc(1, 1), WHITE), IllegalMoveError);

  hist.play(b.loc(4, 4), WHITE);
  hist.play(b.loc(4, 3), BLACK);
  assert.equal(hist.isLegal(hist.board.loc(1, 1), WHITE), true);
});

// Three kos side by side (each 4 columns wide, then a blank column). In each, the holder's stone sits in the middle:
// White at x+1 (Black may capture at x+2), or Black at x+2 (White may capture at x+1).
const KO_ROWS_WWB = ['.XO...XO...XO..', 'XO.O.XO.O.X.XO.', '.XO...XO...XO..'];
// The 6-move cycle: each move retakes a ko other than the one just taken, and the 6th move restores the start.
const TRIPLE_KO_CYCLE = [['B', 2, 1], ['W', 11, 1], ['B', 7, 1], ['W', 1, 1], ['B', 12, 1]];
const CLOSING_MOVE = [6, 1];

for (const [rules, banned] of [['tromp-taylor', true], ['aga', true], ['japanese', false]]) {
  test(`triple ko under ${rules}: the move that repeats the whole-board position is ${banned ? 'banned' : 'allowed'}`, () => {
    const hist = buildHistory({ xSize: 15, ySize: 3, rules, komi: 7, initialStones: stonesFrom(KO_ROWS_WWB), moves: TRIPLE_KO_CYCLE });
    const loc = hist.board.loc(...CLOSING_MOVE);
    assert.equal(hist.board.isLegal(loc, WHITE, false), true, 'legal under simple ko alone');
    assert.equal(hist.isLegal(loc, WHITE), !banned);
    assert.equal(hist.superKoBanned[loc] === 1, banned);
    if (!banned) {
      hist.play(loc, WHITE);
      const start = boardFrom(KO_ROWS_WWB);
      assert.deepEqual([...hist.board.colors], [...start.colors], 'the cycle returns to the starting position');
    }
  });
}

test('ladders: captured on an open board, escapes when a friendly stone sits on its path', () => {
  const open = boardFrom(['.........', '..XX.....', '.XO......', '..X......', '.........', '.........', '.........', '.........', '.........']);
  const w = open.loc(2, 2);
  assert.equal(open.getNumLiberties(w), 1);
  assert.equal(open.clone().searchIsLadderCaptured(w, true, []), true);

  const breaker = open.clone();
  assert.ok(breaker.setStone(breaker.loc(6, 6), WHITE));
  assert.equal(breaker.clone().searchIsLadderCaptured(w, true, []), false);

  const twoLibs = boardFrom(['.........', '..XX.....', '.XO......', '.........', '.........', '.........', '.........', '.........', '.........']);
  const working = [];
  assert.equal(twoLibs.clone().searchIsLadderCapturedAttackerFirst2Libs(twoLibs.loc(2, 2), [], working), true);
  assert.ok(working.length > 0, 'it reports the attacker moves that start a working ladder');
});

test('area: a group with two eyes is pass-alive and owns the territory it alone borders', () => {
  const twoEyes = boardFrom(['.X...', '.X...', 'XX...', '.X...', '.X...']);
  const area = calculateArea(twoEyes, false, true, false, false);
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) assert.equal(area[twoEyes.loc(x, y)], BLACK, `(${x},${y})`);
});

test('area: a group with one eye is not pass-alive; only the "unsafe" options count it', () => {
  const oneEye = boardFrom(['.X...', '.X...', '.X...', '.X...', '.X...']);
  const safe = calculateArea(oneEye, false, true, false, false);
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) assert.equal(safe[oneEye.loc(x, y)], EMPTY, `(${x},${y})`);
  const loose = calculateArea(oneEye, true, true, true, false);
  for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++) assert.equal(loose[oneEye.loc(x, y)], BLACK, `(${x},${y})`);
});
