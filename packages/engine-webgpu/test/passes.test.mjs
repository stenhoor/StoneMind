// Passes in the rules engine and the V7 inputs (KataGo BoardHistory / fillRowV7), for area scoring. Hand-built
// positions whose answers follow from KataGo's rules; native goldens for pass positions are a separate check.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildHistory, encodeHistoryV7, encodePositionV7, BLACK, WHITE, UnsupportedPositionError } from '../dist/index.js';

const base = { xSize: 9, ySize: 9, komi: 7 };

test('two passes end an area-scored game and score it with komi', () => {
  const hist = buildHistory({ ...base, rules: 'tromp-taylor', moves: [['B', 'pass'], ['W', 'pass']] });
  assert.equal(hist.isGameFinished, true);
  // An empty board is nobody's area, so White wins by komi.
  assert.equal(hist.finalWhiteMinusBlackScore, 7);
  assert.equal(hist.winner, WHITE);
});

test('a move between passes resets the count', () => {
  const hist = buildHistory({ ...base, rules: 'tromp-taylor', moves: [['B', 'pass'], ['W', 4, 4], ['B', 'pass']] });
  assert.equal(hist.isGameFinished, false);
  assert.equal(hist.consecutiveEndingPasses, 1);
  assert.equal(hist.passWouldEndGame(WHITE), true);
});

test('area scoring counts stones and the territory they enclose', () => {
  // A Black wall down column 1 encloses column 0; White has one stone elsewhere.
  const initialStones = [['W', 5, 5]];
  for (let y = 0; y < 9; y++) initialStones.push(['B', 1, y]);
  const hist = buildHistory({ ...base, rules: 'tromp-taylor', initialStones, moves: [['B', 'pass'], ['W', 'pass']] });
  assert.equal(hist.isGameFinished, true);
  // Black: 9 wall stones + the 9 points of column 0. White: its one stone (the rest borders both colours).
  assert.equal(hist.finalWhiteMinusBlackScore, 1 - 18 + 7);
  assert.equal(hist.winner, BLACK);
});

test('a pass that would start the territory encore is refused; one pass is fine', () => {
  const one = buildHistory({ ...base, rules: 'japanese', moves: [['B', 'pass']] });
  assert.equal(one.consecutiveEndingPasses, 1);
  assert.throws(() => buildHistory({ ...base, rules: 'japanese', moves: [['B', 'pass'], ['W', 'pass']] }), UnsupportedPositionError);
});

test('inputs: a pass sets the history global instead of a plane', () => {
  const enc = encodePositionV7({ ...base, rules: 'tromp-taylor', moves: [['B', 4, 4], ['W', 'pass']] });
  const g = enc.input.global;
  assert.equal(g[0], 1, 'last move (White) was a pass');
  assert.equal(g[1], 0, 'the move before was not');
  const area = 81;
  assert.equal(enc.input.spatial[10 * area + 4 * 9 + 4], 1, 'plane 10 holds the move before');
  assert.equal(g[14], 1, 'a pass by Black would end the phase');
});

test('inputs: friendly-pass rules hide history and the pass-ends-phase flag when a pass would end the game', () => {
  const spec = { ...base, moves: [['B', 4, 4], ['W', 'pass']] };
  const chinese = encodePositionV7({ ...spec, rules: 'chinese' });
  assert.equal(chinese.input.global[0], 0);
  assert.equal(chinese.input.global[14], 0);
  assert.equal(chinese.input.spatial[10 * 81 + 4 * 9 + 4], 0);
  // conservativePassAndIsRoot does the same under rules without friendly passes.
  const hist = buildHistory({ ...spec, rules: 'tromp-taylor' });
  const root = encodeHistoryV7(hist, { conservativePassAndIsRoot: true });
  assert.equal(root.input.global[0], 0);
  assert.equal(root.input.global[14], 0);
});

test('a cloned history plays on independently of the original', () => {
  const hist = buildHistory({ ...base, rules: 'tromp-taylor', moves: [['B', 4, 4]] });
  const copy = hist.clone();
  copy.play(copy.board.loc(2, 2), WHITE);
  copy.play(1 /* PASS_LOC */, BLACK);
  assert.equal(hist.moves.length, 1);
  assert.equal(hist.board.colors[hist.board.loc(2, 2)], 0);
  assert.equal(hist.consecutiveEndingPasses, 0);
  assert.equal(copy.moves.length, 3);
  assert.equal(copy.consecutiveEndingPasses, 1);
  // The copy continues to apply superko and pass rules.
  copy.play(1, WHITE);
  assert.equal(copy.isGameFinished, true);
  assert.equal(hist.isGameFinished, false);
});
