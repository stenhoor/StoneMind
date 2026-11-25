import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encodeSimplePosition, parseGtp, OutsideSimpleDomainError } from '../dist/index.js';

const plane = (enc, p, x, y) => enc.input.spatial[p * enc.input.xSize * enc.input.ySize + y * enc.input.xSize + x];

test('parses GTP coordinates, skipping I', () => {
  assert.deepEqual(parseGtp('A19', 19), [0, 0]);
  assert.deepEqual(parseGtp('J1', 19), [8, 18]);
  assert.deepEqual(parseGtp('t1', 19), [18, 18]);
  assert.throws(() => parseGtp('I5', 19));
});

test('encodes stones, liberties and the last moves relative to the player to move', () => {
  const enc = encodeSimplePosition({
    xSize: 9, ySize: 9, rules: 'tromp-taylor', komi: 7.5, nextPlayer: 'B',
    moves: [['B', 2, 2], ['W', 6, 6], ['B', 2, 6]],
  });
  assert.equal(plane(enc, 0, 0, 0), 1); // on board
  assert.equal(plane(enc, 1, 2, 2), 1); // own stone (Black to move)
  assert.equal(plane(enc, 2, 6, 6), 1); // opponent stone
  assert.equal(plane(enc, 9, 2, 6), 0); // last move was Black's own, so history stops: it must be the opponent's
  const w = encodeSimplePosition({ xSize: 9, ySize: 9, rules: 'tromp-taylor', komi: 7.5, nextPlayer: 'W', moves: [['B', 2, 2], ['W', 6, 6], ['B', 2, 6]] });
  assert.equal(plane(w, 9, 2, 6), 1); // the opponent's last move
  assert.equal(plane(w, 10, 6, 6), 1); // own move before it
  assert.equal(plane(w, 11, 2, 2), 1);
});

test('encodes komi, rules and the komi parity wave as KataGo does', () => {
  const tt = encodeSimplePosition({ xSize: 5, ySize: 5, rules: 'tromp-taylor', komi: 7.5, nextPlayer: 'W' });
  assert.deepEqual(Array.from(tt.input.global.slice(5, 11)), [0.375, 1, 0.5, 1, 0, 0]);
  assert.equal(tt.input.global[18], 0.5); // 25 points (odd): komi floor 7, delta 0.5
  const jp = encodeSimplePosition({ xSize: 9, ySize: 9, rules: 'japanese', komi: 6.5, nextPlayer: 'B' });
  assert.equal(jp.input.global[5], Math.fround(-6.5 / 20)); // Black's self-komi is negative
  assert.deepEqual(Array.from(jp.input.global.slice(6, 11)), [0, 0, 0, 1, 1]);
  assert.equal(jp.input.global[18], 0); // no parity wave under territory scoring
});

test('refuses positions outside the simple domain instead of encoding them wrongly', () => {
  const base = { xSize: 5, ySize: 5, rules: 'tromp-taylor', komi: 7.5, nextPlayer: 'B' };
  // A group in atari: ladder features could apply.
  assert.throws(() => encodeSimplePosition({ ...base, initialStones: [['W', 0, 0], ['B', 1, 0]] }), OutsideSimpleDomainError);
  // A capture.
  assert.throws(
    () => encodeSimplePosition({ ...base, initialStones: [['W', 0, 0], ['B', 1, 0]], moves: [['B', 0, 1]] }),
    OutsideSimpleDomainError,
  );
  // Area scoring with an empty region bordered by one colour only.
  assert.throws(() => encodeSimplePosition({ ...base, xSize: 3, ySize: 3, initialStones: [['B', 1, 1]] }), OutsideSimpleDomainError);
});
