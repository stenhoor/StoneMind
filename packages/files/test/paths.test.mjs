// Path rules (src/paths.ts), on the built module.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PathError, isWithin, join, nameOf, normalize, parentOf, segments } from '../dist/paths.js';

test('paths are absolute and written one way', () => {
  assert.equal(normalize('/'), '/');
  assert.equal(normalize('/weights/'), '/weights');
  assert.equal(normalize('/weights/kata1.bin.gz'), '/weights/kata1.bin.gz');
  assert.deepEqual(segments('/a/b'), ['a', 'b']);
  assert.deepEqual(segments('/'), []);
});

test('bad paths are refused', () => {
  for (const bad of ['weights', '', '/a//b', '/a/../b', '/./a', '/a\\b', '/a/\u0001', '/' + 'x'.repeat(256)]) {
    assert.throws(() => normalize(bad), PathError, JSON.stringify(bad));
  }
});

test('join, parent and name', () => {
  assert.equal(join('/', 'games', '2026-11-20 vs. Kaito.sgf'), '/games/2026-11-20 vs. Kaito.sgf');
  assert.equal(join('/weights/', 'b18.bin.gz'), '/weights/b18.bin.gz');
  assert.equal(parentOf('/weights/b18.bin.gz'), '/weights');
  assert.equal(parentOf('/top.txt'), '/');
  assert.equal(nameOf('/weights/b18.bin.gz'), 'b18.bin.gz');
  assert.throws(() => parentOf('/'), PathError);
  assert.throws(() => nameOf('/'), PathError);
  assert.throws(() => join('/a', '..'), PathError);
});

test('isWithin', () => {
  assert.equal(isWithin('/a/b', '/a'), true);
  assert.equal(isWithin('/a', '/a'), true);
  assert.equal(isWithin('/ab', '/a'), false);
  assert.equal(isWithin('/a', '/a/b'), false);
  assert.equal(isWithin('/anything', '/'), true);
});
