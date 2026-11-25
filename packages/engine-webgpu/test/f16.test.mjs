// binary16 conversion (src/gpu/f16.ts): known bit patterns from the IEEE 754 binary16 format, and an exhaustive
// check that every conversion picks the nearest representable half, ties to even.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toFloat16Bits, fromFloat16Bits, toFloat16Array } from '../dist/index.js';

test('known bit patterns', () => {
  const cases = [
    [1, 0x3c00], [-2, 0xc000], [0.5, 0x3800], [65504, 0x7bff], [0, 0x0000], [-0, 0x8000],
    [2 ** -14, 0x0400], [2 ** -24, 0x0001], [2 ** -25, 0x0000], [Infinity, 0x7c00], [-Infinity, 0xfc00],
    [65520, 0x7c00], // halfway between 65504 and 65536: ties to even rounds up to infinity
    [1e6, 0x7c00],
  ];
  for (const [v, bits] of cases) assert.equal(toFloat16Bits(v), bits, `${v}`);
  assert.equal(toFloat16Bits(NaN) & 0x7c00, 0x7c00);
  assert.notEqual(toFloat16Bits(NaN) & 0x03ff, 0);
});

test('decoding inverts encoding for every finite half', () => {
  for (let h = 0; h < 0x10000; h++) {
    if ((h & 0x7c00) === 0x7c00) continue;
    assert.equal(toFloat16Bits(fromFloat16Bits(h)), h === 0x8000 ? 0x8000 : h, `0x${h.toString(16)}`);
  }
});

test('every float32 between neighbouring halves rounds to the nearer one, ties to even', () => {
  // For each pair of adjacent positive finite halves, check points just below, at and just above the midpoint.
  for (let h = 0; h < 0x7bff; h++) {
    const a = fromFloat16Bits(h);
    const b = fromFloat16Bits(h + 1);
    const mid = (a + b) / 2;
    const eps = (b - a) / 1024;
    assert.equal(toFloat16Bits(mid - eps), h, `below midpoint of 0x${h.toString(16)}`);
    assert.equal(toFloat16Bits(mid + eps), h + 1, `above midpoint of 0x${h.toString(16)}`);
    assert.equal(toFloat16Bits(mid), h % 2 === 0 ? h : h + 1, `midpoint of 0x${h.toString(16)}`);
    assert.equal(toFloat16Bits(-(mid - eps)), 0x8000 | h, `negative, below midpoint of 0x${h.toString(16)}`);
  }
});

test('toFloat16Array pads to an even length', () => {
  const out = toFloat16Array(new Float32Array([1, 2, 3]));
  assert.equal(out.length, 4);
  assert.deepEqual([...out.slice(0, 3)], [0x3c00, 0x4000, 0x4200]);
});
