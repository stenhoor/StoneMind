// Which GEMM v2 shape each device gets by default (src/gpu/kernels.ts, defaultGemmConfig).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { defaultGemmConfig, GEMM_DEFAULT } from '../dist/index.js';

test('Apple GPUs get 4 × 4, chunk 8, with f16 arithmetic where shader-f16 exists', () => {
  assert.deepEqual(defaultGemmConfig('apple', true), { rm: 4, rn: 4, tk: 8, arith: 'f16' });
  assert.deepEqual(defaultGemmConfig('apple', false), { rm: 4, rn: 4, tk: 8, arith: 'f32' });
});

test('other vendors, and an unknown vendor, keep GEMM_DEFAULT', () => {
  for (const vendor of ['nvidia', 'intel', 'google', '', undefined]) {
    assert.deepEqual(defaultGemmConfig(vendor, true), GEMM_DEFAULT);
    assert.deepEqual(defaultGemmConfig(vendor, false), GEMM_DEFAULT);
  }
});
