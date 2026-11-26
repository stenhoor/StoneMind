// Which GEMM v2 shape each device gets by default (src/gpu/kernels.ts, defaultGemmConfig).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { defaultGemmConfig, GEMM_DEFAULT } from '../dist/index.js';

test('Apple GPUs get 4 × 4, chunk 8, with f16 arithmetic only for fp16 storage', () => {
  assert.deepEqual(defaultGemmConfig('apple', 'f16'), { rm: 4, rn: 4, tk: 8, arith: 'f16' });
  // f32 storage keeps f32 arithmetic: f16 arithmetic there failed the f32 goldens on the iPad.
  assert.deepEqual(defaultGemmConfig('apple', 'f32'), { rm: 4, rn: 4, tk: 8, arith: 'f32' });
});

test('other vendors, and an unknown vendor, keep GEMM_DEFAULT', () => {
  for (const vendor of ['nvidia', 'intel', 'google', '', undefined]) {
    assert.deepEqual(defaultGemmConfig(vendor, 'f16'), GEMM_DEFAULT);
    assert.deepEqual(defaultGemmConfig(vendor, 'f32'), GEMM_DEFAULT);
  }
});
