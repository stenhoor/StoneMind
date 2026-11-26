// The matrix GEMM's shapes by batch size, and which GEMMs they can run (src/gpu/kernels.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GEMM_MMA, MATRIX_GEMM_BY_BATCH, gemmMmaTile, matrixGemmFits } from '../dist/gpu/kernels.js';

test('the batch ranges ascend and cover every batch size', () => {
  const ends = MATRIX_GEMM_BY_BATCH.map((r) => r.maxBatch);
  for (let i = 1; i < ends.length; i++) assert.ok(ends[i] > ends[i - 1]);
  assert.equal(ends[0] >= 1, true);
  assert.equal(ends.at(-1), Infinity);
});

test('every shape in the table reads past N only with direct loads and no copying path', () => {
  for (const { config } of MATRIX_GEMM_BY_BATCH) {
    assert.equal(config.direct, true);
    assert.equal(config.overreadN, true);
    assert.equal(config.noCopy, true);
  }
});

test('a shape fits only GEMMs whose M fills its tiles and whose K fills its steps', () => {
  const shared = MATRIX_GEMM_BY_BATCH.at(-1).config;
  assert.equal(gemmMmaTile(shared).tm, 192);
  assert.equal(matrixGemmFits(shared, 192, 192), true);
  assert.equal(matrixGemmFits(shared, 384, 384), true);
  assert.equal(matrixGemmFits(shared, 96, 384), false);
  assert.equal(matrixGemmFits(shared, 192, 40), false);
  const small = MATRIX_GEMM_BY_BATCH[0].config;
  assert.equal(matrixGemmFits(small, 96, 384), true);
  assert.equal(matrixGemmFits(small, 48, 384), false);
  // Without noCopy, the copying path handles any shape.
  assert.equal(matrixGemmFits({ fm: 2, fn: 1, direct: true }, 48, 40), true);
});

test('noCopy leaves out the copying path and its workgroup memory; share stages through workgroup memory', () => {
  const plain = GEMM_MMA('f16', 'f16', 'f16', { fm: 2, fn: 1, direct: true });
  assert.match(plain, /var<workgroup> As/);
  const noCopy = GEMM_MMA('f16', 'f16', 'f16', MATRIX_GEMM_BY_BATCH[0].config);
  assert.doesNotMatch(noCopy, /var<workgroup> As/);
  assert.doesNotMatch(noCopy, /let full/);
  const shared = GEMM_MMA('f16', 'f16', 'f16', MATRIX_GEMM_BY_BATCH.at(-1).config);
  assert.match(shared, /@builtin\(subgroup_id\)/);
  assert.match(shared, /var<workgroup> Ash/);
  assert.match(shared, /@workgroup_size\(192\)/);
});
