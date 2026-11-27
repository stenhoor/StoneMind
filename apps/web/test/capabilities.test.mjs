// The capability report's summary (src/capabilities/report.ts), on the built modules.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { reportJson, summarize } from '../dist/js/capabilities/report.js';

const base = {
  kind: 'stonemind-capabilities', version: 1, date: '2026-11-27T00:00:00.000Z', userAgent: 'test',
  hardwareConcurrency: 8, devicePixelRatio: 2, screen: { width: 1440, height: 900 },
  webgpu: { api: true, adapter: true, vendor: 'nvidia', architecture: 'ampere', description: '', fallbackAdapter: false, features: ['shader-f16'], limits: { maxStorageBufferBindingSize: 2147483644 } },
  dedicatedWorker: { started: true, webgpu: { api: true, adapter: true, vendor: 'nvidia' }, opfs: { api: true, syncAccessHandle: true } },
  sharedWorker: { api: true, started: true, webgpu: { api: true, adapter: false }, opfs: { api: true, syncAccessHandle: false } },
  opfs: { api: true, createWritable: true },
  storage: { estimate: true, quota: 10 * 1024 ** 3, usage: 5 * 1024 ** 2, persistApi: true, persisted: false },
  webLocks: { api: true, works: true },
  broadcastChannel: { api: true, works: true },
  css: { backdropFilter: true, backdropFilterSvg: false, containerQueries: true },
};

const row = (rows, feature) => rows.find((r) => r.feature === feature);

test('every row has a group, a feature and a yes, no or info result', () => {
  const rows = summarize(base);
  assert.ok(rows.length >= 15);
  for (const r of rows) {
    assert.ok(r.group && r.feature);
    assert.ok(r.ok === true || r.ok === false || r.ok === null);
    assert.equal(typeof r.detail, 'string');
  }
});

test('WebGPU in each context is reported separately', () => {
  const rows = summarize(base);
  assert.equal(row(rows, 'In the page').ok, true);
  assert.match(row(rows, 'In the page').detail, /nvidia · ampere/);
  assert.equal(row(rows, 'In a dedicated worker').ok, true);
  assert.equal(row(rows, 'In a shared worker').ok, false);
  assert.equal(row(rows, 'In a shared worker').detail, 'navigator.gpu, but no adapter');
  assert.equal(row(rows, 'Half precision (shader-f16)').ok, true);
  assert.equal(row(rows, 'Largest storage buffer').detail, '2.0 GB');
});

test('missing shared workers and failed workers say why', () => {
  const noShared = summarize({ ...base, sharedWorker: { api: false, started: false } });
  assert.equal(row(noShared, 'In a shared worker').detail, 'No SharedWorker in this browser');
  const failed = summarize({ ...base, dedicatedWorker: { started: false, error: 'blocked' } });
  assert.equal(row(failed, 'In a dedicated worker').ok, false);
  assert.equal(row(failed, 'In a dedicated worker').detail, 'blocked');
});

test('no WebGPU at all: no adapter details are listed', () => {
  const rows = summarize({ ...base, webgpu: { api: false, adapter: false } });
  assert.equal(row(rows, 'In the page').detail, 'No navigator.gpu');
  assert.equal(row(rows, 'Half precision (shader-f16)'), undefined);
});

test('storage shows usage and quota, and whether files are kept', () => {
  const rows = summarize(base);
  assert.equal(row(rows, 'Quota').detail, '5 MB used of 10.0 GB');
  assert.equal(row(rows, 'Persistent storage').detail, 'Not granted yet');
  assert.equal(row(rows, 'createSyncAccessHandle in a worker').ok, true);
});

test('the report JSON round-trips', () => {
  assert.deepEqual(JSON.parse(reportJson(base)), base);
});
