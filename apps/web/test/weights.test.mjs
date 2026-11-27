// What the Files app reads from KataGo weights (src/files/weights.ts), on the built modules.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { describeNetwork, formatBytes, grouped, parseWeightsName, readWeightsHeader } from '../dist/js/files/weights.js';

const NAME = 'kata1-b18c384nbt-s9996604416-d4316597426';

test('a KataGo weights name gives the run, the network and its training', () => {
  const w = parseWeightsName(`${NAME}.bin.gz`);
  assert.deepEqual(w, { run: 'kata1', blocks: 18, channels: 384, kind: 'nbt', samples: 9996604416n, dataRows: 4316597426n, format: 'binary' });
  assert.equal(describeNetwork(w), 'b18c384nbt: 18 blocks, 384 channels, nested bottleneck');
  assert.equal(parseWeightsName('kata1-b6c96-s175395328-d26788732.txt.gz')?.format, 'text');
  assert.equal(describeNetwork(parseWeightsName('kata1-b6c96-s175395328-d26788732.txt.gz')), 'b6c96: 6 blocks, 96 channels, residual');
});

test('other names are not read as weights', () => {
  for (const name of ['game.sgf', `${NAME}.bin`, `${NAME}.zip`, 'kata1-b18c384nbt.bin.gz', `-b18c384-s1-d1.bin.gz`]) assert.equal(parseWeightsName(name), null, name);
});

test('the model name and version come from the start of the file, gzipped or not', async () => {
  const body = `${NAME}\n15\n${'x '.repeat(10000)}`;
  assert.deepEqual(await readWeightsHeader(new Blob([gzipSync(body)])), { name: NAME, modelVersion: 15 });
  assert.deepEqual(await readWeightsHeader(new Blob([body])), { name: NAME, modelVersion: 15 });
  await assert.rejects(readWeightsHeader(new Blob([gzipSync('not a model at all')])), /does not start like a KataGo model/);
});

test('sizes and counts read the way people write them', () => {
  assert.equal(formatBytes(512), '512 bytes');
  assert.equal(formatBytes(11 * 1024), '11.0 KB');
  assert.equal(formatBytes(98.2 * 1024 * 1024), '98.2 MB');
  assert.equal(formatBytes(150 * 1024 * 1024), '150 MB');
  assert.equal(formatBytes(10 * 1024 ** 3), '10.0 GB');
  assert.equal(grouped(9996604416n, 'en-US'), '9,996,604,416');
});
