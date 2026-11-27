// KataGo's position hashes (src/rules/hash.ts) against KataGo v1.18.2: its own test vectors for the random number
// generators (core/rand.cpp: XorShift1024Mult::test, PCG32::test, and Rand::runTests' first values of Rand("abc")), and
// native's thisHash and symHash for every position of the search goldens, and for positions with a ko and on rectangular
// boards (test/goldens/hashes.json, from scripts/capture-hash-goldens.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { XorShift1024Mult, PCG32, KataGoRand, loadPositionHashes, positionHashes, buildHistory, gtpMove } from '../dist/index.js';
import { queryHistory } from './query-position.mjs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

test('XorShift1024Mult matches KataGo\'s test vector', () => {
  const xorm = new XorShift1024Mult([
  15148282349006049087n, 3601266951833665894n, 16929445066801446424n, 13475938501103070154n, 15713138009143754412n,
  4148159782736716337n, 16035594834001032141n, 5555591070439871209n, 4101130512537511022n, 12821547636792886909n,
  9050874162294428797n, 6187760405891629771n, 10053646276519763308n, 2219782655280501359n, 3719698449347562208n,
  5421263376768154227n
  ]);
  const expected = [
  0x749746d1, 0x9242ca14, 0x98db98a1, 0x1348e491, 0xde60e668, 0x77e37a69, 0xeb51a9d3, 0xd44b4727, 0x341895b0,
  0xc7b1b3f4, 0xe7ef0529, 0x8e72ea7e, 0x5855da19, 0xfffcd2b2, 0xa684e430, 0xb76a7e0d, 0x5af3820e, 0x320b0699,
  0xdbb85ee0, 0xc1dcd25c, 0x4b395e3e, 0x4007756f, 0x76a0c667, 0xaa6041f6, 0x756f94bb, 0x39527d1b, 0x6e1232ef,
  0xb3027668, 0x776ea832, 0x35a0ed1b, 0x1f2f0268, 0xadd59669
  ];
  for (const e of expected) assert.equal(xorm.nextUInt(), e);
});

test('PCG32 matches KataGo\'s test vector', () => {
  const pcg = new PCG32(123n);
  const expected = [
  0xb3766cbd, 0x65fdd305, 0x2a3b9b9c, 0x09a2dee9, 0x1a86aabc, 0x36a98234, 0x82e6e2b4, 0x10c077e5, 0x29755fc7,
  0xf7fa7b5c, 0x1cb7ae7d, 0xcce0e3d9, 0x065ec08b, 0x505d1cdb, 0x8b778f3c, 0xdb72f217
  ];
  for (const e of expected) assert.equal(pcg.nextUInt(), e);
});

test('Rand seeded from a string matches KataGo (MD5 and SHA-256 seeding)', async () => {
  const rand = await KataGoRand.create('abc');
  const expected = [
  476808125, 4218845147, 1770424533, 2748129731, 3501022071, 2408095342, 579150456, 3608809805, 694088449,
  2852964936, 192049595, 1340578043, 649387339, 2025290779, 589057531, 402360262
  ];
  for (const e of expected) assert.equal(rand.nextUInt(), e);
});

test('thisHash and symHash match native KataGo on every search golden position', async () => {
  await loadPositionHashes();
  let checked = 0;
  for (const file of readdirSync(here('./goldens')).filter((f) => /^search-.*-s\d+-d\d+\.json$/.test(f))) {
    const golden = JSON.parse(readFileSync(here(`./goldens/${file}`), 'utf8'));
    for (const q of golden.queries) {
      const hist = queryHistory(q);
      const got = positionHashes(hist.board, hist.nextPla);
      assert.equal(got.thisHash, q.response.rootInfo.thisHash, `${file} ${q.id} thisHash`);
      assert.equal(got.symHash, q.response.rootInfo.symHash, `${file} ${q.id} symHash`);
      checked++;
    }
  }
  assert.equal(checked, 31, 'positions checked');
});

test('thisHash and symHash match native KataGo with a ko and on rectangular boards', async () => {
  await loadPositionHashes();
  const golden = JSON.parse(readFileSync(here('./goldens/hashes.json'), 'utf8'));
  for (const p of golden.positions) {
    const hist = buildHistory({ xSize: p.xSize, ySize: p.ySize, rules: p.rules, komi: p.komi, moves: p.moves.map(([c, l]) => gtpMove(c, l, p.ySize)) });
    const got = positionHashes(hist.board, hist.nextPla);
    assert.equal(got.thisHash, p.thisHash, `${p.id} thisHash`);
    assert.equal(got.symHash, p.symHash, `${p.id} symHash`);
  }
});
