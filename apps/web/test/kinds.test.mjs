// How Files works out a file's kind (src/files/kinds.ts) and keeps it (src/files/catalog.ts), on the built modules.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { gzipSync } from 'node:zlib';
import { evaluateKind, looksLikeKataGoConfig, sgfKind } from '../dist/js/files/kinds.js';
import { emptyCatalog, isAppPath, withEntry, withMove, without } from '../dist/js/files/catalog.js';

const blob = (text) => new Blob([text]);
const NET = 'kata1-b18c384nbt-s9996604416-d4316597426';

test('an SGF naming players or a result is a game', async () => {
  const r = await evaluateKind('final.sgf', blob('(;GM[1]SZ[19]PB[Lee]PW[Cho]RE[B+R];B[pd];W[dp])'));
  assert.equal(r.kind, 'game');
  assert.match(r.why, /Black player \(PB\), White player \(PW\), result \(RE\)/);
  assert.equal(sgfKind('(;GM[1]RE[W+0.5])').kind, 'game');
});

test('an SGF that sets up stones and names no players or result is a problem', async () => {
  const r = await evaluateKind('corner.sgf', blob('(;GM[1]SZ[19]AB[pd][qc]AW[qd][rd]C[Black to live];B[rc])'));
  assert.equal(r.kind, 'problem');
  assert.match(r.why, /sets up stones \(AB, AW\)/);
});

test('an SGF with neither is a game', () => {
  assert.equal(sgfKind('(;GM[1]SZ[19];B[pd];W[dp])').kind, 'game');
});

test('property names are matched whole: PB inside another name does not count', () => {
  assert.equal(sgfKind('(;GM[1]XPB[x]AB[aa])').kind, 'problem');
});

test('a KataGo network is recognised from its name and its first bytes', async () => {
  const r = await evaluateKind(`${NET}.bin.gz`, blob(gzipSync(`${NET}\n14\n${'x '.repeat(100)}`)));
  assert.equal(r.kind, 'weights');
  assert.match(r.why, /model version 14/);
});

test('a KataGo config is recognised from key = value lines with KataGo keys', async () => {
  const cfg = '# Example\nlogDir = gtp_logs\nrules = tromp-taylor\n\nmaxVisits = 500   # visits\nnumSearchThreads = 6\n';
  assert.equal((await evaluateKind('gtp.cfg', blob(cfg))).kind, 'config');
  assert.equal(looksLikeKataGoConfig('[section]\nname = x\n'), false);
  assert.equal(looksLikeKataGoConfig('colour = blue\n'), false);
});

test('files of no kind Files takes are refused, saying which kinds it takes', async () => {
  const r = await evaluateKind('notes.txt', blob('hello'));
  assert.match(r.refused, /notes\.txt is not a kind of file Files takes/);
  assert.match(r.refused, /SGF games and problems, KataGo networks \(\.bin\.gz, \.txt\.gz\) and KataGo configs \(\.cfg\)/);
});

test('a file named like a kind but not one inside is refused', async () => {
  assert.match((await evaluateKind('game.sgf', blob('not an sgf'))).refused, /does not start like an SGF file/);
  assert.match((await evaluateKind(`${NET}.bin.gz`, blob(gzipSync('hello')))).refused, /does not start like one/);
  assert.match((await evaluateKind('x.cfg', blob('hello world'))).refused, /does not read like a KataGo config/);
});

test('the catalog follows moves and deletes, folders included', () => {
  let c = emptyCatalog();
  c = withEntry(c, '/club/a.sgf', { kind: 'game', by: 'evaluated' });
  c = withEntry(c, '/club/b.sgf', { kind: 'problem', by: 'player' });
  c = withEntry(c, '/n.bin.gz', { kind: 'weights', by: 'evaluated' });
  c = withMove(c, '/club', '/2026/club');
  assert.deepEqual(Object.keys(c.files).sort(), ['/2026/club/a.sgf', '/2026/club/b.sgf', '/n.bin.gz']);
  assert.deepEqual(c.files['/2026/club/b.sgf'], { kind: 'problem', by: 'player' });
  c = without(c, '/2026');
  assert.deepEqual(Object.keys(c.files), ['/n.bin.gz']);
  // A move of /club does not touch /clubs.
  c = withMove(withEntry(c, '/clubs/x.sgf', { kind: 'game', by: 'evaluated' }), '/club', '/other');
  assert.ok(c.files['/clubs/x.sgf']);
  assert.ok(JSON.parse(JSON.stringify(c)).files['/n.bin.gz'], 'the catalog is plain JSON');
});

test('the app keeps its own files in a folder the player does not see', () => {
  assert.ok(isAppPath('/.stonemind/catalog.json'));
  assert.ok(!isAppPath('/.stonemindx'));
  assert.ok(!isAppPath('/games'));
});
