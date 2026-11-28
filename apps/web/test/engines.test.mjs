// The engine host's protocol and logic (src/engines/protocol.ts, host.ts), on the built modules, with fake tabs and the
// fake engine. The worker and the tabs are tested in browsers (test/e2e/engines.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { KATAGO_DEFAULTS, engineConfig, fromEngine, toEngine } from '../dist/js/engines/protocol.js';
import { EngineHost } from '../dist/js/engines/host.js';
import { ENDLESS_VISITS, fakeEngine } from '../dist/js/engines/fake.js';

test('query ids are prefixed per tab on the way in and restored on the way out', () => {
  const sent = JSON.parse(toEngine('t1', JSON.stringify({ id: 'q1', moves: [], maxVisits: 10 })).line);
  assert.notEqual(sent.id, 'q1');
  const back = fromEngine(JSON.stringify({ id: sent.id, turnNumber: 0, isDuringSearch: false }));
  assert.equal(back.prefix, 't1');
  assert.equal(JSON.parse(back.line).id, 'q1');
  // A terminate names another query: that id is prefixed too, and restored in the reply.
  const term = JSON.parse(toEngine('t2', JSON.stringify({ id: 'a1', action: 'terminate', terminateId: 'q1' })).line);
  assert.equal(fromEngine(JSON.stringify({ id: term.id, action: 'terminate', terminateId: term.terminateId })).line, JSON.stringify({ id: 'a1', action: 'terminate', terminateId: 'q1' }));
});

test('lines with no id go to every tab; terminate_all is refused, as it would stop other tabs’ queries', () => {
  assert.equal(fromEngine('{"error":"general"}').prefix, null);
  assert.equal(fromEngine('not json').prefix, null);
  const r = toEngine('t1', JSON.stringify({ id: 'x', action: 'terminate_all' }));
  assert.match(JSON.parse(r.refused).error, /terminate each query instead/);
  assert.equal(JSON.parse(r.refused).id, 'x');
});

test('an engine starts with KataGo’s analysis_example.cfg settings, the player’s over them', () => {
  assert.equal(KATAGO_DEFAULTS.numAnalysisThreads, '2');
  assert.equal(KATAGO_DEFAULTS.numSearchThreadsPerAnalysisThread, '16');
  assert.equal(KATAGO_DEFAULTS.maxVisits, '500');
  assert.deepEqual(engineConfig({ maxVisits: '100' }), { ...KATAGO_DEFAULTS, maxVisits: '100' });
});

// ------------------------------------------------------------------ the host

const tick = () => new Promise((r) => setTimeout(r, 0));

function tab(host) {
  const got = [];
  const conn = host.connect({ post: (m) => got.push(m) });
  return {
    got,
    send: (m) => conn.receive(m),
    line: (engineId, q) => conn.receive({ type: 'line', engineId, line: JSON.stringify(q) }),
    lines: () => got.filter((m) => m.type === 'line').map((m) => JSON.parse(m.line)),
    status: () => got.filter((m) => m.type === 'status').at(-1)?.status,
    disconnect: () => conn.disconnect(),
  };
}

function hostWithFake() {
  const engines = [];
  const host = new EngineHost(true, false, {
    fake: async () => {
      const lines = fakeEngine();
      const sent = [];
      const send = lines.send.bind(lines);
      lines.send = (l) => (sent.push(JSON.parse(l)), send(l));
      engines.push({ lines, sent });
      return { lines, info: { model: { name: 'fake', version: 0 } } };
    },
  });
  return { host, engines };
}

const START = { engineId: 'browser', kind: 'fake', weightsPath: '/n.bin.gz' };

test('two tabs share one engine, and each gets only its own replies, even with the same query id', async () => {
  const { host, engines } = hostWithFake();
  const a = tab(host);
  const b = tab(host);
  a.send({ type: 'start', request: START });
  await tick();
  await tick();
  assert.equal(engines.length, 1);
  assert.equal(b.status().engines[0].state, 'running');
  assert.equal(b.status().tabs, 2);
  a.line('browser', { id: 'q1', moves: [], maxVisits: 7 });
  b.line('browser', { id: 'q1', moves: [], maxVisits: 9 });
  await tick();
  assert.deepEqual(a.lines().map((l) => [l.id, l.rootInfo.visits]), [['q1', 7]]);
  assert.deepEqual(b.lines().map((l) => [l.id, l.rootInfo.visits]), [['q1', 9]]);
});

test('a tab that goes has its queries terminated; the last tab going stops the engine', async () => {
  const { host, engines } = hostWithFake();
  const a = tab(host);
  const b = tab(host);
  a.send({ type: 'start', request: START });
  await tick();
  await tick();
  a.line('browser', { id: 'long', moves: [], maxVisits: ENDLESS_VISITS });
  a.disconnect();
  await tick();
  const terminate = engines[0].sent.find((q) => q.action === 'terminate');
  assert.ok(terminate, 'a terminate was sent for the gone tab’s query');
  assert.equal(b.status().tabs, 1);
  assert.equal(b.status().engines[0].state, 'running');
  b.disconnect();
  await tick();
  assert.equal(host.status().engines[0]?.state ?? 'none', 'none');
});

test('one engine at a time for now; stop ends it for every tab; a start for an unknown kind fails plainly', async () => {
  const { host } = hostWithFake();
  const a = tab(host);
  a.send({ type: 'start', request: START });
  await tick();
  await tick();
  a.send({ type: 'start', request: { ...START, engineId: 'second' } });
  await tick();
  assert.match(a.status().engines.find((e) => e.id === 'second').error, /Only 1 engine runs at a time for now/);
  a.send({ type: 'stop', engineId: 'browser' });
  await tick();
  assert.equal(a.status().engines.find((e) => e.id === 'browser').state, 'stopped');
  assert.ok(a.got.some((m) => m.type === 'closed' && m.engineId === 'browser'));
  a.send({ type: 'start', request: { ...START, engineId: 'x', kind: 'webgpu' } });
  await tick();
  assert.match(a.status().engines.find((e) => e.id === 'x').error, /can't run a webgpu engine/);
});

test('a start that fails says why, and echoes the asking tab’s token', async () => {
  const host = new EngineHost(true, true, { fake: async () => { throw new Error('There is no weights file at /x in this browser’s storage.'); } });
  const a = tab(host);
  a.send({ type: 'start', request: { ...START, token: 'abc' } });
  await tick();
  await tick();
  const s = a.status().engines[0];
  assert.equal(s.state, 'failed');
  assert.equal(s.token, 'abc');
  assert.match(s.error, /no weights file at \/x/);
});
