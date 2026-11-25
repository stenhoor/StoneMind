import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AnalysisError, JsonLineSession, parsePairingFragment, expectedFinals } from '../dist/index.js';

const DESCRIPTOR = { id: 'fake', kind: 'local', displayName: 'Fake' };

/** A transport whose engine side is scripted by the test. */
function fakeTransport(onSend) {
  let receive = () => {};
  let closed = () => {};
  const sent = [];
  const transport = {
    send: (line) => {
      const msg = JSON.parse(line);
      sent.push(msg);
      onSend?.(msg, (reply) => queueMicrotask(() => receive(JSON.stringify(reply))));
    },
    onLine: (r) => (receive = r),
    onClose: (r) => (closed = r),
    close: () => closed('closed by client'),
  };
  return { transport, sent, push: (reply) => receive(JSON.stringify(reply)), drop: (reason) => closed(reason) };
}

const query = (id, extra = {}) => ({ id, moves: [], rules: 'tromp-taylor', komi: 7.5, boardXSize: 19, boardYSize: 19, ...extra });
const result = (id, turnNumber, isDuringSearch, visits = 10) => ({
  id,
  turnNumber,
  isDuringSearch,
  moveInfos: [{ move: 'Q16', visits, winrate: 0.5, scoreLead: 0.4, prior: 0.2, order: 0, pv: ['Q16'] }],
  rootInfo: { winrate: 0.5, scoreLead: 0.4, visits, currentPlayer: 'B' },
});

async function collect(iterable) {
  const out = [];
  for await (const r of iterable) out.push(r);
  return out;
}

test('expects one final result per analysed turn', () => {
  assert.equal(expectedFinals({}), 1);
  assert.equal(expectedFinals({ analyzeTurns: [0, 1, 2] }), 3);
});

test('streams partial results, then ends after the final one', async () => {
  const { transport } = fakeTransport((msg, reply) => {
    if (msg.id === 'q1') {
      reply(result('q1', 0, true, 5));
      reply(result('q1', 0, false, 100));
    }
  });
  const session = new JsonLineSession(DESCRIPTOR, transport);
  const results = await collect(session.analyze(query('q1')));
  assert.deepEqual(results.map((r) => [r.isDuringSearch, r.rootInfo.visits]), [[true, 5], [false, 100]]);
});

test('routes interleaved results to their own queries and waits for every turn', async () => {
  const { transport, push } = fakeTransport();
  const session = new JsonLineSession(DESCRIPTOR, transport);
  const a = collect(session.analyze(query('a', { analyzeTurns: [0, 1] })));
  const b = collect(session.analyze(query('b')));
  push(result('b', 0, true));
  push(result('a', 1, false));
  push(result('b', 0, false));
  push(result('a', 0, false));
  assert.deepEqual((await a).map((r) => r.turnNumber), [1, 0]);
  assert.deepEqual((await b).map((r) => r.isDuringSearch), [true, false]);
});

test('aborting sends terminate and still delivers the final result', async () => {
  const { transport, sent, push } = fakeTransport((msg, reply) => {
    if (msg.action === 'terminate') {
      reply(msg);
      reply({ id: msg.terminateId, turnNumber: 0, isDuringSearch: false, noResults: true });
    }
  });
  const session = new JsonLineSession(DESCRIPTOR, transport);
  const controller = new AbortController();
  const stream = session.analyze(query('q'), { signal: controller.signal });
  const done = collect(stream);
  push(result('q', 0, true));
  controller.abort();
  const results = await done;
  assert.equal(results.at(-1).noResults, true);
  assert.ok(sent.some((m) => m.action === 'terminate' && m.terminateId === 'q'));
});

test('actions get their own ids and resolve with the echoed reply', async () => {
  const { transport } = fakeTransport((msg, reply) => {
    if (msg.action === 'query_version') reply({ ...msg, version: '1.18.2', git_hash: 'abc' });
  });
  const session = new JsonLineSession(DESCRIPTOR, transport);
  const reply = await session.action({ action: 'query_version' });
  assert.equal(reply.version, '1.18.2');
  assert.match(reply.id, /^action-\d+$/);
});

test('a field error rejects its query with the field named', async () => {
  const { transport } = fakeTransport((msg, reply) => {
    if (msg.id === 'bad') reply({ error: 'komi out of range', field: 'komi', id: 'bad' });
  });
  const session = new JsonLineSession(DESCRIPTOR, transport);
  await assert.rejects(collect(session.analyze(query('bad', { komi: 999 }))), (e) => {
    assert.ok(e instanceof AnalysisError);
    assert.equal(e.field, 'komi');
    return true;
  });
});

test('warnings reach the query callback without ending it', async () => {
  const warnings = [];
  const { transport } = fakeTransport((msg, reply) => {
    if (msg.id === 'w') {
      reply({ warning: 'rules converted', field: 'rules', id: 'w' });
      reply(result('w', 0, false));
    }
  });
  const session = new JsonLineSession(DESCRIPTOR, transport);
  const results = await collect(session.analyze(query('w'), { onWarning: (w) => warnings.push(w.field) }));
  assert.deepEqual(warnings, ['rules']);
  assert.equal(results.length, 1);
});

test('a dropped connection fails open queries and pending actions', async () => {
  const { transport, drop } = fakeTransport();
  const session = new JsonLineSession(DESCRIPTOR, transport);
  const stream = collect(session.analyze(query('q')));
  const action = session.action({ action: 'clear_cache' });
  drop('bridge stopped');
  await assert.rejects(stream, /bridge stopped/);
  await assert.rejects(action, /bridge stopped/);
  assert.throws(() => session.analyze(query('later')), /bridge stopped/);
});

test('refuses a second open query with the same id', () => {
  const { transport } = fakeTransport();
  const session = new JsonLineSession(DESCRIPTOR, transport);
  session.analyze(query('dup'));
  assert.throws(() => session.analyze(query('dup')), /already open/);
});

test('reads a pairing only from a localhost bridge URL and a well-formed token', () => {
  const token = 'a'.repeat(32);
  assert.deepEqual(parsePairingFragment(`#bridge=${encodeURIComponent('ws://127.0.0.1:6565')}&token=${token}`), {
    url: 'ws://127.0.0.1:6565',
    token,
  });
  assert.equal(parsePairingFragment(`#bridge=${encodeURIComponent('ws://evil.example:6565')}&token=${token}`), null);
  assert.equal(parsePairingFragment(`#bridge=${encodeURIComponent('ws://127.0.0.1:6565')}&token=short`), null);
  assert.equal(parsePairingFragment(''), null);
});
