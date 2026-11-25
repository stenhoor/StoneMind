import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { request } from 'node:http';
import { BRIDGE_PROTOCOL, connectBridge, bridgeTokenProtocol, parsePairingFragment } from '@stonemind/engine';
import { startBridge, acceptKey } from '../dist/index.js';

const FAKE = fileURLToPath(new URL('./fake-katago.mjs', import.meta.url));
const ORIGIN = 'https://stenhoor.github.io';
const TOKEN = '0123456789abcdef0123456789abcdef';

let bridge;
const engineLog = [];

before(async () => {
  bridge = await startBridge({
    command: process.execPath,
    args: [FAKE],
    port: 0,
    token: TOKEN,
    log: (m) => engineLog.push(m),
  });
});
after(() => bridge.close());

/** connectBridge as a browser on `origin` would call it: browsers always send Origin. */
async function connectAs(origin, token = TOKEN) {
  const Original = globalThis.WebSocket;
  globalThis.WebSocket = class extends Original {
    constructor(url, protocols) {
      super(url, { protocols, headers: { origin } });
    }
  };
  try {
    return await connectBridge({ url: `ws://127.0.0.1:${bridge.port}`, token });
  } finally {
    globalThis.WebSocket = Original;
  }
}

/** A raw upgrade request, for checks a real WebSocket client can't make. Resolves with the HTTP status. */
function rawUpgrade(headers) {
  return new Promise((resolve, reject) => {
    const req = request({
      host: '127.0.0.1',
      port: bridge.port,
      headers: {
        connection: 'Upgrade',
        upgrade: 'websocket',
        'sec-websocket-version': '13',
        'sec-websocket-key': 'dGhlIHNhbXBsZSBub25jZQ==',
        host: `127.0.0.1:${bridge.port}`,
        origin: ORIGIN,
        'sec-websocket-protocol': `${BRIDGE_PROTOCOL}, ${bridgeTokenProtocol(TOKEN)}`,
        ...headers,
      },
    });
    req.on('upgrade', (res, socket) => {
      socket.destroy();
      resolve(res.statusCode);
    });
    req.on('response', (res) => resolve(res.statusCode));
    req.on('error', reject);
    req.end();
  });
}

const query = (id, extra = {}) => ({ id, moves: [['B', 'Q16']], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, ...extra });

async function collect(iterable) {
  const out = [];
  for await (const r of iterable) out.push(r);
  return out;
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

test('computes the RFC 6455 accept key', () => {
  // The worked example from RFC 6455 section 1.3.
  assert.equal(acceptKey('dGhlIHNhbXBsZSBub25jZQ=='), 's3pPLMBiTxaQ9kYGzzhZRbK+xOo=');
});

test('the pairing link opens the page with the bridge and token in its fragment', () => {
  const url = new URL(bridge.pairingLink);
  assert.equal(url.origin + url.pathname, 'https://stenhoor.github.io/StoneMind/engine-bench/');
  assert.deepEqual(parsePairingFragment(url.hash), { url: `ws://127.0.0.1:${bridge.port}`, token: TOKEN });
});

test('accepts an allowed origin with the right token', async () => {
  assert.equal(await rawUpgrade({}), 101);
});

test('refuses an origin that is not allowed', async () => {
  assert.equal(await rawUpgrade({ origin: 'https://evil.example' }), 403);
  assert.equal(await rawUpgrade({ origin: '' }), 403);
});

test('refuses a wrong or missing token', async () => {
  assert.equal(await rawUpgrade({ 'sec-websocket-protocol': `${BRIDGE_PROTOCOL}, token.wrongwrongwrongwrong` }), 401);
  assert.equal(await rawUpgrade({ 'sec-websocket-protocol': BRIDGE_PROTOCOL }), 401);
});

test('refuses a Host other than the loopback address (DNS rebinding)', async () => {
  assert.equal(await rawUpgrade({ host: `attacker.example:${bridge.port}` }), 403);
});

test('connectBridge rejects a wrong token', async () => {
  await assert.rejects(connectAs(ORIGIN, 'f'.repeat(32)), /refused the connection/);
});

test('streams partial and final results back under the page’s own id', async () => {
  const engine = await connectAs(ORIGIN);
  const results = await collect(engine.analyze(query('q1', { reportDuringSearchEvery: 0.1, maxVisits: 50 })));
  assert.deepEqual(results.map((r) => [r.id, r.isDuringSearch]), [['q1', true], ['q1', false]]);
  assert.equal(results[1].rootInfo.visits, 50);
  assert.ok(engineLog.some((l) => l.includes('"id":"c') && l.includes('|q1"')), 'the engine saw a prefixed id');
  await engine.close();
});

test('answers actions such as query_version and query_models', async () => {
  const engine = await connectAs(ORIGIN);
  const version = await engine.action({ action: 'query_version' });
  assert.equal(version.version, '1.18.2');
  const models = await engine.action({ action: 'query_models' });
  assert.equal(models.models[0].version, 14);
  await engine.close();
});

test('two pages may use the same query id without seeing each other’s results', async () => {
  const a = await connectAs(ORIGIN);
  const b = await connectAs(ORIGIN);
  const [ra, rb] = await Promise.all([
    collect(a.analyze(query('same', { maxVisits: 11 }))),
    collect(b.analyze(query('same', { maxVisits: 22 }))),
  ]);
  assert.equal(ra.length, 1);
  assert.equal(rb.length, 1);
  assert.equal(ra[0].rootInfo.visits, 11);
  assert.equal(rb[0].rootInfo.visits, 22);
  await a.close();
  await b.close();
});

test('aborting terminates the search and still ends the stream', async () => {
  const engine = await connectAs(ORIGIN);
  const controller = new AbortController();
  const done = collect(engine.analyze(query('long', { maxVisits: 100000 }), { signal: controller.signal }));
  await wait(50);
  controller.abort();
  const results = await done;
  assert.equal(results.at(-1).noResults, true);
  await engine.close();
});

test('a field error from the engine rejects the query', async () => {
  const engine = await connectAs(ORIGIN);
  await assert.rejects(collect(engine.analyze(query('bad', { komi: 999 }))), (e) => e.field === 'komi');
  await engine.close();
});

test('the bridge refuses actions it does not pass on', async () => {
  const engine = await connectAs(ORIGIN);
  await assert.rejects(engine.action({ action: 'shutdown' }), /does not pass on/);
  await engine.close();
});

test('closing a page terminates its open queries in the engine', async () => {
  const engine = await connectAs(ORIGIN);
  engine.analyze(query('orphan', { maxVisits: 100000 }));
  await wait(50);
  await engine.close();
  await wait(100);
  assert.ok(
    engineLog.some((l) => l.includes('"action":"terminate"') && l.includes('|orphan"')),
    'the engine received a terminate for the closed page’s query',
  );
});
