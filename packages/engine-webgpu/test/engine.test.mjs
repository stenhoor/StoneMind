// The browser analysis engine (src/engine/) beyond what the sessions against native check (test/protocol.test.mjs):
// the engine through @stonemind/engine's JsonLineSession, its evaluation cache, several analysis threads, what this
// port refuses, its startup config, and KataGo's rules and JSON forms.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { AnalysisError } from '@stonemind/engine';
import {
  loadModel, forward, KataGoAnalysisLines, AnalysisConfigError, loadPositionHashes, parseKataGoRules,
  rulesToString, supportedRules, RulesParseError, parseJson, dumpJson, formatDouble, JsonFloat, roundDynamic, analysisHistory,
  parseRules, Board, BLACK, WHITE, PASS_LOC,
} from '../dist/index.js';
import { createAnalysisEngine } from '../dist/session.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
await loadPositionHashes();
const CONFIG = { numAnalysisThreads: '1', numSearchThreadsPerAnalysisThread: '1', nnRandomize: 'false', wideRootNoise: '0' };
/** The reference network, counting the positions it evaluates. */
const countingNet = () => {
  const net = { maxBatch: 16, evaluated: 0, evaluate: async (inputs) => ((net.evaluated += inputs.length), inputs.map((i) => forward(model, i))) };
  return net;
};
const query = (fields = {}) => ({ id: 'q', rules: 'tromp-taylor', komi: 7.5, boardXSize: 5, boardYSize: 5, moves: [['B', 'C3']], maxVisits: 6, ...fields });

/** Sends lines and collects what the engine writes until `done` says the lines so far are enough. */
async function session(engine, lines, done) {
  const out = [];
  engine.onLine((l) => out.push(JSON.parse(l)));
  for (const l of lines) engine.send(typeof l === 'string' ? l : JSON.stringify(l));
  for (const start = Date.now(); !done(out) && Date.now() - start < 120_000; ) await new Promise((r) => setTimeout(r, 10));
  engine.close();
  return out;
}
const finals = (n) => (out) => out.filter((o) => o.turnNumber !== undefined && !o.isDuringSearch).length >= n;

test('through JsonLineSession: results, warnings, errors, and actions', async () => {
  const engine = await createAnalysisEngine({ model, modelName: 'b6', net: countingNet(), config: CONFIG });
  assert.deepEqual(engine.descriptor, { id: 'stonemind-webgpu', kind: 'browser', displayName: 'Browser engine' });
  const warnings = [];
  const results = [];
  for await (const r of engine.analyze(query({ id: 'a', analyzeTurns: [0, 1], typo: 1 }), { onWarning: (w) => warnings.push(w) })) results.push(r);
  assert.deepEqual(results.map((r) => [r.turnNumber, r.isDuringSearch, r.rootInfo.visits]), [[0, false, 6], [1, false, 6]]);
  assert.equal(results[1].rootInfo.currentPlayer, 'W');
  assert.deepEqual(warnings.map((w) => w.field), ['typo']);
  await assert.rejects(
    async () => {
      for await (const _ of engine.analyze(query({ id: 'b', komi: 7.25 }))) void _;
    },
    (e) => e instanceof AnalysisError && e.field === 'komi' && e.queryId === 'b',
  );
  const version = await engine.action({ action: 'query_version' });
  assert.equal(version.version, '1.18.2');
  const models = await engine.action({ action: 'query_models' });
  assert.deepEqual(models.models, [{ name: 'b6', internalName: model.name, maxBatchSize: 16, usesHumanSLProfile: false, version: 8, usingFP16: 'false' }]);
  await engine.close();
});

test('through JsonLineSession: an aborted query ends with a final line for each turn', async () => {
  const engine = await createAnalysisEngine({ model, modelName: 'b6', net: countingNet(), config: CONFIG });
  const controller = new AbortController();
  const got = [];
  for await (const r of engine.analyze(query({ id: 'long', analyzeTurns: [0, 1], maxVisits: 100000, reportDuringSearchEvery: 0.05 }), { signal: controller.signal })) {
    got.push(r);
    if (r.isDuringSearch) controller.abort();
  }
  // Turn 0 stopped with what it had searched; turn 1, still queued, never started.
  const finalOf = (t) => got.find((r) => r.turnNumber === t && !r.isDuringSearch);
  assert.ok(finalOf(0).rootInfo.visits >= 1 && finalOf(0).rootInfo.visits < 100000);
  assert.equal(finalOf(1).noResults, true);
  assert.equal(got.filter((r) => !r.isDuringSearch).length, 2);
  await engine.close();
});

test('clear_cache empties the evaluation cache', async () => {
  const net = countingNet();
  const engine = new KataGoAnalysisLines({ model, modelName: 'b6', net, config: CONFIG });
  const out = [];
  engine.onLine((l) => out.push(JSON.parse(l)));
  const waitFor = async (n) => {
    while (!finals(n)(out)) await new Promise((r) => setTimeout(r, 10));
  };
  engine.send(JSON.stringify(query({ id: 'a' })));
  await waitFor(1);
  const first = net.evaluated;
  // The same search again finds every evaluation in the cache, until the cache is cleared.
  engine.send(JSON.stringify(query({ id: 'b' })));
  await waitFor(2);
  assert.equal(net.evaluated, first);
  engine.send(JSON.stringify({ id: 'c', action: 'clear_cache' }));
  engine.send(JSON.stringify(query({ id: 'd' })));
  await waitFor(3);
  assert.equal(net.evaluated, 2 * first);
  assert.deepEqual(out.find((o) => o.id === 'd').moveInfos, out.find((o) => o.id === 'a').moveInfos);
  engine.close();
});

test('the evaluation cache keeps apart positions with different komi or rules', async () => {
  const net = countingNet();
  const fresh = await session(new KataGoAnalysisLines({ model, modelName: 'b6', net, config: CONFIG }), [query({ id: 'k', komi: 0.5 })], finals(1));
  const shared = await session(
    new KataGoAnalysisLines({ model, modelName: 'b6', net, config: CONFIG }),
    [query({ id: 'a' }), query({ id: 'r', rules: 'chinese' }), query({ id: 'k', komi: 0.5 })],
    finals(3),
  );
  assert.deepEqual(shared.find((o) => o.id === 'k'), fresh[0]);
  assert.notEqual(shared.find((o) => o.id === 'r').rootInfo.rawLead, shared.find((o) => o.id === 'a').rootInfo.rawLead);
});

test('two analysis threads give the same results as one', async () => {
  // On different board sizes, so neither search can use an evaluation the other made for a different history.
  const lines = [query({ id: 'a', analyzeTurns: [0, 1] }), query({ id: 'b', boardXSize: 6, boardYSize: 6, moves: [['B', 'B2'], ['W', 'D4']], analyzeTurns: [1, 2] })];
  const one = await session(new KataGoAnalysisLines({ model, modelName: 'b6', net: countingNet(), config: CONFIG }), lines, finals(4));
  const net = countingNet();
  let inFlight = 0;
  let mostInFlight = 0;
  const watched = { maxBatch: 16, evaluate: async (inputs) => {
    inFlight++;
    mostInFlight = Math.max(mostInFlight, inputs.length);
    await new Promise((r) => setTimeout(r, 1));
    inFlight--;
    return net.evaluate(inputs);
  } };
  const two = await session(new KataGoAnalysisLines({ model, modelName: 'b6', net: watched, config: { ...CONFIG, numAnalysisThreads: '2' } }), lines, finals(4));
  const key = (o) => `${o.id}:${o.turnNumber}`;
  assert.deepEqual(new Map(two.map((o) => [key(o), o])), new Map(one.map((o) => [key(o), o])));
  // The two searches' leaves went into the same network calls.
  assert.equal(mostInFlight, 2);
  void inFlight;
});

test('what this engine refuses that KataGo searches', async () => {
  const out = await session(
    new KataGoAnalysisLines({ model, modelName: 'b6', net: countingNet(), config: CONFIG }),
    [
      query({ id: 'stone', initialStones: [['B', 'C3'], ['W', 'C3']] }),
      query({ id: 'noise', overrideSettings: { rootNoiseEnabled: true } }),
      query({ id: 'symmetries', overrideSettings: { rootNumSymmetriesToSample: 2 } }),
      query({ id: 'threads', overrideSettings: { numSearchThreads: 4 } }),
      query({ id: 'offboard', moves: [['B', '(5,0)']] }),
    ],
    finals(1),
  );
  const error = (id) => out.find((o) => o.id === id && o.error);
  assert.equal(error('stone').field, 'initialStones');
  assert.equal(error('noise').error, 'Could not set settings: not supported by this engine: rootNoiseEnabled');
  assert.equal(error('symmetries').error, 'Could not set settings: not supported by this engine: rootNumSymmetriesToSample');
  // numSearchThreads is the search's batch: accepted.
  assert.equal(error('threads'), undefined);
  assert.equal(error('offboard').error, 'Could not parse board location: (5,0)');
});

test('startup config: KataGo\'s required keys and alias, unread keys, and settings not ported', () => {
  const make = (config) => new KataGoAnalysisLines({ model, modelName: 'b6', net: countingNet(), config });
  assert.throws(() => make({ numSearchThreads: '1' }), (e) => e instanceof AnalysisConfigError && /numAnalysisThreads/.test(e.message));
  assert.throws(() => make({ numAnalysisThreads: '1' }), (e) => e instanceof AnalysisConfigError && /numSearchThreads/.test(e.message));
  assert.throws(() => make({ ...CONFIG, numSearchThreads: '2' }), /Cannot specify both numSearchThreadsPerAnalysisThread and numSearchThreads/);
  assert.throws(() => make({ ...CONFIG, maxTime: '5' }), (e) => e instanceof AnalysisConfigError && /maxTime/.test(e.message));
  assert.throws(() => make({ ...CONFIG, reportAnalysisWinratesAs: 'X' }), /Could not parse config value for reportAnalysisWinratesAs: X/);
  assert.deepEqual(make({ ...CONFIG, logDir: 'logs', nnCacheSizePowerOfTwo: '20', typo: '1' }).unusedConfigKeys, ['typo']);
});

test('warnUnusedFields false leaves out the unused field warnings', async () => {
  const out = await session(new KataGoAnalysisLines({ model, modelName: 'b6', net: countingNet(), config: { ...CONFIG, warnUnusedFields: 'false' } }), [query({ typo: 1 })], finals(1));
  assert.equal(out.filter((o) => o.warning).length, 0);
});

test('KataGo\'s rules: names, the JSON object, and the compact form', () => {
  const r = (s, o) => parseKataGoRules(s, o);
  assert.deepEqual(r(' Chinese_OGS '), { koRule: 'POSITIONAL', scoringRule: 'AREA', taxRule: 'NONE', multiStoneSuicideLegal: false, hasButton: false, whiteHandicapBonusRule: 'N', friendlyPassOk: true, komi: 7.5 });
  assert.equal(r('stone scoring').taxRule, 'ALL');
  assert.equal(r('goe').friendlyPassOk, true);
  // The compact form: territory scoring defaults the tax to SEKI and the komi to 6.5; whb N-1 before N.
  const compact = r('koSIMPLE scoreTERRITORY sui0 whbN-1 fpok1');
  assert.deepEqual([compact.koRule, compact.scoringRule, compact.taxRule, compact.multiStoneSuicideLegal, compact.whiteHandicapBonusRule, compact.friendlyPassOk, compact.komi], ['SIMPLE', 'TERRITORY', 'SEKI', false, 'N-1', true, 6.5]);
  assert.equal(r('komi-3.5scoringAREA').komi, -3.5);
  assert.equal(r('button1').komi, 7);
  for (const bad of ['', 'koBOGUS', 'komi1e5', 'sui2', 'whatever']) assert.throws(() => r(bad), RulesParseError, bad);
  // A JSON object, read in key order: "scoring" after "score".
  const fields = { score: 'TERRITORY', scoring: 'AREA', komi: true };
  const reader = { keys: Object.keys(fields).sort(), getString: (k) => fields[k], getBool: (k) => fields[k], getFloat: (k) => +fields[k] };
  const obj = r('{"komi":true,"score":"TERRITORY","scoring":"AREA"}', reader);
  assert.deepEqual([obj.scoringRule, obj.taxRule, obj.komi], ['AREA', 'NONE', 1]);
  assert.equal(rulesToString(r('aga')), 'koSITUATIONALscoreAREAtaxNONEsui0whbN-1fpok1komi7.5');
  assert.deepEqual(supportedRules(r('{"ko":"SPIGHT"}', { keys: ['ko'], getString: () => 'SPIGHT' })), { rules: { ...r('tromp-taylor'), koRule: 'SITUATIONAL' }, supported: false });
});

test('JSON as nlohmann reads and writes it', () => {
  const v = parseJson('{"b":[1,1.0,1e2,-0],"a":2.5}');
  assert.equal(v.b[0], 1);
  assert.ok(v.b[1] instanceof JsonFloat && v.b[2] instanceof JsonFloat);
  assert.equal(dumpJson(v), '{"a":2.5,"b":[1,1.0,100.0,0]}');
  const cases = [[1, '1.0'], [0.1, '0.1'], [1e-5, '1e-05'], [1e-4, '0.0001'], [123.456, '123.456'], [1e15, '1e+15'], [1e14, '100000000000000.0'], [-2.5e-10, '-2.5e-10'], [1.5e300, '1.5e+300'], [NaN, 'null']];
  for (const [x, s] of cases) assert.equal(formatDouble(x), s, String(x));
  // KataGo's rounding: 8 digits past the first, halves away from zero.
  assert.equal(roundDynamic(0.123456789123, 8), 0.123456789);
  assert.equal(roundDynamic(-2.5, 0), -3);
  assert.equal(roundDynamic(123456789012.5, 8), 123456789013);
});

test('assumeMultipleStartingBlackMovesAreHandicap off: Black\'s opening moves are not handicap stones', () => {
  const b = new Board(9, 9);
  const at = (x, y) => b.loc(x, y);
  // Black's moves with White passing in between count as handicap stones, when the setting is on.
  const position = { xSize: 9, ySize: 9, rules: parseRules('chinese'), komi: 7.5, moves: [[BLACK, at(2, 2)], [WHITE, PASS_LOC], [BLACK, at(6, 6)], [WHITE, at(4, 4)]] };
  assert.equal(analysisHistory(position).whiteHandicapBonusScore, 2);
  assert.equal(analysisHistory({ ...position, assumeMultipleStartingBlackMovesAreHandicap: false }).whiteHandicapBonusScore, 0);
});
