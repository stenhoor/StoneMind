// The browser engine's analysis protocol (src/engine/) against native KataGo's analysis engine: the same sessions of
// query lines (scripts/capture-protocol-goldens.mjs), run through KataGoAnalysisLines with the reference forward pass,
// compared line by line for each query id. Errors, warnings and action replies must be the same text; results the same
// fields, written the same way (integer or not), with values within the search check's tolerances (test/search.test.mjs).
//
// The native KataGo the goldens come from is v1.18.2 with later commits from KataGo's master branch (focus moves). The
// differences that brings, and this port's own, are checked as differences, by name, below.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, forward, KataGoAnalysisLines, loadPositionHashes, parseJson, JsonFloat, KATAGO_VERSION, KATAGO_GIT_HASH, roundDynamic, roundStatic } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const golden = JSON.parse(readFileSync(here('./goldens/protocol-g170-b6c96-s175395328-d26788732.json'), 'utf8'));
const model = await loadModel(new Uint8Array(readFileSync(here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz'))));
await loadPositionHashes();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Runs a scenario's steps through a fresh engine, until it has written as many final lines as native did. */
async function runOurs(scenario) {
  const net = { maxBatch: 64, evaluate: async (inputs) => inputs.map((i) => forward(model, i)) };
  const engine = new KataGoAnalysisLines({ model, modelName: 'g170-b6c96-s175395328-d26788732.bin.gz', net, config: golden.config });
  assert.deepEqual(engine.unusedConfigKeys, [], 'every key of the config is read');
  const out = [];
  engine.onLine((l) => out.push(l));
  engine.onClose((reason) => assert.fail(`engine closed: ${reason}`));
  for (const step of scenario.steps) {
    for (const line of step.lines) engine.send(line);
    if (step.waitMs) await sleep(step.waitMs);
  }
  const finals = (lines) => lines.filter((l) => JSON.parse(l).isDuringSearch !== true).length;
  const want = finals(scenario.output);
  for (const start = Date.now(); finals(out) < want && Date.now() - start < 600_000; ) await sleep(20);
  // Anything more would be a line too many.
  await sleep(100);
  engine.close();
  return out;
}

/** Lines by query id (general errors under ''), in order. */
function byId(lines) {
  const m = new Map();
  for (const text of lines) {
    const value = parseJson(text);
    const id = typeof value.id === 'string' ? value.id : '';
    if (!m.has(id)) m.set(id, []);
    m.get(id).push({ text, value });
  }
  return m;
}

const TOL = { winrate: 2e-4, scoreLead: 2e-3, utility: 1e-3, relative: 1e-3, ownership: 2e-4 + 1e-6, ownershipVariance: 1e-3 };
const num = (v) => (v instanceof JsonFloat ? v.value : v);
const near = (got, want, tol, what) => assert.ok(Math.abs(num(got) - num(want)) <= tol, `${what}: ${num(got)} vs ${num(want)}`);
const nearRelative = (got, want, what) => near(got, want, TOL.relative * Math.max(1, Math.abs(num(want))), what);
const nearScore = (got, want, what) => near(got, want, TOL.scoreLead * Math.max(1, Math.abs(num(want))), what);
const nearStdev = (got, want, what) => near(num(got) ** 2, num(want) ** 2, TOL.ownershipVariance, `${what} squared`);
const exact = (got, want, what) => assert.deepEqual(got, want, what);

// The tolerance for each field of a result (the search check's, applied to the JSON).
const MOVE_FIELDS = {
  move: exact, visits: exact, edgeVisits: exact, order: exact, isSymmetryOf: exact, pv: exact, pvVisits: exact, pvEdgeVisits: exact,
  winrate: (g, w, m) => near(g, w, TOL.winrate, m), lcb: (g, w, m) => near(g, w, TOL.winrate, m),
  prior: (g, w, m) => near(g, w, TOL.winrate, m), noResultValue: (g, w, m) => near(g, w, TOL.winrate, m),
  scoreLead: (g, w, m) => near(g, w, TOL.scoreLead, m), scoreMean: (g, w, m) => near(g, w, TOL.scoreLead, m),
  scoreSelfplay: (g, w, m) => near(g, w, TOL.scoreLead, m), scoreStdev: nearScore,
  utility: (g, w, m) => near(g, w, TOL.utility, m), utilityLcb: (g, w, m) => near(g, w, TOL.utility, m),
  playSelectionValue: nearRelative, weight: nearRelative, edgeWeight: nearRelative,
  ownership: (g, w, m) => w.forEach((x, i) => near(g[i], x, TOL.ownership, `${m}[${i}]`)),
  ownershipStdev: (g, w, m) => w.forEach((x, i) => nearStdev(g[i], x, `${m}[${i}]`)),
};
const ROOT_FIELDS = {
  visits: exact, currentPlayer: exact, thisHash: exact, symHash: exact, weight: nearRelative,
  winrate: (g, w, m) => near(g, w, TOL.winrate, m), rawWinrate: (g, w, m) => near(g, w, TOL.winrate, m),
  rawNoResultProb: (g, w, m) => near(g, w, TOL.winrate, m), rawStWrError: (g, w, m) => near(g, w, TOL.winrate, m),
  scoreSelfplay: (g, w, m) => near(g, w, TOL.scoreLead, m), scoreLead: (g, w, m) => near(g, w, TOL.scoreLead, m),
  rawLead: (g, w, m) => near(g, w, TOL.scoreLead, m), rawScoreSelfplay: (g, w, m) => near(g, w, TOL.scoreLead, m),
  scoreStdev: nearScore, rawScoreSelfplayStdev: nearScore, rawStScoreError: nearScore, rawVarTimeLeft: nearScore,
  utility: (g, w, m) => near(g, w, TOL.utility, m),
};
const RESULT_FIELDS = {
  id: exact, turnNumber: exact, isDuringSearch: exact, noResults: exact,
  policy: (g, w, m) => w.forEach((x, i) => (num(x) === -1 ? assert.equal(num(g[i]), -1, `${m}[${i}]`) : near(g[i], x, TOL.winrate, `${m}[${i}]`))),
  ownership: MOVE_FIELDS.ownership, ownershipStdev: MOVE_FIELDS.ownershipStdev,
};

/**
 * The same JSON form: the same keys, arrays of the same length, and each number written as an integer or not alike.
 * `stopped`: a search stopped early, whose moves and lines may be fewer or more.
 */
function sameForm(got, want, what, stopped = false) {
  if (want instanceof JsonFloat || typeof want === 'number') {
    assert.equal(typeof got === 'number' ? 'integer' : got instanceof JsonFloat ? 'float' : typeof got, typeof want === 'number' ? 'integer' : 'float', `${what}: written as`);
  } else if (Array.isArray(want)) {
    assert.ok(Array.isArray(got), `${what}: an array`);
    // The principal variations of a search stopped early may differ in length.
    if (!/\.pv(Visits|EdgeVisits)?$/.test(what) && !(stopped && what.endsWith('.moveInfos'))) assert.equal(got.length, want.length, `${what}: length`);
    want.slice(0, 1).forEach((w, i) => got[i] !== undefined && sameForm(got[i], w, `${what}[${i}]`, stopped));
  } else if (want !== null && typeof want === 'object') {
    // A move searched by one engine and not the other may lack isSymmetryOf in one.
    const keys = (o) => Object.keys(o).filter((k) => !(stopped && k === 'isSymmetryOf')).sort();
    assert.deepEqual(keys(got), keys(want), `${what}: fields`);
    for (const k of keys(want)) sameForm(got[k], want[k], `${what}.${k}`, stopped);
  } else assert.equal(typeof got, typeof want, `${what}: type`);
}

/**
 * Our numbers are rounded as KataGo rounds them: 8 digits past the first (roundDynamic), ownership to 10^-6
 * (roundStatic); rootInfo's weight isn't rounded.
 */
function roundedAsKataGo(v, what) {
  const check = (x, ownership, w) => x instanceof JsonFloat && assert.equal(x.value, ownership ? roundStatic(x.value, 1e6) : roundDynamic(x.value, 8), `${w}: rounded`);
  for (const [k, x] of Object.entries(v)) {
    if (k === 'moveInfos') x.forEach((m) => roundedAsKataGo(m, `${what} ${m.move}`));
    else if (k === 'rootInfo') for (const [f, y] of Object.entries(x)) f !== 'weight' && check(y, false, `${what} rootInfo ${f}`);
    else if (Array.isArray(x)) x.forEach((y, i) => check(y, k.startsWith('ownership'), `${what} ${k}[${i}]`));
    else check(x, false, `${what} ${k}`);
  }
}

/** A final result's values against native's, each field with its tolerance. */
function sameValues(got, want, what) {
  for (const [k, w] of Object.entries(want)) {
    if (k === 'moveInfos') {
      // The order of moves with equal play selection values is the search's; compare by move.
      const ours = new Map(got.moveInfos.map((m) => [m.move, m]));
      for (const wm of w) {
        const gm = ours.get(wm.move);
        assert.ok(gm, `${what} ${wm.move} searched`);
        for (const [f, x] of Object.entries(wm)) MOVE_FIELDS[f](gm[f], x, `${what} ${wm.move} ${f}`);
      }
    } else if (k === 'rootInfo') {
      for (const [f, x] of Object.entries(w)) ROOT_FIELDS[f](got.rootInfo[f], x, `${what} rootInfo ${f}`);
    } else RESULT_FIELDS[k](got[k], w, `${what} ${k}`);
  }
}

// The port's differences from native in errors and action replies, by what they are about.
const JSON_SYNTAX_SUFFIX = ' - could not parse input line as json request: ';
function sameMessageLine(got, want, what) {
  const g = got.value;
  const w = want.value;
  if (typeof w.error === 'string' && w.error.includes(JSON_SYNTAX_SUFFIX)) {
    // The JSON parser's own message differs (nlohmann's against JavaScript's); the rest is KataGo's.
    assert.equal(g.error.slice(g.error.indexOf(JSON_SYNTAX_SUFFIX)), w.error.slice(w.error.indexOf(JSON_SYNTAX_SUFFIX)), what);
  } else if (typeof w.error === 'string' && w.error.startsWith("'action' field must be")) {
    // Native's build has set_focus, added to KataGo after v1.18.2.
    assert.equal(`${g.error} or 'set_focus'`, w.error, what);
  } else if (typeof w.error === 'string' && w.error.startsWith('Could not set settings: Could not parse')) {
    // KataGo's config errors name the config file; this engine has none.
    assert.ok(w.error.startsWith(g.error) && w.error.slice(g.error.length).startsWith(' in config file '), `${what}: ${g.error} vs ${w.error}`);
    assert.equal(g.field, w.field, what);
  } else if (w.action === 'query_version') {
    // This engine reports the version it ports and that version's commit; native's build is a later commit.
    assert.equal(g.version, w.version, `${what} version`);
    assert.equal(KATAGO_VERSION, w.version);
    assert.equal(g.git_hash, KATAGO_GIT_HASH, `${what} git_hash`);
    assert.equal(got.text.replace(KATAGO_GIT_HASH, w.git_hash), want.text, what);
  } else if (w.action === 'query_models') {
    // The model name is the file name given; maxBatchSize is each engine's own.
    for (const f of ['internalName', 'usesHumanSLProfile', 'version', 'usingFP16']) assert.deepEqual(g.models[0][f], w.models[0][f], `${what} ${f}`);
    sameForm(g, w, what);
  } else assert.equal(got.text, want.text, what);
}

for (const scenario of golden.scenarios) {
  test(`protocol matches native KataGo: ${scenario.name}`, async () => {
    const ourLines = await runOurs(scenario);
    const ours = byId(ourLines);
    const native = byId(scenario.output);
    assert.deepEqual([...ours.keys()].sort(), [...native.keys()].sort(), 'the same query ids answered');
    const partial = new Set(scenario.partial ?? []);
    for (const [id, wantLines] of native) {
      const gotLines = ours.get(id);
      const finals = (lines) => lines.filter((l) => l.value.isDuringSearch !== true);
      const during = (lines) => lines.filter((l) => l.value.isDuringSearch === true);
      const expected = scenario.duringCount?.[id];
      if (expected === 'some') {
        assert.ok(during(gotLines).length > 0 && during(wantLines).length > 0, `${id}: reports during the search`);
      } else {
        assert.equal(during(gotLines).length, expected ?? 0, `${id}: reports during the search`);
        assert.equal(during(wantLines).length, expected ?? 0, `${id}: native's reports during the search`);
      }
      // Reports during the search come before the final result, with the visits growing.
      let lastVisits = 0;
      for (const l of gotLines) {
        if (l.value.isDuringSearch === true) {
          sameForm(l.value, during(wantLines)[0].value, `${id} during the search`, true);
          assert.ok(l.value.rootInfo.visits >= lastVisits, `${id}: visits grow`);
          lastVisits = l.value.rootInfo.visits;
        }
      }
      const got = finals(gotLines);
      const want = finals(wantLines);
      assert.equal(got.length, want.length, `${id}: lines`);
      want.forEach((w, i) => {
        const g = got[i];
        const what = `${id} line ${i}`;
        if (!('turnNumber' in w.value)) return sameMessageLine(g, w, what);
        sameForm(g.value, w.value, what, partial.has(id));
        roundedAsKataGo(g.value, what);
        if (partial.has(id) && w.value.moveInfos) {
          // A search stopped early: its form only.
          assert.ok(g.value.rootInfo.visits >= 1, `${what}: some visits`);
          return;
        }
        sameValues(g.value, w.value, what);
      });
    }
    // Final results come in native's order (the priority scenario checks this order).
    const order = (lines) => lines.map((l) => JSON.parse(l)).filter((v) => v.turnNumber !== undefined && v.isDuringSearch === false).map((v) => `${v.id}:${v.turnNumber}`);
    if (scenario.name === 'priority') assert.deepEqual(order(ourLines), order(scenario.output));
  });
}
