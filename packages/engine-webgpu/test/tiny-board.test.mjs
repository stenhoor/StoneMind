// Golden test: KataGo's own `runnnontinyboardtest` (cpp/tests/testsearchmisc.cpp, Tests::runNNOnTinyBoard), run with
// g170-b6c96 and symmetry 3, recorded in KataGo's cpp/tests/results/runNNOnTinyBoardTest.txt at commit d91ea85.
// Expected values below are copied from that file; KataGo prints probabilities and ownership ×1000, rounded.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, countParameters, encodeSimplePosition, evaluateWithSymmetry } from '../dist/index.js';

const MODEL = fileURLToPath(new URL('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz', import.meta.url));
const model = await loadModel(new Uint8Array(readFileSync(MODEL)));

// .....
// ...x.
// ..o..
// .xxo.
// .....
const position = {
  xSize: 5,
  ySize: 5,
  rules: 'tromp-taylor',
  komi: 7.5,
  initialStones: [
    ['B', 3, 1],
    ['W', 2, 2],
    ['B', 1, 3],
    ['B', 2, 3],
    ['W', 3, 3],
  ],
  nextPlayer: 'W',
};

const EXPECTED = {
  win: 90.25,
  loss: 9.75,
  noResult: 0,
  scoreMean: 9.37,
  scoreMeanSq: 242.9,
  lead: 7.87,
  pass: 0,
  // null marks an illegal point, printed "-" by KataGo.
  policy: [
    [0, 0, 0, 0, 0],
    [0, 3, 150, null, 0],
    [0, 237, null, 607, 0],
    [0, null, null, null, 0],
    [0, 0, 0, 1, 0],
  ],
  ownership: [
    [-21, -257, 64, -419, 121],
    [-129, -354, -489, -148, 241],
    [-341, -367, -294, 46, 292],
    [-322, -457, -80, 559, 277],
    [-247, -395, -127, -132, 91],
  ],
};

test('parses the g170-b6c96 test network', () => {
  assert.equal(model.name, 'g170-b6c96-s175395328-d26788732');
  assert.equal(model.modelVersion, 8);
  assert.equal(model.trunk.numBlocks, 6);
  assert.equal(model.trunk.trunkNumChannels, 96);
  assert.equal(model.numPolicyChannels, 1);
  // KataGo's log for this file: "(convnet, 1027911 params)".
  assert.equal(countParameters(model), 1027911);
});

test('matches KataGo on the tiny board (values, policy, ownership)', () => {
  const encoded = encodeSimplePosition(position);
  const ev = evaluateWithSymmetry(model, encoded.input, 3, {
    nextPlayer: 'W',
    legal: encoded.legal,
    koRuleSimple: encoded.koRuleSimple,
    scoringTerritory: encoded.scoringTerritory,
  });

  const near = (actual, expected, tol, what) =>
    assert.ok(Math.abs(actual - expected) <= tol, `${what}: got ${actual}, KataGo printed ${expected}`);
  near(ev.whiteWinProb * 100, EXPECTED.win, 0.01, 'Win %');
  near(ev.whiteLossProb * 100, EXPECTED.loss, 0.01, 'Loss %');
  near(ev.whiteNoResultProb * 100, EXPECTED.noResult, 0.005, 'NoResult %');
  near(ev.whiteScoreMean, EXPECTED.scoreMean, 0.01, 'ScoreMean');
  near(ev.whiteScoreMeanSq, EXPECTED.scoreMeanSq, 0.1, 'ScoreMeanSq');
  near(ev.whiteLead, EXPECTED.lead, 0.01, 'Lead');
  assert.equal(ev.varTimeLeft, -1);

  near(Math.round(ev.policy[25] * 1000), EXPECTED.pass, 0, 'pass');
  for (let y = 0; y < 5; y++) {
    for (let x = 0; x < 5; x++) {
      const p = ev.policy[y * 5 + x];
      const expected = EXPECTED.policy[y][x];
      if (expected === null) assert.equal(p, -1, `policy (${x},${y}) should be illegal`);
      else near(Math.round(p * 1000), expected, 1, `policy (${x},${y}) ×1000`);
      near(Math.round(ev.whiteOwnership[y * 5 + x] * 1000), EXPECTED.ownership[y][x], 1, `ownership (${x},${y}) ×1000`);
    }
  }
});

// KataGo's SymmetryHelpers::getSymLoc: flip x (bit 1), flip y (bit 0), then transpose (bit 2).
function symLoc(x, y, size, s) {
  if (s & 2) x = size - 1 - x;
  if (s & 1) y = size - 1 - y;
  return s & 4 ? [y, x] : [x, y];
}

test('each symmetry equals evaluating the mirrored board, mapped back (exact)', () => {
  const opts = (enc) => ({ nextPlayer: 'W', legal: enc.legal, koRuleSimple: false, scoringTerritory: false });
  const original = encodeSimplePosition(position);
  for (let s = 0; s < 8; s++) {
    const viaSymmetry = evaluateWithSymmetry(model, original.input, s, opts(original));
    const mirrored = encodeSimplePosition({
      ...position,
      initialStones: position.initialStones.map(([p, x, y]) => [p, ...symLoc(x, y, 5, s)]),
    });
    const direct = evaluateWithSymmetry(model, mirrored.input, 0, opts(mirrored));
    assert.ok(Math.abs(viaSymmetry.whiteWinProb - direct.whiteWinProb) < 1e-6, `symmetry ${s}: win`);
    assert.ok(Math.abs(viaSymmetry.whiteLead - direct.whiteLead) < 1e-4, `symmetry ${s}: lead`);
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        const [sx, sy] = symLoc(x, y, 5, s);
        const a = viaSymmetry.policy[y * 5 + x];
        const b = direct.policy[sy * 5 + sx];
        assert.ok(Math.abs(a - b) < 1e-6, `symmetry ${s}: policy at (${x},${y})`);
        const oa = viaSymmetry.whiteOwnership[y * 5 + x];
        const ob = direct.whiteOwnership[sy * 5 + sx];
        assert.ok(Math.abs(oa - ob) < 1e-6, `symmetry ${s}: ownership at (${x},${y})`);
      }
    }
  }
});
