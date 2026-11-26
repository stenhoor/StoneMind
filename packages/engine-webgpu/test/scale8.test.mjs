// KataGo's scale8 transform (src/model/scale8.ts) must leave the network's function unchanged: the reference forward
// pass on the scaled model, post-processed with its outputScaleMultiplier (×8), matches native KataGo's goldens at the
// same tolerances as the unscaled model, and its raw outputs are the original's divided by 8.
//
// The b6 test net (model version 8) uses ReLU throughout, so only b18c384nbt (Mish) exercises mish_scale8. Like the
// goldens test, b18 runs only when STONEMIND_B18_MODEL points at its model file; otherwise it is skipped by name.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, applyScale8, encodePositionV7, evaluateWithSymmetry, forward, gtpMove } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const positions = JSON.parse(readFileSync(here('./goldens/positions.json'), 'utf8')).positions;
const TOL = { prob: 2e-4, lead: 2e-3, policy: 2e-4, ownership: 2e-4 };

const NETS = [
  { golden: './goldens/g170-b6c96-s175395328-d26788732.json', model: here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz') },
  { golden: './goldens/kata1-b18c384nbt-s9996604416-d4316597426.json', model: process.env.STONEMIND_B18_MODEL ?? '' },
];

const encode = (position, m) => {
  const size = position.boardSize;
  const moves = position.moves.map(([color, loc]) => gtpMove(color, loc, size));
  return encodePositionV7({ xSize: size, ySize: size, rules: position.rules, komi: position.komi, moves }, m.preferPassAliveUnderSuicideRules);
};

for (const net of NETS) {
  if (!net.model || !existsSync(net.model)) {
    test(`scale8: ${net.golden}`, { skip: 'model file not present (set STONEMIND_B18_MODEL for b18c384nbt)' }, () => {});
    continue;
  }
  const golden = JSON.parse(readFileSync(here(net.golden), 'utf8'));
  const model = await loadModel(new Uint8Array(readFileSync(net.model)));
  const scaled = applyScale8(model);
  const usesMish = model.trunk.trunkTipActivation.activation === 'mish';

  test(`${model.name}: scale8 changes biases, activations and the output multiplier, and leaves the original untouched`, () => {
    assert.equal(scaled.postProcessParams.outputScaleMultiplier, model.postProcessParams.outputScaleMultiplier * 8);
    assert.equal(scaled.trunk.trunkTipActivation.activation, usesMish ? 'mish_scale8' : model.trunk.trunkTipActivation.activation);
    assert.equal(scaled.trunk.trunkTipBN.mergedBias[0], Math.fround(model.trunk.trunkTipBN.mergedBias[0] * 0.125));
    assert.equal(scaled.trunk.trunkTipBN.mergedScale, model.trunk.trunkTipBN.mergedScale);
    assert.notEqual(model.trunk.trunkTipActivation.activation, 'mish_scale8');
    if (usesMish) assert.throws(() => applyScale8(scaled), /already/);
  });

  for (const position of positions) {
    if (!golden.positions.some((p) => p.id === position.id)) {
      test(`${model.name}: scale8 on ${position.id}`, { skip: 'no native golden captured for this position yet' }, () => {});
      continue;
    }
    test(`${model.name}: scale8 (reference, symmetry 0) on ${position.id} matches native KataGo and the unscaled raw outputs / 8`, () => {
      const encoded = encode(position, model);
      const raw = forward(model, encoded.input);
      const rawScaled = forward(scaled, encoded.input);
      // Raw outputs are 1/8 of the original's, up to fp32 rounding through the network.
      const close = (a, b, what) => {
        for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] * 8 - b[i]) <= 1e-4 * Math.max(1, Math.abs(b[i])), `${what}[${i}] ${a[i] * 8} vs ${b[i]}`);
      };
      close(rawScaled.value, raw.value, 'value');
      close(rawScaled.scoreValue, raw.scoreValue, 'scoreValue');
      close(rawScaled.ownership, raw.ownership, 'ownership');
      close(rawScaled.policy[0], raw.policy[0], 'policy');

      const want = golden.positions.find((p) => p.id === position.id).evaluations.find((e) => e.symmetry === 0);
      const got = evaluateWithSymmetry(scaled, encoded.input, 0, {
        nextPlayer: encoded.nextPlayer, legal: encoded.legal, koRuleSimple: encoded.koRuleSimple, scoringTerritory: encoded.scoringTerritory,
      });
      const size = position.boardSize;
      const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs KataGo ${b}`);
      near(got.whiteWinProb, want.whiteWin, TOL.prob, 'whiteWin');
      near(got.whiteLead, want.whiteLead, TOL.lead, 'whiteLead');
      for (let i = 0; i < size * size; i++) {
        if (want.policy[i] !== null) near(got.policy[i], want.policy[i], TOL.policy, `policy[${i}]`);
        near(got.whiteOwnership[i], want.whiteOwnership[i], TOL.ownership, `ownership[${i}]`);
      }
    });
  }
}
