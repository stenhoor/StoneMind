// Compare the reference network with golden outputs captured from native KataGo (scripts/capture-goldens.mjs):
// every position in test/goldens/positions.json, all 8 symmetries.
//
// Inputs come from the full V7 encoder (rules engine: captures, ko and superko, ladders, pass-alive area). A network
// whose model file isn't present locally is skipped by name: b18c384nbt (98 MB) is not downloaded in CI; set
// STONEMIND_B18_MODEL to its path to run it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadModel, encodePositionV7, evaluateWithSymmetry, parseGtp } from '../dist/index.js';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const positions = JSON.parse(readFileSync(here('./goldens/positions.json'), 'utf8')).positions;

// KataGo prints win/loss to 6 decimals and lead to 3; small fp32 summation-order differences come on top.
const TOL = { prob: 2e-4, lead: 2e-3, policy: 2e-4, ownership: 2e-4 };

const NETS = [
  { golden: './goldens/g170-b6c96-s175395328-d26788732.json', model: here('./fixtures/g170-b6c96-s175395328-d26788732.bin.gz') },
  { golden: './goldens/kata1-b18c384nbt-s9996604416-d4316597426.json', model: process.env.STONEMIND_B18_MODEL ?? '' },
];

for (const net of NETS) {
  if (!existsSync(here(net.golden))) continue;
  if (!net.model || !existsSync(net.model)) {
    test(`${net.golden}`, { skip: 'model file not present (set STONEMIND_B18_MODEL for b18c384nbt)' }, () => {});
    continue;
  }
  const golden = JSON.parse(readFileSync(here(net.golden), 'utf8'));
  const model = await loadModel(new Uint8Array(readFileSync(net.model)));

  for (const position of positions) {
    const captured = golden.positions.find((p) => p.id === position.id);
    const size = position.boardSize;
    const moves = position.moves.map(([color, loc]) => [color, ...parseGtp(loc, size)]);
    const nextPlayer = moves.length % 2 === 0 ? 'B' : 'W';

    const encoded = encodePositionV7(
      { xSize: size, ySize: size, rules: position.rules, komi: position.komi, moves },
      model.preferPassAliveUnderSuicideRules,
    );
    assert.equal(encoded.nextPlayer, nextPlayer);

    test(`${model.name}: ${position.id} matches native KataGo under all 8 symmetries`, () => {
      assert.ok(captured, `no golden output for ${position.id}`);
      for (const want of captured.evaluations) {
        const got = evaluateWithSymmetry(model, encoded.input, want.symmetry, {
          nextPlayer,
          legal: encoded.legal,
          koRuleSimple: encoded.koRuleSimple,
          scoringTerritory: encoded.scoringTerritory,
        });
        const where = `${position.id}, symmetry ${want.symmetry}`;
        const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${where}: ${what} ${a} vs KataGo ${b}`);
        near(got.whiteWinProb, want.whiteWin, TOL.prob, 'whiteWin');
        near(got.whiteLossProb, want.whiteLoss, TOL.prob, 'whiteLoss');
        near(got.whiteLead, want.whiteLead, TOL.lead, 'whiteLead');
        near(got.whiteScoreMean, want.whiteScoreSelfplay, TOL.lead, 'whiteScoreSelfplay');
        near(got.policy[size * size], want.policyPass, TOL.policy, 'policyPass');
        for (let i = 0; i < size * size; i++) {
          const p = want.policy[i];
          if (p === null) assert.equal(got.policy[i], -1, `${where}: point ${i} should be illegal`);
          else near(got.policy[i], p, TOL.policy, `policy[${i}]`);
          near(got.whiteOwnership[i], want.whiteOwnership[i], TOL.ownership, `ownership[${i}]`);
        }
      }
    });
  }
}
