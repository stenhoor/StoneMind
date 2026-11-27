// The t distribution behind value weighting, against KataGo's own test values (core/fancymath.cpp).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { tDistCdf, VALUE_WEIGHT_DISTRIBUTION } from '../dist/search/distribution.js';

const golden = JSON.parse(readFileSync(fileURLToPath(new URL('./goldens/tdist-cdf.json', import.meta.url)), 'utf8'));

for (const [dof, values] of Object.entries(golden.cdf)) {
  test(`tDistCdf matches KataGo with ${dof} degrees of freedom`, () => {
    assert.equal(values.length, 41);
    values.forEach((want, i) => {
      const z = -6 + i * 0.3;
      // KataGo prints 10 decimal places.
      assert.ok(Math.abs(tDistCdf(z, Number(dof)) - want) <= 6e-11, `z ${z}: ${tDistCdf(z, Number(dof))} vs ${want}`);
    });
  });
}

test('the value-weight table interpolates the 3-degree t distribution', () => {
  assert.equal(VALUE_WEIGHT_DISTRIBUTION.getCdf(-60), 0);
  assert.equal(VALUE_WEIGHT_DISTRIBUTION.getCdf(60), 1);
  assert.ok(Math.abs(VALUE_WEIGHT_DISTRIBUTION.getCdf(0) - 0.5) < 1e-12);
  for (const z of [-3.3, -1, -0.2, 0.7, 2.5]) assert.ok(Math.abs(VALUE_WEIGHT_DISTRIBUTION.getCdf(z) - tDistCdf(z, 3)) < 1e-4);
});
