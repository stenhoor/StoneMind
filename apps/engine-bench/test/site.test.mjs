import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const bench = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'site', 'engine-bench');
const html = readFileSync(join(bench, 'index.html'), 'utf8');

test('every local file the page links to is in the site', () => {
  const refs = [...html.matchAll(/(?:href|src)="([^"]+)"/g)].map((m) => m[1]).filter((u) => !/^(https?:|#|\.\/$)/.test(u));
  assert.ok(refs.length >= 6);
  for (const ref of refs) assert.ok(existsSync(join(bench, ref)), `missing ${ref}`);
});

test('the import map points at modules that exist', () => {
  const map = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]);
  for (const target of Object.values(map.imports)) assert.ok(existsSync(join(bench, target)), `missing ${target}`);
});

test('every relative import in the shipped modules resolves to a file', () => {
  for (const dir of ['js', join('vendor', 'engine')]) {
    for (const file of readdirSync(join(bench, dir)).filter((f) => f.endsWith('.js'))) {
      const source = readFileSync(join(bench, dir, file), 'utf8');
      for (const [, spec] of source.matchAll(/from\s+['"](\.[^'"]+)['"]/g)) {
        assert.ok(existsSync(resolve(bench, dir, spec)), `${dir}/${file} imports missing ${spec}`);
      }
    }
  }
});

test('no shipped module imports a bare package other than the engine', () => {
  for (const file of readdirSync(join(bench, 'js')).filter((f) => f.endsWith('.js'))) {
    const source = readFileSync(join(bench, 'js', file), 'utf8');
    for (const [, spec] of source.matchAll(/from\s+['"]([^.'"][^'"]*)['"]/g)) {
      assert.equal(spec, '@stonemind/engine', `js/${file} imports ${spec}, which the import map doesn't provide`);
    }
  }
});
