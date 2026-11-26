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

// The GPU check page from packages/engine-webgpu, under gpu-check/.
const gpuCheck = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'site', 'gpu-check');

test('the GPU check page, its modules and its data are in the site', () => {
  const page = join(gpuCheck, 'test', 'gpu');
  const gpuHtml = readFileSync(join(page, 'index.html'), 'utf8');
  for (const [, ref] of gpuHtml.matchAll(/(?:href|src)="(\.[^"]+)"/g)) assert.ok(existsSync(join(page, ref)), `missing ${ref}`);
  const check = readFileSync(join(page, 'check.js'), 'utf8');
  for (const [, spec] of check.matchAll(/from\s+['"](\.[^'"]+)['"]/g)) assert.ok(existsSync(resolve(page, spec)), `check.js imports missing ${spec}`);
  const gemm = readFileSync(join(page, 'gemm-bench.js'), 'utf8');
  for (const [, spec] of gemm.matchAll(/from\s+['"](\.[^'"]+)['"]/g)) assert.ok(existsSync(resolve(page, spec)), `gemm-bench.js imports missing ${spec}`);
  assert.ok(existsSync(join(gpuCheck, 'test', 'goldens', 'positions.json')));
  assert.ok(existsSync(join(gpuCheck, 'test', 'goldens', 'kata1-b18c384nbt-s9996604416-d4316597426.json')));
  assert.ok(!existsSync(join(gpuCheck, 'test', 'goldens', 'kata1-b18c384nbt-s9996604416-d4316597426-cuda.json')));
});

test('every relative import in the GPU check engine modules resolves to a file', () => {
  const walk = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]));
  const files = walk(join(gpuCheck, 'dist')).filter((f) => f.endsWith('.js'));
  assert.ok(files.length > 10);
  for (const file of files) {
    const source = readFileSync(file, 'utf8');
    for (const [, spec] of source.matchAll(/from\s+['"]([^'"]+)['"]/g)) {
      assert.ok(spec.startsWith('.'), `${file} imports the bare specifier ${spec}`);
      assert.ok(existsSync(resolve(dirname(file), spec)), `${file} imports missing ${spec}`);
    }
  }
});

test('the onnxruntime-web baseline page and the files it loads are in the site', () => {
  const ort = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'site', 'ort-bench');
  for (const file of ['index.html', 'ort-bench.js', 'vendor/LICENSE.txt', 'vendor/ort.webgpu.min.mjs', 'vendor/ort.wasm.min.mjs']) {
    assert.ok(existsSync(join(ort, file)), `missing ort-bench/${file}`);
  }
  // Each build's runtime, as its loader names it.
  for (const loader of ['ort.webgpu.min.mjs', 'ort.wasm.min.mjs']) {
    const source = readFileSync(join(ort, 'vendor', loader), 'utf8');
    for (const [name] of source.matchAll(/ort-wasm-simd-threaded[a-z.]*\.mjs/g)) {
      assert.ok(existsSync(join(ort, 'vendor', name)), `${loader} loads missing ${name}`);
      assert.ok(existsSync(join(ort, 'vendor', name.replace(/\.mjs$/, '.wasm'))), `missing ${name.replace(/\.mjs$/, '.wasm')}`);
    }
  }
  // The page imports the engine from the GPU check's copy.
  const page = readFileSync(join(ort, 'ort-bench.js'), 'utf8');
  for (const [, spec] of page.matchAll(/from\s+['"](\.[^'"]+)['"]/g)) assert.ok(existsSync(resolve(ort, spec)), `ort-bench.js imports missing ${spec}`);
});

test('the home page and every local file it links to are in the site', () => {
  const site = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'site');
  const home = readFileSync(join(site, 'index.html'), 'utf8');
  const refs = [...home.matchAll(/(?:href|src)="([^"]+)"/g)].map((m) => m[1]).filter((u) => !/^(https?:|#|\.\/$)/.test(u));
  assert.ok(refs.length >= 8);
  for (const ref of refs) assert.ok(existsSync(join(site, ref)), `home page links to missing ${ref}`);
});
