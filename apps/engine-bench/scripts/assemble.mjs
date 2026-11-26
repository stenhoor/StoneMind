// Assemble the static site in dist/site: the bench page, its compiled modules, and verbatim copies of the brand files
// and the engine package; and the WebGPU engine's GPU check page (gpu-check/). The page loads plain ES modules through an import map; there is no bundler.
import { cpSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const app = join(dirname(fileURLToPath(import.meta.url)), '..');
const repo = join(app, '..', '..');
const site = join(app, 'dist', 'site');
const bench = join(site, 'engine-bench');

rmSync(site, { recursive: true, force: true });
mkdirSync(bench, { recursive: true });

cpSync(join(app, 'src', 'index.html'), join(bench, 'index.html'));
cpSync(join(app, 'src', 'bench.css'), join(bench, 'bench.css'));
cpSync(join(app, 'dist', 'js'), join(bench, 'js'), { recursive: true });

const brand = join(repo, 'packages', 'brand');
cpSync(join(brand, 'tokens', 'tokens.css'), join(bench, 'brand', 'tokens.css'));
cpSync(join(brand, 'styles', 'components.css'), join(bench, 'brand', 'components.css'));
cpSync(join(brand, 'logos'), join(bench, 'brand', 'logos'), { recursive: true });

const engineDist = join(repo, 'packages', 'engine', 'dist');
mkdirSync(join(bench, 'vendor', 'engine'), { recursive: true });
for (const file of readdirSync(engineDist)) {
  if (file.endsWith('.js')) cpSync(join(engineDist, file), join(bench, 'vendor', 'engine', file));
}

// The engine's GPU check page (packages/engine-webgpu/test/gpu), with the engine's compiled modules and the goldens it
// compares against, laid out as in the package so the page's relative paths are unchanged. Open it at
// gpu-check/test/gpu/ and pick a model file. The owner's CUDA goldens are a record only, so they are left out.
const webgpu = join(repo, 'packages', 'engine-webgpu');
const gpuCheck = join(site, 'gpu-check');
const copyJs = (from, to) => {
  for (const entry of readdirSync(from, { withFileTypes: true })) {
    if (entry.isDirectory()) copyJs(join(from, entry.name), join(to, entry.name));
    else if (entry.name.endsWith('.js')) {
      mkdirSync(to, { recursive: true });
      cpSync(join(from, entry.name), join(to, entry.name));
    }
  }
};
copyJs(join(webgpu, 'dist'), join(gpuCheck, 'dist'));
mkdirSync(join(gpuCheck, 'test', 'gpu'), { recursive: true });
for (const file of ['index.html', 'check.js']) cpSync(join(webgpu, 'test', 'gpu', file), join(gpuCheck, 'test', 'gpu', file));
mkdirSync(join(gpuCheck, 'test', 'goldens'), { recursive: true });
for (const file of readdirSync(join(webgpu, 'test', 'goldens'))) {
  if (file.endsWith('.json') && !file.endsWith('-cuda.json')) cpSync(join(webgpu, 'test', 'goldens', file), join(gpuCheck, 'test', 'goldens', file));
}
writeFileSync(
  join(gpuCheck, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>StoneMind GPU check</title>' +
    '<meta http-equiv="refresh" content="0; url=test/gpu/"><a href="test/gpu/">StoneMind GPU check</a>\n',
);

// The onnxruntime-web baseline (ort-bench/): a developer page that runs KataGo's ONNX graph under onnxruntime-web, the
// approved benchmark-only outside code (docs/spikes/0001-browser-katago-webgpu.md, "Approved outside code"). It uses
// the GPU check's copy of the engine for encoding and post-processing. Only the two builds it loads are copied: the
// WebGPU build and the WASM build, each with its runtime.
const ortBench = join(site, 'ort-bench');
mkdirSync(join(ortBench, 'vendor'), { recursive: true });
for (const file of ['index.html', 'ort-bench.js']) cpSync(join(app, 'ort-bench', file), join(ortBench, file));
const ortDist = join(repo, 'node_modules', 'onnxruntime-web', 'dist');
for (const file of [
  'ort.webgpu.min.mjs',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort.wasm.min.mjs',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
]) {
  cpSync(join(ortDist, file), join(ortBench, 'vendor', file));
}
cpSync(join(app, 'ort-bench', 'ONNXRUNTIME-LICENSE.txt'), join(ortBench, 'vendor', 'LICENSE.txt'));

// The site root: GitHub Pages serves the repository's site at /StoneMind/, so point it at the bench.
writeFileSync(
  join(site, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>StoneMind</title>' +
    '<meta http-equiv="refresh" content="0; url=engine-bench/"><a href="engine-bench/">StoneMind engine bench</a>\n',
);
writeFileSync(join(site, '.nojekyll'), '');
console.log(`assembled ${site}`);
