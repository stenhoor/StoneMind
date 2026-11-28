#!/usr/bin/env node
// Which sub-projects a change affects, for CI: the workspaces whose files changed, and every workspace that depends on
// one of them (directly or not). Node built-ins only.
//
//   node scripts/ci-affected.mjs <base> [<head>]   (default head: HEAD)
//
// Prints JSON: { all, packages, gpu, e2e, files }.
// - all: true when a shared file changed (the root package.json, the TypeScript base config, .nvmrc), when CI's own files
//   changed and no workspace did, or when the base is unknown. Every workspace is then affected.
// - packages: the affected workspaces' names, in the root `workspaces` order (dependency order).
// - gpu: whether @stonemind/engine-webgpu's GPU check is needed: everything is affected, or a file changed that only the
//   GPU check tests (GPU_ONLY). The check (npm run gpu-check, as CI runs it) runs the WebGPU network on the goldens
//   positions and compares it with the reference forward pass and native's outputs. The node tests already compare the
//   loader, the reference pass, the input encoder, the rules and the search with native, so the GPU check is needed for
//   what they don't run: the WebGPU network and the code it uses (src/gpu, src/model, src/reference), the page and its
//   script, its test network and goldens, and the package's build settings. The names the page imports from the
//   package's exports are checked by a node test (test/gpu-page-imports.test.mjs), so src/index.ts isn't in the list.
// - e2e: the affected workspaces that have an `e2e` script (browser tests), where it, or a workspace it depends on,
//   changed in a file other than Markdown.
// A change to package-lock.json that touches only our own workspaces' entries (a workspace adding another as a
// dependency, say) counts as a change to those workspaces; any other lockfile change, such as an outside package's
// version, still affects everything.
// Files under docs/ and Markdown files at the root affect nothing. Inside a workspace every file counts, Markdown
// included (some tests read their package's Markdown).
// With GITHUB_OUTPUT set, also writes `any`, `packages` (space-separated) and `gpu` there.
import { execFileSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = join(import.meta.dirname, '..');
const [base, head = 'HEAD'] = process.argv.slice(2);
const GPU_PACKAGE = '@stonemind/engine-webgpu';
const GPU_DIR = 'packages/engine-webgpu/';
const GPU_ONLY = [
  /^src\/(gpu|model|reference)\//, /^test\/gpu\//, /^scripts\/gpu-check\.mjs$/, /^test\/fixtures\//,
  /^test\/goldens\/(positions|g170-b6c96-s175395328-d26788732)\.json$/, /^package\.json$/, /^tsconfig[^/]*\.json$/,
];
const LOCKFILE = 'package-lock.json';
const SHARED = [/^package\.json$/, /^tsconfig\.base\.json$/, /^\.nvmrc$/];
// CI's own files. Changed alongside workspaces, they are tested by that change's run; changed alone, everything runs.
const CI = [/^\.github\//, /^scripts\/ci-affected\.mjs$/];

const rootPkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const workspaces = rootPkg.workspaces.map((dir) => {
  const pkg = JSON.parse(readFileSync(join(root, dir, 'package.json'), 'utf8'));
  const deps = Object.keys({ ...pkg.dependencies, ...pkg.devDependencies, ...pkg.peerDependencies, ...pkg.optionalDependencies });
  return { dir, name: pkg.name, deps, e2e: Boolean(pkg.scripts?.e2e) };
});
const names = new Set(workspaces.map((w) => w.name));

const unknownBase = !base || /^0+$/.test(base);
let files = [];
if (!unknownBase) {
  files = execFileSync('git', ['diff', '--name-only', `${base}...${head}`], { cwd: root, encoding: 'utf8' }).split('\n').filter(Boolean);
}

// The workspaces whose lockfile entries changed, or null if anything else in the lockfile changed.
function lockfileWorkspaces() {
  const read = (rev) => {
    try {
      return JSON.parse(execFileSync('git', ['show', `${rev}:${LOCKFILE}`], { cwd: root, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 }));
    } catch {
      return null;
    }
  };
  const before = read(base);
  const after = read(head);
  if (!before?.packages || !after?.packages) return null;
  const { packages: a, ...restA } = before;
  const { packages: b, ...restB } = after;
  if (JSON.stringify(restA) !== JSON.stringify(restB)) return null;
  const touched = new Set();
  for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
    if (JSON.stringify(a[key]) === JSON.stringify(b[key])) continue;
    // A workspace's own entry ("apps/web"), or its link in node_modules ("node_modules/@stonemind/web").
    const w = workspaces.find((w) => key === w.dir || key === `node_modules/${w.name}`);
    if (!w) return null;
    touched.add(w.name);
  }
  return touched;
}

let lockfile = new Set();
let all = unknownBase || files.some((f) => SHARED.some((re) => re.test(f)));
if (!all && files.includes(LOCKFILE)) {
  const touched = lockfileWorkspaces();
  if (touched) lockfile = touched;
  else all = true;
}

// Workspaces changed directly, and whether a change in each was more than Markdown.
const changed = new Map();
for (const f of files) {
  const w = workspaces.find((w) => f.startsWith(w.dir + '/'));
  if (!w) continue;
  changed.set(w.name, (changed.get(w.name) ?? false) || !f.endsWith('.md'));
}

for (const name of lockfile) changed.set(name, true);
if (!all && changed.size === 0 && files.some((f) => CI.some((re) => re.test(f)))) all = true;

// Close over dependents: a workspace is affected if it changed or depends on an affected one.
const affected = new Map(all ? workspaces.map((w) => [w.name, true]) : changed);
for (let grew = true; grew; ) {
  grew = false;
  for (const w of workspaces) {
    for (const d of w.deps) {
      if (!names.has(d) || !affected.has(d)) continue;
      const code = affected.get(d) || (affected.get(w.name) ?? false);
      if (!affected.has(w.name) || affected.get(w.name) !== code) {
        affected.set(w.name, code);
        grew = true;
      }
    }
  }
}

const packages = workspaces.map((w) => w.name).filter((n) => affected.has(n));
const gpu = all || files.some((f) => f.startsWith(GPU_DIR) && GPU_ONLY.some((re) => re.test(f.slice(GPU_DIR.length))));
const e2e = workspaces.filter((w) => w.e2e && affected.get(w.name) === true).map((w) => w.name);
console.log(JSON.stringify({ all, packages, gpu, e2e, files }, null, 2));
if (process.env.GITHUB_OUTPUT) {
  appendFileSync(process.env.GITHUB_OUTPUT, `any=${packages.length > 0}\npackages=${packages.join(' ')}\ngpu=${gpu}\ne2e=${e2e.join(' ')}\n`);
}
