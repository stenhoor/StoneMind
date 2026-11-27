#!/usr/bin/env node
// Which sub-projects a change affects, for CI: the workspaces whose files changed, and every workspace that depends on
// one of them (directly or not). Node built-ins only.
//
//   node scripts/ci-affected.mjs <base> [<head>]   (default head: HEAD)
//
// Prints JSON: { all, packages, gpu, e2e, files }.
// - all: true when a shared file changed (root package files, the TypeScript base config, .nvmrc, CI itself) or the base
//   is unknown. Every workspace is then affected.
// - packages: the affected workspaces' names, in the root `workspaces` order (dependency order).
// - gpu: whether @stonemind/engine-webgpu's GPU check is needed, i.e. it, or a workspace it depends on, changed in a
//   file other than Markdown.
// - e2e: the affected workspaces that have an `e2e` script (browser tests), where it, or a workspace it depends on,
//   changed in a file other than Markdown.
// Files under docs/ and Markdown files at the root affect nothing. Inside a workspace every file counts, Markdown
// included (some tests read their package's Markdown).
// With GITHUB_OUTPUT set, also writes `any`, `packages` (space-separated) and `gpu` there.
import { execFileSync } from 'node:child_process';
import { appendFileSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = join(import.meta.dirname, '..');
const [base, head = 'HEAD'] = process.argv.slice(2);
const GPU_PACKAGE = '@stonemind/engine-webgpu';
const SHARED = [/^package\.json$/, /^package-lock\.json$/, /^tsconfig\.base\.json$/, /^\.nvmrc$/, /^\.github\//, /^scripts\/ci-affected\.mjs$/];

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
const all = unknownBase || files.some((f) => SHARED.some((re) => re.test(f)));

// Workspaces changed directly, and whether a change in each was more than Markdown.
const changed = new Map();
for (const f of files) {
  const w = workspaces.find((w) => f.startsWith(w.dir + '/'));
  if (!w) continue;
  changed.set(w.name, (changed.get(w.name) ?? false) || !f.endsWith('.md'));
}

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
const gpu = affected.get(GPU_PACKAGE) === true;
const e2e = workspaces.filter((w) => w.e2e && affected.get(w.name) === true).map((w) => w.name);
console.log(JSON.stringify({ all, packages, gpu, e2e, files }, null, 2));
if (process.env.GITHUB_OUTPUT) {
  appendFileSync(process.env.GITHUB_OUTPUT, `any=${packages.length > 0}\npackages=${packages.join(' ')}\ngpu=${gpu}\ne2e=${e2e.join(' ')}\n`);
}
