// Assemble the static site in dist/site: the page, its compiled modules, and verbatim copies of the brand files and
// the engine package. The page loads plain ES modules through an import map; there is no bundler.
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

// The site root: GitHub Pages serves the repository's site at /StoneMind/, so point it at the bench.
writeFileSync(
  join(site, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>StoneMind</title>' +
    '<meta http-equiv="refresh" content="0; url=engine-bench/"><a href="engine-bench/">StoneMind engine bench</a>\n',
);
writeFileSync(join(site, '.nojekyll'), '');
console.log(`assembled ${site}`);
