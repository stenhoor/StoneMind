// Assemble the app in dist/site: the app page, the capability page, the stylesheet, the compiled modules, the file kit's
// modules, and verbatim copies of the brand files. The page loads plain ES modules; there is no bundler.
import { cpSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const app = join(dirname(fileURLToPath(import.meta.url)), '..');
const repo = join(app, '..', '..');
const site = join(app, 'dist', 'site');

rmSync(site, { recursive: true, force: true });
mkdirSync(site, { recursive: true });
cpSync(join(app, 'src', 'index.html'), join(site, 'index.html'));
cpSync(join(app, 'src', 'app.css'), join(site, 'app.css'));
cpSync(join(app, 'dist', 'js'), join(site, 'js'), { recursive: true });
// The capability page (build plan, step 2) and its worker probe, a classic script copied as is.
mkdirSync(join(site, 'capabilities'), { recursive: true });
cpSync(join(app, 'src', 'capabilities', 'index.html'), join(site, 'capabilities', 'index.html'));
cpSync(join(app, 'src', 'capabilities', 'worker-probe.js'), join(site, 'js', 'capabilities', 'worker-probe.js'));

// The file kit's compiled modules, which the page's import map names (its sync writer is a worker beside them).
const files = join(repo, 'packages', 'files', 'dist');
mkdirSync(join(site, 'vendor', 'files'), { recursive: true });
for (const f of readdirSync(files)) if (f.endsWith('.js')) cpSync(join(files, f), join(site, 'vendor', 'files', f));

const brand = join(repo, 'packages', 'brand');
cpSync(join(brand, 'tokens', 'tokens.css'), join(site, 'brand', 'tokens.css'));
cpSync(join(brand, 'styles', 'components.css'), join(site, 'brand', 'components.css'));
cpSync(join(brand, 'logos'), join(site, 'brand', 'logos'), { recursive: true });
console.log(`assembled ${site}`);
