// Assemble the app in dist/site: the page, its stylesheet, its compiled modules, and verbatim copies of the brand
// files. The page loads plain ES modules; there is no bundler.
import { cpSync, mkdirSync, rmSync } from 'node:fs';
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

const brand = join(repo, 'packages', 'brand');
cpSync(join(brand, 'tokens', 'tokens.css'), join(site, 'brand', 'tokens.css'));
cpSync(join(brand, 'styles', 'components.css'), join(site, 'brand', 'components.css'));
cpSync(join(brand, 'logos'), join(site, 'brand', 'logos'), { recursive: true });
console.log(`assembled ${site}`);
