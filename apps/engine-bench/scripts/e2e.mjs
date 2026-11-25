// End-to-end check, run by hand (Playwright is not a project dependency): start the bridge with the fake KataGo from
// the bridge's tests, serve the site, open the pairing link in Chromium, analyse a position, and check the results.
//
//   npm run build && node apps/engine-bench/scripts/e2e.mjs
//
// It needs a Playwright with Chromium that Node can import (for example one installed outside the repository).
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { startBridge } from '../../../packages/katago-bridge/dist/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, '..', '..', '..');
const require = createRequire(join(process.cwd(), 'noop.js'));
const { chromium } = require('playwright');

const server = spawn(process.execPath, [join(here, 'serve.mjs')], { env: { ...process.env, PORT: '8080' }, stdio: 'inherit' });
await new Promise((r) => setTimeout(r, 500));

const bridge = await startBridge({
  command: process.execPath,
  args: [join(repo, 'packages', 'katago-bridge', 'test', 'fake-katago.mjs')],
  port: 0,
  pageUrl: 'http://localhost:8080/engine-bench/',
  log: (m) => console.log(`[bridge] ${m}`),
});

const browser = await chromium.launch();
let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`[page] ${m.text()}`));
  await page.goto(bridge.pairingLink);
  await page.waitForSelector('#status[data-state="connected"]', { timeout: 10000 });
  const after = new URL(page.url());
  if (after.hash) throw new Error('the token stayed in the address bar');
  console.log('connected; version shown:', await page.textContent('#info-version'));
  await page.selectOption('#position', 'opening');
  await page.fill('#max-visits', '120');
  await page.click('#run');
  await page.waitForFunction(() => document.querySelectorAll('#moves tr').length > 0 && document.querySelector('#status')?.getAttribute('data-state') === 'connected');
  console.log('visits:', await page.textContent('#visits'), 'top move:', await page.textContent('#moves tr td'));
  await page.screenshot({ path: process.env.SCREENSHOT ?? join(repo, 'apps', 'engine-bench', 'dist', 'e2e.png'), fullPage: true });
  console.log('e2e passed');
} catch (e) {
  failed = true;
  console.error('e2e failed:', e);
} finally {
  await browser.close();
  await bridge.close();
  server.kill();
  process.exitCode = failed ? 1 : 0;
}
