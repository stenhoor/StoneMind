// The capability page in real browsers (build plan, Milestone 1, step 2). Each browser's report is saved to
// dist/reports/<browser>.json and its summary printed, so a CI run keeps what each engine supports.
//
//   npm run e2e -w @stonemind/web          (BROWSERS and STONEMIND_CHROMIUM as for frame.test.mjs)
import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import * as playwright from 'playwright';
import { SITE, serve } from '../../scripts/server.mjs';

const BROWSERS = (process.env.BROWSERS ?? 'chromium').split(',').map((s) => s.trim()).filter(Boolean);
const REPORTS = join(SITE, '..', 'reports');

let server;
let url;
before(async () => ({ server, url } = await serve()));
after(() => server?.close());

for (const name of BROWSERS) {
  describe(name, () => {
    let browser;
    before(async () => {
      const executablePath = name === 'chromium' ? process.env.STONEMIND_CHROMIUM : undefined;
      browser = await playwright[name].launch(executablePath ? { executablePath } : {});
    });
    after(() => browser?.close());

    test('the capability page checks everything and offers the report', async () => {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(`${url}capabilities/`);
      await page.waitForSelector('body[data-ready="true"]', { timeout: 60_000 });
      const report = JSON.parse(await page.locator('#json').textContent());
      assert.equal(report.kind, 'stonemind-capabilities');
      for (const key of ['webgpu', 'dedicatedWorker', 'sharedWorker', 'opfs', 'storage', 'webLocks', 'broadcastChannel', 'css']) {
        assert.ok(report[key], `report has ${key}`);
      }
      // Every probe finished: booleans, not hangs. A worker that could not start says why.
      assert.equal(typeof report.webgpu.api, 'boolean');
      assert.ok(report.dedicatedWorker.started || report.dedicatedWorker.error, 'dedicated worker started or said why not');
      assert.ok(await page.getByRole('row').count() >= 15);
      assert.deepEqual(errors, []);

      mkdirSync(REPORTS, { recursive: true });
      writeFileSync(join(REPORTS, `${name}.json`), JSON.stringify(report, null, 2) + '\n');
      const rows = await page.$$eval('#results tr', (trs) => trs.map((tr) => [...tr.children].map((c) => c.textContent).join(' | ')));
      console.log(`\n${name} (${report.userAgent})\n  ${rows.join('\n  ')}`);
      await page.close();
    });

    test('the StoneMind menu opens the capability page (Browser Info), and About no longer links to it', async () => {
      const page = await browser.newPage();
      await page.goto(url);
      await page.getByRole('button', { name: 'StoneMind menu' }).click();
      await page.getByRole('menuitem', { name: 'Browser Info' }).click();
      await page.waitForURL(/capabilities\/$/);
      await page.goto(url);
      await page.getByRole('button', { name: 'StoneMind menu' }).click();
      await page.getByRole('menuitem', { name: 'About StoneMind' }).click();
      assert.equal(await page.locator('.sheet a[href="capabilities/"]').count(), 0);
      await page.close();
    });
  });
}
