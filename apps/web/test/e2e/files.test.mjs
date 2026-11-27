// The Files app in real browsers, with Playwright (build plan, Milestone 1, step 4).
//
//   npm run e2e -w @stonemind/web                      # Chromium
//   BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/web   # as CI runs it
import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import * as playwright from 'playwright';
import { serve } from '../../scripts/server.mjs';

const BROWSERS = (process.env.BROWSERS ?? 'chromium').split(',').map((s) => s.trim()).filter(Boolean);

// A small stand-in for a KataGo network: the name and model version a real one starts with, then filler.
const NAME = 'kata1-b18c384nbt-s9996604416-d4316597426';
const dir = mkdtempSync(join(tmpdir(), 'stonemind-files-'));
const weights = join(dir, `${NAME}.bin.gz`);
writeFileSync(weights, gzipSync(Buffer.concat([Buffer.from(`${NAME}\n15\n`), Buffer.alloc(300_000, 7)])));
const game = join(dir, 'game.sgf');
writeFileSync(game, '(;GM[1]SZ[19];B[pd];W[dp])');

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

    const openFiles = async (context) => {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(url);
      await page.getByRole('button', { name: 'Apps', exact: true }).click();
      await page.locator('#bar-apps').getByRole('button', { name: 'Files' }).click();
      await page.getByRole('toolbar', { name: 'Files' }).waitFor();
      return { page, errors };
    };
    const sheet = (page) => page.locator('.sheet[data-app="files"]');
    const importFile = async (page, path) => {
      const chooser = page.waitForEvent('filechooser');
      await sheet(page).getByRole('button', { name: 'Import…' }).click();
      await (await chooser).setFiles(path);
    };
    const row = (page, label) => sheet(page).getByRole('button', { name: label, exact: true });

    test('the library folders are there, and Weights opens first', async () => {
      const context = await browser.newContext();
      const { page, errors } = await openFiles(context);
      const library = sheet(page).getByRole('navigation', { name: 'Library' });
      for (const f of ['Games', 'Problems', 'Weights', 'Configs']) await library.getByRole('button', { name: f }).waitFor();
      assert.equal(await library.getByRole('button', { name: 'Weights' }).getAttribute('aria-current'), 'true');
      assert.match(await sheet(page).textContent(), /This folder is empty/);
      assert.equal(await page.locator('.sm-menubar-app').textContent(), 'Files');
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('importing weights shows the network, its training and the model version', async () => {
      const context = await browser.newContext();
      const { page, errors } = await openFiles(context);
      await importFile(page, weights);
      await row(page, `File ${NAME}.bin.gz`).click();
      const details = sheet(page).getByRole('complementary', { name: 'Details' });
      await details.getByText('b18c384nbt: 18 blocks, 384 channels, nested bottleneck').waitFor();
      const text = async () => (await details.textContent()).replace(/ | /g, ' ');
      await details.getByText('Model version').waitFor();
      await page.waitForFunction(() => /Model version15/.test(document.querySelector('.files-inspector')?.textContent ?? ''));
      assert.match(await text(), /Training runkata1/);
      assert.match(await text(), /Training samples \(s\)9.996.604.416/);
      assert.match(await text(), /Training data rows \(d\)4.316.597.426/);
      assert.match(await sheet(page).locator('tbody').textContent(), /b18c384nbt/);
      // The checksum is the file's own SHA-256.
      await details.getByRole('button', { name: 'Compute' }).click();
      const { createHash } = await import('node:crypto');
      const expected = createHash('sha256').update(readFileSync(weights)).digest('hex');
      await details.getByText(expected).waitFor();
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('a new folder, a rename, a download and a confirmed delete', async () => {
      const context = await browser.newContext({ acceptDownloads: true });
      const { page, errors } = await openFiles(context);
      await sheet(page).getByRole('navigation', { name: 'Library' }).getByRole('button', { name: 'Games' }).click();
      await importFile(page, game);
      await row(page, 'File game.sgf').waitFor();

      await sheet(page).getByRole('button', { name: 'New folder…' }).click();
      await page.getByRole('dialog').getByLabel('Name').fill('Club games');
      await page.getByRole('dialog').getByRole('button', { name: 'Create' }).click();
      await row(page, 'Folder Club games').waitFor();

      await row(page, 'File game.sgf').click();
      await sheet(page).getByRole('button', { name: 'Rename…' }).click();
      await page.getByRole('dialog').getByLabel('New name').fill('first.sgf');
      await page.getByRole('dialog').getByRole('button', { name: 'Rename' }).click();
      await row(page, 'File first.sgf').waitFor();
      assert.equal(await row(page, 'File game.sgf').count(), 0);

      const download = page.waitForEvent('download');
      await sheet(page).getByRole('button', { name: 'Download' }).click();
      const saved = await download;
      assert.equal(saved.suggestedFilename(), 'first.sgf');
      assert.equal(readFileSync(await saved.path(), 'utf8'), readFileSync(game, 'utf8'));

      // Cancel keeps the file; Delete removes it.
      await sheet(page).getByRole('button', { name: 'Delete…' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Cancel' }).click();
      await row(page, 'File first.sgf').waitFor();
      await sheet(page).getByRole('button', { name: 'Delete…' }).click();
      assert.match(await page.getByRole('dialog').textContent(), /Delete first\.sgf\?/);
      await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click();
      await row(page, 'File first.sgf').waitFor({ state: 'detached' });

      // Into the new folder and back by the path.
      await row(page, 'Folder Club games').dblclick();
      await sheet(page).getByRole('navigation', { name: 'Folder' }).getByRole('button', { name: 'Club games' }).waitFor();
      await sheet(page).getByRole('navigation', { name: 'Folder' }).getByRole('button', { name: 'Games', exact: true }).click();
      await row(page, 'Folder Club games').waitFor();
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('columns sort, and the keyboard alone makes, opens and leaves a folder', async () => {
      const context = await browser.newContext();
      const { page, errors } = await openFiles(context);
      await sheet(page).getByRole('navigation', { name: 'Library' }).getByRole('button', { name: 'Problems' }).click();
      const big = join(dir, 'big.sgf');
      writeFileSync(big, `(;GM[1]SZ[19]C[${'x'.repeat(5000)}])`);
      await importFile(page, [big, game]);
      await row(page, 'File game.sgf').waitFor();
      const names = () => sheet(page).locator('tbody .files-name').allTextContents();
      assert.deepEqual(await names(), ['big.sgf', 'game.sgf']);
      await sheet(page).getByRole('button', { name: 'Size' }).click();
      assert.equal(await sheet(page).locator('th', { hasText: 'Size' }).getAttribute('aria-sort'), 'descending');
      assert.deepEqual(await names(), ['big.sgf', 'game.sgf']);
      await sheet(page).getByRole('button', { name: /^Size/ }).click();
      assert.equal(await sheet(page).locator('th', { hasText: 'Size' }).getAttribute('aria-sort'), 'ascending');
      assert.deepEqual(await names(), ['game.sgf', 'big.sgf']);

      // New folder: Enter on the button, type, Enter submits.
      await sheet(page).getByRole('button', { name: 'New folder…' }).focus();
      await page.keyboard.press('Enter');
      await page.getByRole('dialog').getByLabel('Name').waitFor();
      await page.keyboard.type('Tsumego');
      await page.keyboard.press('Enter');
      await row(page, 'Folder Tsumego').waitFor();
      // Escape in a dialog closes the dialog, not Files.
      await sheet(page).getByRole('button', { name: 'New folder…' }).focus();
      await page.keyboard.press('Enter');
      await page.getByRole('dialog').waitFor();
      await page.keyboard.press('Escape');
      await page.getByRole('dialog').waitFor({ state: 'detached' });
      assert.equal(await sheet(page).count(), 1);
      // Enter on a folder opens it; the path leads back.
      await row(page, 'Folder Tsumego').focus();
      await page.keyboard.press('Enter');
      await sheet(page).getByRole('navigation', { name: 'Folder' }).getByRole('button', { name: 'Tsumego' }).waitFor();
      assert.match(await sheet(page).textContent(), /This folder is empty/);
      await sheet(page).getByRole('navigation', { name: 'Folder' }).getByRole('button', { name: 'Problems' }).focus();
      await page.keyboard.press('Enter');
      await row(page, 'File big.sgf').waitFor();
      // Escape outside a dialog closes Files.
      await row(page, 'File big.sgf').focus();
      await page.keyboard.press('Escape');
      await sheet(page).waitFor({ state: 'detached' });
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('a file imported in one tab appears in the other', async () => {
      const context = await browser.newContext();
      const a = await openFiles(context);
      const b = await openFiles(context);
      await importFile(a.page, game);
      await row(b.page, 'File game.sgf').waitFor();
      assert.deepEqual([...a.errors, ...b.errors], []);
      await context.close();
    });

    test('the sidebar shows this browser’s storage, and the menu bar has the File menu', async () => {
      const context = await browser.newContext();
      const { page, errors } = await openFiles(context);
      await sheet(page).getByRole('meter', { name: 'Storage used' }).waitFor();
      assert.match(await sheet(page).locator('.files-storage').textContent(), /used of|not known/);
      await sheet(page).getByRole('button', { name: 'Import…' }).focus();
      await page.locator('sm-menu-bar').getByRole('button', { name: 'File', exact: true }).click();
      const menu = page.getByRole('menu');
      for (const item of ['Import…', 'New folder…', 'Download', 'Rename…', 'Delete…']) await menu.getByRole('menuitem', { name: item }).waitFor();
      assert.deepEqual(errors, []);
      await context.close();
    });
  });
}
