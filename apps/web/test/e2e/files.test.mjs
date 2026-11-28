// The Files app in real browsers, with Playwright (build plan, Milestone 1, step 4).
//
//   npm run e2e -w @stonemind/web                      # Chromium
//   BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/web   # as CI runs it
import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import * as playwright from 'playwright';
import { serve } from '../../scripts/server.mjs';

const BROWSERS = (process.env.BROWSERS ?? 'chromium').split(',').map((s) => s.trim()).filter(Boolean);

// Small stand-ins for each kind of file.
const NET = 'kata1-b18c384nbt-s9996604416-d4316597426';
const dir = mkdtempSync(join(tmpdir(), 'stonemind-files-'));
const file = (name, data) => {
  writeFileSync(join(dir, name), data);
  return join(dir, name);
};
const weights = file(`${NET}.bin.gz`, gzipSync(Buffer.concat([Buffer.from(`${NET}\n15\n`), Buffer.alloc(300_000, 7)])));
const game = file('final.sgf', '(;GM[1]SZ[19]PB[Lee]PW[Cho]RE[B+R];B[pd];W[dp])');
const big = file('long.sgf', `(;GM[1]SZ[19]PB[A]PW[B]C[${'x'.repeat(5000)}])`);
const problem = file('corner.sgf', '(;GM[1]SZ[19]AB[pd][qc]AW[qd][rd]C[Black to live];B[rc])');
const config = file('gtp.cfg', '# KataGo\nlogDir = gtp_logs\nmaxVisits = 500\nnumSearchThreads = 6\n');
const notes = file('notes.txt', 'hello');

let server;
let url;
before(async () => ({ server, url } = await serve()));
after(() => server?.close());

for (const name of BROWSERS) {
  describe(name, () => {
    // Each test gets a fresh, persistent profile, as a player's is. Playwright's default contexts are ephemeral, and
    // WebKit (like Safari in a private window) has no OPFS in an ephemeral one (microsoft/playwright#18235).
    const newContext = (options = {}) => {
      const executablePath = name === 'chromium' ? process.env.STONEMIND_CHROMIUM : undefined;
      return playwright[name].launchPersistentContext(mkdtempSync(join(tmpdir(), 'stonemind-web-')), { ...options, ...(executablePath ? { executablePath } : {}) });
    };

    const openFiles = async (context, before) => {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(url);
      if (before) await page.evaluate(before);
      // The Apps bar shows at startup.
      await page.locator('#bar-apps').getByRole('button', { name: 'Files' }).click();
      await page.getByRole('toolbar', { name: 'Files' }).waitFor();
      return { page, errors };
    };
    const sheet = (page) => page.locator('.sheet[data-app="files"]');
    const importFiles = async (page, paths) => {
      const chooser = page.waitForEvent('filechooser');
      await sheet(page).getByRole('button', { name: 'Import…' }).click();
      await (await chooser).setFiles(paths);
    };
    const row = (page, label) => sheet(page).getByRole('button', { name: label, exact: true });
    const kinds = (page) => sheet(page).getByRole('navigation', { name: 'Kinds' });
    const kind = (page, plural) => kinds(page).getByRole('button', { name: new RegExp(`^${plural}, \\d+$`) });
    const count = async (page, plural) => Number((await kind(page, plural).getAttribute('aria-label')).split(', ')[1]);
    const status = (page) => sheet(page).getByRole('status').first();

    test('kinds and folders are separate: kinds list files, folders hold them', async () => {
      const context = await newContext();
      // An earlier version made empty folders named after the kinds; they are removed. One with a file in it stays.
      const { page, errors } = await openFiles(context, async () => {
        const root = await navigator.storage.getDirectory();
        for (const d of ['games', 'problems', 'weights', 'configs']) await root.getDirectoryHandle(d, { create: true });
        const w = await (await (await root.getDirectoryHandle('games')).getFileHandle('kept.sgf', { create: true })).createWritable();
        await w.write('(;GM[1]PB[a]PW[b])');
        await w.close();
      });
      for (const k of ['Games', 'Problems', 'Weights', 'Configs']) await kind(page, k).waitFor();
      const folders = sheet(page).getByRole('navigation', { name: 'Folders' });
      await folders.getByRole('button', { name: 'This browser' }).waitFor();
      assert.equal(await folders.getByRole('button', { name: /problems|weights|configs/ }).count(), 0);
      await row(page, 'Folder games').waitFor();
      // The file the player kept there is worked out as a game.
      assert.equal(await count(page, 'Games'), 1);
      assert.equal(await page.locator('.sm-menubar-app').textContent(), 'Files');
      assert.equal(await page.locator('.desk-empty').isVisible(), false);
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('each file’s kind comes from its name and content, not from what is shown; unknown files are refused', async () => {
      const context = await newContext();
      const { page, errors } = await openFiles(context);
      // Weights is shown when a game, a problem, a network, a config and a text file are imported.
      await kind(page, 'Weights').click();
      await importFiles(page, [game, problem, weights, config, notes]);
      // The refusal shows at once; the imported files are listed once they are written.
      await status(page).getByText(/Not imported: notes\.txt is not a kind of file Files takes/).waitFor();
      await status(page).getByText(/Imported gtp\.cfg/).waitFor();
      const text = await status(page).textContent();
      assert.match(text, /Imported final\.sgf as a game into This browser\. It names the Black player \(PB\)/);
      assert.match(text, /Imported corner\.sgf as a problem into This browser\. It sets up stones/);
      assert.match(text, /Imported gtp\.cfg as a KataGo config/);
      assert.equal(await count(page, 'Games'), 1);
      assert.equal(await count(page, 'Problems'), 1);
      assert.equal(await count(page, 'Weights'), 1);
      assert.equal(await count(page, 'Configs'), 1);
      // Weights lists only the network.
      assert.deepEqual(await sheet(page).locator('tbody .files-name').allTextContents(), [`${NET}.bin.gz`]);
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('a network shows its size, training and model version, and its checksum on request', async () => {
      const context = await newContext();
      const { page, errors } = await openFiles(context);
      await importFiles(page, [weights]);
      await kind(page, 'Weights').click();
      await row(page, `File ${NET}.bin.gz`).click();
      const details = sheet(page).getByRole('complementary', { name: 'Details' });
      await details.getByText('b18c384nbt: 18 blocks, 384 channels, nested bottleneck').waitFor();
      await page.waitForFunction(() => /Model version15/.test(document.querySelector('.files-inspector')?.textContent ?? ''));
      const text = (await details.textContent()).replace(/ | /g, ' ');
      assert.match(text, /KindKataGo network/);
      assert.match(text, /Training samples \(s\)9.996.604.416/);
      assert.match(text, /Training data rows \(d\)4.316.597.426/);
      assert.match(await sheet(page).locator('tbody').textContent(), /b18c384nbt/);
      await details.getByRole('button', { name: 'Compute' }).click();
      await details.getByText(createHash('sha256').update(readFileSync(weights)).digest('hex')).waitFor();
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('the player can change a game to a problem, and the choice lasts', async () => {
      const context = await newContext();
      const { page, errors } = await openFiles(context);
      await importFiles(page, [game]);
      await kind(page, 'Games').click();
      await row(page, 'File final.sgf').click();
      await sheet(page).getByRole('group', { name: 'Kind' }).getByRole('button', { name: 'Problem' }).click();
      await page.waitForFunction(() => document.querySelector('[aria-label^="Problems, "]')?.getAttribute('aria-label') === 'Problems, 1');
      assert.equal(await count(page, 'Games'), 0);
      await kind(page, 'Problems').click();
      await row(page, 'File final.sgf').click();
      await sheet(page).getByText('Chosen by you.').waitFor();
      await page.reload();
      await page.locator('#bar-apps').getByRole('button', { name: 'Files' }).click();
      await kind(page, 'Problems').waitFor();
      assert.equal(await count(page, 'Problems'), 1);
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('folders: new folder, import into it, move into it, rename, download and a confirmed delete', async () => {
      const context = await newContext({ acceptDownloads: true });
      const { page, errors } = await openFiles(context);
      await sheet(page).getByRole('button', { name: 'New folder…' }).click();
      await page.getByRole('dialog').getByLabel('Name').fill('Club');
      await page.getByRole('dialog').getByRole('button', { name: 'Create' }).click();
      await row(page, 'Folder Club').waitFor();
      await sheet(page).getByRole('navigation', { name: 'Folders' }).getByRole('button', { name: 'Club' }).click();
      await importFiles(page, [problem]);
      await row(page, 'File corner.sgf').waitFor();
      assert.match(await sheet(page).locator('tbody').textContent(), /Problem/);

      // A file in This browser moved into Club.
      await sheet(page).getByRole('navigation', { name: 'Folder', exact: true }).getByRole('button', { name: 'This browser' }).click();
      await importFiles(page, [game]);
      await row(page, 'File final.sgf').click();
      await sheet(page).getByRole('button', { name: 'Move to…' }).click();
      await page.getByRole('dialog').getByLabel('To folder').selectOption('/Club');
      await page.getByRole('dialog').getByRole('button', { name: 'Move' }).click();
      await row(page, 'File final.sgf').waitFor({ state: 'detached' });
      // Still a game, now in Club.
      await kind(page, 'Games').click();
      await row(page, 'File final.sgf').waitFor();
      assert.match(await sheet(page).locator('tbody').textContent(), /\/Club/);

      await row(page, 'File final.sgf').click();
      await sheet(page).getByRole('button', { name: 'Rename…' }).click();
      await page.getByRole('dialog').getByLabel('New name').fill('first.sgf');
      await page.getByRole('dialog').getByRole('button', { name: 'Rename' }).click();
      await row(page, 'File first.sgf').waitFor();
      assert.equal(await count(page, 'Games'), 1);

      const download = page.waitForEvent('download');
      await sheet(page).getByRole('button', { name: 'Download' }).click();
      const saved = await download;
      assert.equal(saved.suggestedFilename(), 'first.sgf');
      assert.equal(readFileSync(await saved.path(), 'utf8'), readFileSync(game, 'utf8'));

      await sheet(page).getByRole('button', { name: 'Delete…' }).click();
      await page.getByRole('dialog').getByRole('button', { name: 'Cancel' }).click();
      await row(page, 'File first.sgf').waitFor();
      await sheet(page).getByRole('button', { name: 'Delete…' }).click();
      assert.match(await page.getByRole('dialog').textContent(), /Delete first\.sgf\?/);
      await page.getByRole('dialog').getByRole('button', { name: 'Delete' }).click();
      await row(page, 'File first.sgf').waitFor({ state: 'detached' });
      assert.equal(await count(page, 'Games'), 0);
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('columns sort, and the keyboard alone makes, opens and leaves a folder', async () => {
      const context = await newContext();
      const { page, errors } = await openFiles(context);
      await importFiles(page, [big, game]);
      await row(page, 'File final.sgf').waitFor();
      await row(page, 'File long.sgf').waitFor();
      const names = () => sheet(page).locator('tbody .files-name').allTextContents();
      assert.deepEqual(await names(), ['final.sgf', 'long.sgf']);
      await sheet(page).getByRole('button', { name: 'Size' }).click();
      assert.equal(await sheet(page).locator('th', { hasText: 'Size' }).getAttribute('aria-sort'), 'descending');
      assert.deepEqual(await names(), ['long.sgf', 'final.sgf']);
      await sheet(page).getByRole('button', { name: /^Size/ }).click();
      assert.equal(await sheet(page).locator('th', { hasText: 'Size' }).getAttribute('aria-sort'), 'ascending');
      assert.deepEqual(await names(), ['final.sgf', 'long.sgf']);

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
      await row(page, 'Folder Tsumego').focus();
      await page.keyboard.press('Enter');
      await sheet(page).getByRole('navigation', { name: 'Folder', exact: true }).getByRole('button', { name: 'Tsumego' }).waitFor();
      assert.match(await sheet(page).textContent(), /This folder is empty/);
      await sheet(page).getByRole('navigation', { name: 'Folder', exact: true }).getByRole('button', { name: 'This browser' }).focus();
      await page.keyboard.press('Enter');
      await row(page, 'File long.sgf').waitFor();
      // Escape outside a dialog closes Files, and the Apps bar shows again.
      await row(page, 'File long.sgf').focus();
      await page.keyboard.press('Escape');
      await sheet(page).waitFor({ state: 'detached' });
      assert.equal(await page.$eval('#bar-apps', (b) => b.hidden), false);
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('a file imported in one tab appears in the other', async () => {
      const context = await newContext();
      const a = await openFiles(context);
      const b = await openFiles(context);
      await importFiles(a.page, [game]);
      await row(b.page, 'File final.sgf').waitFor();
      await b.page.waitForFunction(() => document.querySelector('[aria-label^="Games, "]')?.getAttribute('aria-label') === 'Games, 1');
      assert.deepEqual([...a.errors, ...b.errors], []);
      await context.close();
    });

    test('the sidebar shows this browser’s storage, and the menu bar has File and Go menus', async () => {
      const context = await newContext();
      const { page, errors } = await openFiles(context);
      await sheet(page).getByRole('meter', { name: 'Storage used' }).waitFor();
      assert.match(await sheet(page).locator('.files-storage').textContent(), /used of|not known/);
      await sheet(page).getByRole('button', { name: 'Import…' }).focus();
      await page.locator('sm-menu-bar').getByRole('button', { name: 'File', exact: true }).click();
      for (const item of ['Import…', 'New folder…', 'Move to…', 'Download', 'Rename…', 'Delete…']) await page.getByRole('menu').getByRole('menuitem', { name: item }).waitFor();
      await page.keyboard.press('Escape');
      await page.locator('sm-menu-bar').getByRole('button', { name: 'Go', exact: true }).click();
      for (const item of ['Games', 'Problems', 'Weights', 'Configs', 'This browser']) await page.getByRole('menu').getByRole('menuitemradio', { name: item }).waitFor();
      assert.deepEqual(errors, []);
      await context.close();
    });
  });
}
