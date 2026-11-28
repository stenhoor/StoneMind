// The frame in real browsers, with Playwright (build plan, Milestone 1, step 1).
//
//   npm run e2e -w @stonemind/web                      # Chromium
//   BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/web   # as CI runs it
//
// Each browser named in BROWSERS must be installed for Playwright (`npx -w @stonemind/web playwright install`).
// STONEMIND_CHROMIUM points at another Chromium, as for the engine's GPU check.
import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import * as playwright from 'playwright';
import { serve } from '../../scripts/server.mjs';

const BROWSERS = (process.env.BROWSERS ?? 'chromium').split(',').map((s) => s.trim()).filter(Boolean);
const WIDTHS = { phone: 390, tablet: 820, desktop: 1440 };

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

    const open = async (width = WIDTHS.desktop) => {
      const page = await browser.newPage({ viewport: { width, height: 900 } });
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(url);
      await page.waitForSelector('sm-menu-bar .sm-menubar-clock');
      return { page, errors };
    };

    test('the Apps bar shows at startup; at most one bar is shown, and pressing the lit button hides it', async () => {
      const { page, errors } = await open();
      const shown = () => page.$$eval('.frame-bar', (bars) => bars.filter((b) => !b.hidden).map((b) => b.id));
      assert.deepEqual(await shown(), ['bar-apps']);
      await page.locator('sm-menu-bar').getByRole('button', { name: 'Workspaces' }).click();
      assert.deepEqual(await shown(), ['bar-workspaces']);
      await page.getByRole('button', { name: 'Apps', exact: true }).click();
      assert.deepEqual(await shown(), ['bar-apps']);
      assert.equal(await page.locator('sm-menu-bar').getByRole('button', { name: 'Workspaces' }).getAttribute('aria-pressed'), 'false');
      await page.getByRole('button', { name: 'Apps', exact: true }).click();
      assert.deepEqual(await shown(), []);
      assert.deepEqual(errors, []);
      await page.close();
    });

    for (const [label, width] of Object.entries(WIDTHS)) {
      test(`the menu bar stays on one line at ${label} width`, async () => {
        const { page } = await open(width);
        const { height, mode } = await page.$eval('sm-menu-bar', (bar) => ({ height: bar.getBoundingClientRect().height, mode: bar.dataset.mode }));
        assert.ok(height <= 60, `menu bar is ${height}px tall`);
        assert.equal(mode, { phone: 'narrow', tablet: 'compact', desktop: 'full' }[label]);
        // Every visible item in the menu bar sits on the same line.
        const tops = await page.$$eval('sm-menu-bar > *', (items) =>
          items.filter((i) => i.getClientRects().length && getComputedStyle(i).display !== 'none').map((i) => {
            const r = i.getBoundingClientRect();
            return r.top + r.height / 2;
          }));
        for (const t of tops) assert.ok(Math.abs(t - tops[0]) <= 4, `items not on one line: ${tops.join(', ')}`);
        await page.close();
      });
    }

    test('narrow: menus fold into one Menus button', async () => {
      const { page } = await open(WIDTHS.phone);
      await page.getByRole('button', { name: 'Menus' }).click();
      const menu = page.getByRole('menu', { name: 'Menus' });
      assert.ok(await menu.isVisible());
      assert.ok(await menu.getByRole('group', { name: 'Window' }).isVisible());
      await page.keyboard.press('Escape');
      assert.equal(await menu.count(), 0);
      await page.close();
    });

    test('menus work from the keyboard', async () => {
      const { page } = await open();
      const windowMenu = page.locator('sm-menu-bar').getByRole('button', { name: 'Window', exact: true });
      await windowMenu.focus();
      await page.keyboard.press('ArrowDown');
      const menu = page.getByRole('menu', { name: 'Window' });
      assert.ok(await menu.isVisible());
      // The first item (Workspaces bar) is focused; choose it.
      await page.keyboard.press('Enter');
      assert.equal(await page.$eval('#bar-workspaces', (b) => b.hidden), false);
      assert.equal(await menu.count(), 0);
      assert.equal(await page.evaluate(() => document.activeElement?.textContent), 'Window');
      await page.close();
    });

    test('theme and backdrop are set in one place: StoneMind › Settings…, not the View menu', async () => {
      const { page } = await open();
      assert.equal(await page.locator('sm-menu-bar').getByRole('button', { name: 'View', exact: true }).count(), 0);
      await page.getByRole('button', { name: 'StoneMind menu' }).click();
      await page.getByRole('menuitem', { name: 'Settings…' }).click();
      assert.equal(await page.locator('.sheet[data-app="settings"]').count(), 1);
      await page.close();
    });

    test('bar buttons work from the keyboard', async () => {
      const { page } = await open();
      await page.locator('sm-menu-bar').getByRole('button', { name: 'Layouts' }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await page.$eval('#bar-layouts', (b) => b.hidden), false);
      await page.keyboard.press('Space');
      assert.equal(await page.$eval('#bar-layouts', (b) => b.hidden), true);
      await page.close();
    });

    test('singleton apps open once, and Settings changes the theme and backdrop', async () => {
      const { page } = await open();
      const settings = page.locator('#bar-apps').getByRole('button', { name: 'Settings' });
      await settings.click();
      await settings.click();
      assert.equal(await page.locator('.sheet').count(), 1);
      assert.equal(await page.locator('.sm-menubar-app').textContent(), 'Settings');
      await page.getByRole('radio', { name: 'Night' }).check();
      assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'night');
      await page.getByRole('button', { name: 'Washi paper' }).click();
      assert.equal(await page.evaluate(() => document.documentElement.dataset.backdrop), 'washi');
      await page.keyboard.press('Escape');
      assert.equal(await page.locator('.sheet').count(), 0);
      assert.equal(await page.locator('.sm-menubar-app').isVisible(), false);
      // The screen is empty again, so the Apps bar shows, even if it was hidden while the app was open.
      assert.equal(await page.$eval('#bar-apps', (b) => b.hidden), false);
      await settings.click();
      await page.getByRole('button', { name: 'Apps', exact: true }).click();
      assert.equal(await page.$eval('#bar-apps', (b) => b.hidden), true);
      await page.getByRole('button', { name: 'Close Settings' }).click();
      assert.equal(await page.$eval('#bar-apps', (b) => b.hidden), false);
      // Settings last across a reload.
      await page.reload();
      assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), 'night');
      assert.equal(await page.evaluate(() => document.documentElement.dataset.backdrop), 'washi');
      await page.close();
    });

    test('gobans and panels say they are not built yet, and cannot be pressed', async () => {
      const { page } = await open();
      for (const n of ['Review', 'Play', 'Problems', 'Game tree', 'Analysis', 'Notes']) {
        assert.ok(await page.locator('#bar-apps').getByRole('button', { name: n }).isDisabled(), n);
      }
      await page.close();
    });
  });
}
