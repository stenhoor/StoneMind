// The engine host in real browsers, with Playwright (build plan, Milestone 1, step 5), using the fake engine
// (src/engines/fake.ts), which speaks KataGo's protocol without a GPU. The real browser engine is checked on the
// owner's machines.
//
//   npm run e2e -w @stonemind/web                      # Chromium
//   BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/web   # as CI runs it
import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as playwright from 'playwright';
import { serve } from '../../scripts/server.mjs';

const BROWSERS = (process.env.BROWSERS ?? 'chromium').split(',').map((s) => s.trim()).filter(Boolean);

let server;
let url;
before(async () => ({ server, url } = await serve()));
after(() => server?.close());

const START = { engineId: 'browser', kind: 'fake', weightsPath: '/n.bin.gz' };

for (const name of BROWSERS) {
  describe(name, () => {
    // A persistent profile, as a player's is: Playwright's default contexts are ephemeral, and WebKit has no OPFS in
    // one (microsoft/playwright#18235).
    // Chromium is given WebGPU as the engine's GPU check gives it (a software adapter where there is no GPU).
    const newContext = () => {
      const executablePath = name === 'chromium' ? process.env.STONEMIND_CHROMIUM : undefined;
      const args = name === 'chromium' ? ['--enable-unsafe-webgpu'] : [];
      return playwright[name].launchPersistentContext(mkdtempSync(join(tmpdir(), 'stonemind-engines-')), { args, ...(executablePath ? { executablePath } : {}) });
    };

    /** A tab of the app connected to the engine host, with a weights file in storage for the fake engine to "load". */
    const openTab = async (context, mode = 'shared') => {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(url);
      await page.evaluate(async (mode) => {
        const root = await navigator.storage.getDirectory();
        const w = await (await root.getFileHandle('n.bin.gz', { create: true })).createWritable();
        await w.write('stand-in');
        await w.close();
        const { EngineHostClient } = await import('./js/engines/client.js');
        window.host = await EngineHostClient.connect(mode);
      }, mode);
      return { page, errors };
    };
    const status = (page) => page.evaluate(() => window.host.status);
    const start = (page, request = START) => page.evaluate((r) => window.host.start(r), request);
    // One analysis through the tab's AnalysisEngine: the visits of each final result.
    const analyse = (page, id, maxVisits) => page.evaluate(async ([id, maxVisits]) => {
      const out = [];
      for await (const r of window.host.engine('browser').analyze({ id, moves: [], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, maxVisits })) out.push([r.id, r.rootInfo.visits]);
      return out;
    }, [id, maxVisits]);

    test('two tabs share one engine, and each gets only its own results, even with the same query id', async () => {
      const context = await newContext();
      const a = await openTab(context);
      const b = await openTab(context);
      const s = await start(a.page);
      assert.equal(s.state, 'running');
      const seen = await status(b.page);
      assert.equal(seen.shared, true);
      assert.equal(seen.tabs, 2);
      assert.equal(seen.engines[0].state, 'running');
      assert.deepEqual(seen.engines[0].config.numSearchThreadsPerAnalysisThread, '16');
      const [ra, rb] = await Promise.all([analyse(a.page, 'q1', 7), analyse(b.page, 'q1', 9)]);
      assert.deepEqual(ra, [['q1', 7]]);
      assert.deepEqual(rb, [['q1', 9]]);
      assert.deepEqual([...a.errors, ...b.errors], []);
      await context.close();
    });

    test('the engine outlives the tab that started it', async () => {
      const context = await newContext();
      const a = await openTab(context);
      const b = await openTab(context);
      await start(a.page);
      await a.page.close();
      await b.page.waitForFunction(() => window.host.status.tabs === 1);
      assert.equal((await status(b.page)).engines[0].state, 'running');
      assert.deepEqual(await analyse(b.page, 'after', 5), [['after', 5]]);
      // A tab opened now finds it running too.
      const c = await openTab(context);
      assert.equal((await status(c.page)).engines[0].state, 'running');
      assert.deepEqual(b.errors, []);
      await context.close();
    });

    test('a start with no weights file fails and says why; stop ends the engine for every tab', async () => {
      const context = await newContext();
      const a = await openTab(context);
      const b = await openTab(context);
      const failed = await start(a.page, { ...START, weightsPath: '/missing.bin.gz' });
      assert.equal(failed.state, 'failed');
      assert.match(failed.error, /There is no weights file at \/missing\.bin\.gz in this browser’s storage/);
      await start(a.page);
      await a.page.evaluate(() => window.host.stop('browser'));
      await b.page.waitForFunction(() => window.host.status.engines[0]?.state === 'stopped');
      await context.close();
    });

    test('the real browser engine runs a KataGo network from this browser’s storage where WebGPU works, and says plainly where it doesn’t', async () => {
      // KataGo's own small test network (b6c96), which the engine package fetches for its tests.
      const network = readFileSync(join(import.meta.dirname, '..', '..', '..', '..', 'packages', 'engine-webgpu', 'test', 'fixtures', 'g170-b6c96-s175395328-d26788732.bin.gz'));
      const context = await newContext();
      const { page, errors } = await openTab(context, 'auto');
      await page.evaluate(async (b64) => {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const w = await (await (await navigator.storage.getDirectory()).getFileHandle('b6c96.bin.gz', { create: true })).createWritable();
        await w.write(bytes);
        await w.close();
      }, network.toString('base64'));
      const host = await status(page);
      const s = await start(page, { engineId: 'browser', kind: 'webgpu', weightsPath: '/b6c96.bin.gz', config: { numAnalysisThreads: '1', numSearchThreadsPerAnalysisThread: '4' } });
      if (host.webgpu || !host.shared) {
        if (s.state === 'running') {
          assert.deepEqual(s.model, { name: 'g170-b6c96-s175395328-d26788732', version: 8 });
          const results = await page.evaluate(async () => {
            const out = [];
            for await (const r of window.host.engine('browser').analyze({ id: 'real', moves: [['B', 'Q16']], rules: 'chinese', komi: 7.5, boardXSize: 19, boardYSize: 19, maxVisits: 16 })) out.push(r.rootInfo.visits);
            return out;
          });
          assert.deepEqual(results, [16]);
          const version = await page.evaluate(() => window.host.engine('browser').action({ action: 'query_version' }));
          assert.match(version.version, /^1\.\d+/);
        } else assert.match(s.error, /WebGPU|adapter/i, `${name}: a failed start says why: ${s.error}`);
      } else {
        assert.equal(s.state, 'failed');
        assert.match(s.error, /WebGPU|adapter/i);
      }
      if (name === 'chromium') assert.equal(s.state, 'running', `Chromium with WebGPU runs the engine: ${s.error ?? ''}`);
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('the Engines console starts the engine, sends KataGo queries as typed, and times each result', async () => {
      const network = readFileSync(join(import.meta.dirname, '..', '..', '..', '..', 'packages', 'engine-webgpu', 'test', 'fixtures', 'g170-b6c96-s175395328-d26788732.bin.gz'));
      const context = await newContext();
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(url);
      await page.evaluate(async (b64) => {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const w = await (await (await navigator.storage.getDirectory()).getFileHandle('b6c96.bin.gz', { create: true })).createWritable();
        await w.write(bytes);
        await w.close();
      }, network.toString('base64'));
      await page.locator('#bar-apps').getByRole('button', { name: 'Engines' }).click();
      const weights = page.getByLabel('Weights');
      await weights.locator('option[value="/b6c96.bin.gz"]').waitFor({ state: 'attached' });
      const settings = page.getByLabel('KataGo settings (key = value)');
      // KataGo's defaults are filled in.
      assert.match(await settings.inputValue(), /numSearchThreadsPerAnalysisThread = 16/);
      // A line that isn't "key = value" is refused before starting.
      await settings.fill('numAnalysisThreads 1');
      await page.getByRole('button', { name: 'Start' }).click();
      await page.getByText(/Line 1 of the settings is not "key = value"/).waitFor();
      await settings.fill('numAnalysisThreads = 1\nnumSearchThreadsPerAnalysisThread = 4');
      await page.getByRole('button', { name: 'Start' }).click();
      const state = page.locator('.console-state strong');
      await page.waitForFunction(() => /Running|Did not start/.test(document.querySelector('.console-state strong')?.textContent ?? ''), null, { timeout: 120_000 });
      const send = page.getByRole('button', { name: 'Send' });
      if ((await state.textContent()) === 'Running') {
        const command = page.getByLabel(/Queries: one JSON line each/);
        await command.fill('{"action":"query_version"}\n{"id":"b8","moves":[],"rules":"chinese","komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":8}');
        await command.press('Control+Enter');
        const log = page.getByRole('log', { name: 'Engine log' });
        await log.getByText(/"version":"1\.\d+/).waitFor();
        await log.getByText(/^\d+ ms · 8 visits · \d+ visits\/s$/).waitFor({ timeout: 120_000 });
        // The log is the queries and replies as JSON lines; query_version was given an id.
        const text = await page.evaluate(() => document.querySelector('.console-log')?.textContent ?? '');
        assert.match(text, /\{"action":"query_version","id":"c\d+"\}/);
      } else {
        // Where there is no WebGPU, it says so, and nothing can be sent.
        assert.notEqual(name, 'chromium', 'Chromium with WebGPU runs the engine');
        await page.locator('.console-status').getByText(/WebGPU|adapter/i).first().waitFor();
        assert.ok(await send.isDisabled());
      }
      assert.deepEqual(errors, []);
      await context.close();
    });

    test('a host for one tab alone, where a shared one can’t run WebGPU, keeps its engine to itself', async () => {
      const context = await newContext();
      const a = await openTab(context, 'dedicated');
      const b = await openTab(context, 'shared');
      assert.equal((await status(a.page)).shared, false);
      await start(a.page);
      assert.deepEqual(await analyse(a.page, 'solo', 3), [['solo', 3]]);
      assert.equal((await status(b.page)).engines.length, 0);
      await context.close();
    });
  });
}
