// The file kit in real browsers (build plan, Milestone 1, step 3).
//
//   npm run e2e -w @stonemind/files                                  # Chromium
//   BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/files # as CI runs it
//
// STONEMIND_CHROMIUM points at another Chromium. Both write methods are tested where the browser has them:
// 'writable' (createWritable from the page) and 'worker' (createSyncAccessHandle in a dedicated worker).
import { after, before, describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as playwright from 'playwright';

const BROWSERS = (process.env.BROWSERS ?? 'chromium').split(',').map((s) => s.trim()).filter(Boolean);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.map': 'application/json' };

let server;
let url;
before(async () => {
  server = createServer(async (req, res) => {
    const file = join(ROOT, normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)));
    if (!file.startsWith(ROOT)) return res.writeHead(403).end();
    let body;
    try {
      body = await readFile(file);
    } catch {
      return res.writeHead(404).end();
    }
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' }).end(body);
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  url = `http://localhost:${server.address().port}/test/browser/index.html`;
});
after(() => server?.close());

for (const name of BROWSERS) {
  describe(name, () => {
    let context;
    after(() => context?.close());

    const open = async () => {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(e));
      await page.goto(url);
      await page.waitForSelector('body[data-ready="true"]');
      return { page, errors };
    };

    // Each test gets a fresh profile, so OPFS starts empty. The profile is persistent, as a player's is: Playwright's
    // default contexts are ephemeral, and WebKit (like Safari in a private window) has no OPFS in an ephemeral one
    // (microsoft/playwright#18235).
    const fresh = async () => {
      await context?.close();
      const executablePath = name === 'chromium' ? process.env.STONEMIND_CHROMIUM : undefined;
      context = await playwright[name].launchPersistentContext(mkdtempSync(join(tmpdir(), 'stonemind-opfs-')), { acceptDownloads: true, ...(executablePath ? { executablePath } : {}) });
    };

    for (const method of ['writable', 'worker']) {
      describe(`writing with ${method}`, () => {
        let supported = true;
        before(async () => {
          await fresh();
          const { page } = await open();
          supported = await page.evaluate((m) => (m === 'worker' ? typeof Worker !== 'undefined' : 'createWritable' in FileSystemFileHandle.prototype), method);
          await page.close();
        });

        test('write, read, list, stat, move and remove', async (t) => {
          if (!supported) return t.skip(`${name} has no ${method} write`);
          const { page, errors } = await open();
          const result = await page.evaluate(async (m) => {
            const s = await files.OpfsFileStore.open({ writeMethod: m });
            await s.write('/games/a.sgf', '(;GM[1]SZ[19])');
            await s.write('/games/b.sgf', new Uint8Array([40, 59, 41]));
            await s.mkdir('/weights');
            const list = (await s.list('/')).map((e) => `${e.kind}:${e.name}`);
            const games = (await s.list('/games')).map((e) => `${e.name}:${e.size}`);
            const text = await (await s.read('/games/a.sgf')).text();
            await s.move('/games/a.sgf', '/archive/2026/a.sgf');
            const moved = [await s.stat('/games/a.sgf'), (await s.stat('/archive/2026/a.sgf'))?.size];
            await s.move('/archive', '/old');
            const movedDir = (await s.list('/old/2026')).map((e) => e.name);
            await s.remove('/old', { recursive: true });
            let refused = '';
            try { await s.remove('/games'); } catch (e) { refused = e.name; }
            const errs = [];
            for (const bad of [() => s.read('/nope.sgf'), () => s.move('/games/b.sgf', '/games/b.sgf/x'), () => s.write('/', 'x')]) {
              try { await bad(); } catch (e) { errs.push(e.name); }
            }
            return { list, games, text, moved, movedDir, after: (await s.list('/')).map((e) => e.name), refused, errs, method: s.writeMethod };
          }, method);
          assert.equal(result.method, method);
          assert.deepEqual(result.list, ['directory:games', 'directory:weights']);
          assert.deepEqual(result.games, ['a.sgf:14', 'b.sgf:3']);
          assert.equal(result.text, '(;GM[1]SZ[19])');
          assert.deepEqual(result.moved, [null, 14]);
          assert.deepEqual(result.movedDir, ['a.sgf']);
          assert.deepEqual(result.after, ['games', 'weights']);
          assert.ok(result.refused, 'removing a folder that is not empty without recursive is refused');
          assert.deepEqual(result.errs, ['NotFoundError', 'PathError', 'PathError']);
          assert.deepEqual(errors, []);
          await page.close();
        });

        test('a large file streams in, with progress, and reads back the same', async (t) => {
          if (!supported) return t.skip(`${name} has no ${method} write`);
          const { page } = await open();
          const result = await page.evaluate(async (m) => {
            const s = await files.OpfsFileStore.open({ writeMethod: m });
            const SIZE = 48 * 1024 * 1024;
            const piece = 3 * 1024 * 1024 + 7; // not a multiple of the store's 1 MB chunks
            let made = 0;
            const stream = new ReadableStream({
              pull(c) {
                if (made >= SIZE) return c.close();
                const n = Math.min(piece, SIZE - made);
                const b = new Uint8Array(n);
                for (let i = 0; i < n; i++) b[i] = (made + i) % 251;
                made += n;
                c.enqueue(b);
              },
            });
            const progress = [];
            await s.write('/weights/big.bin', stream, { onProgress: (w) => progress.push(w) });
            const blob = await s.read('/weights/big.bin');
            const sample = new Uint8Array(await blob.slice(40 * 1024 * 1024, 40 * 1024 * 1024 + 8192).arrayBuffer());
            return { size: blob.size, last: progress.at(-1), steps: progress.length, rising: progress.every((v, i) => i === 0 || v > progress[i - 1]), sample: [sample[0], sample[4096]] };
          }, method);
          assert.equal(result.size, 48 * 1024 * 1024);
          assert.equal(result.last, result.size);
          assert.ok(result.steps >= 48, `progress reported ${result.steps} times`);
          assert.ok(result.rising);
          assert.deepEqual(result.sample, [(40 * 1024 * 1024) % 251, (40 * 1024 * 1024 + 4096) % 251]);
          await page.close();
        });

        test('an aborted write leaves no new file behind', async (t) => {
          if (!supported) return t.skip(`${name} has no ${method} write`);
          const { page } = await open();
          const result = await page.evaluate(async (m) => {
            const s = await files.OpfsFileStore.open({ writeMethod: m });
            const ac = new AbortController();
            const stream = new ReadableStream({ pull(c) { c.enqueue(new Uint8Array(1024 * 1024)); } });
            let error = '';
            try {
              await s.write('/weights/cut.bin', stream, { signal: ac.signal, onProgress: (w) => { if (w >= 4 * 1024 * 1024) ac.abort(); } });
            } catch (e) { error = e.name; }
            return { error, stat: await s.stat('/weights/cut.bin') };
          }, method);
          assert.equal(result.error, 'AbortError');
          assert.equal(result.stat, null);
          await page.close();
        });
      });
    }

    test('two tabs writing the same file never interleave', async () => {
      await fresh();
      const a = await open();
      const b = await open();
      const write = (page, byte) => page.evaluate(async (v) => {
        const s = await files.OpfsFileStore.open();
        const chunk = new Uint8Array(1024 * 1024).fill(v);
        let n = 0;
        const stream = new ReadableStream({ async pull(c) { if (n++ === 16) return c.close(); await new Promise((r) => setTimeout(r, 5)); c.enqueue(chunk.slice()); } });
        await s.write('/shared.bin', stream);
      }, byte);
      await Promise.all([write(a.page, 1), write(b.page, 2)]);
      const bytes = await a.page.evaluate(async () => {
        const s = await files.OpfsFileStore.open();
        const blob = await s.read('/shared.bin');
        const all = new Uint8Array(await blob.arrayBuffer());
        return { size: blob.size, values: [...new Set(all)] };
      });
      assert.equal(bytes.size, 16 * 1024 * 1024);
      assert.equal(bytes.values.length, 1, `the file mixes both writes: ${bytes.values}`);
      await a.page.close();
      await b.page.close();
    });

    test('a change in one tab is announced in the other', async () => {
      await fresh();
      const a = await open();
      const b = await open();
      const heard = b.page.evaluate(async () => {
        const s = await files.OpfsFileStore.open();
        return new Promise((resolve) => s.addEventListener('change', (e) => resolve(e.detail), { once: true }));
      });
      await b.page.waitForTimeout(100);
      await a.page.evaluate(async () => (await files.OpfsFileStore.open()).write('/notes/x.md', '# x'));
      assert.deepEqual(await heard, { paths: ['/notes/x.md'], remote: true });
      await a.page.close();
      await b.page.close();
    });

    test('download saves the file the browser holds', async () => {
      await fresh();
      const { page } = await open();
      await page.evaluate(async () => (await files.OpfsFileStore.open()).write('/games/c.sgf', '(;GM[1])'));
      const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.evaluate(async () => files.downloadBlob(await (await files.OpfsFileStore.open()).read('/games/c.sgf'), 'c.sgf')),
      ]);
      assert.equal(download.suggestedFilename(), 'c.sgf');
      const path = await download.path();
      assert.equal(await readFile(path, 'utf8'), '(;GM[1])');
      await page.close();
    });

    test('storage info reports usage, quota and persistence', async () => {
      await fresh();
      const { page } = await open();
      const info = await page.evaluate(() => files.storageInfo());
      assert.equal(typeof info.persisted, 'boolean');
      assert.equal(typeof info.quota, 'number');
      await page.close();
    });
  });
}
