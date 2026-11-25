// Run the WebGPU network on the golden positions in a browser and compare it with the reference forward pass and
// with native KataGo's outputs; optionally benchmark evaluations per second by batch size.
//
//   npm run build -w @stonemind/engine-webgpu
//   node packages/engine-webgpu/scripts/gpu-check.mjs [--bench [--batches 1,8,64]] [--model <net.bin.gz> [--golden <goldens.json>]]
//   node packages/engine-webgpu/scripts/gpu-check.mjs --serve [--model ...]   # serve the page; open it in any browser
//
// With --model, the slow TypeScript reference comparison is skipped unless --reference is given; the goldens decide.
//
// Without --serve it drives headless Chromium through Playwright (a devDependency of this package; install its browser
// once with `npx playwright install chromium`, or point STONEMIND_CHROMIUM at a Chromium executable). Extra Chromium
// flags can be passed with --chromium-args "...". Without --model it uses the test network (npm run fixtures).
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const pkg = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};

const model = option('--model') ? resolve(option('--model')) : undefined;
const golden = option('--golden') ? resolve(option('--golden')) : undefined;
for (const f of [model, golden]) if (f && !existsSync(f)) throw new Error(`no such file: ${f}`);

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.map': 'application/json' };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  let file;
  if (path === '/model.bin.gz') file = model;
  else if (path === '/golden.json') file = golden;
  else {
    file = normalize(join(pkg, path));
    if (!file.startsWith(pkg)) file = undefined;
  }
  if (!file || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(Number(option('--port') ?? 0), '127.0.0.1', r));
const query = new URLSearchParams();
if (model) query.set('model', '/model.bin.gz');
if (golden) query.set('golden', '/golden.json');
else if (model) query.set('golden', '/none.json');
if (flag('--reference')) query.set('reference', '1');
const url = `http://127.0.0.1:${server.address().port}/test/gpu/index.html${query.size ? `?${query}` : ''}`;

if (flag('--serve')) {
  console.log(`Serving the GPU check page at ${url}`);
  console.log('Open it in a browser with WebGPU, then press a button. Ctrl+C stops the server.');
} else {
  const { chromium } = await import('playwright');
  const extra = option('--chromium-args')?.split(' ').filter(Boolean) ?? [];
  const executablePath = process.env.STONEMIND_CHROMIUM;
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', ...extra], ...(executablePath ? { executablePath } : {}) });
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error(`[page] ${e.message}`));
  await page.goto(url);
  const batches = option('--batches')?.split(',').map(Number);
  const report = await page.evaluate((opts) => window.runCheck(opts), { bench: flag('--bench'), ...(batches ? { batches } : {}) });
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
  server.close();
  if (!report.ok) process.exitCode = 1;
}
