// Serve dist/site on http://localhost:8080 for local development. The bridge allows this origin by default.
import { createServer } from 'node:http';
import { readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'site');
const port = Number(process.env.PORT ?? 8080);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.map': 'application/json', '.json': 'application/json', '.wasm': 'application/wasm' };
// Reports posted by the onnxruntime-web baseline page are saved here.
const reportOut = process.env.REPORT_OUT ?? 'ort-bench-report.json';

createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/report') {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    await writeFile(reportOut, Buffer.concat(chunks));
    console.log(`report saved to ${reportOut}`);
    res.writeHead(200).end();
    return;
  }
  // The onnxruntime-web baseline is served cross-origin isolated, so its WASM provider can use threads, as it can on
  // any site that sends these headers (GitHub Pages can't). Everything it loads is same-origin.
  const isolated = req.url?.startsWith('/ort-bench/') ? { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' } : {};
  const path = normalize(decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname));
  let file = join(root, path);
  if (!file.startsWith(root)) {
    res.writeHead(403).end();
    return;
  }
  try {
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': types[extname(file)] ?? 'application/octet-stream', ...isolated }).end(body);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found\n');
  }
}).listen(port, '127.0.0.1', () => console.log(`engine bench: http://localhost:${port}/engine-bench/ · onnxruntime-web baseline: http://localhost:${port}/ort-bench/`));
