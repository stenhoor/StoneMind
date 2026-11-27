// A static file server for dist/site, used by `npm run serve` and the browser tests. Node built-ins only.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { dirname, extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SITE = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist', 'site');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.map': 'application/json', '.json': 'application/json' };

/** Serves `root` on 127.0.0.1:`port` (0 picks a free port). Resolves with the server and its base URL. */
export function serve({ root = SITE, port = 0 } = {}) {
  const server = createServer(async (req, res) => {
    const path = normalize(decodeURIComponent(new URL(req.url ?? '/', 'http://x').pathname));
    let file = join(root, path);
    if (!file.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
    try {
      if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
      const body = await readFile(file);
      res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream' }).end(body);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found\n');
    }
  });
  return new Promise((resolve) => {
    server.listen(port, '127.0.0.1', () => {
      const address = server.address();
      const actual = typeof address === 'object' && address ? address.port : port;
      resolve({ server, url: `http://localhost:${actual}/` });
    });
  });
}
