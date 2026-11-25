import { spawn, type ChildProcess } from 'node:child_process';
import { timingSafeEqual } from 'node:crypto';
import { createServer, type IncomingMessage } from 'node:http';
import type { Socket } from 'node:net';
import { BRIDGE_PROTOCOL } from '@stonemind/engine';
import { acceptWebSocket, type WebSocketConnection } from './websocket.js';

/** Pages allowed to connect unless the bridge is told otherwise. */
export const DEFAULT_ALLOWED_ORIGINS: readonly string[] = [
  'https://stenhoor.github.io',
  'https://stonemind.app',
  'http://localhost:8080',
  'http://127.0.0.1:8080',
];

/** The page the pairing link opens unless told otherwise. */
export const DEFAULT_PAGE_URL = 'https://stenhoor.github.io/StoneMind/engine-bench/';

export interface BridgeOptions {
  /** The engine command, e.g. the path to `katago`. */
  command: string;
  /** Its arguments, e.g. ["analysis", "-model", "...", "-config", "..."]. */
  args: readonly string[];
  /** Port on 127.0.0.1; 0 picks a free one. */
  port?: number;
  allowedOrigins?: readonly string[];
  pageUrl?: string;
  /** The pairing token. Generated when omitted; tests may fix it. */
  token?: string;
  /** Where the bridge reports what it does. Never receives the token. */
  log?: (message: string) => void;
}

export interface RunningBridge {
  port: number;
  /** The link to open: the page, with the bridge URL and token in its fragment. */
  pairingLink: string;
  token: string;
  close(): Promise<void>;
}

const encoder = new TextEncoder();

function newToken(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function sameSecret(a: string, b: string): boolean {
  const x = encoder.encode(a);
  const y = encoder.encode(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

function refuse(socket: Socket, status: string): void {
  socket.write(`HTTP/1.1 ${status}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n`);
  socket.destroy();
}

/** The id prefix that marks which page connection a query belongs to. */
const ID_PATTERN = /^c(\d+)\|([\s\S]*)$/;

interface Client {
  ws: WebSocketConnection;
  /** Open analysis query ids (the page's own ids) → finals still expected. */
  open: Map<string, number>;
}

/**
 * Start the local bridge: a native KataGo analysis engine, reachable from allowed pages over a WebSocket on
 * 127.0.0.1 after pairing with the token.
 *
 * Several pages may share the engine: each page's query ids are prefixed on the way in and restored on the way out,
 * so pages never see or stop each other's queries.
 */
export async function startBridge(options: BridgeOptions): Promise<RunningBridge> {
  const log = options.log ?? (() => {});
  const allowedOrigins = new Set(options.allowedOrigins ?? DEFAULT_ALLOWED_ORIGINS);
  const token = options.token ?? newToken();
  const clients = new Map<number, Client>();
  let nextClient = 0;
  let port = 0;

  // ------------------------------------------------------------------ engine
  const engine: ChildProcess = spawn(options.command, options.args, { stdio: ['pipe', 'pipe', 'pipe'] });
  let engineAlive = true;
  const toEngine = (message: Record<string, unknown>) => {
    if (engineAlive) engine.stdin?.write(JSON.stringify(message) + '\n');
  };

  const routeFromEngine = (line: string) => {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      log(`engine wrote a non-JSON line: ${line.slice(0, 200)}`);
      return;
    }
    const match = typeof message.id === 'string' ? ID_PATTERN.exec(message.id) : null;
    if (!match) {
      // A general error, or a line for no page: every page should know.
      for (const c of clients.values()) c.ws.send(line);
      return;
    }
    const client = clients.get(Number(match[1]));
    if (!client) return; // the page has gone
    message.id = match[2];
    for (const field of ['terminateId', 'targetId']) {
      const value = message[field];
      const inner = typeof value === 'string' ? ID_PATTERN.exec(value) : null;
      if (inner) message[field] = inner[2];
    }
    const id = match[2]!;
    if (typeof message.turnNumber === 'number' && message.isDuringSearch === false && client.open.has(id)) {
      const left = client.open.get(id)! - 1;
      if (left <= 0) client.open.delete(id);
      else client.open.set(id, left);
    }
    client.ws.send(JSON.stringify(message));
  };

  const decoder = new TextDecoder();
  let pending = '';
  engine.stdout?.on('data', (chunk) => {
    pending += decoder.decode(chunk, { stream: true });
    let nl: number;
    while ((nl = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, nl).trim();
      pending = pending.slice(nl + 1);
      if (line) routeFromEngine(line);
    }
  });
  const errDecoder = new TextDecoder();
  engine.stderr?.on('data', (chunk) => {
    for (const line of errDecoder.decode(chunk, { stream: true }).split('\n')) if (line.trim()) log(`katago: ${line.trim()}`);
  });
  engine.stdin?.on('error', () => {});
  const engineExited = new Promise<void>((resolve) => {
    engine.on('exit', (code, signal) => {
      engineAlive = false;
      log(`engine exited (${signal ?? `code ${code}`})`);
      for (const c of clients.values()) c.ws.close(1011, 'the engine stopped');
      resolve();
    });
    engine.on('error', (error) => {
      engineAlive = false;
      log(`could not start the engine: ${error.message}`);
      for (const c of clients.values()) c.ws.close(1011, 'the engine could not start');
      resolve();
    });
  });

  // ------------------------------------------------------------- page side
  const fromPage = (cid: number, client: Client, text: string) => {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(text) as Record<string, unknown>;
    } catch {
      client.ws.send(JSON.stringify({ error: 'the bridge expects one JSON object per message' }));
      return;
    }
    if (typeof message !== 'object' || message === null || Array.isArray(message) || typeof message.id !== 'string') {
      client.ws.send(JSON.stringify({ error: 'every query needs a string "id"' }));
      return;
    }
    const id = message.id;
    const prefixed = (value: string) => `c${cid}|${value}`;
    const action = message.action;

    if (action === undefined) {
      if (client.open.has(id)) {
        client.ws.send(JSON.stringify({ error: 'a query with this id is still open', field: 'id', id }));
        return;
      }
      const turns = Array.isArray(message.analyzeTurns) ? message.analyzeTurns.length : 1;
      client.open.set(id, Math.max(1, turns));
      toEngine({ ...message, id: prefixed(id) });
      return;
    }
    switch (action) {
      case 'query_version':
      case 'clear_cache':
      case 'query_models':
        toEngine({ ...message, id: prefixed(id) });
        return;
      case 'terminate':
      case 'set_focus': {
        const field = action === 'terminate' ? 'terminateId' : 'targetId';
        if (typeof message[field] !== 'string') {
          client.ws.send(JSON.stringify({ error: `"${field}" must be a string`, field, id }));
          return;
        }
        toEngine({ ...message, id: prefixed(id), [field]: prefixed(message[field] as string) });
        return;
      }
      case 'terminate_all':
        // Only this page's queries: other pages share the engine.
        for (const openId of client.open.keys()) {
          const t: Record<string, unknown> = { id: prefixed(`${id}/${openId}`), action: 'terminate', terminateId: prefixed(openId) };
          if (message.turnNumbers !== undefined) t.turnNumbers = message.turnNumbers;
          toEngine(t);
        }
        client.ws.send(JSON.stringify(message));
        return;
      default:
        client.ws.send(JSON.stringify({ error: `the bridge does not pass on the action "${String(action)}"`, field: 'action', id }));
    }
  };

  const server = createServer((_req, res) => {
    res.writeHead(426, { 'Content-Type': 'text/plain' });
    res.end('This is the StoneMind KataGo bridge. Connect with a WebSocket from an allowed page.\n');
  });

  server.on('upgrade', (req: IncomingMessage, socket: Socket, head: Uint8Array) => {
    const origin = String(req.headers.origin ?? '');
    const host = String(req.headers.host ?? '');
    const offered = String(req.headers['sec-websocket-protocol'] ?? '')
      .split(',')
      .map((p) => p.trim());
    const offeredToken = offered.find((p) => p.startsWith('token.'))?.slice('token.'.length) ?? '';

    // Defends against DNS rebinding: the page must have addressed the loopback interface by name or number.
    if (![`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`].includes(host)) {
      log(`refused a connection for host "${host}"`);
      return refuse(socket, '403 Forbidden');
    }
    if (!allowedOrigins.has(origin)) {
      log(`refused a connection from origin "${origin || '(none)'}"`);
      return refuse(socket, '403 Forbidden');
    }
    if (!offered.includes(BRIDGE_PROTOCOL) || !sameSecret(offeredToken, token)) {
      log(`refused a connection from ${origin}: wrong or missing pairing token`);
      return refuse(socket, '401 Unauthorized');
    }
    if (String(req.headers.upgrade ?? '').toLowerCase() !== 'websocket' || !req.headers['sec-websocket-key']) {
      return refuse(socket, '400 Bad Request');
    }
    if (!engineAlive) return refuse(socket, '503 Service Unavailable');

    const cid = ++nextClient;
    const client: Client = { ws: acceptWebSocket(req, socket, head, BRIDGE_PROTOCOL), open: new Map() };
    clients.set(cid, client);
    log(`page connected from ${origin} (connection ${cid})`);
    client.ws.onMessage((text) => fromPage(cid, client, text));
    client.ws.onClose(() => {
      clients.delete(cid);
      for (const openId of client.open.keys()) {
        toEngine({ id: `c${cid}|close/${openId}`, action: 'terminate', terminateId: `c${cid}|${openId}` });
      }
      log(`page disconnected (connection ${cid})`);
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(options.port ?? 0, '127.0.0.1', () => resolve());
  });
  const address = server.address();
  port = typeof address === 'object' && address ? address.port : 0;

  const pageUrl = options.pageUrl ?? DEFAULT_PAGE_URL;
  const pairingLink = `${pageUrl}#bridge=${encodeURIComponent(`ws://127.0.0.1:${port}`)}&token=${token}`;
  log(`listening on ws://127.0.0.1:${port} for ${[...allowedOrigins].join(', ')}`);

  return {
    port,
    pairingLink,
    token,
    async close() {
      for (const c of clients.values()) c.ws.close(1001, 'the bridge is stopping');
      await new Promise<void>((resolve) => server.close(() => resolve()));
      if (engineAlive) {
        engine.stdin?.end();
        const timer = setTimeout(() => engine.kill('SIGTERM'), 3000);
        await engineExited;
        clearTimeout(timer);
      }
    },
  };
}
