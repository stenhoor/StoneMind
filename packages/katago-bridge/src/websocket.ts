// A minimal WebSocket server endpoint (RFC 6455): text messages, fragmentation, ping/pong and close.
// Enough for one line-based protocol on localhost; binary messages and extensions are refused.
import { createHash } from 'node:crypto';
import type { IncomingMessage } from 'node:http';
import type { Socket } from 'node:net';

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
/** The largest message the bridge accepts from a page, in bytes. KataGo queries are far smaller. */
export const MAX_MESSAGE_BYTES = 4 * 1024 * 1024;

export interface WebSocketConnection {
  send(text: string): void;
  close(code?: number, reason?: string): void;
  onMessage(listener: (text: string) => void): void;
  onClose(listener: (code: number, reason: string) => void): void;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });

function concat(a: Uint8Array, b: Uint8Array): Uint8Array {
  if (a.length === 0) return b;
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function frame(opcode: number, payload: Uint8Array): Uint8Array {
  const n = payload.length;
  const headerLength = n < 126 ? 2 : n < 65536 ? 4 : 10;
  const out = new Uint8Array(headerLength + n);
  const view = new DataView(out.buffer);
  out[0] = 0x80 | opcode; // FIN + opcode; server frames are never masked
  if (n < 126) out[1] = n;
  else if (n < 65536) {
    out[1] = 126;
    view.setUint16(2, n);
  } else {
    out[1] = 127;
    view.setBigUint64(2, BigInt(n));
  }
  out.set(payload, headerLength);
  return out;
}

/** The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key. */
export function acceptKey(key: string): string {
  return createHash('sha1').update(key + GUID).digest('base64');
}

/**
 * Complete the WebSocket handshake for an upgrade request that the caller has already checked, and speak the
 * protocol on the socket.
 */
export function acceptWebSocket(req: IncomingMessage, socket: Socket, head: Uint8Array, protocol: string): WebSocketConnection {
  const key = String(req.headers['sec-websocket-key'] ?? '');
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\n' +
      'Upgrade: websocket\r\n' +
      'Connection: Upgrade\r\n' +
      `Sec-WebSocket-Accept: ${acceptKey(key)}\r\n` +
      `Sec-WebSocket-Protocol: ${protocol}\r\n\r\n`,
  );
  socket.setNoDelay(true);

  const messageListeners: ((text: string) => void)[] = [];
  const closeListeners: ((code: number, reason: string) => void)[] = [];
  let buffer: Uint8Array = new Uint8Array(0);
  let fragments: Uint8Array[] = [];
  let fragmentBytes = 0;
  let inMessage = false;
  let closed = false;

  const finish = (code: number, reason: string) => {
    if (closed) return;
    closed = true;
    for (const l of closeListeners) l(code, reason);
  };

  const close = (code = 1000, reason = '') => {
    if (closed) return;
    const reasonBytes = encoder.encode(reason).slice(0, 123);
    const payload = new Uint8Array(2 + reasonBytes.length);
    new DataView(payload.buffer).setUint16(0, code);
    payload.set(reasonBytes, 2);
    socket.write(frame(0x8, payload));
    socket.end();
    finish(code, reason);
  };

  const parse = () => {
    for (;;) {
      if (buffer.length < 2) return;
      const b0 = buffer[0]!;
      const b1 = buffer[1]!;
      const fin = (b0 & 0x80) !== 0;
      const opcode = b0 & 0x0f;
      if ((b0 & 0x70) !== 0) return close(1002, 'reserved bits set');
      if ((b1 & 0x80) === 0) return close(1002, 'client frames must be masked');
      let length = b1 & 0x7f;
      let offset = 2;
      const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
      if (length === 126) {
        if (buffer.length < 4) return;
        length = view.getUint16(2);
        offset = 4;
      } else if (length === 127) {
        if (buffer.length < 10) return;
        const big = view.getBigUint64(2);
        if (big > BigInt(MAX_MESSAGE_BYTES)) return close(1009, 'message too big');
        length = Number(big);
        offset = 10;
      }
      if (buffer.length < offset + 4 + length) return;
      const mask = buffer.subarray(offset, offset + 4);
      const payload = buffer.slice(offset + 4, offset + 4 + length);
      for (let i = 0; i < payload.length; i++) payload[i]! ^= mask[i & 3]!;
      buffer = buffer.subarray(offset + 4 + length);

      if (opcode >= 0x8) {
        if (!fin || length > 125) return close(1002, 'bad control frame');
        if (opcode === 0x8) {
          const code = payload.length >= 2 ? new DataView(payload.buffer).getUint16(0) : 1005;
          return close(code === 1005 ? 1000 : code);
        }
        if (opcode === 0x9) socket.write(frame(0xa, payload));
        continue; // pong: ignore
      }
      if (opcode === 0x2) return close(1003, 'binary messages are not supported');
      if (opcode === 0x1) {
        if (inMessage) return close(1002, 'new message inside a fragmented one');
        inMessage = true;
      } else if (opcode !== 0x0 || !inMessage) {
        return close(1002, 'unexpected frame');
      }
      fragmentBytes += payload.length;
      if (fragmentBytes > MAX_MESSAGE_BYTES) return close(1009, 'message too big');
      fragments.push(payload);
      if (fin) {
        const whole = fragments.reduce(concat, new Uint8Array(0));
        fragments = [];
        fragmentBytes = 0;
        inMessage = false;
        let text: string;
        try {
          text = decoder.decode(whole);
        } catch {
          return close(1007, 'invalid UTF-8');
        }
        for (const l of messageListeners) l(text);
        if (closed) return;
      }
    }
  };

  socket.on('data', (chunk) => {
    buffer = concat(buffer, chunk);
    parse();
  });
  socket.on('close', () => finish(1006, 'connection lost'));
  socket.on('error', () => finish(1006, 'connection error'));
  if (head.length > 0) {
    buffer = concat(buffer, head);
    queueMicrotask(parse);
  }

  return {
    send: (text) => {
      if (!closed) socket.write(frame(0x1, encoder.encode(text)));
    },
    close,
    onMessage: (l) => void messageListeners.push(l),
    onClose: (l) => void closeListeners.push(l),
  };
}
