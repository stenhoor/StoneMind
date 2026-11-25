// The few Node.js built-in APIs the bridge uses, declared here so the package needs no outside type packages.
// They match Node 22's documented signatures for the overloads used; extend them when the bridge uses more.

declare module 'node:net' {
  export interface Socket {
    write(data: Uint8Array | string): boolean;
    end(data?: Uint8Array | string): void;
    destroy(): void;
    setNoDelay(noDelay?: boolean): this;
    on(event: 'data', listener: (chunk: Uint8Array) => void): this;
    on(event: 'close' | 'end', listener: () => void): this;
    on(event: 'error', listener: (error: Error) => void): this;
  }
}

declare module 'node:http' {
  import type { Socket } from 'node:net';

  export interface IncomingMessage {
    headers: Record<string, string | string[] | undefined>;
    url?: string;
    method?: string;
  }

  export interface ServerResponse {
    writeHead(status: number, headers?: Record<string, string>): this;
    end(body?: string): void;
  }

  export interface Server {
    listen(port: number, host: string, callback?: () => void): this;
    address(): { port: number; address: string } | string | null;
    close(callback?: (error?: Error) => void): this;
    on(event: 'upgrade', listener: (req: IncomingMessage, socket: Socket, head: Uint8Array) => void): this;
    on(event: 'error', listener: (error: Error) => void): this;
    once(event: 'error', listener: (error: Error) => void): this;
  }

  export function createServer(handler: (req: IncomingMessage, res: ServerResponse) => void): Server;
}

declare module 'node:crypto' {
  export function createHash(algorithm: 'sha1'): { update(data: string): { digest(encoding: 'base64'): string } };
  export function timingSafeEqual(a: Uint8Array, b: Uint8Array): boolean;
}

declare module 'node:child_process' {
  interface Readable {
    on(event: 'data', listener: (chunk: Uint8Array) => void): this;
  }
  interface Writable {
    write(data: string): boolean;
    end(): void;
    on(event: 'error', listener: (error: Error) => void): this;
  }
  export interface ChildProcess {
    readonly pid?: number;
    readonly stdin: Writable | null;
    readonly stdout: Readable | null;
    readonly stderr: Readable | null;
    kill(signal?: string): boolean;
    on(event: 'exit', listener: (code: number | null, signal: string | null) => void): this;
    on(event: 'error', listener: (error: Error) => void): this;
  }
  export function spawn(command: string, args: readonly string[], options?: { stdio?: ['pipe', 'pipe', 'pipe'] }): ChildProcess;
}

declare module 'node:util' {
  type OptionConfig = { type: 'string' | 'boolean'; multiple?: boolean; short?: string; default?: string | boolean | string[] };
  export function parseArgs(config: {
    args?: string[];
    options: Record<string, OptionConfig>;
    allowPositionals?: boolean;
  }): { values: Record<string, string | boolean | string[] | undefined>; positionals: string[] };
}

declare const process: {
  argv: string[];
  exitCode: number | undefined;
  stdout: { write(text: string): boolean };
  stderr: { write(text: string): boolean };
  on(event: 'SIGINT' | 'SIGTERM', listener: () => void): void;
};
