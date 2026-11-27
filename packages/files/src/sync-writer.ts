// A dedicated worker that writes one file with `createSyncAccessHandle`, for browsers whose pages can't use
// `createWritable` (build plan, Milestone 1, step 3). The page sends:
//   { open: string[] }                 the path's segments; the file is created (or emptied)
//   { chunk: ArrayBuffer }             appended; answered with { written } once it is on disk
//   { close: true }                    flushed and closed; answered with { closed: true }
//   { abort: true }                    closed and the file removed; answered with { aborted: true }
// Any failure is answered with { error }.
export {};

interface SyncHandle {
  write(data: ArrayBufferView, options: { at: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

let handle: SyncHandle | null = null;
let dir: FileSystemDirectoryHandle | null = null;
let name = '';
let offset = 0;

const reply = (message: unknown) => (self as unknown as Worker).postMessage(message);

self.onmessage = async (event: MessageEvent) => {
  const m = event.data as { open?: string[]; chunk?: ArrayBuffer; close?: boolean; abort?: boolean };
  try {
    if (m.open) {
      let d = await navigator.storage.getDirectory();
      const parts = m.open.slice();
      name = parts.pop() ?? '';
      for (const part of parts) d = await d.getDirectoryHandle(part, { create: true });
      dir = d;
      const file = await d.getFileHandle(name, { create: true });
      handle = await (file as FileSystemFileHandle & { createSyncAccessHandle(): Promise<SyncHandle> }).createSyncAccessHandle();
      handle.truncate(0);
      offset = 0;
      reply({ opened: true });
    } else if (m.chunk && handle) {
      offset += handle.write(new Uint8Array(m.chunk), { at: offset });
      reply({ written: offset });
    } else if (m.close && handle) {
      handle.flush();
      handle.close();
      handle = null;
      reply({ closed: true });
    } else if (m.abort) {
      handle?.close();
      handle = null;
      if (dir && name) await dir.removeEntry(name).catch(() => {});
      reply({ aborted: true });
    }
  } catch (e) {
    reply({ error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) });
  }
};
