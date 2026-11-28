// KataGo weights: what the file name says, and what the start of the file says.
//
// - File names: KataGo's python/train.py names exported networks "%s-s%d-d%d" % (exportprefix, global_step_samples,
//   total_num_data_rows), and the prefix is the run plus the network's size, e.g. "kata1-b18c384nbt". So
//   "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz" is run kata1, 18 blocks, 384 channels, "nbt" (nested bottleneck),
//   trained on 9,996,604,416 samples with 4,316,597,426 rows of training data.
// - File contents: after gunzip, a model file starts with its name and then its model version
//   (packages/engine-webgpu/src/model/parse.ts reads them the same way).

export interface WeightsName {
  readonly run: string;
  readonly blocks: number;
  readonly channels: number;
  /** Letters after the channel count, such as "nbt"; empty for the plain residual net. */
  readonly kind: string;
  /** Training samples (s) and rows of training data (d), as written in the name. */
  readonly samples: bigint;
  readonly dataRows: bigint;
  readonly format: 'binary' | 'text';
}

const NAME = /^(?<run>[A-Za-z0-9_]+)-b(?<blocks>\d+)c(?<channels>\d+)(?<kind>[a-z]*)-s(?<s>\d+)-d(?<d>\d+)\.(?<fmt>bin|txt)\.gz$/;

/** Reads a KataGo weights file name, or returns null if it isn't one in KataGo's naming. */
export function parseWeightsName(fileName: string): WeightsName | null {
  const m = NAME.exec(fileName);
  const g = m?.groups;
  if (!g) return null;
  return {
    run: g['run'] as string,
    blocks: Number(g['blocks']),
    channels: Number(g['channels']),
    kind: g['kind'] as string,
    samples: BigInt(g['s'] as string),
    dataRows: BigInt(g['d'] as string),
    format: g['fmt'] === 'bin' ? 'binary' : 'text',
  };
}

const KINDS: Readonly<Record<string, string>> = { '': 'residual', nbt: 'nested bottleneck' };

/** "b18c384nbt: 18 blocks, 384 channels, nested bottleneck". */
export function describeNetwork(w: WeightsName): string {
  const kind = KINDS[w.kind] ?? w.kind;
  return `b${w.blocks}c${w.channels}${w.kind}: ${w.blocks} blocks, ${w.channels} channels, ${kind}`;
}

/** A whole number with thousands separators, in the browser's own format. */
export function grouped(n: bigint | number, locale?: string): string {
  return new Intl.NumberFormat(locale).format(n);
}

export interface WeightsHeader {
  readonly name: string;
  readonly modelVersion: number;
}

/**
 * Reads the model name and version from the start of a weights file (gzipped or not), decompressing only the first
 * few kilobytes. Rejects if the file doesn't start like a KataGo model.
 */
export async function readWeightsHeader(file: Blob): Promise<WeightsHeader> {
  const start = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  const gzipped = start[0] === 0x1f && start[1] === 0x8b;
  const raw = file.stream();
  const stream = gzipped ? raw.pipeThrough(new DecompressionStream('gzip') as unknown as ReadableWritablePair<Uint8Array, Uint8Array>) : raw;
  const reader = stream.getReader();
  const decoder = new TextDecoder('latin1');
  let text = '';
  try {
    while (text.length < 4096) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      const tokens = text.trimStart().split(/\s+/);
      if (tokens.length >= 3) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  const [name, version] = text.trimStart().split(/\s+/);
  const modelVersion = Number(version);
  if (!name || !Number.isInteger(modelVersion)) throw new Error('This file does not start like a KataGo model');
  return { name, modelVersion };
}

/** The SHA-256 of a file, as hex. Reads the whole file into memory. */
export async function sha256(file: Blob): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** "98.2 MB", "11 KB", "512 bytes". */
export function formatBytes(n: number | undefined): string {
  if (n === undefined) return '';
  if (n < 1024) return `${n} bytes`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}
