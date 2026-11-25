// Download KataGo's small test network, g170-b6c96 (CC0: a g170-run network), pinned to a KataGo commit and checked
// by SHA-256. Its reference output is tests/results/runNNOnTinyBoardTest.txt at the same commit.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const KATAGO_COMMIT = 'd91ea855110dae533f0aada947b2b7d78cc8a4e1';
const FILE = 'g170-b6c96-s175395328-d26788732.bin.gz';
const SHA256 = 'f5d32604e3675c480c7c8f6aa579a1ea857135628a0afccc8fa56330fbacd38d';
const url = `https://raw.githubusercontent.com/lightvector/KataGo/${KATAGO_COMMIT}/cpp/tests/models/${FILE}`;

const dir = join(dirname(fileURLToPath(import.meta.url)), '..', 'test', 'fixtures');
const path = join(dir, FILE);
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');

if (existsSync(path) && sha(readFileSync(path)) === SHA256) process.exit(0);
mkdirSync(dir, { recursive: true });
const res = await fetch(url);
if (!res.ok) throw new Error(`download failed: ${res.status} ${url}`);
const bytes = new Uint8Array(await res.arrayBuffer());
if (sha(bytes) !== SHA256) throw new Error(`${FILE}: SHA-256 mismatch`);
writeFileSync(path, bytes);
console.log(`fetched ${FILE}`);
