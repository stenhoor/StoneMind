/**
 * KataGo's position hashes, as its analysis engine reports them in `rootInfo` (`thisHash`, `symHash`). Ported from KataGo
 * v1.18.2: its random number generator (`core/rand.cpp`, `core/rand_helpers.h`, `core/md5.cpp`), the hash tables of
 * `Board::initHash`, and `Board::getSitHashWithSimpleKo`.
 *
 * KataGo builds its hash tables at startup from fixed seed strings, so they're the same in every KataGo build. Its seeding
 * hashes the seed with SHA-256; that comes from the platform's Web Crypto (`crypto.subtle`), which is asynchronous, so the
 * tables are built once by `loadPositionHashes()` before `positionHashes()` can be used. Web Crypto needs a secure
 * context in a browser (https or localhost).
 */
import { NULL_LOC, BLACK, WHITE, EMPTY, type Board, type Stone } from './board.js';

const MASK64 = (1n << 64n) - 1n;
const MASK32 = 0xffffffffn;

// ---------------------------------------------------------------- MD5 (KataGo core/md5.cpp)

const MD5_R = [
  7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11,
  16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
// KataGo's k[]: the binary integer parts of the sines of the integers 1 to 64.
const MD5_K = [
  0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8,
  0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
  0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87,
  0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
  0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039,
  0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
  0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
  0xeb86d391,
];

/** KataGo `MD5::get`: the four 32-bit words of the digest of `msg`. */
export function md5Words(msg: Uint8Array): [number, number, number, number] {
  let h0 = 0x67452301;
  let h1 = 0xefcdab89;
  let h2 = 0x98badcfe;
  let h3 = 0x10325476;
  let newLen = msg.length * 8 + 1;
  while (newLen % 512 !== 448) newLen++;
  newLen /= 8;
  const buf = new Uint8Array(newLen + 64);
  buf.set(msg);
  buf[msg.length] = 128;
  const view = new DataView(buf.buffer);
  // The length in bits, little-endian, as KataGo's memcpy on a little-endian machine writes it.
  const bitsLen = BigInt(msg.length) * 8n;
  view.setUint32(newLen, Number(bitsLen & MASK32), true);
  view.setUint32(newLen + 4, Number((bitsLen >> 32n) & MASK32), true);
  for (let offset = 0; offset < newLen; offset += 64) {
    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    for (let i = 0; i < 64; i++) {
      let f: number;
      let g: number;
      if (i < 16) {
        f = (b & c) | (~b & d);
        g = i;
      } else if (i < 32) {
        f = (d & b) | (~d & c);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;
      } else {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;
      }
      const temp = d;
      d = c;
      c = b;
      const x = (a + f + MD5_K[i]! + view.getUint32(offset + g * 4, true)) >>> 0;
      const r = MD5_R[i]!;
      b = (b + ((x << r) | (x >>> (32 - r)))) >>> 0;
      a = temp;
    }
    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
  }
  return [h0, h1, h2, h3];
}

// ---------------------------------------------------------------- generators (KataGo core/rand_helpers.h)

/** KataGo `XorShift1024Mult` (xorshift1024*). */
export class XorShift1024Mult {
  readonly #a: bigint[];
  #idx = 0;
  constructor(initA: readonly bigint[]) {
    if (initA.length !== 16) throw new Error('XorShift1024Mult needs 16 words');
    this.#a = [...initA];
  }
  nextUInt(): number {
    let a0 = this.#a[this.#idx]!;
    this.#idx = (this.#idx + 1) & 15;
    let a1 = this.#a[this.#idx]!;
    a1 ^= (a1 << 31n) & MASK64;
    a1 ^= a1 >> 11n;
    a0 ^= a0 >> 30n;
    this.#a[this.#idx] = a0 ^ a1;
    const result = (this.#a[this.#idx]! * 1181783497276652981n) & MASK64;
    return Number(result >> 32n);
  }
}

/** KataGo `PCG32`. */
export class PCG32 {
  #s: bigint;
  constructor(state: bigint) {
    this.#s = state & MASK64;
  }
  nextUInt(): number {
    this.#s = (this.#s * 6364136223846793005n + 1442695040888963407n) & MASK64;
    const s = this.#s;
    const x = Number((((s >> 18n) ^ s) >> 27n) & MASK32);
    const rot = Number(s >> 59n);
    return rot === 0 ? x : ((x >>> rot) | (x << (32 - rot))) >>> 0;
  }
}

/** KataGo's SHA2::get256 into four 64-bit words: each the next 8 bytes of the digest, big-endian. */
async function sha256Words(text: string): Promise<bigint[]> {
  const digest = new DataView(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)));
  return [0, 8, 16, 24].map((i) => digest.getBigUint64(i, false));
}

/** KataGo `Rand`, seeded from a string as `Rand::init(const string&)` does. Only what the hash tables use is ported. */
export class KataGoRand {
  readonly #xorm: XorShift1024Mult;
  readonly #pcg32: PCG32;
  private constructor(xorm: XorShift1024Mult, pcg32: PCG32) {
    this.#xorm = xorm;
    this.#pcg32 = pcg32;
  }

  static async create(seed: string): Promise<KataGoRand> {
    const s = `|${md5Words(new TextEncoder().encode(seed))[0]}|${seed}`;
    let counter = 0;
    let words: bigint[] = [];
    let next = 4;
    const getNonzero = async (): Promise<bigint> => {
      let value: bigint;
      do {
        if (next >= 4) {
          words = await sha256Words(`${counter}${s}`);
          counter += 37;
          next = 0;
        }
        value = words[next]!;
        next += 1;
      } while (value === 0n);
      return value;
    };
    const initA: bigint[] = [];
    for (let i = 0; i < 16; i++) initA.push(await getNonzero());
    const xorm = new XorShift1024Mult(initA);
    const pcg32 = new PCG32(await getNonzero());
    return new KataGoRand(xorm, pcg32);
  }

  /** KataGo `Rand::nextUInt`. */
  nextUInt(): number {
    return (this.#pcg32.nextUInt() + this.#xorm.nextUInt()) >>> 0;
  }

  /** KataGo `Rand::nextUInt64`. */
  nextUInt64(): bigint {
    const lower = BigInt(this.nextUInt());
    const upper = BigInt(this.nextUInt()) << 32n;
    return lower | upper;
  }
}

// ---------------------------------------------------------------- hash tables (KataGo Board::initHash)

/** KataGo `Hash128`. */
interface Hash128 {
  hash0: bigint;
  hash1: bigint;
}

const ZERO: Hash128 = { hash0: 0n, hash1: 0n };
const xor = (a: Hash128, b: Hash128): Hash128 => ({ hash0: a.hash0 ^ b.hash0, hash1: a.hash1 ^ b.hash1 });
/** KataGo `Hash128::operator<`: the high word first. */
const less = (a: Hash128, b: Hash128): boolean => a.hash1 < b.hash1 || (a.hash1 === b.hash1 && a.hash0 < b.hash0);
/** KataGo `Global::uint64ToHexString`, high word then low word, as `rootInfo` prints a hash. */
const hex = (h: Hash128): string => h.hash1.toString(16).toUpperCase().padStart(16, '0') + h.hash0.toString(16).toUpperCase().padStart(16, '0');

/** The largest board side this engine supports (KataGo's default `MAX_LEN`). */
const MAX_LEN = 19;
/** KataGo `Board::MAX_ARR_SIZE` for MAX_LEN 19. */
const MAX_ARR_SIZE = (MAX_LEN + 1) * (MAX_LEN + 2) + 1;

interface Tables {
  player: Hash128[];
  board: Hash128[][];
  koLoc: Hash128[];
  sizeX: Hash128[];
  sizeY: Hash128[];
}

let tables: Tables | null = null;
let loading: Promise<void> | null = null;

/**
 * Build KataGo's hash tables (`Board::initHash`), once. Only the tables the situation hash reads are kept. The others
 * come from separately seeded streams or follow these in the same stream, so leaving them out changes nothing here.
 */
export function loadPositionHashes(): Promise<void> {
  loading ??= (async () => {
    let rand = await KataGoRand.create('Board::initHash()');
    const nextHash = (): Hash128 => {
      const hash0 = rand.nextUInt64();
      const hash1 = rand.nextUInt64();
      return { hash0, hash1 };
    };
    const player: Hash128[] = [];
    for (let i = 0; i < 4; i++) player.push(nextHash());
    // ZOBRIST_ENCORE_HASH, not used here.
    for (let i = 0; i < 3; i++) nextHash();
    const board: Hash128[][] = [];
    const koLoc: Hash128[] = [];
    for (let i = 0; i < MAX_ARR_SIZE; i++) {
      const row: Hash128[] = [];
      for (let j = 0; j < 4; j++) {
        const onBoard = j === BLACK || j === WHITE;
        row.push(onBoard ? nextHash() : ZERO);
        // ZOBRIST_KO_MARK_HASH, not used here.
        if (onBoard) nextHash();
      }
      board.push(row);
      koLoc.push(nextHash());
    }
    rand = await KataGoRand.create('Board::initHash() for ZOBRIST_SIZE hashes');
    const sizeX: Hash128[] = [];
    const sizeY: Hash128[] = [];
    for (let i = 0; i < MAX_LEN + 1; i++) {
      sizeX.push(nextHash());
      sizeY.push(nextHash());
    }
    tables = { player, board, koLoc, sizeX, sizeY };
  })();
  return loading;
}

/**
 * KataGo `getSitHashWithSimpleKo` of the board under `symmetry` (as `SymmetryHelpers::getSymBoard` builds it): the
 * board's size and stones, its simple ko point, and the player to move.
 */
function sitHash(t: Tables, board: Board, pla: Stone, symmetry: number): Hash128 {
  const transpose = (symmetry & 4) !== 0;
  const flipX = (symmetry & 2) !== 0;
  const flipY = (symmetry & 1) !== 0;
  const symXSize = transpose ? board.ySize : board.xSize;
  const symYSize = transpose ? board.xSize : board.ySize;
  const symLocOf = (x: number, y: number): number => {
    let sx = flipX ? board.xSize - x - 1 : x;
    let sy = flipY ? board.ySize - y - 1 : y;
    if (transpose) [sx, sy] = [sy, sx];
    return sx + 1 + (sy + 1) * (symXSize + 1);
  };
  let h = xor(t.sizeX[symXSize]!, t.sizeY[symYSize]!);
  let symKoLoc = NULL_LOC;
  for (let y = 0; y < board.ySize; y++) {
    for (let x = 0; x < board.xSize; x++) {
      const loc = board.loc(x, y);
      const c = board.colors[loc]!;
      const symLoc = symLocOf(x, y);
      if (c !== EMPTY) h = xor(h, t.board[symLoc]![c]!);
      if (loc === board.koLoc) symKoLoc = symLoc;
    }
  }
  if (symKoLoc !== NULL_LOC) h = xor(h, t.koLoc[symKoLoc]!);
  return xor(h, t.player[pla]!);
}

/**
 * The hashes KataGo's analysis reports in `rootInfo`: `thisHash`, the position with the player to move, and `symHash`,
 * the smallest of that hash over the board's 8 symmetries. Needs `loadPositionHashes()` first.
 */
export function positionHashes(board: Board, pla: Stone): { thisHash: string; symHash: string } {
  if (!tables) throw new Error('call loadPositionHashes() before positionHashes()');
  if (board.xSize > MAX_LEN || board.ySize > MAX_LEN) throw new Error(`boards up to ${MAX_LEN} lines only`);
  const thisHash = sitHash(tables, board, pla, 0);
  let symHash = thisHash;
  for (let symmetry = 1; symmetry < 8; symmetry++) {
    const h = sitHash(tables, board, pla, symmetry);
    if (less(h, symHash)) symHash = h;
  }
  return { thisHash: hex(thisHash), symHash: hex(symHash) };
}
