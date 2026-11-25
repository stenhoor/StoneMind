/**
 * IEEE 754 binary16 ("half") conversion, for uploading fp16 weights. Written here because `Float16Array` is not
 * available in every runtime the engine targets (Node 22 has none).
 */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** The binary16 bit pattern nearest to `value` (round to nearest, ties to even; overflow to ±infinity; NaN kept). */
export function toFloat16Bits(value: number): number {
  f32[0] = value;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;
  if (exp === 0xff) return sign | (mant ? 0x7e00 : 0x7c00); // NaN (quiet) or infinity
  // Unbiased exponent; binary16 normals cover 2^-14 .. 2^15.
  const e = exp - 127;
  if (e > 15) return sign | 0x7c00;
  if (e >= -14) {
    // Normal: keep 10 mantissa bits, round the 13 dropped bits to nearest even.
    let h = ((e + 15) << 10) | (mant >>> 13);
    const rest = mant & 0x1fff;
    if (rest > 0x1000 || (rest === 0x1000 && (h & 1))) h += 1; // a carry into the exponent is still correct
    return sign | h;
  }
  if (e < -25) return sign; // below half the smallest subnormal: rounds to zero
  // Subnormal: value = m · 2^-24 with m < 1024. Shift the full 24-bit significand into place, round to nearest even.
  const sig = mant | 0x800000;
  const shift = -e - 1; // 14..24 for e in -15..-25: dropped bits = shift
  let h = sig >>> shift;
  const rest = sig & ((1 << shift) - 1);
  const half = 1 << (shift - 1);
  if (rest > half || (rest === half && (h & 1))) h += 1;
  return sign | h;
}

/** The number a binary16 bit pattern stands for. */
export function fromFloat16Bits(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Convert a Float32Array to binary16 bits, padded to an even length so the byte size is a multiple of 4. */
export function toFloat16Array(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length + (values.length & 1));
  for (let i = 0; i < values.length; i++) out[i] = toFloat16Bits(values[i]!);
  return out;
}
