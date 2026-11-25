/**
 * The 8 board symmetries, ported from `copyWithSymmetry` in KataGo's `cpp/neuralnet/nninputs.cpp` (KataGo 1.18, MIT,
 * © David J Wu). Bit 0 flips y, bit 1 flips x, bit 2 transposes (square boards only).
 *
 * KataGo transforms the inputs with a symmetry, runs the net, and maps the spatial outputs back with the reverse copy.
 */

/**
 * Copy channel-major planes [c][h][w] under a symmetry.
 * @param reverse false for inputs (original → symmetric), true for outputs (symmetric → original).
 */
export function copyWithSymmetry(
  src: Float32Array,
  channels: number,
  hSize: number,
  wSize: number,
  symmetry: number,
  reverse: boolean,
): Float32Array {
  const transpose = (symmetry & 4) !== 0 && hSize === wSize;
  let flipX = (symmetry & 2) !== 0;
  let flipY = (symmetry & 1) !== 0;
  if (transpose && !reverse) [flipX, flipY] = [flipY, flipX];
  const dst = new Float32Array(src.length);
  const plane = hSize * wSize;
  let hBase = 0;
  let hStride = wSize;
  let wBase = 0;
  let wStride = 1;
  if (flipY) {
    hBase = (hSize - 1) * hStride;
    hStride = -hStride;
  }
  if (flipX) {
    wBase = (wSize - 1) * wStride;
    wStride = -wStride;
  }
  if (transpose) [hStride, wStride] = [wStride, hStride];
  for (let c = 0; c < channels; c++) {
    const base = c * plane;
    for (let h = 0; h < hSize; h++) {
      for (let w = 0; w < wSize; w++) {
        dst[base + hBase + h * hStride + wBase + w * wStride] = src[base + h * wSize + w]!;
      }
    }
  }
  return dst;
}
