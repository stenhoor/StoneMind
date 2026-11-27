/**
 * The Student's t cumulative distribution as KataGo's search tabulates it for value weighting: KataGo's
 * `FancyMath::tdistcdf` (through its regularized incomplete beta function, by the modified Lentz continued fraction)
 * sampled into a `DistributionTable` and read back by linear interpolation (`cpp/core/fancymath.cpp`,
 * `cpp/search/distributiontable.{h,cpp}`, KataGo 1.18, MIT, © David J Wu).
 *
 * JavaScript has no `lgamma`, which KataGo takes from the C library; `logGamma` here is the Lanczos approximation
 * (g = 7, 9 coefficients), accurate to about 1e-15.
 */

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** ln Γ(x) for x > 0. */
export function logGamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
  x -= 1;
  let a = LANCZOS[0]!;
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i]! / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b);
}

/** KataGo `evaluateContinuedFractionHelper`: the modified Lentz algorithm. */
function continuedFraction(numer: (n: number) => number, denom: (n: number) => number, tolerance: number, maxTerms: number): number {
  const tiny = 1e-300;
  let ret = denom(0);
  if (ret === 0) ret = tiny;
  let c = ret;
  let d = 0;
  for (let n = 1; n < maxTerms; n++) {
    const nextNumer = numer(n);
    const nextDenom = denom(n);
    d = nextDenom + nextNumer * d;
    if (d === 0) d = tiny;
    c = nextDenom + nextNumer / c;
    if (c === 0) c = tiny;
    d = 1 / d;
    const mult = c * d;
    ret = ret * mult;
    if (Math.abs(mult - 1) <= tolerance) break;
  }
  return ret;
}

/** KataGo `incompleteBetaContinuedFraction`. */
function incompleteBetaContinuedFraction(x: number, a: number, b: number): number {
  const numer = (n: number) => {
    if (n % 2 === 0) {
      const m = n / 2;
      return (m * (b - m) * x) / (a + 2 * m - 1) / (a + 2 * m);
    }
    const m = (n - 1) / 2;
    return (-(a + m) * (a + b + m) * x) / (a + 2 * m) / (a + 2 * m + 1);
  };
  return continuedFraction(numer, () => 1, 1e-15, 100000);
}

/** KataGo `FancyMath::regularizedIncompleteBeta`. */
export function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (!(x >= 0 && x <= 1 && a > 0 && b > 0)) return NaN;
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const logx = Math.log(x);
  const logy = Math.log(1 - x);
  if (x <= (a + 1) / (a + b + 2)) return Math.exp(logx * a + logy * b - logBeta(a, b)) / a / incompleteBetaContinuedFraction(x, a, b);
  return 1 - Math.exp(logy * b + logx * a - logBeta(a, b)) / b / incompleteBetaContinuedFraction(1 - x, b, a);
}

/** KataGo `FancyMath::tdistcdf`. */
export function tDistCdf(x: number, degreesOfFreedom: number): number {
  const v = degreesOfFreedom;
  if (!(v > 0)) return NaN;
  const half = regularizedIncompleteBeta(v / (x * x + v), v / 2, 0.5) / 2;
  return x >= 0 ? 1 - half : half;
}

/** KataGo `DistributionTable`, the cumulative half only: `size` samples from `minZ` to `maxZ`, interpolated linearly. */
export class CdfTable {
  readonly #cdf: Float64Array;

  constructor(cdf: (z: number) => number, readonly minZ: number, readonly maxZ: number, readonly size: number) {
    this.#cdf = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      if (i === 0) this.#cdf[i] = 0;
      else if (i === size - 1) this.#cdf[i] = 1;
      else this.#cdf[i] = cdf(minZ + (i * (maxZ - minZ)) / (size - 1));
    }
  }

  getCdf(z: number): number {
    const d = ((this.size - 1) * (z - this.minZ)) / (this.maxZ - this.minZ);
    if (d <= 0) return 0;
    const idx = Math.trunc(d);
    if (idx >= this.size - 1) return 1;
    const lambda = d - idx;
    const y0 = this.#cdf[idx]!;
    const y1 = this.#cdf[idx + 1]!;
    return y0 + lambda * (y1 - y0);
  }
}

/** KataGo `VALUE_WEIGHT_DEGREES_OF_FREEDOM`. */
const VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;

/** KataGo's `valueWeightDistribution`: the t distribution with 3 degrees of freedom, from −50 to 50 in 2000 samples. */
export const VALUE_WEIGHT_DISTRIBUTION = new CdfTable((z) => tDistCdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM), -50, 50, 2000);
