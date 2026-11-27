/**
 * JSON as KataGo's analysis engine reads and writes it (it uses nlohmann::json). Two things differ from JavaScript's own
 * JSON and matter to the protocol:
 * - nlohmann keeps whether a number was written as an integer (`100`) or not (`100.0`, `1e2`), and KataGo's integer
 *   fields refuse the second. `parseJson` keeps that: a number written as an integer is a JavaScript number, any other
 *   number a `JsonFloat`.
 * - nlohmann writes objects with their keys sorted and floating-point numbers in its own format (`2.0`, `1e-05`).
 *   `dumpJson` writes the same way, so the engine's lines read like native KataGo's.
 */

/** A JSON number that was not written as an integer, or a floating-point value to write. */
export class JsonFloat {
  constructor(readonly value: number) {}
}

/** A parsed JSON value: integers are numbers, other numbers `JsonFloat`. */
export type JsonValue = null | boolean | number | JsonFloat | string | JsonValue[] | JsonObject;
export interface JsonObject {
  [key: string]: JsonValue;
}

// nlohmann reads an integer that fits neither int64 nor uint64 as a float.
const INT_LIMIT = 2 ** 64;

/**
 * Parse JSON, keeping integers apart from other numbers. Integers beyond 2^53 lose precision, as JavaScript numbers do
 * (nlohmann keeps them exactly up to 2^64).
 * @throws SyntaxError for text that isn't JSON.
 */
export function parseJson(text: string): JsonValue {
  return JSON.parse(text, (_key, value: unknown, context?: { source?: string }) => {
    if (typeof value !== 'number') return value;
    const source = context?.source;
    const integral = source !== undefined ? !/[.eE]/.test(source) : Number.isInteger(value);
    return integral && Math.abs(value) < INT_LIMIT ? value : new JsonFloat(value);
  }) as JsonValue;
}

export const isObject = (v: JsonValue | undefined): v is JsonObject =>
  typeof v === 'object' && v !== null && !Array.isArray(v) && !(v instanceof JsonFloat);
export const isNumber = (v: JsonValue | undefined): v is number | JsonFloat => typeof v === 'number' || v instanceof JsonFloat;
/** nlohmann `is_number_integer`. */
export const isInteger = (v: JsonValue | undefined): v is number => typeof v === 'number';
export const numberValue = (v: number | JsonFloat): number => (typeof v === 'number' ? v : v.value);
export const has = (o: JsonObject, key: string): boolean => Object.prototype.hasOwnProperty.call(o, key);

/** The error nlohmann throws when a value can't be read as the type asked for. */
export class JsonTypeError extends Error {}

/**
 * nlohmann `get<int>()`: any number (a non-integer is truncated) or a boolean, converted to a C++ int as x86-64 does.
 * @throws JsonTypeError otherwise.
 */
export function getInt(v: JsonValue | undefined): number {
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'number') return Number(BigInt.asIntN(32, BigInt(v)));
  if (v instanceof JsonFloat) {
    const t = Math.trunc(v.value);
    // Out of range, x86-64's conversion gives INT_MIN.
    return Number.isFinite(t) && t >= -(2 ** 31) && t < 2 ** 31 ? t : -(2 ** 31);
  }
  throw new JsonTypeError('type must be number');
}

/**
 * nlohmann `get<int64_t>()`: any number, a non-integer truncated (no booleans).
 * @throws JsonTypeError otherwise.
 */
export function getInt64(v: JsonValue | undefined): number {
  if (typeof v === 'number') return v;
  if (v instanceof JsonFloat) {
    const t = Math.trunc(v.value);
    return Number.isFinite(t) && t >= -(2 ** 63) && t < 2 ** 63 ? t : -(2 ** 63);
  }
  throw new JsonTypeError('type must be number');
}

/** nlohmann `get<vector<T>>()`: an array, each element read with `get`. @throws JsonTypeError otherwise. */
export function getArray<T>(v: JsonValue | undefined, get: (e: JsonValue) => T): T[] {
  if (!Array.isArray(v)) throw new JsonTypeError('type must be array');
  return v.map(get);
}

/** nlohmann `get<string>()`. @throws JsonTypeError for anything else. */
export function getString(v: JsonValue | undefined): string {
  if (typeof v !== 'string') throw new JsonTypeError('type must be string');
  return v;
}

/** nlohmann `get<bool>()`. @throws JsonTypeError for anything else. */
export function getBool(v: JsonValue | undefined): boolean {
  if (typeof v !== 'boolean') throw new JsonTypeError('type must be boolean');
  return v;
}

/** nlohmann `get<float>()`: any number, or a boolean (as 0 or 1). @throws JsonTypeError otherwise. */
export function getFloat(v: JsonValue | undefined): number {
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (isNumber(v)) return Math.fround(numberValue(v));
  throw new JsonTypeError('type must be number');
}

/**
 * A floating-point number as nlohmann writes it: the shortest digits that read back the same (nlohmann's Grisu2 can,
 * rarely, write a longer form), in plain notation with at least one decimal place when the point falls within 15 digits
 * left or 4 right of the first digit, else as `d.ddde±XX`. Not finite: `null`.
 */
export function formatDouble(x: number): string {
  if (!Number.isFinite(x)) return 'null';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const sign = x < 0 ? '-' : '';
  const [mantissa, exp] = Math.abs(x).toExponential().split('e') as [string, string];
  const digits = mantissa.replace('.', '');
  const k = digits.length;
  // The decimal point falls after the first n digits.
  const n = Number(exp) + 1;
  let body: string;
  if (k <= n && n <= 15) body = digits + '0'.repeat(n - k) + '.0';
  else if (0 < n && n <= 15) body = `${digits.slice(0, n)}.${digits.slice(n)}`;
  else if (-4 < n && n <= 0) body = `0.${'0'.repeat(-n)}${digits}`;
  else {
    const e = n - 1;
    const abs = Math.abs(e);
    body = `${digits[0]}${k > 1 ? '.' + digits.slice(1) : ''}e${e < 0 ? '-' : '+'}${abs < 10 ? '0' : ''}${abs}`;
  }
  return sign + body;
}

/** nlohmann `dump()`: compact, object keys sorted, numbers as nlohmann writes them. */
export function dumpJson(v: JsonValue): string {
  if (v === null) return 'null';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return String(v);
  if (v instanceof JsonFloat) return formatDouble(v.value);
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(dumpJson).join(',')}]`;
  const keys = Object.keys(v).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${dumpJson(v[k]!)}`).join(',')}}`;
}
