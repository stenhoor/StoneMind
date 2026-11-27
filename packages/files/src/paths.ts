// Paths in a FileStore: absolute, '/'-separated, no empty, "." or ".." segments. Free of the DOM, for node --test.

/** Why a path was refused. */
export class PathError extends Error {
  override readonly name = 'PathError';
}

const BAD_CHARS = /[\u0000-\u001f\\]/;

/** The path's segments, checked. "/" is the root and has none. */
export function segments(path: string): string[] {
  if (typeof path !== 'string' || !path.startsWith('/')) throw new PathError(`Paths start with "/": ${JSON.stringify(path)}`);
  const parts = path.split('/').filter((s, i, all) => !(s === '' && (i === 0 || i === all.length - 1)));
  for (const s of parts) {
    if (s === '') throw new PathError(`Empty folder name in ${JSON.stringify(path)}`);
    if (s === '.' || s === '..') throw new PathError(`"${s}" is not allowed in ${JSON.stringify(path)}`);
    if (BAD_CHARS.test(s)) throw new PathError(`Control characters and "\\" are not allowed in ${JSON.stringify(path)}`);
    if (s.length > 255) throw new PathError(`A name is longer than 255 characters in ${JSON.stringify(path)}`);
  }
  return parts;
}

/** The path in its one written form: "/a/b" (no trailing slash; the root is "/"). */
export function normalize(path: string): string {
  return '/' + segments(path).join('/');
}

export function join(dir: string, ...names: string[]): string {
  return normalize('/' + [...segments(dir), ...names].join('/'));
}

/** The folder holding `path` ("/" for a top-level entry). The root has no parent. */
export function parentOf(path: string): string {
  const parts = segments(path);
  if (parts.length === 0) throw new PathError('The root has no parent');
  return '/' + parts.slice(0, -1).join('/');
}

export function nameOf(path: string): string {
  const parts = segments(path);
  const last = parts[parts.length - 1];
  if (last === undefined) throw new PathError('The root has no name');
  return last;
}

/** Whether `inner` is `outer` or inside it. */
export function isWithin(inner: string, outer: string): boolean {
  const a = segments(inner);
  const b = segments(outer);
  return b.length <= a.length && b.every((s, i) => a[i] === s);
}
