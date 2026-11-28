// The GPU check page (test/gpu/*.js) imports from the built package. CI runs the page only when the WebGPU network or the
// page itself changes (scripts/ci-affected.mjs), not when src/index.ts does, so this checks that every name the page
// imports is still exported.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const dir = fileURLToPath(new URL('./gpu/', import.meta.url));
for (const file of readdirSync(dir).filter((f) => f.endsWith('.js'))) {
  const source = readFileSync(dir + file, 'utf8');
  for (const [, names, from] of source.matchAll(/import\s*\{([^}]*)\}\s*from\s*'(\.\.\/\.\.\/dist\/[^']+)'/g)) {
    test(`the GPU check page's ${file} finds what it imports from ${from.slice(6)}`, async () => {
      const module = await import(new URL(from, `file://${dir}`).href);
      const missing = names.split(',').map((n) => n.trim().split(/\s+as\s+/)[0]).filter((n) => n && !(n in module));
      assert.deepEqual(missing, [], 'names not exported');
    });
  }
}
