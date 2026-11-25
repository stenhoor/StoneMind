import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCss } from '../scripts/build-tokens.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const tokens = JSON.parse(readFileSync(join(root, 'tokens', 'tokens.json'), 'utf8'));
const themes = tokens.color.themes;

/** Parse #RRGGBB or rgba(r, g, b, a) into [r, g, b, a]. */
function parse(value) {
  const hex = /^#([0-9a-f]{6})$/i.exec(value);
  if (hex) return [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16)).concat(1);
  const rgba = /^rgba?\(([^)]+)\)$/.exec(value);
  if (rgba) {
    const [r, g, b, a = 1] = rgba[1].split(',').map((s) => parseFloat(s));
    return [r, g, b, a];
  }
  throw new Error(`unsupported colour ${value}`);
}

const composite = ([r, g, b, a], [ur, ug, ub]) => [r * a + ur * (1 - a), g * a + ug * (1 - a), b * a + ub * (1 - a), 1];

function luminance([r, g, b]) {
  const lin = (c) => {
    c /= 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

function value(name, theme) {
  const token = tokens.color.tokens.find((t) => t.name === name);
  assert.ok(token, `unknown token ${name}`);
  const v = typeof token.value === 'object' ? token.value[theme.id] : token.value;
  assert.ok(v, `token ${name} has no value for theme ${theme.id}`);
  return parse(v);
}

test('every colour token has a value for every theme', () => {
  for (const t of tokens.color.tokens) {
    if (typeof t.value !== 'object') continue;
    for (const theme of themes) assert.ok(t.value[theme.id], `${t.name} is missing ${theme.id}`);
  }
});

test('exactly one default theme, and it is light', () => {
  const defaults = themes.filter((t) => t.default);
  assert.equal(defaults.length, 1);
  assert.equal(defaults[0].scheme, 'light');
});

for (const theme of themes) {
  test(`contrast rules hold for ${theme.id} on opaque grounds`, () => {
    for (const { fg, bg, min } of tokens.contrast.pairs) {
      for (const ground of bg) {
        const ratio = contrast(value(fg, theme), value(ground, theme));
        assert.ok(ratio >= min, `${theme.id}: ${fg} on ${ground} is ${ratio.toFixed(2)}:1, needs ${min}:1`);
      }
    }
  });

  test(`contrast rules hold for ${theme.id} on glass over every backdrop`, () => {
    const { tint, backdrops, pairs } = tokens.contrast.glass;
    for (const backdrop of backdrops) {
      const ground = composite(value(tint, theme), value(backdrop, theme));
      for (const { fg, min } of pairs) {
        const ratio = contrast(value(fg, theme), ground);
        assert.ok(ratio >= min, `${theme.id}: ${fg} on glass over ${backdrop} is ${ratio.toFixed(2)}:1, needs ${min}:1`);
      }
    }
  });

  test(`white-on-primary glass button is AA for ${theme.id}`, () => {
    for (const backdrop of tokens.contrast.glass.backdrops) {
      const ground = composite(value('primary-glass', theme), value(backdrop, theme));
      const ratio = contrast(value('on-primary', theme), ground);
      assert.ok(ratio >= 4.5, `${theme.id}: on-primary on primary-glass over ${backdrop} is ${ratio.toFixed(2)}:1`);
    }
  });
}

test('tokens.css is generated from tokens.json and up to date', () => {
  assert.equal(readFileSync(join(root, 'tokens', 'tokens.css'), 'utf8'), buildCss(tokens));
});

test('a dark theme applies under prefers-color-scheme only when no theme is chosen', () => {
  const css = buildCss(tokens);
  assert.match(css, /@media \(prefers-color-scheme: dark\) \{\n  :root:not\(\[data-theme\]\)/);
});

test('every logo asset listed in BRAND.md exists', () => {
  const brand = readFileSync(join(root, 'BRAND.md'), 'utf8');
  const listed = [...brand.matchAll(/`logos\/([a-z0-9-]+\.svg)`/g)].map((m) => m[1]);
  assert.ok(listed.length >= 6, 'BRAND.md lists the logo files');
  for (const file of listed) assert.ok(existsSync(join(root, 'logos', file)), `logos/${file} is missing`);
});
