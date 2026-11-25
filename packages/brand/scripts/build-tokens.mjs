#!/usr/bin/env node
// Generate tokens/tokens.css from tokens/tokens.json.
//
// Usage: node scripts/build-tokens.mjs           write tokens/tokens.css
//        node scripts/build-tokens.mjs --check   exit 1 if tokens/tokens.css is out of date
//
// The default theme goes on :root and [data-theme="<id>"]; every other theme on [data-theme="<id>"].
// A dark theme also applies under prefers-color-scheme: dark when the page has not chosen a theme
// (no data-theme attribute), so a player's explicit choice always wins over the system setting.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const tokensPath = join(root, 'tokens', 'tokens.json');
const cssPath = join(root, 'tokens', 'tokens.css');

export function buildCss(tokens) {
  const themes = tokens.color.themes;
  const base = themes.find((t) => t.default) ?? themes[0];

  const colourBlock = (theme, indent = '  ') =>
    tokens.color.tokens
      .map((t) => {
        const v = typeof t.value === 'object' ? (t.value[theme.id] ?? t.value[base.id]) : t.value;
        return `${indent}--${t.name}: ${v};`;
      })
      .join('\n');

  const parts = [
    `/* ${tokens.name}: generated from tokens/tokens.json by scripts/build-tokens.mjs. Do not edit by hand. */`,
    `:root, [data-theme="${base.id}"] {\n  color-scheme: ${base.scheme};\n${colourBlock(base)}\n}`,
  ];
  for (const theme of themes.filter((t) => t !== base)) {
    parts.push(`[data-theme="${theme.id}"] {\n  color-scheme: ${theme.scheme};\n${colourBlock(theme)}\n}`);
    if (theme.scheme === 'dark') {
      parts.push(
        `@media (prefers-color-scheme: dark) {\n  :root:not([data-theme]) {\n    color-scheme: dark;\n${colourBlock(theme, '    ')}\n  }\n}`,
      );
    }
  }

  const statics = [];
  for (const family of ['spacing', 'radius']) {
    for (const t of tokens[family]?.tokens ?? []) statics.push(`  --${t.name}: ${t.value};`);
  }
  for (const [key, stack] of Object.entries(tokens.type.families)) statics.push(`  --font-${key}: ${stack};`);
  statics.push(`  --glass-blur: ${tokens.glass.blur};`, `  --glass-saturate: ${tokens.glass.saturate};`);
  parts.push(`:root {\n${statics.join('\n')}\n}`);

  const typeClasses = [];
  for (const group of tokens.type.groups) {
    for (const s of group.styles) {
      const props = [
        `font-family: var(--font-${s.family ?? group.family})`,
        `font-size: ${s.fontSize}`,
        `line-height: ${s.lineHeight}`,
        `font-weight: ${s.fontWeight}`,
      ];
      if (s.letterSpacing) props.push(`letter-spacing: ${s.letterSpacing}`);
      typeClasses.push(`.${s.name} { ${props.join('; ')}; }`);
    }
  }
  parts.push(typeClasses.join('\n'));
  return parts.join('\n\n') + '\n';
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const css = buildCss(JSON.parse(readFileSync(tokensPath, 'utf8')));
  if (process.argv.includes('--check')) {
    if (readFileSync(cssPath, 'utf8') !== css) {
      console.error('tokens/tokens.css is out of date: run `npm run build -w @stonemind/brand`.');
      process.exit(1);
    }
    console.log('tokens/tokens.css is up to date');
  } else {
    writeFileSync(cssPath, css);
    console.log('wrote tokens/tokens.css');
  }
}
