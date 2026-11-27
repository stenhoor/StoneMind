// The frame's rules (src/frame/logic.ts), on the built modules.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  BAR_IDS, COMPACT_BELOW, NARROW_BELOW, clockText, deviceProfile, menuBarMode, parseBackdrop, parseTheme, toggleBar,
} from '../dist/js/frame/logic.js';

test('at most one bar is active: pressing a bar shows it, pressing it again hides it', () => {
  assert.equal(toggleBar(null, 'apps'), 'apps');
  assert.equal(toggleBar('apps', 'apps'), null);
  for (const a of BAR_IDS) for (const b of BAR_IDS) assert.equal(toggleBar(a, b), a === b ? null : b);
});

test('menu bar mode follows its own width', () => {
  assert.equal(menuBarMode(390), 'narrow');
  assert.equal(menuBarMode(NARROW_BELOW - 1), 'narrow');
  assert.equal(menuBarMode(NARROW_BELOW), 'compact');
  assert.equal(menuBarMode(COMPACT_BELOW - 1), 'compact');
  assert.equal(menuBarMode(COMPACT_BELOW), 'full');
  assert.equal(menuBarMode(1440), 'full');
});

test('device profile uses the same widths', () => {
  assert.equal(deviceProfile(390), 'mobile');
  assert.equal(deviceProfile(820), 'tablet');
  assert.equal(deviceProfile(1440), 'desktop');
});

test('the clock shows the time only when narrow', () => {
  const d = new Date(2026, 10, 27, 9, 5);
  const narrow = clockText(d, 'narrow', 'en-GB');
  const full = clockText(d, 'full', 'en-GB');
  assert.match(narrow, /^0?9:05$/);
  assert.match(full, /Fri/);
  assert.match(full, /27/);
  assert.match(full, /0?9:05/);
});

test('unknown stored settings fall back to the defaults', () => {
  assert.equal(parseBackdrop(null), 'plain');
  assert.equal(parseBackdrop('marble'), 'plain');
  assert.equal(parseBackdrop('wood'), 'wood');
  assert.equal(parseTheme(null), 'system');
  assert.equal(parseTheme('night'), 'night');
  assert.equal(parseTheme('neon'), 'system');
});
