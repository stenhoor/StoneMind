// The player's frame settings: theme and backdrop.
//
// Until the materials kit exists (design doc, Kits), these are kept in localStorage. The theme uses the brand's own key,
// `stonemind.theme` (packages/brand/components/ThemePicker), so the theme script in <head> can apply it before first
// paint. Recorded in docs/OPEN-QUESTIONS.md.
import { type Backdrop, type Theme, parseBackdrop, parseTheme } from './logic.js';

const THEME_KEY = 'stonemind.theme';
const BACKDROP_KEY = 'stonemind.backdrop';

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable (private windows, blocked site data). The setting then lasts for this page only.
  }
}

export function loadTheme(): Theme {
  return parseTheme(read(THEME_KEY));
}

/** Applies a theme to the page and saves it. "system" removes data-theme, so the tokens follow the device. */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === 'system') delete root.dataset['theme'];
  else root.dataset['theme'] = theme;
  write(THEME_KEY, theme === 'system' ? null : theme);
}

export function loadBackdrop(): Backdrop {
  return parseBackdrop(read(BACKDROP_KEY));
}

export function applyBackdrop(backdrop: Backdrop): void {
  document.documentElement.dataset['backdrop'] = backdrop;
  write(BACKDROP_KEY, backdrop === 'plain' ? null : backdrop);
}
