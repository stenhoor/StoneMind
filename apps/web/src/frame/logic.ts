// The frame's rules, kept free of the DOM so they can be tested with node --test.

/** The three buttons at the right of the menu bar. At most one is active; its bar shows under the menu bar. */
export type BarId = 'workspaces' | 'layouts' | 'apps';

export const BAR_IDS: readonly BarId[] = ['workspaces', 'layouts', 'apps'];

/** Pressing a bar button: it becomes the active bar, or, if it already was, no bar is active. */
export function toggleBar(current: BarId | null, pressed: BarId): BarId | null {
  return current === pressed ? null : pressed;
}

/**
 * How much the menu bar shows at a given width of its own (not the window's), so it always stays on one line.
 * - full: everything.
 * - compact: the status items and the layout name hide.
 * - narrow: also the wordmark becomes the stone alone, the menus fold into one Menus button and the clock shows the
 *   time only.
 */
export type MenuBarMode = 'full' | 'compact' | 'narrow';

export const NARROW_BELOW = 760;
export const COMPACT_BELOW = 1180;

export function menuBarMode(width: number): MenuBarMode {
  if (width < NARROW_BELOW) return 'narrow';
  if (width < COMPACT_BELOW) return 'compact';
  return 'full';
}

/** The device profile for the width of the app (design doc, Device profiles). Same widths as the menu bar's. */
export type DeviceProfile = 'mobile' | 'tablet' | 'desktop';

export function deviceProfile(width: number): DeviceProfile {
  if (width < NARROW_BELOW) return 'mobile';
  if (width < COMPACT_BELOW) return 'tablet';
  return 'desktop';
}

export const PROFILE_NAMES: Readonly<Record<DeviceProfile, string>> = {
  mobile: 'Mobile profile',
  tablet: 'Tablet profile',
  desktop: 'Desktop profile',
};

/** The clock's text, in the browser's own date and time format. Narrow shows the time only. */
export function clockText(date: Date, mode: MenuBarMode, locale?: string): string {
  const options: Intl.DateTimeFormatOptions =
    mode === 'narrow'
      ? { hour: 'numeric', minute: '2-digit' }
      : { weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' };
  return new Intl.DateTimeFormat(locale, options).format(date);
}

/** How often the clock updates, in milliseconds. */
export const CLOCK_INTERVAL = 15_000;

/**
 * The backdrop behind the glass (design doc, Backdrop). A player setting; plain table is the default.
 */
export type Backdrop = 'plain' | 'wood' | 'washi';

export const BACKDROPS: readonly Backdrop[] = ['plain', 'wood', 'washi'];

export const BACKDROP_NAMES: Readonly<Record<Backdrop, string>> = {
  plain: 'Plain table',
  wood: 'Wood grain',
  washi: 'Washi paper',
};

export function parseBackdrop(value: string | null): Backdrop {
  return BACKDROPS.find((b) => b === value) ?? 'plain';
}

/** The brand's themes (packages/brand/BRAND.md). "system" follows the device's light or dark setting. */
export type Theme = 'system' | 'washi' | 'hinoki' | 'night';

export const THEMES: readonly Theme[] = ['system', 'washi', 'hinoki', 'night'];

export const THEME_NAMES: Readonly<Record<Theme, string>> = {
  system: 'System',
  washi: 'Washi & Shu',
  hinoki: 'Hinoki & Indigo',
  night: 'Night',
};

export function parseTheme(value: string | null): Theme {
  return THEMES.find((t) => t === value) ?? 'system';
}
