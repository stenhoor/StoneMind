// The app shell (build plan, Milestone 1, step 1): the frame from the App Interface Design.
//
// The menu bar, the one bar under it (Workspaces, Layouts or Apps, at most one), the backdrop, and the singleton apps.
// Workspaces, goban types and optional panels arrive in later milestones; until then their places say so plainly.
import { type Backdrop, type BarId, BACKDROPS, BACKDROP_NAMES, BAR_IDS, PROFILE_NAMES, THEMES, THEME_NAMES, type Theme, deviceProfile, toggleBar } from './frame/logic.js';
import { type FocusedApp, SmMenuBar } from './frame/menu-bar.js';
import type { Menu } from './frame/menu.js';
import { applyBackdrop, applyTheme, loadBackdrop, loadTheme } from './frame/settings.js';

type SingletonId = 'files' | 'engines' | 'layouts' | 'settings' | 'about';

const SINGLETONS: Readonly<Record<SingletonId, string>> = {
  files: 'Files',
  engines: 'Engines',
  layouts: 'Layouts',
  settings: 'Settings',
  about: 'About StoneMind',
};

const state = {
  bar: null as BarId | null,
  theme: loadTheme(),
  backdrop: loadBackdrop(),
  sheet: null as SingletonId | null,
  focused: null as SingletonId | null,
};

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`missing ${sel}`);
  return el;
};

const menuBar = $<SmMenuBar>('sm-menu-bar');
const sheetHost = $<HTMLElement>('#sheet-host');

function setTheme(theme: Theme): void {
  state.theme = theme;
  applyTheme(theme);
  renderSheet();
}

function setBackdrop(backdrop: Backdrop): void {
  state.backdrop = backdrop;
  applyBackdrop(backdrop);
  renderSheet();
}

function setBar(bar: BarId | null): void {
  state.bar = bar;
  for (const id of BAR_IDS) $<HTMLElement>(`#bar-${id}`).hidden = id !== bar;
  menuBar.bar = bar;
}

// ------------------------------------------------------------------ menus

function frameMenus(): Menu[] {
  return [
    {
      label: 'View',
      items: [
        ...BACKDROPS.map((b) => ({ label: `Backdrop: ${BACKDROP_NAMES[b]}`, checked: () => state.backdrop === b, run: () => setBackdrop(b) })),
        ...THEMES.map((t) => ({ label: `Theme: ${THEME_NAMES[t]}`, checked: () => state.theme === t, run: () => setTheme(t) })),
      ],
    },
    {
      label: 'Window',
      items: [
        ...BAR_IDS.map((id) => ({
          label: { workspaces: 'Workspaces bar', layouts: 'Layouts bar', apps: 'Apps bar' }[id],
          checked: () => state.bar === id,
          run: () => setBar(toggleBar(state.bar, id)),
        })),
        ...(state.sheet ? [{ label: `Close ${SINGLETONS[state.sheet]}`, run: () => closeSheet() }] : []),
      ],
    },
  ];
}

function focusedApp(): FocusedApp {
  if (state.focused) {
    const id = state.focused;
    return { name: SINGLETONS[id], menus: [{ label: 'Window', items: [{ label: `Close ${SINGLETONS[id]}`, run: () => closeSheet() }] }] };
  }
  return { name: '', menus: frameMenus() };
}

function renderMenuBar(): void {
  menuBar.app = focusedApp();
}

menuBar.stoneMindMenu = [
  { label: 'About StoneMind', run: () => openSheet('about') },
  { label: 'Settings…', run: () => openSheet('settings') },
];

// ------------------------------------------------------- singleton apps

let opener: HTMLElement | null = null;

function openSheet(id: SingletonId): void {
  opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  state.sheet = id;
  renderSheet();
  sheetHost.querySelector<HTMLElement>('h2')?.focus();
}

function closeSheet(): void {
  state.sheet = null;
  state.focused = null;
  renderSheet();
  renderMenuBar();
  if (opener?.isConnected) opener.focus();
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

function settingsBody(): Node[] {
  const theme = el('fieldset', { className: 'sm-theme-picker' }, el('legend', { className: 'sm-label' }, 'Theme'));
  for (const t of THEMES) {
    const input = el('input', { type: 'radio', name: 'theme', value: t, checked: state.theme === t });
    input.addEventListener('change', () => setTheme(t));
    theme.append(el('label', {}, input, THEME_NAMES[t]));
  }
  const backdrop = el('div', { className: 'sm-segmented' });
  backdrop.setAttribute('role', 'group');
  backdrop.setAttribute('aria-labelledby', 'backdrop-label');
  for (const b of BACKDROPS) {
    const button = el('button', { type: 'button', textContent: BACKDROP_NAMES[b] });
    button.setAttribute('aria-pressed', String(state.backdrop === b));
    button.addEventListener('click', () => setBackdrop(b));
    backdrop.append(button);
  }
  return [theme, el('div', { className: 'sheet-field' }, el('span', { className: 'sm-label', id: 'backdrop-label' }, 'Backdrop'), backdrop)];
}

function sheetBody(id: SingletonId): Node[] {
  switch (id) {
    case 'settings':
      return settingsBody();
    case 'about':
      return [el('p', {}, 'StoneMind is a Go coaching companion in early development. This is the app’s first frame: Files, Engines and the gobans are still to come.')];
    default:
      return [el('p', { className: 'sm-secondary' }, `${SINGLETONS[id]} is not built yet.`)];
  }
}

function renderSheet(): void {
  const id = state.sheet;
  if (!id) {
    sheetHost.replaceChildren();
    sheetHost.hidden = true;
    return;
  }
  // Keep focus where it was when the sheet re-renders for a changed setting.
  const active = document.activeElement;
  const activeKey = active instanceof HTMLElement && sheetHost.contains(active) ? (active.getAttribute('value') ?? active.textContent) : null;
  const title = el('h2', { className: 'sheet-title', tabIndex: -1, id: 'sheet-title' }, SINGLETONS[id]);
  const close = el('button', { type: 'button', className: 'sm-btn sm-icon-btn sheet-close', title: `Close ${SINGLETONS[id]}` });
  close.setAttribute('aria-label', `Close ${SINGLETONS[id]}`);
  close.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
  close.addEventListener('click', closeSheet);
  const sheet = el('section', { className: 'sm-glass sheet' }, el('div', { className: 'sheet-head' }, title, close), ...sheetBody(id));
  sheet.setAttribute('aria-labelledby', 'sheet-title');
  sheet.dataset['app'] = id;
  sheetHost.replaceChildren(sheet);
  sheetHost.hidden = false;
  if (activeKey) {
    for (const c of sheet.querySelectorAll<HTMLElement>('input, button')) {
      if ((c.getAttribute('value') ?? c.textContent) === activeKey) {
        c.focus();
        break;
      }
    }
  }
}

sheetHost.addEventListener('focusin', () => {
  if (state.sheet && state.focused !== state.sheet) {
    state.focused = state.sheet;
    renderMenuBar();
  }
});
sheetHost.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.sheet) {
    e.preventDefault();
    closeSheet();
  }
});
$<HTMLElement>('#desk').addEventListener('pointerdown', () => {
  if (state.focused) {
    state.focused = null;
    renderMenuBar();
  }
});

// ------------------------------------------------------------- apps bar

for (const button of document.querySelectorAll<HTMLButtonElement>('#bar-apps [data-singleton]')) {
  const id = button.dataset['singleton'] as SingletonId;
  button.addEventListener('click', () => {
    if (state.sheet === id) sheetHost.querySelector<HTMLElement>('h2')?.focus();
    else openSheet(id);
  });
}

// ------------------------------------------------------- profile, start

const profile = $<HTMLElement>('#layouts-profile');
const onResize = () => {
  const name = PROFILE_NAMES[deviceProfile(document.documentElement.clientWidth)];
  profile.textContent = name;
  menuBar.status = [name];
};
new ResizeObserver(onResize).observe(document.documentElement);

menuBar.addEventListener('sm-bar-toggle', (e) => setBar(toggleBar(state.bar, (e as CustomEvent<{ bar: BarId }>).detail.bar)));
applyBackdrop(state.backdrop);
setBar(null);
renderMenuBar();
onResize();
