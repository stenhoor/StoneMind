// <sm-menu-bar>: the menu bar at the top of the app (design doc, The frame).
//
// It sits inside the page's <header>. Left to right: the wordmark (the StoneMind menu), the focused app's name and menus, the status items, the layout
// name, the Workspaces, Layouts and Apps buttons (at most one active), and the clock. It watches its own width and sets
// data-mode (full, compact, narrow) so it always stays on one line; app.css hides what each mode leaves out.
//
// Properties: `app` (the focused app's name and menus), `stoneMindMenu`, `status`, `layout`, `bar`.
// Events: `sm-bar-toggle` with detail { bar } when one of the three buttons is pressed.
import { type BarId, BAR_IDS, CLOCK_INTERVAL, type MenuBarMode, clockText, menuBarMode } from './logic.js';
import { type Menu, type MenuItem, closeMenu, menuButton } from './menu.js';

export interface FocusedApp {
  readonly name: string;
  readonly menus: readonly Menu[];
}

const BAR_LABELS: Readonly<Record<BarId, string>> = { workspaces: 'Workspaces', layouts: 'Layouts', apps: 'Apps' };

const ICONS: Readonly<Record<BarId, string>> = {
  workspaces: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="8" width="14" height="12" rx="2"/><path d="M7 8V5a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-3"/></svg>',
  layouts: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M13 3v18M13 12h8"/></svg>',
  apps: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="3" y="3" width="5" height="5" rx="1.2"/><rect x="9.5" y="3" width="5" height="5" rx="1.2"/><rect x="16" y="3" width="5" height="5" rx="1.2"/><rect x="3" y="9.5" width="5" height="5" rx="1.2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1.2"/><rect x="16" y="9.5" width="5" height="5" rx="1.2"/><rect x="3" y="16" width="5" height="5" rx="1.2"/><rect x="9.5" y="16" width="5" height="5" rx="1.2"/><rect x="16" y="16" width="5" height="5" rx="1.2"/></svg>',
};

export class SmMenuBar extends HTMLElement {
  #app: FocusedApp = { name: '', menus: [] };
  #stoneMindMenu: readonly MenuItem[] = [];
  #status: readonly string[] = [];
  #layout = '';
  #bar: BarId | null = null;
  #mode: MenuBarMode = 'full';
  #timer: ReturnType<typeof setInterval> | undefined;
  #observer: ResizeObserver | undefined;
  readonly #parts = {
    appName: document.createElement('span'),
    menus: document.createElement('div'),
    allMenus: document.createElement('div'),
    allMenusLabel: document.createElement('span'),
    status: document.createElement('div'),
    layout: document.createElement('span'),
    clock: document.createElement('time'),
    bars: new Map<BarId, HTMLButtonElement>(),
  };

  connectedCallback(): void {
    if (this.childElementCount) return;
    this.className = 'sm-glass sm-bar sm-menubar';
    const p = this.#parts;

    const logo = document.createElement('span');
    logo.className = 'sm-menubar-logo';
    logo.innerHTML =
      '<img class="sm-logo-light mb-wordmark" src="brand/logos/wordmark.svg" alt="" width="112" height="18">' +
      '<img class="sm-logo-dark mb-wordmark" src="brand/logos/wordmark-on-dark.svg" alt="" width="112" height="18">' +
      '<img class="sm-logo-light mb-mark" src="brand/logos/lit-stone-mark.svg" alt="" width="26" height="26">' +
      '<img class="sm-logo-dark mb-mark" src="brand/logos/lit-stone-mark-on-dark.svg" alt="" width="26" height="26">';
    const logoButton = menuButton('StoneMind', logo, () => ({ items: this.#stoneMindMenu }), 'sm-menu-button sm-menubar-logo-button');
    logoButton.firstElementChild?.setAttribute('aria-label', 'StoneMind menu');

    p.appName.className = 'sm-menubar-app';
    p.menus.className = 'sm-menubar-menus';
    p.allMenus.className = 'sm-menubar-allmenus';
    p.allMenus.append(menuButton('Menus', p.allMenusLabel, () => ({ groups: this.#app.menus })));
    p.status.className = 'sm-menubar-status';
    p.layout.className = 'sm-menubar-layout sm-chip';

    const spacer = document.createElement('span');
    spacer.className = 'sm-menubar-spacer';

    const bars = document.createElement('div');
    bars.className = 'sm-menubar-bars';
    for (const id of BAR_IDS) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'sm-menubar-toggle';
      b.dataset['bar'] = id;
      b.innerHTML = ICONS[id];
      b.addEventListener('click', () => this.dispatchEvent(new CustomEvent('sm-bar-toggle', { detail: { bar: id }, bubbles: true })));
      p.bars.set(id, b);
      bars.append(b);
    }

    p.clock.className = 'sm-menubar-clock';
    this.append(logoButton, p.appName, p.menus, p.allMenus, spacer, p.status, p.layout, bars, p.clock);

    this.#observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      const mode = menuBarMode(width);
      if (mode !== this.#mode) {
        this.#mode = mode;
        closeMenu();
        this.#render();
      }
    });
    this.#observer.observe(this);
    this.#mode = menuBarMode(this.getBoundingClientRect().width);
    this.#timer = setInterval(() => this.#renderClock(), CLOCK_INTERVAL);
    this.#render();
  }

  disconnectedCallback(): void {
    clearInterval(this.#timer);
    this.#observer?.disconnect();
  }

  get mode(): MenuBarMode {
    return this.#mode;
  }

  set app(app: FocusedApp) {
    this.#app = app;
    this.#render();
  }

  set stoneMindMenu(items: readonly MenuItem[]) {
    this.#stoneMindMenu = items;
  }

  set status(items: readonly string[]) {
    this.#status = items;
    this.#render();
  }

  set layout(name: string) {
    this.#layout = name;
    this.#render();
  }

  set bar(bar: BarId | null) {
    this.#bar = bar;
    this.#render();
  }

  #render(): void {
    if (!this.isConnected || !this.childElementCount) return;
    const p = this.#parts;
    this.dataset['mode'] = this.#mode;
    // No app name when no app is focused: the wordmark already says StoneMind.
    p.appName.textContent = this.#app.name;
    p.appName.hidden = !this.#app.name;
    p.allMenusLabel.textContent = this.#app.name || 'Menus';
    // Rebuilding the menu buttons would drop the focus of the one in use (after choosing from its menu, say), so it is
    // given back to the new button of the same name.
    const active = document.activeElement;
    const focused = active instanceof HTMLElement && p.menus.contains(active) ? active.textContent : null;
    p.menus.replaceChildren(...this.#app.menus.map((m) => menuButton(m.label, m.label, () => ({ items: m.items }))));
    if (focused !== null) [...p.menus.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent === focused)?.focus();
    p.allMenus.hidden = this.#app.menus.length === 0;
    p.status.replaceChildren(
      ...this.#status.map((s) => {
        const item = document.createElement('span');
        item.className = 'sm-menubar-status-item';
        item.textContent = s;
        return item;
      }),
    );
    p.layout.textContent = this.#layout ? `Layout: ${this.#layout}` : '';
    p.layout.hidden = !this.#layout;
    for (const [id, b] of p.bars) {
      const on = this.#bar === id;
      b.setAttribute('aria-pressed', String(on));
      b.setAttribute('aria-label', BAR_LABELS[id]);
      b.title = BAR_LABELS[id];
    }
    this.#renderClock();
  }

  #renderClock(): void {
    const now = new Date();
    const c = this.#parts.clock;
    c.dateTime = now.toISOString();
    c.textContent = clockText(now, this.#mode);
  }
}

customElements.define('sm-menu-bar', SmMenuBar);
