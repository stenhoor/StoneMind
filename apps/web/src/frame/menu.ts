// Drop-down menus for the menu bar: a button with aria-haspopup="menu" that opens a role="menu" list.
//
// Keyboard: Enter, Space or Arrow Down on the button opens the menu and focuses its first item; Arrow Up and Down move
// between items (wrapping), Home and End jump; Enter or Space chooses; Escape closes and returns focus to the button;
// Tab closes. A click outside closes it. Menus are opaque, not glass: they open over the bars, and glass never sits on
// glass (BRAND.md, Material).

export interface MenuItem {
  readonly label: string;
  /** A radio item shows a tick when checked and is announced as menuitemradio. */
  readonly checked?: () => boolean;
  readonly disabled?: boolean;
  readonly run?: () => void;
}

export interface Menu {
  readonly label: string;
  readonly items: readonly MenuItem[];
}

/** A menu's items, or several menus shown as labelled groups (the narrow menu bar's single Menus button). */
export type MenuContent = { readonly items: readonly MenuItem[] } | { readonly groups: readonly Menu[] };

let open: { button: HTMLButtonElement; list: HTMLElement; onDoc: (e: PointerEvent) => void } | null = null;

export function closeMenu(focusButton = false): void {
  if (!open) return;
  const { button, list, onDoc } = open;
  open = null;
  list.remove();
  button.setAttribute('aria-expanded', 'false');
  document.removeEventListener('pointerdown', onDoc, true);
  if (focusButton) button.focus();
}

function itemButton(item: MenuItem): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'sm-menu-item';
  b.tabIndex = -1;
  const radio = item.checked !== undefined;
  b.setAttribute('role', radio ? 'menuitemradio' : 'menuitem');
  const tick = document.createElement('span');
  tick.className = 'sm-menu-tick';
  tick.setAttribute('aria-hidden', 'true');
  if (radio) {
    const on = item.checked?.() ?? false;
    b.setAttribute('aria-checked', String(on));
    tick.textContent = on ? '✓' : '';
  }
  const text = document.createElement('span');
  text.textContent = item.label;
  b.append(tick, text);
  if (item.disabled) b.setAttribute('aria-disabled', 'true');
  b.addEventListener('click', () => {
    if (item.disabled) return;
    closeMenu(true);
    item.run?.();
  });
  return b;
}

function openMenu(button: HTMLButtonElement, content: MenuContent, label: string): void {
  closeMenu();
  const list = document.createElement('div');
  list.className = 'sm-menu';
  list.setAttribute('role', 'menu');
  list.setAttribute('aria-label', label);
  if ('items' in content) {
    for (const item of content.items) list.append(itemButton(item));
  } else {
    for (const menu of content.groups) {
      const group = document.createElement('div');
      group.setAttribute('role', 'group');
      group.setAttribute('aria-label', menu.label);
      const heading = document.createElement('div');
      heading.className = 'sm-menu-heading';
      heading.setAttribute('aria-hidden', 'true');
      heading.textContent = menu.label;
      group.append(heading, ...menu.items.map(itemButton));
      list.append(group);
    }
  }
  const items = () => [...list.querySelectorAll<HTMLButtonElement>('.sm-menu-item')];
  list.addEventListener('keydown', (e) => {
    const all = items();
    const at = all.indexOf(document.activeElement as HTMLButtonElement);
    const go = (i: number) => all[(i + all.length) % all.length]?.focus();
    if (e.key === 'ArrowDown') go(at + 1);
    else if (e.key === 'ArrowUp') go(at - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(all.length - 1);
    else if (e.key === 'Escape') closeMenu(true);
    else if (e.key === 'Tab') {
      closeMenu();
      return;
    } else return;
    e.preventDefault();
  });
  const onDoc = (e: PointerEvent) => {
    if (!list.contains(e.target as Node) && e.target !== button && !button.contains(e.target as Node)) closeMenu();
  };
  document.addEventListener('pointerdown', onDoc, true);
  button.insertAdjacentElement('afterend', list);
  button.setAttribute('aria-expanded', 'true');
  open = { button, list, onDoc };
  items()[0]?.focus();
}

/**
 * A button that opens a menu, inside a positioned wrapper the menu opens under. `content` is read each time the menu
 * opens, so ticks are always current. Returns the wrapper; the button is its first child.
 */
export function menuButton(label: string, text: string | Node, content: () => MenuContent, className = 'sm-menu-button'): HTMLSpanElement {
  const wrap = document.createElement('span');
  wrap.className = 'sm-menu-wrap';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = className;
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  button.append(text);
  const toggle = () => {
    if (open?.button === button) closeMenu(true);
    else openMenu(button, content(), label);
  };
  button.addEventListener('click', toggle);
  button.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && open?.button !== button) {
      e.preventDefault();
      openMenu(button, content(), label);
    }
  });
  wrap.append(button);
  return wrap;
}
