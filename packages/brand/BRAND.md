# StoneMind brand

**Unlock your Go.**

StoneMind's brand is the Lit Stone mark, three themes taken from the Go table (Washi & Shu, Hinoki & Indigo, Night), and a glass material for the app. Players choose their theme.

This package follows the smaltworks brand pattern ([smalt-brand](https://github.com/stenhoor/smalt-brand)): `tokens.json` is the source of truth, `tokens.css` is generated from it, and components are static HTML and CSS references.

## One product, one brand

StoneMind is one product. Its sub-projects (the packages and apps in this monorepo) are parts of it, not products, and they are not branded on their own:

- No sub-project has a mark, wordmark, colour, badge or tagline of its own. The app (stonemind.app) and the website (stonemind.org) both carry the StoneMind brand from this package.
- Package names (`@stonemind/ontology`, `@stonemind/provider-sdk` and so on) are technical names for developers. They never appear in the product's interface or marketing.
- This is where StoneMind departs from the smaltworks family it otherwise follows: there, each product has its own mark and layer colour.

## The name

Write **StoneMind** in prose. The wordmark is set in lowercase, **stonemind**, with the stone standing in for the "o". Never write "Stone Mind" or "STONEMIND".

## Voice

- Plain and specific. Say what StoneMind does for a player.
- Describe what StoneMind is, never what it isn't.
- **No claims that can't be backed up.** Write about a feature or integration in the present tense only once it exists. Until then, describe it as a goal ("designed to", "planned"). This covers the website, READMEs and the app.
- Coaching copy explains itself: say why a note or recommendation was made and what evidence it rests on.
- Style is not weakness: never describe a player's preferences as faults.

## Logo: the Lit Stone

A single slate stone with its shine: the spark of insight.

| Asset | Use |
|---|---|
| `logos/lit-stone-mark.svg` | The mark on light grounds: a `stone-black` body with a washi shine. |
| `logos/lit-stone-mark-on-dark.svg` | The mark on dark grounds: the same stone with a light rim and shine. |
| `logos/wordmark.svg` | The wordmark, with the stone as its "o". Light grounds. |
| `logos/wordmark-on-dark.svg` | The wordmark on dark grounds. |
| `logos/app-icon.svg` | App and PWA icons, social avatars: the stone on a kaya tile. |
| `logos/favicon-32.svg` | Browser tabs. |
| `logos/favicon-16.svg` | 16px tabs. Drops the small highlight dot and thickens the shine. |

The wordmark is Manrope 800 with -0.03em tracking, converted to paths (from Google Fonts, OFL), so it renders the same everywhere with no font loading. Because the wordmark already contains the mark, there is no separate lockup.

- Clear space on every side: the stone's radius.
- In a page, include both versions with `.sm-logo-light` and `.sm-logo-dark` (from `styles/components.css`); the one for the active theme shows.
- Don't recolour, rotate, outline or add effects. Don't put the light-ground versions on dark grounds.

In a README (GitHub strips inline styles, so switch the image by theme):

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="packages/brand/logos/wordmark-on-dark.svg">
  <img alt="StoneMind" src="packages/brand/logos/wordmark.svg" width="280">
</picture>
```

## Colour: three themes

| Theme | `data-theme` | Scheme | Character |
|---|---|---|---|
| Washi & Shu | `washi` (default) | light | Washi paper, a kaya board, sumi ink, and a vermilion seal-red (shu) accent. |
| Hinoki & Indigo | `hinoki` | light | A hinoki cypress board and indigo (ai-iro) ink: quieter and scholarly. |
| Night | `night` | dark | Dark ground with a lighter shu accent. **Provisional:** not yet reviewed. |

With no `data-theme` set, a page follows the device: Washi & Shu in light mode, Night in dark mode. A player's choice (the ThemePicker) always wins.

**Readability rules.** `test/tokens.test.mjs` checks every rule for every theme with the WCAG 2 contrast formula:

1. `text` ≥ 12:1, `text-2` ≥ 7:1, `text-3` ≥ 4.5:1, on both `bg` and `surface`.
2. `link`, `success`, `warning` and `danger` ≥ 4.5:1 on `bg` and `surface`. `focus` ≥ 3:1 on `bg`.
3. On glass, measured with `glass-tint` composited over the table, the board, a black stone and a white stone: `text` ≥ 7:1, and `text-2` and `link` ≥ 4.5:1.
4. `on-primary` ≥ 4.5:1 on `primary`, and on `primary-glass` over every backdrop.
5. `text-3` is never used on glass.
6. Status is never colour alone: `success`, `warning` and `danger` always come with an icon and a word.
7. `danger` is crimson, never the shu accent, so an error never looks like a highlight.

Roles: `primary` for the one main action per view; `link` for links; `accent` for board marks (last move, suggested move); `board`, `board-line`, `stone-black` and `stone-white` for the board itself.

## Type

- **Manrope** (`--font-brand`): the wordmark and headings (`.display`, `.h1`, `.h2`, `.h3`). Weights 700–800, tight tracking.
- **Inter** (`--font-sans`): body and interface (`.body`, `.small`, `.label`). Body is 16px / 1.55, at most 70 characters a line.

Both are Google Fonts (OFL):

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Manrope:wght@700;800&display=swap">
```

## Material: glass

The app's panels are frosted glass floating over the Go board and the table it sits on. The board is the scene, and the interface floats around it.

- Blur 22px, saturation 170% (`--glass-blur`, `--glass-saturate`).
- Tint from `glass-tint`. It's opaque enough to keep the readability rules over any backdrop, so don't lower it.
- A 1px light edge, an inner top rim, a top sheen, and a wide shadow tinted by the theme.
- Glass panels sit over the table and the board. Never put glass over glass, and never animate the blur.

## Interface rules

- **HTML and CSS first.** Interfaces are semantic HTML styled with `styles/components.css` on `tokens/tokens.css`. Use real elements: `<button>`, `<a href>`, `<input>` with a `<label>`, `<fieldset>` with a `<legend>`.
- **No UI frameworks or component libraries.** Behaviour is small, plain TypeScript modules in the app that use the platform's own APIs. As across smaltworks, ask the owner before adding any outside code, and say what it is, its licence and size, and why our own code can't do the job.
- **Build colours from tokens,** never hand-picked values. A component stylesheet that needs a colour the tokens don't have adds a token first.
- Touch targets are at least 44px. Focus is a 2px `focus` ring with a 2px offset; never remove it.
- Respect `prefers-reduced-motion`.

## Tokens

`tokens/tokens.json` is the source of truth: colours per theme, the contrast rules, glass, type, spacing and radius. `tokens/tokens.css` is generated from it by `npm run build` (`scripts/build-tokens.mjs`). The default theme sits on `:root`, every theme on `[data-theme="<id>"]`, and the dark theme also applies under `prefers-color-scheme: dark` when no theme is chosen. There is one class per type style (`.display`, `.h1`, `.body` and so on). Never edit `tokens.css` by hand: CI checks that it matches.

## Components

`components/` holds `Button`, `GlassPanel`, `Chip`, `SegmentedControl` and `ThemePicker` as static HTML and CSS references on `tokens.css` and `styles/components.css`. Open any `preview.html` in a browser. They follow the saved theme, or the system setting when none is saved. Each component's README gives its rules.
