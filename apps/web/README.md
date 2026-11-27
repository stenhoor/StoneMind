# @stonemind/web

The StoneMind app, planned for **stonemind.app**. It is built in the steps of
[docs/StoneMind-App-Build-Plan.md](../../docs/StoneMind-App-Build-Plan.md), from the
[App Interface Design](../../docs/StoneMind-App-Interface-Design.md). It runs locally for now.

So far (Milestone 1, step 1) it is the frame:

- **The menu bar:** the wordmark (the StoneMind menu), the focused app's menus, the device profile, the Workspaces,
  Layouts and Apps buttons (at most one active) and a clock. It watches its own width and stays on one line: below
  1180px the status hides; below 760px the wordmark becomes the stone and the menus fold into one button named after the
  focused app.
- **The bar under it** for whichever button is active.
- **The Apps bar** in its three groups: optional panels, goban types and singleton apps. Panels and gobans are not built
  yet, so their buttons are disabled.
- **Singleton apps** open over the workspaces, one instance each. Settings works (theme and backdrop); Files, Engines
  and Layouts say they are not built yet.
- **The backdrop:** plain table (the default), wood grain or washi paper.

Plain HTML and CSS on `@stonemind/brand`, and small TypeScript modules on platform APIs; no framework and no bundler.

## Commands

```bash
npm run serve -w @stonemind/web      # build, then serve on http://localhost:8080 (PORT to change)
npm run check -w @stonemind/web      # typecheck and the logic tests (node --test)
npm run e2e -w @stonemind/web        # the browser tests (Playwright), in Chromium
BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/web   # as CI runs them
```

The browser tests need Playwright's browsers (`npx -w @stonemind/web playwright install chromium firefox webkit`).
`STONEMIND_CHROMIUM` points them at another Chromium.

## Layout

- `src/index.html`, `src/app.css`: the page and its layout, on the brand's tokens.
- `src/frame/logic.ts`: the frame's rules (bars, menu bar modes, device profiles, clock), free of the DOM.
- `src/frame/menu-bar.ts`: `<sm-menu-bar>`. `src/frame/menu.ts`: drop-down menus with keyboard support.
- `src/frame/settings.ts`: theme and backdrop, in `localStorage` until the materials kit exists
  (docs/OPEN-QUESTIONS.md, 17).
- `src/main.ts`: the page: bars, the Apps bar, singleton apps, menus.
- `scripts/assemble.mjs`: builds `dist/site`. `scripts/server.mjs`, `scripts/serve.mjs`: the local server.
- `test/*.test.mjs`: logic tests. `test/e2e/*.test.mjs`: browser tests.

## Depends on

- `@stonemind/brand`: `tokens.css`, `components.css` and the logos, copied unchanged into `dist/site/brand/`.
- Dev only: `playwright` 1.65.0 (Apache-2.0), for the browser tests; never shipped.

## Used by

Players, once it is published at stonemind.app.
