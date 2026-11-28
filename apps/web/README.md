# @stonemind/web

The StoneMind app, planned for **stonemind.app**. It is built in the steps of
[docs/StoneMind-App-Build-Plan.md](../../docs/StoneMind-App-Build-Plan.md), from the
[App Interface Design](../../docs/StoneMind-App-Interface-Design.md). It runs locally for now.

So far it is the frame (Milestone 1, step 1), the capability page (step 2) and the Files app (step 4).

The frame:

- **The menu bar:** the wordmark (the StoneMind menu), the focused app's menus, the device profile, the Workspaces,
  Layouts and Apps buttons (at most one active) and a clock. It watches its own width and stays on one line: below
  1180px the status hides; below 760px the wordmark becomes the stone and the menus fold into one button named after the
  focused app.
- **The bar under it** for whichever button is active.
- **The Apps bar:** each app is an icon tile with its name, in three groups (optional panels, goban types and singleton
  apps) divided by hairlines; on a narrow bar each group takes its own line. Panels and gobans are not built
  yet, so their buttons are disabled.
- **Singleton apps** open over the workspaces, one instance each. Settings works (theme and backdrop) and so does Files
  (below); Engines and Layouts say they are not built yet.
- **The backdrop:** plain table (the default), wood grain or washi paper.

**The Files app** (Milestone 1, step 4), from the Apps bar, is a file browser in the style of macOS Finder over the
file kit (`@stonemind/files`), on this browser's origin private file system:

- **Folders hold files; filters narrow what shows.** The player always browses a folder, starting at This browser,
  and its folders always show. Games, Problems, Weights and Configs are filters: kinds of file, toggled on and off, any
  number at once. With some on, the folder shows only files of those kinds; with none on (Show all files), every file.
  The filters stay on from folder to folder, and each shows how many files of its kind the open folder holds. They are
  also in the View menu.
- **A breadcrumb trail** at the top runs from This browser to the open folder; each folder above it is a link back up,
  and an Up button (and Go › Up one folder) goes one folder up. Folders are opened from the list.
- **The sidebar** has the filters and this browser's storage: space used of the quota, whether the
  browser keeps the files, and a **Keep my files** button that asks it to.
- **A file's kind is worked out from its name and content** when it is imported, never from what is on screen: an SGF
  naming players (PB, PW) or a result (RE) is a game, one that only sets up stones (AB, AW) is a problem; a `.bin.gz` or
  `.txt.gz` that starts like a KataGo model is a network; a `.cfg` of `key = value` lines with KataGo settings is a
  config. Anything else is refused, with a message saying which kinds Files takes. The player can change a game to a
  problem and back in the inspector. Kinds are kept in `/.stonemind/catalog.json` (docs/OPEN-QUESTIONS.md, 20).
- A path, and columns sorted by name, size or date: each file's kind, and, when networks are shown, the network's
  size from its name.
- An inspector for the selected file. For KataGo weights: the training run, the network (blocks, channels, kind), the
  training samples (`s`) and rows of training data (`d`) from the name, the format, and the model version read from
  the start of the file; a SHA-256 on request.
- Import from the file picker or by dropping files, streamed (a large network never sits whole in memory) with progress
  and Cancel, into the open folder; new folder, move to a folder, rename,
  download, and delete after a confirmation.
- Files fills the whole desk under the bars.
- When Files is focused, the menu bar shows its File, View and Go menus. A change in another tab shows at once.

**The capability page** (Milestone 1, step 2), at `/capabilities/`, opened from the StoneMind menu (Browser Info), checks what this
browser offers: WebGPU in the page, a dedicated worker and a shared worker (with the default adapter, the adapter given
for `high-performance`, `shader-f16` and buffer limits); OPFS with `createWritable` and, in a worker, `createSyncAccessHandle`; the storage quota and persistence; Web
Locks and `BroadcastChannel`; and `backdrop-filter`, SVG filters in it, and container queries. It shows the results and
offers them as a JSON report to copy or download. It stores nothing and sends nothing. The browser tests save each
engine's report to `dist/reports/` and print it.

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
- `src/files/view.ts`: the Files app. `src/files/kinds.ts`: the kinds, and how a file's kind is worked out.
  `src/files/catalog.ts`: where each file's kind is kept. `src/files/weights.ts`: what a KataGo weights file's name and first bytes say,
  free of the DOM.
- `src/capabilities/`: the capability page: `index.html`, `main.ts` (the probes), `report.ts` (the report and its
  summary, free of the DOM) and `worker-probe.js` (a classic script, run as a dedicated and as a shared worker).
- `scripts/assemble.mjs`: builds `dist/site`, with the file kit's modules in `vendor/files/` for the page's import map. `scripts/server.mjs`, `scripts/serve.mjs`: the local server.
- `test/*.test.mjs`: logic tests. `test/e2e/*.test.mjs`: browser tests.

## Depends on

- `@stonemind/brand`: `tokens.css`, `components.css` and the logos, copied unchanged into `dist/site/brand/`.
- `@stonemind/files`: the file kit (`OpfsFileStore`, import, download, storage), its compiled modules copied into
  `dist/site/vendor/files/` and loaded through an import map.
- Dev only: `playwright` 1.65.0 (Apache-2.0), for the browser tests; never shipped.

## Used by

Players, once it is published at stonemind.app.
