# StoneMind App Build Plan: Milestone 1 (Infrastructure)

Version: 0.1

Status: Draft, for review before any code

Date: 2026-11-27

---

# Purpose

This is the plan for the first build of the StoneMind app (`apps/web`, stonemind.app). It turns the
[App Interface Design](StoneMind-App-Interface-Design.md) (version 0.2) into steps, one pull request each, and says
what each step proves and how it is tested.

---

# Decided for this plan (owner, 2026-11-27)

- **Infrastructure first.** Milestone 1 builds the frame, the Files app and the Engines app, before any goban. This is
  the design doc's order of work, steps 2 and 3.
- **Browsers:** Chrome and Edge, Safari (Mac and iPad), Firefox, and phones.
- **Tests:** Playwright drives the real pages in browsers, as a dev dependency only, never shipped. Logic is tested
  with `node --test`.
- **Hosting:** local only for now. The app runs from a local development server until it is worth publishing.

---

# How it is built

- **Where:** `apps/web`, package `@stonemind/web`, as `docs/MONOREPO.md` already plans. New kits go in `packages/*`.
- **Build:** the same way `apps/engine-bench` is built: `tsc` to plain ES modules, plus a small assemble script that
  copies the pages, the brand files and the built packages into `dist/`. No bundler.
- **Interface:** semantic HTML and CSS on `@stonemind/brand` tokens and `components.css`; behaviour in small
  TypeScript modules on platform APIs; custom elements in light DOM. No outside code beyond Playwright.
- **Playwright** is already a dev dependency of `@stonemind/engine-webgpu` (version 1.65.0). `apps/web` uses the same
  version.
- **Each step** updates `docs/MONOREPO.md` and the README "Depends on" and "Used by" sections where it adds a package or
  a cross-package dependency, and adds new packages to the root `workspaces` array in dependency order.

---

# Where each kind of test runs

| Test | Where it runs | What it can show |
|---|---|---|
| Logic (`node --test` on built `dist/`) | Locally and in CI | Layout trees, file name parsing, the engine queue and protocol, anything without a browser |
| Playwright in Chromium | This cloud environment, locally and in CI | Pages, custom elements, OPFS, Web Locks, BroadcastChannel and multi-tab behaviour |
| Playwright in Firefox and WebKit | CI only (their browsers are not installed in this cloud environment) | The same pages in those engines. Playwright's WebKit is not Safari, so it does not replace a check on a real Mac or iPad |
| Real GPU and real devices | The owner's machines and devices, through the capability page (step 2) | WebGPU speed and correctness, and what each browser actually supports |

Notes:

- CI runners have no GPU. Engine tests in CI use a fake evaluator, so the queue, sharing and messaging are tested
  without WebGPU. The browser engine itself is already tested by `@stonemind/engine-webgpu`'s own goldens.
- Runs on the owner's NVIDIA machine go to the local Claude session as issues. Checks on the iPad and phones are run by
  the owner with the capability page.
- CI stays on Node 24-based actions.

---

# Steps

Each step is one pull request. A step does not start until the one before it is merged.

## Step 1: App shell and frame

**Builds:** `apps/web` with its build, local server and test setup, and the frame from the design doc:

- the menu bar: wordmark, the focused app's menus, the clock, and the Workspaces, Layouts and Apps buttons, with at most
  one active;
- the bar under the menu bar for whichever button is active;
- the backdrop;
- the one-line menu bar on narrow screens.

The bars hold placeholders until later steps fill them. The Apps bar shows its three groups, with only the singleton
apps working.

**Tests:**

- Playwright in Chromium, Firefox and WebKit:
  - at most one bar button is active;
  - the menu bar stays on one line at phone, tablet and desktop widths;
  - every control can be reached and used from the keyboard.
- `node --test` for the menu bar's width rules.

**Proves:** the build, the test setup in all three browser engines, and the frame.

## Step 2: Capability page

**Builds:** a page in `apps/web` that reports, as JSON and on screen, what the browser it runs in supports:

- WebGPU in the page, in a dedicated worker and in a shared worker, with the adapter's details;
- OPFS: `createWritable` and `createSyncAccessHandle`;
- `navigator.storage.persist` and the quota;
- Web Locks and `BroadcastChannel`;
- SVG filters in `backdrop-filter`.

**Tests:** Playwright runs it in all three engines in CI and keeps the reports. The owner runs it on the real devices
(Chrome, Edge, Safari on Mac and iPad, Firefox, phones); the NVIDIA machine's run goes to the local session as an
issue.

**Proves:** which browser features steps 3 to 6 can rely on, from evidence, including whether a shared worker can run
WebGPU anywhere.

## Step 3: File kit

**Builds:** `packages/files` (`@stonemind/files`):

- OPFS access: list, read, write, move and delete;
- streamed writes for large files such as KataGo weights;
- import from the file picker and by drag and drop;
- export by download;
- quota and persistence;
- Web Locks around writes, so tabs don't overwrite each other.

**Tests:**

- `node --test` for path handling and the kit's logic;
- Playwright in Chromium, plus Firefox and WebKit in CI:
  - importing a large file;
  - two tabs writing the same file;
  - export.

**Proves:** that player files can be stored and kept safe across tabs in every target browser.

## Step 4: Files app

**Builds:** the Files singleton app in the design doc's Finder style:

- folders that hold files, and filters for the kinds of file (Games, Problems, Weights, Configs: what a file is,
  worked out from its name and content; any number on at once), sortable lists and the inspector;
- streamed weights import with progress;
- weights metadata: network size and type from the file name, the training samples (`s`) and data rows (`d`), and the
  model version read from the file;
- a SHA-256 checksum;
- storage quota and "keep my files".

Games and Problems list their files. Their SGF columns come with the SGF kit in a later milestone; until then they show
the file name, size and date.

**Tests:**

- `node --test` for weights name parsing;
- Playwright:
  - importing a weights file;
  - the inspector's values;
  - sorting;
  - keyboard use.

**Proves:** that the player can bring KataGo weights into the app and see what they are.

## Step 5: Engine host

**Builds:** the engine pool from the design doc:

- **The browser engine:** at most one per browser.
  - **Baseline:** one tab runs `@stonemind/engine-webgpu` in a dedicated worker, chosen with Web Locks, and another tab
    takes over if it closes.
  - **Upgrade:** a shared worker where step 2 showed it can run WebGPU.
- **Requests from other tabs:** they reach the running engine through `BroadcastChannel`.
- **A queue with priorities:** live analysis of the focused workspace first, then of other open workspaces, then
  background reviews.
- **Native KataGo:** connects through the bridge's existing pairing link.
- **Both engines:** implement `AnalysisEngine` from `@stonemind/engine`.

**Tests:**

- `node --test` for the queue and the messages between tabs;
- Playwright in Chromium with a fake evaluator:
  - two tabs share one engine;
  - closing the tab that runs it hands it to another;
  - the queue order.
- The real GPU check runs on the owner's machines with the capability page's engine section.

**Proves:** one engine shared safely by every tab, on real GPUs and with the bridge.

## Step 6: Engines app

**Builds:** the Engines singleton app:

- the engines and which workspace uses which;
- settings in plain words with the KataGo key underneath, starting at KataGo's defaults;
- the monitor of a running engine: speed, batch fill, the time per network call by stage from `lastCallTiming`, the
  queue, and which tabs send work;
- connecting and disconnecting the bridge;
- choosing weights from the Files app.

Where a browser offers no WebGPU, or the browser engine fails to start in it, Engines says so plainly and offers the
bridge. The remote service is not shown until it exists, so the app makes no claim ahead of the product.

**Tests:**

- Playwright with the fake evaluator: the monitor's values, the settings, and the no-WebGPU message;
- the owner checks it on real devices.

**Proves:** that Milestone 1 is done (below).

---

# Milestone 1 is done when

In every target browser:

1. The app loads at the local server with its frame.
2. A player can import KataGo weights into the browser's storage and see their details in Files.
3. Where WebGPU works, a player can start the browser engine from Engines, with those weights, and watch its monitor.
   Two open tabs share the one engine.
4. Where WebGPU does not work, Engines says so and the bridge works instead.

---

# After Milestone 1

These follow the design doc's order of work and get their own plan when Milestone 1 is done:

- the rules kit;
- the SGF and tree kits;
- the goban kit;
- the workspace kit (workspaces, tiling presets, device profiles);
- the optional panels;
- the three goban types: Review, Play and Problems.

---

# Risks

- **Firefox:** Firefox 161 on Linux offers WebGPU, in the page and in both kinds of worker (step 2's report from the
  owner's laptop, `docs/reports/capabilities/`). MDN's compatibility data said it had none on Linux, so that data was
  out of date. The browser engine itself had trouble running in Firefox during spike 0001 and is not yet confirmed
  there; step 5 tests it. Where it can't run, Engines offers the bridge (step 6).
- **Shared workers:** Edge 158 and Firefox 161 on Linux give a shared worker a WebGPU adapter (step 2's reports).
  Safari is not checked yet. The Web Locks baseline covers any browser where it does not work.
- **Phones:** shared workers arrive in Chrome Android only at version 148, by MDN's data; the baseline does not need
  them. Phone GPUs are untested with the browser engine.
- **Playwright's WebKit is not Safari,** so Safari behaviour is confirmed only on the owner's Mac and iPad.
- **OPFS differences between browsers,** for example which write methods each supports, are not checked yet; step 2
  finds out before the file kit relies on them.
