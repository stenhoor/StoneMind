# StoneMind App Interface Design

Version: 0.1

Status: Draft, for review before any code

Date: 2026-11-27

---

# Purpose

This document sets out how the StoneMind app's interface is built: where it is hosted, how its pieces run in the
browser, and the kits (packages) it is made from. It is the design to agree before work on the interface starts.

Each point is marked:

- **Decided:** the owner's decision, with its date.
- **Proposed:** a suggestion, open until the owner accepts it.
- **Open:** a question still to answer.

It builds on the [Technical Architecture](StoneMind-Technical-Architecture.md), [MONOREPO.md](MONOREPO.md), the
[brand](../packages/brand/BRAND.md) and [CLAUDE.md](../CLAUDE.md), and does not repeat them.

---

# Constraints carried over

These come from earlier decisions and still hold:

- **Origin.** The app lives at its own origin, `https://stonemind.app`, because OPFS storage belongs to one origin
  (MONOREPO.md, Deployment Targets). Moving the app to another address later would leave players' stored files behind,
  so the address must stay fixed.
- **HTML and CSS first.** Interfaces are semantic HTML and CSS on the brand tokens and `components.css`. No UI
  frameworks or component libraries; behaviour is small, plain TypeScript on platform APIs. Outside code only with the
  owner's approval (CLAUDE.md, BRAND.md).
- **The board is the scene.** The app's panels are glass over the table and the board (BRAND.md, Material).
- **Open formats.** SGF for games, Markdown for notes, JSON for configuration (Technical Architecture).
- **No claims ahead of the product.** Copy describes features only once they exist (BRAND.md, Voice).

---

# Hosting

**Decided (2026-11-27): stonemind.app is static files.** It can be hosted anywhere that serves static files over HTTPS.

Nothing in the app needs a server:

- The app is HTML, CSS and plain JavaScript modules. The browser engine is TypeScript with its WebGPU shaders, with no
  WebAssembly.
- It needs no special HTTP headers. Cross-origin isolation (COOP/COEP) is only needed for multi-threaded WebAssembly
  or `SharedArrayBuffer`, which the app does not use.
- OPFS and WebGPU need a secure context, which any HTTPS host gives.
- Native KataGo is reached through the bridge on the player's own computer (`@stonemind/katago-bridge`, a WebSocket
  on 127.0.0.1), not through a server of ours.

**Open: which static host.** stonemind.org already uses this repository's GitHub Pages site, which takes one custom
domain (to be confirmed in GitHub's documentation before it is relied on). The host can change later, as long as the
address stays `https://stonemind.app`.

---

# Tabs and panels

**Decided (2026-11-27): both.** The app runs in a browser tab with its components as panels, and any panel can also
run in a tab of its own.

**Proposed:**

- **Shared session across tabs.** The game session (below) is shared between tabs. Each tab holds a copy, kept in step
  with `BroadcastChannel` messages.
- **Coordinated storage.** Writes to the same OPFS files from several tabs are coordinated with the Web Locks API.
- Both are standard browser APIs; no outside code.

---

# Components

**Decided (2026-11-27): native custom elements.** Pluggable components are custom elements, for example `<sm-goban>`
and `<sm-tree>`. Custom elements are a platform API, not a framework.

**Proposed:**

- **Light DOM,** so the brand's CSS applies to them as it does to the rest of the page.
- **One game session object** that the components share: the SGF game tree, the current position, and the analysis
  stream. Components listen and announce changes through `EventTarget` events.
- With that, a panel (tree viewer, comments, move analysis, territory) can be added or replaced without changing the
  board.

---

# Kits

Each kit is a package in this monorepo, following its conventions: plain ES modules from TypeScript, tests with
`node --test`, CSS on the brand tokens, and no outside code. `docs/MONOREPO.md` is updated as each one is created.

## Rules kit

**Decided (2026-11-27):** the rules kit **defines** the rules. It is not tied to KataGo: rules matter to the browser
engine, native KataGo, games imported from other sites, the board and problems.

**Proposed contents:**

- **Rule sets as data:** ko rule, scoring, group tax, suicide, button, white handicap bonus and friendly passes.
  KataGo's rules model covers the common rule sets (Japanese, Chinese, AGA, New Zealand, Tromp-Taylor and others) and is
  a reasonable starting superset, without tying the kit to KataGo.
- **Mappings in:** SGF `RU[]` values to rule sets. The strings vary by site, so the table is researched from the SGF
  FF[4] specification and real files from each source, not guessed. Unknown values are flagged, never assumed.
- **Mappings out:** a rule set to each engine's form, such as KataGo's analysis-query `rules` field and GTP's
  `kata-set-rules`.

The browser engine's rules port (`@stonemind/engine-webgpu`, `src/rules/`) stays where it is, because it must reproduce
KataGo exactly and its goldens prove it does. **Proposed:** it takes its rule-set definitions from the rules kit instead
of its own `rules.ts`.

## Goban kit

**Decided (2026-11-27):** the goban kit **implements** the rules in the interface: playing moves, captures, ko and
counting, under a rule set from the rules kit. That is the clean break: the rules kit defines, the goban kit enforces.

**Proposed contents:** the board rendered in SVG and CSS on the brand's board tokens; input; overlays (marks, last move,
candidate moves, ownership); coordinates.

**Open:** SGF import and problem checking also need to apply moves without a board on screen. Should the goban kit's
move logic be usable on its own, without rendering?

## File kit

**Proposed contents:** OPFS access (list, read, write, move, delete), streamed writes for large files (KataGo weights
are about 100 MB), import from the file picker and drag and drop, export by download, storage quota and persistence
(`navigator.storage`), and the **file browser** component. Its first use is KataGo weights and configuration files.

## SGF kit

**Proposed contents:** reading and writing SGF (FF[4]) and the game tree: variations, comments and markup. This is the
`@stonemind/sgf` already planned in MONOREPO.md.

## Tree kit

**Proposed contents:** a general tree model and its navigation, and the **tree viewer** component. The SGF kit's game
tree builds on it.

## Materials kit

**Proposed contents:** player preferences, such as theme, board and stone appearance and coordinates. They are stored
as JSON in OPFS and applied through the brand tokens, building on `@stonemind/brand`. The app's ThemePicker lives here.

---

# The engine panel

**Proposed:** a panel that chooses the engine (the browser engine or native KataGo through the bridge) and the weights
from OPFS, sets KataGo's settings (defaulting to KataGo's own), and shows the engine's status and speed. Both engines
implement `AnalysisEngine` from `@stonemind/engine`, so one panel can drive either.

---

# The goban's three uses

**Decided (as stated by the owner, 2026-11-27):** the goban is the key interface for:

1. game analysis and reviews;
2. playing against the AI, as teaching games or normal play;
3. problem solving.

**Proposed:** the three differ only in which panels they load and in the session's mode, so they share one board.

---

# Order of work

**Proposed**, one pull request per step:

1. This design document.
2. The file kit and the OPFS file browser, first for weights and configuration files.
3. The engine panel.
4. The rules kit.
5. The SGF and tree kits.
6. The goban kit.
7. Panels: tree viewer, comments and notes, move analysis, territory.
8. The three uses: review, play, problems.

---

# Open questions

1. Which static host serves stonemind.app (Hosting).
2. Whether the goban kit's move logic can run without rendering (Goban kit).
3. The SGF `RU[]` mapping table, researched per source (Rules kit).
