# StoneMind App Interface Design

Version: 0.2

Status: Draft, for review before any code

Date: 2026-11-27

---

# Purpose

This document sets out how the StoneMind app's interface is built: where it is hosted, how its pieces run in the
browser, how the screen is organised, and the kits (packages) it is made from. It is the design to agree before work on
the interface starts.

Each point is marked:

- **Decided:** the owner's decision, with its date.
- **Proposed:** a suggestion, open until the owner accepts it.
- **Open:** a question still to answer.

It builds on the [Technical Architecture](StoneMind-Technical-Architecture.md), [MONOREPO.md](MONOREPO.md), the
[brand](../packages/brand/BRAND.md) and [CLAUDE.md](../CLAUDE.md), and does not repeat them.

Version 0.2 adds the frame, the three goban types (Review, Play, Problems) and their workspaces, optional panels and
singleton apps, tiling presets, device profiles and the engine pool, from four rounds of mockups reviewed with the owner
on 2026-11-27. The mockups are on a private design canvas
(https://claude.ai/artifact/SmPzj8PXCPWqyGe14AebMw); round 4 is the latest, and rounds 1 to 3 are kept there for
comparison. Mockups show sample data only.

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

# The frame

The frame is what stays on screen whatever the player is doing: the menu bar, and at most one bar under it.

## Menu bar

**Decided (2026-11-27):**

- The menu bar is at the top.
- The StoneMind wordmark is at its left and opens the StoneMind menu (settings, layouts and the like).
- The menus belong to the focused panel, as in macOS: focusing the game tree shows the tree's menus, focusing the notes
  shows the notes' menus.
- A clock sits at its right.
- Three buttons sit beside the clock: **Workspaces**, **Layouts** and **Apps** (the 3×3 launcher button). At most one
  is active at a time, or none. The active button's bar shows under the menu bar; pressing it again hides the bar.

**Proposed:**

- The clock uses the browser's own date and time format (`Intl.DateTimeFormat`) and updates every 15 seconds.
- On a narrow screen the menu bar keeps to one line: the wordmark becomes the stone alone, the menus fold into one
  Menus button, the status items and layout name hide, and the clock shows the time only. The widths at which this
  happens are set by the menu bar's own width, not the window's.

## The three bars

- **Workspaces:** the open workspaces as numbered cards (below), and a button for a new one.
- **Layouts:** the presets for the current workspace, Reset splits, the workspace's engine, and the device profile in
  use.
- **Apps:** the launcher, in three groups.

**Decided (2026-11-27): the Apps bar's three groups.**

| Group | What it holds | What launching does |
|---|---|---|
| Left: optional panels | Panels a goban can use, such as the game tree, analysis and notes. They are plugins: none is required. | Adds the panel to the focused workspace. |
| Centre: goban types | **Review**, **Play** and **Problems** (below). | Opens a new workspace of that type. |
| Right: singleton apps | Apps that manage and configure the environment: Files, Engines, Layouts and Settings. | Opens the app. Each has one instance only, and none belongs to a workspace. |

**Proposed:** the coach and territory panels join the optional panels once they exist. A singleton app opens over the
workspaces and closes back to them.

**Proposed:** the bars are glass. Earlier rounds offered launcher positions (above or below the menu bar, either side,
the bottom); the three-button design replaced them, and other positions are not planned unless the owner asks.

## Backdrop

**Decided (2026-11-27):** the backdrop behind the glass is a player setting. Plain table is the default; wood grain and
washi paper are options.

**Decided (2026-11-27):** no decoration on the table. Stone bowls and loose stones were tried and dropped.

---

# Workspaces

**Decided (2026-11-27):**

- A workspace holds one goban and the optional panels the player adds to it.
- There are three types of goban, and so three types of workspace:

  | Type | What it is for | Its source |
  |---|---|---|
  | **Review** | Opening a game for analysis and editing | An SGF file in OPFS |
  | **Play** | Playing a game | A game against the AI |
  | **Problems** | Solving problems | A problem set |

- The game tree, analysis and notes are optional panels, not part of any type. The panels in a workspace share its
  goban's game session.
- Players move between workspaces both ways: a strip of workspaces they scroll through (as in niri) and numbers that
  jump straight to one (as in Hyprland).

**Proposed:**

- Launching a goban type from the Apps bar opens a new workspace of that type. Opening an SGF file from Files opens a
  Review workspace for it. Each workspace is named from its source (for an SGF, from its players or file name).
- Keys 1 to 9 switch to workspaces 1 to 9.
- Each workspace keeps its own preset, split sizes, tabs, maximised tile and engine.

**Open:**

- What a live game connects to (which server or service) is not decided.
- Which type a live game is: Play when the player is in the game, Review when watching it, or both.

---

# Tiling and layout presets

The panels of a workspace tile the space; they do not float or overlap. Hyprland is the model for arranging panels
around one goban.

**Decided (2026-11-27):**

- Panels are placed by layout presets that the player defines. The player sets a default preset and can switch between
  a number of presets.
- The player can widen or narrow every split.
- Windows have minimise and maximise/restore buttons, in place of a "put back" link.

**Proposed:**

- **A preset is a tree of splits.** Each split is horizontal or vertical, with a starting ratio. Each leaf is a tile
  holding one panel, or several panels as tabs.
- **Dragging a split** changes that workspace's ratio, not the preset's. Reset splits returns to the preset's ratios.
  Splits also move with the arrow keys when focused.
- **Maximise** gives one tile the whole workspace; restore puts the layout back.
- **Minimise** in a panel that has been moved to a tab of its own puts it back in its workspace.
- **Presets belong to a goban type.** Each type has a default preset and can have more. The goban's own controls (the
  game panel in Play, the problem panel in Problems) come with the type; the other tiles hold optional panels.
- **Starting presets:**

  | Type | Preset | Layout |
  |---|---|---|
  | Review | Review (default) | Goban left; right: analysis above, game tree and notes side by side below |
  | Review | Study | Goban; game tree over analysis; notes |
  | Review | Compact | Goban; one tile with analysis, game tree and notes as tabs |
  | Play | Play (default) | Goban; game panel above, coach and notes as tabs below |
  | Problems | Problems (default) | Goban cropped to the problem; problem panel above, game tree and notes as tabs below |

- **Adding an optional panel** that the preset has no tile for puts it as a tab in the focused tile.

- **The preset editor** shows a preset as a diagram. Click a tile, then split it right or below, add a panel as a tab,
  or remove it. Any preset can be made the default.
- **Rules** choose which of a type's presets a new workspace opens with, for example by source or by device profile.
- **Storage:** presets are JSON in OPFS, one file per device profile (for example `settings/layouts-desktop.json`).

---

# Device profiles

**Decided (2026-11-27):** there are configurations for mobile, tablet and desktop.

**Proposed:**

- The profile follows the width of the app: Mobile below 760px, Tablet up to 1180px, Desktop above. Both widths are
  player settings.
- Desktop and Tablet each have their own presets. Tablet presets are simpler, mostly the goban beside one tabbed tile.
- Mobile shows one panel at a time, with the workspace's panels as tabs. Its presets set the order of those tabs.
- Export saves the profiles to one JSON file on the player's computer; import reads one back.

**Open:** keeping profiles and other data in step across devices through the player's own file system is not solved
here.

---

# Components

**Decided (2026-11-27): native custom elements.** Pluggable components are custom elements, for example `<sm-goban>`
and `<sm-tree>`. Custom elements are a platform API, not a framework.

**Proposed:**

- **Light DOM,** so the brand's CSS applies to them as it does to the rest of the page.
- **One game session object** per workspace that its components share: the SGF game tree, the current position, and
  the analysis stream. Components listen and announce changes through `EventTarget` events.
- With that, a panel can be added, moved or replaced without changing the board.

---

# Tabs and panels

**Decided (2026-11-27): both.** The app runs in a browser tab with its components as panels, and any panel can also
run in a tab of its own.

**Proposed:**

- **Shared session across tabs.** A workspace's game session is shared between tabs. Each tab holds a copy, kept in
  step with `BroadcastChannel` messages.
- **Coordinated storage.** Writes to the same OPFS files from several tabs are coordinated with the Web Locks API.
- Both are standard browser APIs; no outside code.

---

# The panels

## Goban

**Decided (2026-11-27):** the goban is flat. A 3D board with a visible edge was tried and dropped: it did not look real
and took screen space.

**Proposed:**

- Player settings: coordinates, move numbers, last-move mark (ring, dot, number or none), stone style (shaded or flat),
  wood grain, candidate moves, ownership, and point size.
- Coordinates sit outside the half-stone margin, so stones never touch them.
- The board sizes itself to its tile and stays centred in it.
- The same component draws a cropped view for problems: the real board edges are drawn, and the lines run on past the
  crop's other sides.

## Game tree

**Proposed:**

- The main line runs along the top. Each variation gets its own lane below the move it branches from: a line drops
  straight down from that move, then a short diagonal reaches the variation's first move. Lanes are chosen so no line
  or move crosses another.
- Moves show their numbers; a dot marks a move with a comment; the current move has a shu ring.

## Notes

**Decided (2026-11-27):** notes are Markdown, with Read and Edit as a toggle, never side by side.

**Proposed:** an Edit toolbar for heading, bold, italic and list items, and for inserting the current board point. In
Read, board points written in the text (such as Q11) show as chips.

**Open:** whether a player's notes are kept in the SGF (`C[]`) or beside it as Markdown files. Markdown in a `C[]`
comment shows as plain text in other Go programs.

## Analysis

**Proposed:** the live search of the current position (candidate moves with winrate, score lead and visits), and a
score graph for the whole game from a full review.

## Problems

**Proposed:**

- The board is cropped to the problem's corner or side.
- The player clicks a point to play. A right move is confirmed with its reason. A wrong move is answered with the
  refutation and why it fails.
- Try again, Hint, Solution and Next problem; a log of the player's tries; progress through the collection.

**Open:** where problems come from, and how a collection marks right and wrong answers. The conventions differ between
sources, so they are researched from real files before the SGF kit reads them.

## Files

**Proposed:** a file browser in the style of macOS Finder.

- A sidebar with Library folders (Games, Problems, Weights, Configs), smart folders (for example "Not yet reviewed"),
  and places (this browser's storage with its quota, the bridge computer).
- A path, list views and sortable columns.
- **Columns depend on the folder.** Games show the SGF's own fields: players and ranks (PB, BR, PW, WR), result (RE),
  date (DT), time (TM, OT), rules (RU), komi (KM), handicap (HA), event (EV) and place (PC). Weights show the network's
  size and type from the file name, and fields read from the file itself.
- **An inspector** for the selected file. For a game, a board preview and the SGF values, with `RU[]` shown beside the
  rule set the rules kit maps it to. For weights, the network details, the checksum and the speed measured on this
  device.

The `s` and `d` parts of a KataGo weight file name are the network's training samples and the total rows of training
data. KataGo's `python/train.py` builds the name as `"%s-s%d-d%d" % (exportprefix, global_step_samples,
total_num_data_rows)`.

---

# Material: glass

**Decided (2026-11-27):**

- Three glass strengths are player settings: **Brand** (tint 0.78, the brand's value), **Frosted** (0.62) and
  **Clear** (0.45).
- Edge distortion (the glass bending what is behind it near its edges, most at the corners) is an extra for browsers
  that support it. Chromium browsers (Chrome, Edge) show it. Safari does not, and shows plain frosted glass.

**Proposed:**

- The distortion is an SVG filter in `backdrop-filter`, driven by a displacement map the page draws once, with a
  plain `blur()` and `saturate()` declared first as the fallback.
- Glass panels carry an angled edge highlight, a top sheen and a darker lower rim.

**Open:**

- The brand says the glass tint must not be lowered below 0.78, because the readability tests in `packages/brand/test/`
  are measured at that value. Frosted and Clear have not been run through those tests; text over a black stone is the
  hard case. The brand and its tests need updating before the lower tints ship.
- Firefox has not been checked for the edge distortion.

---

# Engines

**Decided (2026-11-27):**

- In the first build, each workspace attaches to one engine.
- The design allows three kinds of engine: the browser engine (WebGPU), native KataGo through the bridge, and a remote
  service such as zbaduk.

**Proposed:**

- **Engines are shared by all workspaces,** not owned by one.
- **The browser engine: at most one per browser.** A second copy would load the network's weights into GPU memory
  again and compete for the same GPU, so total speed would not rise. It is shared by every tab. The baseline, which
  every target browser supports: one tab runs it in a dedicated worker, chosen with Web Locks, and another takes over
  if it closes. Where a `SharedWorker` can use WebGPU, the engine runs there instead.
- **Native KataGo through the bridge: any number,** since each runs as its own process on the player's computer (for
  example a second network).
- **A remote service** goes through `@stonemind/provider-sdk`: before connecting, the player sees what would be sent
  (the moves of the position being analysed) and must agree.
- **Queue:** live analysis of the focused workspace first, then live analysis of other open workspaces, then
  whole-game reviews in the background.
- **The Engines app** (a singleton app) lists the engines and which workspace uses which, and has a monitor for a running engine: speed,
  how full the GPU batches are, time per network call by stage (the stages the browser engine already measures in
  `lastCallTiming`), the work queue, and which tabs send work. Its settings start at KataGo's own defaults.
- Settings are shown in plain words, with the KataGo key underneath.

**Open:**

- Whether a `SharedWorker` can use WebGPU in each browser. The capability page's reports from the owner's Linux laptop
  (`docs/reports/capabilities/`) show it does in Edge 158 and in Firefox 161; Safari and phones are not checked yet.
  Firefox 161 offers WebGPU on Linux, although MDN's compatibility data said it did not; whether the browser engine
  runs well in Firefox is not yet confirmed.
- Whether zbaduk offers an API that StoneMind can use.
- Using more than one engine in a workspace at once (for example, to compare two networks) is left for later.
- The bridge already pairs with a page through a pairing link that carries a token (`packages/katago-bridge`,
  README, Security); the Engines app reuses it.

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

**Proposed contents:** the goban panel (above): the board drawn on the brand's board tokens, input, overlays (marks,
last move, candidate moves, ownership), coordinates and the cropped problem view.

**Open:** SGF import and problem checking also need to apply moves without a board on screen. Should the goban kit's
move logic be usable on its own, without rendering?

## File kit

**Proposed contents:** OPFS access (list, read, write, move, delete), streamed writes for large files (KataGo weights
are about 100 MB), import from the file picker and drag and drop, export by download, storage quota and persistence
(`navigator.storage`), and the Files app (above). Its first use is KataGo weights and configuration files.

## SGF kit

**Proposed contents:** reading and writing SGF (FF[4]) and the game tree: variations, comments and markup. This is the
`@stonemind/sgf` already planned in MONOREPO.md.

## Tree kit

**Proposed contents:** a general tree model and its navigation, the lane layout for drawing it, and the game tree panel.
The SGF kit's game tree builds on it.

## Workspace kit

**Proposed (new in 0.2):** workspaces, their sources and game sessions, the tiling of panels from a preset's split tree,
dragging splits, tabs, maximise and minimise, the preset editor, rules, and device profiles. It holds no Go knowledge of
its own: panels come from the other kits.

## Materials kit

**Proposed contents:** player preferences, such as theme, glass strength, backdrop, board and stone appearance and
coordinates. They are stored as JSON in OPFS and applied through the brand tokens, building on `@stonemind/brand`. The
app's ThemePicker lives here.

---

# The goban's three uses

**Decided (as stated by the owner, 2026-11-27):** the goban is the key interface for:

1. game analysis and reviews;
2. playing against the AI, as teaching games or normal play;
3. problem solving.

**Decided (2026-11-27):** the three uses are the three goban types, Review, Play and Problems (Workspaces, above).
They share one goban component and differ in their source, the controls that come with the type, and their presets.

**Open (from round 1):**

- What makes a game a teaching game: hints on request, a warning after a big mistake, take-backs, or all three.
- How the AI's strength is set: visits only, or a human-like network (KataGo's human SL network needs its own weights
  file).
- Whether play against the AI has clocks.

---

# Order of work

**Proposed**, one pull request per step:

1. This design document.
2. The file kit and the Files app, first for weights and configuration files.
3. The Engines app, with the browser engine shared across tabs.
4. The rules kit.
5. The SGF and tree kits.
6. The goban kit.
7. The workspace kit: workspaces, tiling presets and device profiles.
8. Panels: game tree, notes, analysis, territory.
9. The three uses: review, play, problems.

---

# Open questions

1. Which static host serves stonemind.app (Hosting).
2. What a live game connects to, and whether it is a Play or a Review workspace (Workspaces).
3. Syncing profiles and data across devices through the player's file system (Device profiles).
4. Whether notes are kept in the SGF or beside it as Markdown (Notes).
5. Where problems come from, and how each source marks right and wrong answers (Problems).
6. Readability of the Frosted and Clear glass, and the brand change they need; edge distortion in Firefox (Material).
7. WebGPU in a `SharedWorker`; zbaduk's API; more than one engine per workspace (Engines).
8. Whether the goban kit's move logic can run without rendering (Goban kit).
9. The SGF `RU[]` mapping table, researched per source (Rules kit).
10. Teaching games, AI strength and clocks (The goban's three uses).
