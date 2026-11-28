# StoneMind Monorepo and Sub-Project Interdependencies

Version: 0.1

Status: Living document. Update it in the same change that adds, removes, or rewires a sub-project.

---

# Principles

StoneMind is a monorepo of independent sub-projects.

- **Each sub-project is an independent effort.** It has its own `package.json`, version,
  README, build, typecheck, and tests, and can be built, tested, and released on its own.
- **Dependencies are explicit.** A sub-project may only use another sub-project through a
  declared `dependencies` entry and that sub-project's public entry point (`exports`).
  No relative imports across sub-project folders (`../../other-package/src/...`).
- **Dependencies point inward.** Contracts and data (`domain`, `storage`, `provider-sdk`)
  sit at the centre; engines, providers, and apps depend on them, never the reverse.
- **No cycles.** The dependency graph between sub-projects must stay acyclic.
- **One product, one brand.** StoneMind is a single product. Sub-projects are independent engineering efforts, not
  products: none has a brand, mark or colour of its own, and package names never appear in the product's interface.
  Every sub-project with a user-facing surface takes its brand from `@stonemind/brand`.
- **Every interdependency is documented** here and in both sub-projects' READMEs
  ("Depends on" / "Used by").

---

# Layout

```text
StoneMind/
├── docs/                    Specifications (charter, architecture, data model, ...)
├── packages/                Libraries: contracts, data, engines, providers
│   ├── brand/               @stonemind/brand
│   ├── domain/              @stonemind/domain
│   ├── storage/             @stonemind/storage
│   ├── files/               @stonemind/files: the file kit (OPFS)
│   ├── ontology/            @stonemind/ontology
│   ├── provider-sdk/        @stonemind/provider-sdk
│   ├── engine/              @stonemind/engine
│   ├── engine-webgpu/       @stonemind/engine-webgpu (experimental)
│   └── katago-bridge/       @stonemind/katago-bridge
├── apps/                    Deployable applications
│   ├── engine-bench/        @stonemind/engine-bench (GitHub Pages)
│   └── web/                 @stonemind/web: the app (stonemind.app); runs locally for now. Planned: site → stonemind.org
├── tsconfig.base.json       Shared compiler settings (each sub-project extends it)
└── package.json             npm workspaces root; orchestration scripts only
```

The root `package.json` lists workspaces in dependency order, so `npm run build`
from the root builds dependencies before dependents.

---

# Current Sub-Projects

| Sub-project | Kind | Deployed as | Depends on | Used by |
|---|---|---|---|---|
| `@stonemind/brand` | Brand (CSS, JSON, SVG) | npm package; files copied unchanged by apps | none | engine-bench, web; planned: site |
| `@stonemind/domain` | Library (types) | npm package | none | ontology, provider-sdk |
| `@stonemind/storage` | Library (ports) | npm package | none | files; planned: Smaltworks adapter, engines |
| `@stonemind/files` | Library (the file kit) | npm package; loaded in pages without a bundler | storage (types only); dev only: playwright 1.65.0 | web |
| `@stonemind/ontology` | Library (data + logic) | npm package; `seed.json` is also a standalone data artifact | domain | none yet (future ontology engine, web app) |
| `@stonemind/provider-sdk` | Library (contracts + registry) | npm package for third-party plugin authors | domain | none yet (future providers, engines, web app) |
| `@stonemind/engine` | Library (contract + bridge client) | npm package; loaded in pages through an import map | none | katago-bridge, engine-webgpu, engine-bench |
| `@stonemind/engine-webgpu` | Library (experimental engine) | npm package; loaded in pages without a bundler | engine (the `AnalysisEngine` contract; only the `./session` entry point loads it when run) | engine-bench (its site build publishes the GPU check page) |
| `@stonemind/katago-bridge` | Local tool (Node CLI) | Run from a checkout on the player's computer | engine | players running native KataGo |
| `@stonemind/web` | App (static pages) | Not deployed yet: runs locally (`npm run serve -w @stonemind/web`); planned at **stonemind.app** | brand, files; dev only: playwright 1.65.0 (Apache-2.0, approved 2026-11-27, browser tests) | none |
| `@stonemind/engine-bench` | App (static page) | GitHub Pages at https://stonemind.org/ (a placeholder home page), /engine-bench/, /gpu-check/ and /ort-bench/ | engine, brand, engine-webgpu; outside: onnxruntime-web 1.31.1 (MIT, benchmark-only, approved 2026-11-25), copied into /ort-bench/ only | none |

```text
          @stonemind/domain          @stonemind/storage          @stonemind/brand (no dependencies)
            ▲           ▲
            │           │
 @stonemind/ontology   @stonemind/provider-sdk
```

## Interface contracts between sub-projects

| From → To | What is relied on | Breaking-change impact |
|---|---|---|
| ontology → domain | `Concept`, `ConceptRelationship`, `RelationshipType`, `ConceptCategory`, `StyleDimension` | Renaming a relationship type or category invalidates `seed.json` |
| apps → brand (planned) | `tokens.css` custom properties, `components.css` classes (`sm-*`), logo file names | Renaming a token or class breaks every app; change brand first, then each app, in the same release |
| katago-bridge → engine | `BRIDGE_PROTOCOL` (the WebSocket subprotocol) and the pairing link format read by `parsePairingFragment` | Changing either breaks pairing between a bridge and a page of a different version |
| files → storage | The `FileStore` port and its types (`FileEntry`, `FileData`, `FileWriteOptions`) | Typecheck catches breaks; the file kit's browser tests exercise the port |
| web → files | `OpfsFileStore`, `importFiles`, `pickFiles`, `filesFromDrop`, `downloadBlob`, `storageInfo`, `requestPersistence` and the path helpers; the layout of the package's `dist/` (all its `.js` files, the sync writer among them), copied to `vendor/files/` and named in the page's import map | Typecheck catches API breaks; the Files app's browser tests run on the copied modules |
| web → brand | `tokens/tokens.css`, `styles/components.css` and `logos/`, copied unchanged by the app's build; the theme key `stonemind.theme` | Moving those files breaks the build; the browser tests load the page |
| engine-bench → engine | `connectBridge`, `parsePairingFragment`, the `AnalysisEngine` API and protocol types | Typecheck catches breaks; the site test checks the import map |
| engine-webgpu → engine | `LineTransport`, `EngineDescriptor` and `JsonLineSession`, which `createAnalysisEngine` (`@stonemind/engine-webgpu/session`) puts the browser engine behind | Typecheck catches breaks; `test/engine.test.mjs` runs the engine through `JsonLineSession`. The main entry point imports engine for types only, so pages loading it need no import map |
| engine-bench → engine-webgpu | The layout of the package's `dist/`, `test/gpu/` and `test/goldens/`, which the site build copies to `gpu-check/`; and, for the protocol check page, `loadModel`, `requestEngineDevice`, `GpuNetwork` and `createAnalysisEngine` (`./session`), loaded from `vendor/engine-webgpu/` through its import map | Moving those files breaks the site build; typecheck catches API breaks; the site test checks each page's imports and data |
| provider-sdk → domain | `Game`, `GameAnalysis`, `Observation`, `EvidenceRecord`, `Resource`, `ResourceRelationship` | Third-party providers must be updated; bump provider-sdk major version |

---

# Planned Sub-Projects

From docs/StoneMind-Technical-Architecture.md (V1 scope) and the roadmap milestones.
Expected dependencies are listed so they can be checked when each is created.

| Planned sub-project | Path | Purpose | Expected dependencies | Milestone |
|---|---|---|---|---|
| `@stonemind/site` | `apps/site` | Project website, deployed to **stonemind.org**: research, specifications, docs, community | brand (may render content from `docs/`) | M0–M1 |
| `@stonemind/sgf` | `packages/sgf` | SGF parsing/serialisation | none | M1 |
| `@stonemind/storage-smaltworks` | `packages/storage-smaltworks` | Smaltworks + OPFS implementation of storage ports | storage, Smaltworks | M1 |
| `@stonemind/provider-katago` | `packages/provider-katago` | KataGo analysis provider: puts an `AnalysisEngine` (browser or local) behind the provider SDK, with its data disclosure | provider-sdk, engine, domain, sgf | M1 |
| `@stonemind/evidence-engine` | `packages/evidence-engine` | Observations → indicators → hypotheses pipeline | domain, ontology | M1–M2 |
| `@stonemind/player-graph` | `packages/player-graph` | Player Understanding Graph engine | domain, ontology, storage | M2 |
| `@stonemind/recommendation-engine` | `packages/recommendation-engine` | Goal-aware recommendations | domain, ontology, player-graph | M3 |
| `@stonemind/style-engine` | `packages/style-engine` | Style Graph inference | domain, ontology | M4 |
| AI providers (`provider-ollama`, `provider-anthropic`, ...) | `packages/provider-*` | Bring Your Own AI | provider-sdk | as needed |

---

# Deployment Targets

| Domain | Serves | Sub-project |
|---|---|---|
| `stonemind.app` | The application. It gets its own origin because OPFS storage is scoped per origin: player data lives under `stonemind.app` and is never shared with the website. | `apps/web` |
| `stonemind.org` | The main website: research, specifications, documentation, community. | `apps/site` |

Keep the app on its own origin. Moving it to a different domain or subdomain later would
leave existing players' OPFS data behind at the old origin, so that change would need a
migration and export path first.

---

# Working Rules

## Adding a sub-project

1. Create `packages/<name>/` (library) or `apps/<name>/` (deployable).
2. Copy the standard files from an existing package: `package.json`, `tsconfig.json`,
   `tsconfig.build.json`, `vitest.config.ts`, `README.md`.
3. Declare internal dependencies in `dependencies` (exact workspace version).
4. Add the folder to the root `workspaces` array **after** everything it depends on.
5. Update the tables above and the "Used by" section of each dependency's README.

## Development vs. published resolution

Each package's `exports` has a `stonemind-source` condition pointing at `src/index.ts`.
Typecheck (`tsconfig.base.json` `customConditions`) and tests (`vitest.config.ts`
`resolve.conditions`) use it, so changes in a dependency are visible immediately without
rebuilding. `tsconfig.build.json` clears the condition, so builds and published consumers use
the dependency's compiled `dist/`. That is why build order matters.

## CI

CI runs only the checks a change needs (`scripts/ci-affected.mjs`):
- **Affected sub-projects:** the ones whose files changed, plus every sub-project that depends on one of them,
  directly or not. Dependencies are read from each package's `package.json`, so declaring them there (step 3 above)
  is what keeps CI correct.
- **The GPU check** runs only when a file changed that no other check tests: the WebGPU network and the code it uses
  (`src/gpu`, `src/model`, `src/reference`), the check's page and script, its test network and goldens, or the
  package's build settings (the list is `GPU_ONLY` in `scripts/ci-affected.mjs`). The node tests compare the rest of
  the engine with native KataGo, and a node test checks that every name the page imports is still exported.
- **Browser tests** (an `e2e` script) run in Chromium, Firefox and WebKit for each affected sub-project that has one,
  when it, or something it depends on, changed in a file other than Markdown.
- **Shared files** (the root `package.json`, `tsconfig.base.json`, `.nvmrc`) run everything.
- **`package-lock.json`:** a change only to our own workspaces' entries (one workspace adding another as a dependency)
  counts as a change to those workspaces; any other change, such as an outside package's version, runs everything.
- **CI's own files** (`.github/`, `scripts/ci-affected.mjs`) run everything only when nothing else changed; alongside
  a workspace change they are tested by that change's run.
- **Time limits:** each browser test file has 5 minutes (`--test-timeout` in the `e2e` scripts: `node --test` applies
  it to each file, not to each test inside one), the browser-test step 15 minutes, and the job 30, so a hang fails the
  run. Playwright's own waits fail after 30 s.
- **Before pushing,** run what CI will run for the change in all three browsers
  (`BROWSERS=chromium,firefox,webkit npm run e2e -w <package>`).
  Playwright's browsers download from `cdn.playwright.dev`.
- **Files only under `docs/`, or Markdown at the root,** run nothing.

## Versioning

Each sub-project is versioned independently (semver). A breaking change to a contract
listed under "Interface contracts" requires a major bump of the providing package and an
update to every dependent in the same change.
