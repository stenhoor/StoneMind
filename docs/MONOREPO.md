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
│   ├── ontology/            @stonemind/ontology
│   └── provider-sdk/        @stonemind/provider-sdk
├── apps/                    Deployable applications (none yet): web → stonemind.app, site → stonemind.org
├── tsconfig.base.json       Shared compiler settings (each sub-project extends it)
└── package.json             npm workspaces root; orchestration scripts only
```

The root `package.json` lists workspaces in dependency order, so `npm run build`
from the root builds dependencies before dependents.

---

# Current Sub-Projects

| Sub-project | Kind | Deployed as | Depends on | Used by |
|---|---|---|---|---|
| `@stonemind/brand` | Brand (CSS, JSON, SVG) | npm package; files copied unchanged by apps | none | none yet (apps/web, apps/site) |
| `@stonemind/domain` | Library (types) | npm package | none | ontology, provider-sdk |
| `@stonemind/storage` | Library (ports) | npm package | none | none yet (future Smaltworks adapter, engines) |
| `@stonemind/ontology` | Library (data + logic) | npm package; `seed.json` is also a standalone data artifact | domain | none yet (future ontology engine, web app) |
| `@stonemind/provider-sdk` | Library (contracts + registry) | npm package for third-party plugin authors | domain | none yet (future providers, engines, web app) |

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
| provider-sdk → domain | `Game`, `GameAnalysis`, `Observation`, `EvidenceRecord`, `Resource`, `ResourceRelationship` | Third-party providers must be updated; bump provider-sdk major version |

---

# Planned Sub-Projects

From docs/StoneMind-Technical-Architecture.md (V1 scope) and the roadmap milestones.
Expected dependencies are listed so they can be checked when each is created.

| Planned sub-project | Path | Purpose | Expected dependencies | Milestone |
|---|---|---|---|---|
| `@stonemind/web` | `apps/web` | Browser-first application, deployed to **stonemind.app** | brand, domain, ontology, provider-sdk, storage, engines | M1 |
| `@stonemind/site` | `apps/site` | Project website, deployed to **stonemind.org**: research, specifications, docs, community | brand (may render content from `docs/`) | M0–M1 |
| `@stonemind/sgf` | `packages/sgf` | SGF parsing/serialisation | none | M1 |
| `@stonemind/storage-smaltworks` | `packages/storage-smaltworks` | Smaltworks + OPFS implementation of storage ports | storage, Smaltworks | M1 |
| `@stonemind/provider-katago` | `packages/provider-katago` | KataGo analysis provider | provider-sdk, domain, sgf | M1 |
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

## Versioning

Each sub-project is versioned independently (semver). A breaking change to a contract
listed under "Interface contracts" requires a major bump of the providing package and an
update to every dependent in the same change.
