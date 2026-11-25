# CLAUDE.md

Guidance for AI coding assistants working in this repository.

## What this is

StoneMind is a local-first, browser-first, player-centric Go coaching companion. The
specifications in `docs/` are the source of truth. Read `docs/StoneMind-Charter.md`
(especially its "Invariants" section) before making design decisions.

## Repository structure

- npm-workspaces monorepo. Each sub-project in `packages/*` (libraries) or `apps/*`
  (deployables, none yet) is an **independent effort**, with its own package.json, version,
  README, tsconfig, vitest config and tests.
- `docs/MONOREPO.md` is the authoritative map of sub-projects and their interdependencies.
  **Any change that adds a sub-project or a cross-package dependency must update it**,
  along with the "Depends on" / "Used by" sections of the affected package READMEs.
- Cross-package imports use the package name (`@stonemind/domain`), never relative paths
  into another package. Dependencies point inward (contracts → engines → apps), with no cycles.
- The root `workspaces` array is in dependency order. Add new packages after their dependencies.

## Commands

```bash
npm install          # at root, once
npm run check        # typecheck + tests, all sub-projects
npm run build        # build all, in dependency order
npm run check -w @stonemind/ontology   # one sub-project
```

## Conventions

- TypeScript, strict (`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`,
  `verbatimModuleSyntax`). ESM only. Use `.js` extensions in relative imports and `import type` for types.
- Dev-time resolution uses the `stonemind-source` export condition (points at `src/`).
  Builds use `dist/`. See `docs/MONOREPO.md`.
- Domain records must stay JSON-serialisable (ISO date strings, string ids), because all player data must be exportable.
- No sub-project may require a specific AI or cloud provider. External calls go through
  `@stonemind/provider-sdk` providers, which declare a `DataDisclosure` and require consent.
- Storage goes through `@stonemind/storage` ports. Smaltworks isn't available as a dependency yet.
- Ontology edits go in `packages/ontology/src/seed.json`. Run its tests afterwards.
  Known seed issues are listed in the tests and in `docs/OPEN-QUESTIONS.md`.
- When the code has to make a decision the specs leave open, record it in `docs/OPEN-QUESTIONS.md`.
