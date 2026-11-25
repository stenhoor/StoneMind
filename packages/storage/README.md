# @stonemind/storage

Narrow storage *ports* (`RelationalStore`, `GraphStore`, `AnalyticsStore`) that StoneMind engines depend on. A future `@stonemind/storage-smaltworks` package will implement them on Smaltworks (SQL + Cypher + DataFrames) persisted to OPFS.

Specification: docs/StoneMind-Technical-Architecture.md (Storage Architecture)

## Depends on

None.

## Used by

Nothing yet. Planned: the Smaltworks adapter and the engines.

## Deployment

Published as an npm package (types only).

## Development

Run these from this directory (`packages/storage`) after `npm install` at the repository root:

```bash
npm run check      # typecheck + tests
npm run build      # emit dist/
```

## Notes

- Keep the ports minimal: they describe what StoneMind needs, not what Smaltworks offers.
- `AnalyticsStore` is a placeholder until the DataFrame API is known (`docs/OPEN-QUESTIONS.md` §9).

See `docs/MONOREPO.md` for how this package fits with the other sub-projects.
