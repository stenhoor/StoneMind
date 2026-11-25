# @stonemind/domain

The StoneMind data model as TypeScript types: player, goals, ontology, understanding, style, evidence, games, tsumego, teaching sessions, resources, recommendations and study plans.

Specification: docs/StoneMind-Data-Model-Specification.md

## Depends on

None.

## Used by

`@stonemind/ontology`, `@stonemind/provider-sdk`

## Deployment

Published as an npm package. Almost entirely types; the only runtime export is `DEPENDENCY_RELATIONSHIP_TYPES`.

## Development

Run these from this directory (`packages/domain`) after `npm install` at the repository root:

```bash
npm run check      # typecheck + tests
npm run build      # emit dist/
```

## Notes

- Records are plain JSON-serialisable objects. Dates are ISO strings and ids are strings, so everything can be exported (player ownership).
- A change here affects every dependent. Check the contracts table in `docs/MONOREPO.md` before renaming anything.
- Fields that go beyond the spec are listed in `docs/OPEN-QUESTIONS.md` §8.

See `docs/MONOREPO.md` for how this package fits with the other sub-projects.
