# @stonemind/ontology

The Go Ontology seed (`src/seed.json`, encoded from `docs/StoneMind-Ontology-Seed.md`), the seven style dimensions, and validation: name resolution, dangling references, duplicates, and learning-dependency cycle detection.

Specification: docs/StoneMind-Ontology-Seed.md, docs/StoneMind-Ontology-Design.md, docs/StoneMind-Style-Model.md

## Depends on

`@stonemind/domain`: `Concept`, `ConceptRelationship`, `RelationshipType`, `ConceptCategory`, `StyleDimension`.

## Used by

Nothing yet. Planned: the ontology engine, the evidence engine, the player graph and the web app.

## Deployment

Published as an npm package. `src/seed.json` is also usable on its own as an open-format data artifact (it is copied to `dist/seed.json`).

## Development

Run these from this directory (`packages/ontology`) after `npm install` at the repository root:

```bash
npm run check      # typecheck + tests
npm run build      # emit dist/
```

## Notes

- `seed.json` relation targets are concept **names** as written in the seed doc. `resolveSeed()` maps them to ids, using `aliases` where needed, and reports any it cannot resolve.
- The tests fix the current known issues (one cycle, 21 unresolved names) so that new problems fail CI. See `docs/OPEN-QUESTIONS.md` §1–6.
- Contributors editing the ontology should only need to touch `src/seed.json` and run `npm test`.

See `docs/MONOREPO.md` for how this package fits with the other sub-projects.
