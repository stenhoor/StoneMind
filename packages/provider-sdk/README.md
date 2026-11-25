# @stonemind/provider-sdk

Contracts for every external integration: AI, analysis, evidence, resource and export providers. Also `ProviderRegistry`, which enforces the project's provider rules: providers are optional, players choose them, and data goes to an external provider only after the player consents.

Specification: docs/StoneMind-Provider-and-Plugin-SDK.md, docs/StoneMind-Technical-Architecture.md (Integration Architecture)

## Depends on

`@stonemind/domain`: `Game`, `GameAnalysis`, `Observation`, `EvidenceRecord`, `Resource`, `ResourceRelationship`.

## Used by

Nothing yet. Planned: every `provider-*` package, the engines and the web app.

## Deployment

Published as an npm package for first- and third-party plugin authors. Treat its exported interfaces as a public API and follow semver strictly.

## Development

Run these from this directory (`packages/provider-sdk`) after `npm install` at the repository root:

```bash
npm run check      # typecheck + tests
npm run build      # emit dist/
```

## Notes

- Every provider declares a `DataDisclosure`: what is sent, where, and why (Rule 4).
- `ProviderRegistry.get()` throws `ConsentRequiredError` for a `remote` or `self-hosted` provider until `grantConsent()` is called.
- Interface differences between the docs are noted in `docs/OPEN-QUESTIONS.md` §10.

See `docs/MONOREPO.md` for how this package fits with the other sub-projects.
