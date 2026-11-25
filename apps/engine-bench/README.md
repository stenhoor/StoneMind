# @stonemind/engine-bench

The engine bench page from [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md), published on GitHub Pages at
https://stenhoor.github.io/StoneMind/engine-bench/.

**Phase 1:** it connects to the native KataGo on your computer through `@stonemind/katago-bridge`. It analyses a
position and shows the top moves, with the visits/s measured in the page. Later phases add the browser engine and the
comparisons.

## Depends on

`@stonemind/engine` (loaded in the page through an import map) and `@stonemind/brand` (tokens, components and logos,
copied unchanged).

## Used by

Nothing: it is a deployable page.

## Deployment

`npm run build` assembles a static site in `dist/site` with no bundler. The `Pages` workflow publishes it on every push
to `main`, which needs **Settings → Pages → Source: GitHub Actions**.

## Development

```bash
npm run serve      # build, then http://localhost:8080/engine-bench/
```

```bash
npm run check      # typecheck, build, then check that every file the page references is in the site
```

`scripts/e2e.mjs` is an end-to-end check in Chromium: fake KataGo → bridge → page. It needs a Playwright that Node can
import; Playwright is not a project dependency.

## Notes

- HTML and CSS on the brand tokens, plus one TypeScript module, with no framework.
- Visits/s includes the bridge and WebSocket overhead. It is a page-side measure, comparable across engines.
