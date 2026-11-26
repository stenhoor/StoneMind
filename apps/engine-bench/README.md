# @stonemind/engine-bench

The engine bench page from [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md), published on GitHub Pages at
https://stenhoor.github.io/StoneMind/engine-bench/.

**Phase 1:** it connects to the native KataGo on your computer through `@stonemind/katago-bridge`. It analyses a
position and shows the top moves, with the visits/s measured in the page. Later phases add the browser engine and the
comparisons.

**Home page** (`home/`): the site root, https://stonemind.org/ (the custom domain of the GitHub Pages site). A placeholder
under construction: the goals from the charter, written as goals, with links to the charter, the repository and the
developer pages. It uses its own copy of the brand files. A Light / Dark switch (the brand's SegmentedControl) sets
the colour mode: Light (Washi & Shu) by default, whatever the device's setting, and a visitor's choice is remembered
(`stonemind.theme`).

## Depends on

`@stonemind/engine` (loaded in the page through an import map) and `@stonemind/brand` (tokens, components and logos,
copied unchanged).

`@stonemind/engine-webgpu`: the site build also publishes that package's GPU check page at
https://stenhoor.github.io/StoneMind/gpu-check/, copying its compiled modules, `test/gpu/` and its goldens (not the
CUDA record) with the package's own layout, so a browser on any device can run the check by picking a model file.

**onnxruntime-web baseline** (`ort-bench/`, published at https://stenhoor.github.io/StoneMind/ort-bench/): spike
0001's pass mark 1. A developer page that runs KataGo's own ONNX graph (`katago dumponnx -skip-scale8`, from a KataGo
build with the ONNX or TensorRT backend) under onnxruntime-web 1.31.1, with its WebGPU and WASM execution providers. For
each, it checks the 19×19 golden positions against native KataGo at the engine's goldens tolerances, then measures
evaluations per second by batch size, the same way as the GPU check page (the first 19×19 golden position repeated,
one warm-up, the median of 5 runs). It needs the `.onnx` file and the same network's `.bin.gz` (for post-processing).
onnxruntime-web is the approved benchmark-only outside code (MIT); only its WebGPU and WASM builds are copied, with its
licence, and the product never loads it. `npm run serve` serves it cross-origin isolated, so the WASM provider can use
threads (GitHub Pages can't send those headers), and saves the reports it posts to `REPORT_OUT`.

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
