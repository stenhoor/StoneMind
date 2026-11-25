# @stonemind/engine-webgpu

The experimental browser engine from [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md). **Phases 2 to 4** (phase 4 in progress):

- **Model loader** (`src/model/`): parses official KataGo `.bin.gz` / `.txt.gz` files (model versions 8–17;
  convolutional nets), using the platform `DecompressionStream`. It follows the reading order of KataGo's `desc.cpp`.
  Transformer blocks and RMSNorm trunks are refused clearly; they come in phase 6.
- **Reference network** (`src/reference/`): a plain fp32 TypeScript forward pass, post-processing and the 8 symmetries,
  following KataGo's CPU (Eigen) backend and `nneval.cpp`. It is slow on purpose: it is the correctness reference for
  the WGSL kernels in phase 4.
- **Rules engine** (`src/rules/`): a port of the parts of KataGo's `Board` and `BoardHistory` that the network's inputs
  depend on: captures, suicide, simple ko, positional and situational superko, the ladder search, pass-alive area, the
  named rule sets, territory scoring's per-stone adjustment and the white handicap bonus. Passes, the territory-scoring
  encore and rule sets with a button are refused with `UnsupportedPositionError` rather than approximated.
- **V7 input encoder** (`src/features/v7.ts`): `encodePositionV7` builds KataGo's 22 spatial planes and 19 globals from
  a position (setup stones and moves), following `NNInputs::fillRowV7`. The earlier simple encoder
  (`src/features/simple.ts`), for positions with no captures, ko, ladders or one-colour regions, is still exported.
- **WebGPU network** (`src/gpu/`), fp32, first version:
  - `requestEngineDevice` asks for `shader-f16`, `subgroups` and `timestamp-query` when the adapter has them.
  - `GpuNetwork.create(engine, model, { maxBatch })` uploads the weights and builds a plan for a 19×19 board (smaller
    boards sit in its top-left corner, masked, as in KataGo).
  - `evaluate(inputs)` runs a batch with one submit and one readback, and returns the same `RawOutputs` as the
    reference forward pass.
  - Kernels: a tiled implicit-GEMM convolution, batch norm with activation and mask, global pooling and small
    matrix-vector products. Winograd F(4×4,3×3), fused epilogues and fp16 come next, checked against these.
  - `src/gpu/webgpu.ts` declares the part of the WebGPU API the engine uses, because TypeScript's DOM library doesn't
    include WebGPU yet.

## Verification

`test/tiny-board.test.mjs` reproduces KataGo's own `runnnontinyboardtest` (g170-b6c96, symmetry 3). The parameter
count, win/loss, score, lead, and every policy and ownership value match KataGo's recorded output
(`cpp/tests/results/runNNOnTinyBoardTest.txt` at commit d91ea85) to the printed precision. A second test checks all 8
symmetries exactly against mirrored boards.

`test/goldens.test.mjs` compares the reference network with outputs captured from native KataGo 1.18.2 (CPU/Eigen
build, fp32) on the positions in `test/goldens/positions.json`, under all 8 symmetries: win/loss, lead, score, every
policy and ownership value. Inputs come from `encodePositionV7`. All 10 positions match for g170-b6c96. The
b18c384nbt goldens (captured by the owner, `test/goldens/kata1-b18c384nbt-…json`) run only when
`STONEMIND_B18_MODEL` points at that model file, because the 98 MB model is not downloaded in CI; otherwise that test
is skipped by name.

`test/rules.test.mjs` checks the rules engine on hand-built positions: captures, suicide, simple ko, a triple-ko cycle
under positional, situational and simple ko, a ladder with and without a ladder breaker, and pass-alive area.

**Capturing goldens** for another network (for example b18c384nbt on a machine with KataGo):

```bash
node packages/engine-webgpu/scripts/capture-goldens.mjs --katago <katago> --config <gtp config> --model <net.bin.gz>
```

It writes `test/goldens/<net name>.json`, using `kata-raw-nn <symmetry> 0` with FP16 turned off, and records the
KataGo version and the network's SHA-256.

**The GPU check** (`scripts/gpu-check.mjs`, run by hand) loads `test/gpu/index.html` in a browser. It runs the
WebGPU network on the golden positions in one batch, compares every output with the reference forward pass, and
compares the post-processed results with native KataGo's (symmetry 0) at the goldens test's tolerances. `--bench`
adds evaluations per second by batch size.

```bash
npm run build -w @stonemind/engine-webgpu
npm run gpu-check -w @stonemind/engine-webgpu                         # headless Chromium, via Playwright
node packages/engine-webgpu/scripts/gpu-check.mjs --serve --model <net.bin.gz> --golden <goldens.json>
```

With `--serve` it only serves the page and prints its address, to open in any browser with WebGPU. Headless runs use
Playwright, a devDependency of this package (approved for this on 2026-11-25); install its browser once with
`npx playwright install chromium`, or set `STONEMIND_CHROMIUM` to a Chromium executable. CI runs the check on every
pull request (`npm run gpu-check`). Without a GPU, Chromium's software adapter (SwiftShader) runs it: correct, but far
too slow to say anything about speed.

The test network (CC0, a g170-run net) is downloaded by `npm run fixtures` from KataGo's repository at that commit and
checked by SHA-256. It is not committed.

## Depends on

No StoneMind packages. Dev only: Playwright (Apache-2.0), for the headless GPU check.

## Used by

Nothing yet. Planned: `apps/engine-bench` (phase 4).

## Development

```bash
npm run check      # typecheck, build, fetch the test network, node --test
```

## Notes

- Written fresh in TypeScript. KataGo's code (MIT, © David J Wu) is the reference for the file format and the maths,
  as the comments in each file say.
- The b18c384nbt goldens are committed, but the check against them runs only on a machine that has the model
  (`STONEMIND_B18_MODEL`, or `--model` and `--golden` for the GPU check).
