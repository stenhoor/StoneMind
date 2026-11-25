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
  - Kernels: 3×3 convolutions by Winograd F(4×4,3×3) (an input transform, 36 batched GEMMs, an output transform that
    also adds the residual or bias); other sizes by a tiled implicit-GEMM convolution; batch norm with activation and
    mask; global pooling; small matrix-vector products.
  - Fusion: where a batch norm and activation feed only a 3×3 convolution, the Winograd input transform applies them
    as it reads, so the normalised tensor is never written out.
  - Options `winograd: false` and `fuse: false` turn these off.
  - **fp16 storage** (`precision: 'f16'`, needs the device feature `shader-f16`): activations, Winograd tensors and
    convolution weights are stored as f16; every kernel converts to f32 on load and computes in f32. It applies
    KataGo's **scale8** transform first (`src/model/scale8.ts`, ported from `ModelDesc::applyScale8ToReduceActivations`):
    activations run at 1/8 scale with `mish_scale8`, and raw outputs are multiplied back by 8 on readback, so results
    are unscaled either way. `scale8: true` also works in f32.
  - Mish is computed as x · n/(n + 2) with n = eˣ(eˣ + 2), which equals x · tanh(log(1 + eˣ)) exactly but needs only
    `exp`: `log(1 + eˣ)` lost most of its precision for negative x in f32, and WGSL has no `log1p`.
  - `src/gpu/webgpu.ts` declares the part of the WebGPU API the engine uses, because TypeScript's DOM library doesn't
    include WebGPU yet.

## Verification

`test/tiny-board.test.mjs` reproduces KataGo's own `runnnontinyboardtest` (g170-b6c96, symmetry 3). The parameter
count, win/loss, score, lead, and every policy and ownership value match KataGo's recorded output
(`cpp/tests/results/runNNOnTinyBoardTest.txt` at commit d91ea85) to the printed precision. A second test checks all 8
symmetries exactly against mirrored boards.

`test/goldens.test.mjs` compares the reference network with outputs captured from native KataGo 1.18.2 (CPU/Eigen
build, fp32) on the positions in `test/goldens/positions.json`, under all 8 symmetries: win/loss, lead, score, every
policy and ownership value. Inputs come from `encodePositionV7`. All 10 positions match for g170-b6c96 and for
b18c384nbt (all 8 symmetries; the b18c384nbt run was checked on 2026-11-25). The b18c384nbt test runs only when `STONEMIND_B18_MODEL` points at that model file, because the 98 MB model
is not downloaded in CI; otherwise it is skipped by name.

Goldens must come from KataGo's CPU (Eigen) build, which the reference follows. KataGo's CUDA and Eigen backends differ
on b18c384nbt by more than the test's tolerances (up to about 2e-3 in ownership; see
[the measurement](../../docs/spikes/results/2026-11-b18-goldens-backends.md)). The owner's CUDA capture is kept as
`test/goldens/kata1-b18c384nbt-s9996604416-d4316597426-cuda.json` for the record; tests don't use it.

fp32 variants must match the goldens at the test's tolerances. fp16 variants are judged as KataGo judges an fp16
configuration against an fp32 reference: its `GpuErrorStats` metrics, 99th percentile and maximum, against the limits
of `runBackendErrorTest` (`cpp/tests/testnnevalcanary.cpp`, KataGo 1.18). Those are: win rate 2%/5%, lead and score
1/3 points, top-policy probability 2.5%/6%, policy KL 0.002/0.004, ownership 3.5%/20%. The goldens have no score stdev or
short-term error outputs, so those metrics are not checked. The report gives each metric's ratio to its limit. The check
also evaluates the WGSL activations over a grid of inputs against KataGo's formulas in double precision (within 1e-5).

`test/scale8.test.mjs` checks that the scale8 transform leaves the reference network's function unchanged (raw outputs
divided by 8; post-processed outputs match the goldens). Only b18c384nbt uses Mish, so only it exercises `mish_scale8`;
it runs when `STONEMIND_B18_MODEL` is set. `test/f16.test.mjs` checks the fp32 → fp16 conversion exhaustively.

`test/rules.test.mjs` checks the rules engine on hand-built positions: captures, suicide, simple ko, a triple-ko cycle
under positional, situational and simple ko, a ladder with and without a ladder breaker, and pass-alive area.

**Capturing goldens** for another network (for example b18c384nbt on a machine with KataGo):

```bash
node packages/engine-webgpu/scripts/capture-goldens.mjs --katago <Eigen katago> --config <gtp config> --model <net.bin.gz> --backend EIGEN
```

It writes `test/goldens/<net name>.json`, using `kata-raw-nn <symmetry> 0` with FP16 turned off, and records the
KataGo version, the backend and the network's SHA-256.

**The GPU check** (`scripts/gpu-check.mjs`, run by hand) loads `test/gpu/index.html` in a browser. It runs the
WebGPU network on the golden positions in one batch (each variant: Winograd fused, Winograd unfused, implicit GEMM, fp32 with scale8, and fp16 fused and unfused where the device has `shader-f16`), compares every output with the reference forward pass, and
compares the post-processed results with native KataGo's (symmetry 0) at the goldens test's tolerances. `--bench`
adds evaluations per second by batch size.

```bash
npm run build -w @stonemind/engine-webgpu
npm run gpu-check -w @stonemind/engine-webgpu                         # headless Chromium, via Playwright
node packages/engine-webgpu/scripts/gpu-check.mjs --serve --model <net.bin.gz> --golden <goldens.json>
```

With the benchmark, on a device with `timestamp-query`, the report also has a `profile` at batches 1, 16 and 64: the
plan's GPU time from timestamps at the start and end of its single compute pass, against wall time; and each dispatch
timed in its own pass, summed by kernel, with the ten slowest. `timerTickNs` is the greatest common divisor of the
measured intervals; a large value means the browser coarsens its timestamps. `GpuNetwork.profile(inputs, 'pass' |
'ops')` does the timing.

With `--model`, the slow TypeScript reference comparison is skipped unless `--reference` is given; the goldens are
the correctness check. With `--serve` it only serves the page and prints its address, to open in any browser with WebGPU; each report is
posted back and saved to `--out` (default `gpu-check-report.json`). Headless runs use
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
