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
  named rule sets, territory scoring's per-stone adjustment and the white handicap bonus, and passes: consecutive
  ending passes, simple ko's spight-like ending, and the end and score of an area-scored game. The territory-scoring
  encore (a pass that would end the main phase under territory scoring), ending a game with a group tax, no-result long
  cycles and rule sets with a button are refused with `UnsupportedPositionError` rather than approximated.
- **V7 input encoder** (`src/features/v7.ts`): `encodePositionV7` builds KataGo's 22 spatial planes and 19 globals from
  a position (setup stones and moves, `[player, 'pass']` for a pass; `gtpMove` converts GTP moves), following
  `NNInputs::fillRowV7`, passes included (the pass-history globals, the pass-ends-phase flag, and history hidden under
  friendly-pass rules). `encodeHistoryV7` encodes a `GameHistory` directly, with KataGo's conservative-pass and
  passing-hack options. The earlier simple encoder
  (`src/features/simple.ts`), for positions with no captures, ko, ladders or one-colour regions, is still exported.
- **Search** (`src/search/`): `Search`, an asynchronous port of the core of KataGo's search that keeps a batch of
  leaves in flight for the GPU with virtual losses, as KataGo's search threads do. It covers KataGo's PUCT (log-growing
  exploration constant, parent-utility-stdev factor), first-play urgency, win/loss and score utility (`scorevalue.ts`,
  KataGo's `ScoreValue` table), stats recomputed from children, terminal values, the evaluation cache (keyed as KataGo
  keys it, without move history), random or fixed symmetries, and the analysis engine's `ignorePreRootHistory` and
  `conservativePass`. Area scoring only for now. Not yet ported, each to be added and measured: graph search, noise
  pruning, value weighting, uncertainty weighting, subtree value bias, LCB ordering, policy optimism, wide root noise,
  passing hacks, root symmetry pruning, tree reuse. Evaluation goes through a `NetEvaluator`: the WebGPU network, or
  the reference forward pass.
- **Review** (`src/search/review.ts`): `reviewGame` searches every turn of a game, several turns at once, each with a
  small batch (16 by default), sharing one evaluation cache across turns as KataGo's analysis engine does.
  `SharedEvaluator` combines the concurrent searches' leaves into one network call of up to the network's batch size.
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
  - **GEMM v2** (`gemm: 'v2'`, the default; `'v1'` keeps the first kernels): one generic batched matrix multiply
    (`GEMM_V2`) for the Winograd GEMMs and for 1×1 convolutions, which become a plain GEMM per batch position instead
    of going through the implicit-GEMM convolution. 64 × 128 tiles, 4 × 8 outputs per thread, and a fully unrolled
    inner product with scalar accumulators, so no index there is a run-time value apart from each thread's offsets
    into workgroup memory (WebKit clamps every non-constant array index: `Source/WebGPU/WGSL/BoundsCheck.cpp`).
    `gemmConfig` sets its shape: rows and columns per thread (`rm`, `rn`: 4 or 8), the reduction chunk (`tk`: 8 or
    16) and the arithmetic (`arith`: `f32`; or, where the device has `shader-f16`, `f16` or `f16x2`, the latter on pairs
    of output columns as `vec2<f16>`; both sum each chunk in f16 and add it into f32 accumulators). The default is chosen per device by `defaultGemmConfig`: on Apple GPUs 4 × 4, chunks of 8, with f16 arithmetic
    when `precision` is `'f16'` (fastest on the owner's iPad) and f32 arithmetic otherwise; elsewhere `GEMM_DEFAULT`, 4 × 8, chunks of 16, f32.
  - **Winograd transforms v2** (`winoTransforms: 'v2'`, default `'v1'` until measured): each workgroup copies a few
    channels of one board into workgroup memory with adjacent reads, applying a fused batch norm once per point (v1
    recomputes it for each overlapping tile), and writes the output board back with adjacent writes.
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
under positional, situational and simple ko, a ladder with and without a ladder breaker, and pass-alive area. It also
plays random games on 5×5 and 9×9 and checks the board's incremental chains (stones, liberties, heads) against a flood
fill after every move and undo. `test/passes.test.mjs` checks passes: the end and score of area-scored games, the
territory encore refused, and the pass inputs. The goldens include four positions with passes.

`test/search.test.mjs` checks the search against native KataGo's analysis engine (CPU build, captured with
`scripts/capture-search-goldens.mjs`): both run one search thread, symmetry 0, and the features this search doesn't
have yet switched off, on 9×9 and 19×19 positions under tromp-taylor, Chinese (friendly passes, simple ko) and AGA
(situational superko) rules, including a root after a pass and a root after the end of the game. The same moves get
exactly the same visits, with the same principal variations, and win rates and leads within the network goldens'
tolerances. A one-constant change to the search (`cpuctExplorationLog` 0.45 → 0.4) makes it fail.

**Search benchmark.** The GPU check page's **Search benchmark** button (or `scripts/gpu-check.mjs --search`) runs the
search on the WebGPU network:
- **Live:** visits per second on one 19×19 position (`fight-19`), 2000 visits, at batches 16, 32 and 64.
- **Review:** the fixed 250-move game of `test/goldens/agreement-game.json` at 100 visits a turn, in three modes: one turn
  at a time at batch 64, one at a time at batch 16, and four at once at batch 16 (`reviewGame`). The report includes the
  game and each mode's recommended moves.

`scripts/native-search-bench.mjs` runs native KataGo's analysis engine on the same work from that report: the same
position and visits, and every turn of the same game at the same visits. It also reports how often each browser review
mode recommends native's move. It clears KataGo's evaluation cache before
each timed query, as the browser starts each search with an empty one.

**Top-move agreement.** The page's **Agreement positions** button (or `scripts/gpu-check.mjs --agreement`) records the
search's recommended move on every 5th position of `test/goldens/agreement-game.json` (the game from the owner's
search benchmark), at 100 and 500 visits, batches 16 and 64. `scripts/native-agreement.mjs` runs native KataGo on the
same positions twice per visit count, clearing its cache in between, and once at 4000 visits. It reports how often the
browser agrees with native, how often native agrees with itself (the control, since KataGo's multithreaded search isn't
deterministic), and how often each agrees with the 4000-visit reference.

**Capturing goldens** for another network (for example b18c384nbt on a machine with KataGo):

```bash
node packages/engine-webgpu/scripts/capture-goldens.mjs --katago <Eigen katago> --config <gtp config> --model <net.bin.gz> --backend EIGEN
```

It writes `test/goldens/<net name>.json`, using `kata-raw-nn <symmetry> 0` with FP16 turned off, and records the
KataGo version, the backend and the network's SHA-256.

**The GPU check** (`scripts/gpu-check.mjs`, run by hand) loads `test/gpu/index.html` in a browser. It runs the
WebGPU network on the golden positions in one batch (each variant: Winograd fused and unfused, implicit GEMM, fp32 with scale8, GEMM v2 fused and unfused, and fp16 fused and unfused with each GEMM where the device has `shader-f16`; `--variants <text>` or `?variants=` runs only those whose JSON contains the text), compares every output with the reference forward pass, and
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

**Tuning** (`--tune`, or the page's "Tune the GEMM" button) runs every GEMM v2 shape (`rm`, `rn`, `tk`, and `arith`
f32 and, where available, f16) with Winograd fused and the device's best storage precision. Each is checked for
correctness and benchmarked at batches 16 and 64, without the profile.

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

`apps/engine-bench`: its site build publishes the GPU check page (`test/gpu/`) with this package's compiled modules
and goldens at https://stenhoor.github.io/StoneMind/gpu-check/. There, a model is picked with the page's file input,
and the report can be downloaded. The page works the same served by `scripts/gpu-check.mjs`.

## Development

```bash
npm run check      # typecheck, build, fetch the test network, node --test
```

## Notes

- Written fresh in TypeScript. KataGo's code (MIT, © David J Wu) is the reference for the file format and the maths,
  as the comments in each file say.
- The b18c384nbt goldens are committed, but the check against them runs only on a machine that has the model
  (`STONEMIND_B18_MODEL`, or `--model` and `--golden` for the GPU check).
