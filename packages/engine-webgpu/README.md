# @stonemind/engine-webgpu

**What this engine is for.** It runs KataGo's networks and search in the browser, for players who don't want to, or
can't, install or build KataGo natively. It aims to get the most performance a browser allows. It is not a way to get
maximum performance: native KataGo on the same machine is faster. In very limited testing (one laptop GPU, one
browser), a game review took about 1.4 to 1.9 times as long as native KataGo, depending on browser settings.

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
  ending passes, simple ko's spight-like ending, and the end and score of an area-scored game. Moves are played, recorded and
  undone with KataGo's own chain bookkeeping (`Board::playMoveRecorded` and `Board::undo`), and pass-alive area uses
  KataGo's arrays and linked lists (`docs/spikes/results/2026-11-b18-search-js.md`). The territory-scoring
  encore (a pass that would end the main phase under territory scoring), ending a game with a group tax, no-result long
  cycles and rule sets with a button are refused with `UnsupportedPositionError` rather than approximated.
- **V7 input encoder** (`src/features/v7.ts`): `encodePositionV7` builds KataGo's 22 spatial planes and 19 globals from
  a position (setup stones and moves, `[player, 'pass']` for a pass; `gtpMove` converts GTP moves), following
  `NNInputs::fillRowV7`, passes included (the pass-history globals, the pass-ends-phase flag, and history hidden under
  friendly-pass rules). `encodeHistoryV7` encodes a `GameHistory` directly, with KataGo's conservative-pass and
  passing-hack options. Its `ladders` option, not a KataGo option, reuses ladder results by board: KataGo searches
  ladders on the current board and the two before it for every input, and a search passes a cache so that a node's
  board, already searched when the node was encoded, isn't searched again for its children and grandchildren. The
  inputs are the same either way (`test/ladder-cache.test.mjs`). The earlier simple encoder
  (`src/features/simple.ts`), for positions with no captures, ko, ladders or one-colour regions, is still exported.
- **Search** (`src/search/`): `Search`, an asynchronous port of the core of KataGo's search that keeps a batch of
  leaves in flight for the GPU with virtual losses, as KataGo's search threads do. It covers KataGo's PUCT (log-growing
  exploration constant, parent-utility-stdev factor), first-play urgency, win/loss and score utility (`scorevalue.ts`,
  KataGo's `ScoreValue` table), stats recomputed from children, terminal values, the evaluation cache (keyed as KataGo
  keys it, without move history), random or fixed symmetries, and the analysis engine's `ignorePreRootHistory` and
  `conservativePass`. The analysis orders moves and builds principal variations as KataGo does, from its play
  selection values with LCB selection (`useLcbForSelection`, on by default as in native's analysis engine), and reports
  each move's `lcb`, `utilityLcb` and `playSelectionValue`. Area scoring only for now. Not yet ported, each to be added
  and measured: graph search, noise pruning, value weighting, uncertainty weighting, subtree value bias, policy
  optimism, wide root noise, passing hacks, root symmetry pruning, tree reuse. Evaluation goes through a `NetEvaluator`: the WebGPU network, or
  the reference forward pass.
- **Review** (`src/search/review.ts`): `reviewGame` searches every turn of a game, several turns at once, each with a
  small batch, sharing one evaluation cache across turns as KataGo's analysis engine does. The defaults are native
  KataGo's `analysis_example.cfg`: 2 turns at once, batch 16.
  `SharedEvaluator` stands in for KataGo's neural-net server: it collects the concurrent searches' leaves into network
  calls of up to the network's batch size, with up to the network's `maxInFlight` calls at once. When two calls can
  start, searches that asked together are answered by separate calls, so one search picks its next leaves while the
  other's call runs on the GPU. Each search still has at most its batch of leaves in flight.
- **WebGPU network** (`src/gpu/`), fp32, first version:
  - `requestEngineDevice` asks for `shader-f16`, `subgroups` and `timestamp-query` when the adapter has them.
  - `GpuNetwork.create(engine, model, { maxBatch })` uploads the weights and builds a plan for a 19×19 board (smaller
    boards sit in its top-left corner, masked, as in KataGo).
  - `evaluate(inputs)` runs a batch with one submit and one readback, and returns the same `RawOutputs` as the
    reference forward pass. Up to `maxInFlight` calls (option, default 2) run at once, each with its own readback
    buffer, so one call's inputs are written and its work submitted while another's outputs are read back. Calls are
    submitted in the order they're made; `profile` runs alone.
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
  - **GEMM v3** (`gemm: 'v3'`, the default except on Apple GPUs, from `defaultGemm`): GEMM v2's tiles and
    parameters, with each thread owning runs of 4 adjacent rows and columns so it reads its operands from workgroup
    memory as `vec4`s, `vec4` accumulators, f32 arithmetic, and the next chunk of A and B read into registers while
    the current one is multiplied. It does the same fma steps in the same order as v2's f32 arithmetic, so the outputs
    are the same. `gemmV3Config` sets its shape (`rm`, `rn`: 4 or 8; `tk`: 8 or 16; default `GEMM_V3_DEFAULT`,
    4 × 8, chunks of 16). `wy` (8 or 16, default 16) sets the threads along M, so the tile is wy·rm × 16·rn, and
    `double: true` alternates two buffers of workgroup memory so each chunk needs one barrier. Neither was faster on
    the owner's RTX 3080 Laptop GPU (docs/spikes/results/2026-11-b18-kernel-experiments.md). Apple GPUs keep GEMM v2
    with their measured shape until v3 is measured there.
  - **GEMM v2** (`gemm: 'v2'`; `'v1'` keeps the first kernels): one generic batched matrix multiply
    (`GEMM_V2`) for the Winograd GEMMs and for 1×1 convolutions, which become a plain GEMM per batch position instead
    of going through the implicit-GEMM convolution. 64 × 128 tiles, 4 × 8 outputs per thread, and a fully unrolled
    inner product with scalar accumulators, so no index there is a run-time value apart from each thread's offsets
    into workgroup memory (WebKit clamps every non-constant array index: `Source/WebGPU/WGSL/BoundsCheck.cpp`).
    `gemmConfig` sets its shape: rows and columns per thread (`rm`, `rn`: 4 or 8), the reduction chunk (`tk`: 8 or
    16) and the arithmetic (`arith`: `f32`; or, where the device has `shader-f16`, `f16` or `f16x2`, the latter on pairs
    of output columns as `vec2<f16>`; both sum each chunk in f16 and add it into f32 accumulators). The default is chosen per device by `defaultGemmConfig`: on Apple GPUs 4 × 4, chunks of 8, with f16 arithmetic
    when `precision` is `'f16'` (fastest on the owner's iPad) and f32 arithmetic otherwise; elsewhere `GEMM_DEFAULT`, 4 × 8, chunks of 16, f32.
  - **Matrix hardware** (`matrix`, default `'auto'`): where the device offers subgroup matrices with an f16 → f32
    16 × 16 × 16 shape, `precision` is `'f16'` and `gemm` is `'v2'` or `'v3'`, the Winograd GEMMs and 1×1 convolutions run
    on subgroup matrices (`GEMM_MMA`: f16 in, f32 sums), on the same weights as `gemm`. Each call uses the shape
    `MATRIX_GEMM_BY_BATCH` gives for its batch size (2 × 1 matrices at batch 1, 2 × 2 up to 8, 4 × 2 per subgroup and
    3 × 2 subgroups sharing their loads through workgroup memory above that). A GEMM whose shape fits none of them keeps
    `gemm` at every batch size, so a position's result doesn't depend on its call's batch size. Browsers offer the
    feature (`chromium-experimental-subgroup-matrix`) only behind a switch today; on the owner's laptop, Edge with
    `--enable-unsafe-webgpu` offers it, and the device requests it when offered. Elsewhere, and with `'off'`, nothing
    changes. On the owner's laptop it was faster than GEMM v3 at every batch measured (1–64), the batch-16 pass took
    22–25% less time, and the review 15.8% less (`docs/spikes/results/2026-11-b18-matrix-ladder.md`). The first matrix
    GEMM, the Winograd GEMMs only with transposed weights (`winoGemm: 'mma'`), stays as an option
    (`docs/spikes/results/2026-11-b18-matrix.md`).
  - **Winograd transforms v2** (`winoIn`, `winoOut`, `winoTransforms` sets both): each workgroup copies a few channels
    of one board into workgroup memory with adjacent reads, applying a fused batch norm once per point (v1 recomputes it
    for each overlapping tile), and writes the output board back with adjacent writes. The v2 input transform was
    faster than v1 in every steady measurement, the output transforms in neither order consistently
    (`docs/spikes/results/2026-11-b18-steady.md`).
  - **Winograd transforms v3** (the default for both): v2 reading the input boards, or writing the output boards, as
    one contiguous span per workgroup, four values at a time. The input transform also stages the fused batch norm's
    per-channel values and the mask (one bit per point) in workgroup memory once, and issues its board reads before
    that staging. The output transform covers 4 channels of 2 boards per workgroup, so its reads of the GEMM's output
    are runs of 50 tiles, and issues its residual reads before its writes. On the owner's laptop, at batch 16: input
    −25% against v2, output −24% against v1, batch-16 pass −6.9% in the budget benchmark, review −3.7%
    (`docs/spikes/results/2026-11-b18-transform-ladder.md`).
  - **Batch norm v3** (`bnAct`, the default): batch norm, activation and mask with 8 values per thread, loaded and
    stored 4 at a time (v2: 4 per thread; v1: one). It's used for layers whose channels are a multiple of 8, which is
    all of b18's. On the owner's laptop, each alone: batch 16 at 78.4–85.1% of copy bandwidth (median 83.7%) against
    v1's 55.1–56.1%, and batch 64 at 93.3–95.1% against 57.2–58.1%. The review's change was too small to measure
    (`docs/spikes/results/2026-11-b18-bn-act.md`).
  - Mish is computed as x · n/(n + 2) with n = eˣ(eˣ + 2), which equals x · tanh(log(1 + eˣ)) exactly but needs only
    `exp`: `log(1 + eˣ)` lost most of its precision for negative x in f32, and WGSL has no `log1p`.
  - `src/gpu/webgpu.ts` declares the part of the WebGPU API the engine uses, because TypeScript's DOM library doesn't
    include WebGPU yet.

## Browser flags for more speed

**Use these at your own risk.** They change how the browser treats your GPU, for every site, not only StoneMind.
StoneMind takes no responsibility for any stability, performance or hardware problems they cause. The engine works
without them.

These are command-line flags for Chromium-based browsers. They were measured only in Microsoft Edge 158 on the owner's
Linux laptop (RTX 3080 Laptop GPU, alongside a second GPU); other browsers, systems and GPUs may behave differently.

- **`--use-webgpu-power-preference=force-high-performance`**: makes WebGPU use the high-performance GPU on a computer
  with two. The engine already asks for it (`powerPreference: 'high-performance'` in `src/gpu/device.ts`), but on the
  owner's laptop Edge used the NVIDIA GPU only when started with this flag.
- **`--enable-unsafe-webgpu`**: turns on WebGPU features that browsers consider experimental or unsafe, and with them
  matrix hardware (`chromium-experimental-subgroup-matrix`, offered on the owner's laptop only with this flag). The
  engine then uses it automatically (`matrix: 'auto'`). With it, on the owner's laptop, the batch-16 network pass took
  22–25% less time and a 250-move review 16–19% less (`docs/spikes/results/2026-11-b18-matrix-ladder.md`). The flag
  also turns off WebGPU safety restrictions for every site the browser visits.

Every measurement on the owner's laptop was also made with `--ignore-gpu-blocklist`, which overrides the browser's
list of GPUs and drivers it considers unreliable. Whether WebGPU or matrix hardware needs it there hasn't been tested.

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
- **Review:** the fixed 250-move game of `test/goldens/agreement-game.json` at 100 visits a turn, in four modes: one turn
  at a time at batch 64, one at a time at batch 16, two at once at batch 16 (`reviewGame`'s default, native KataGo's
  `analysis_example.cfg`), and four at once at batch 16. The report includes the
  game and each mode's recommended moves.

`scripts/native-search-bench.mjs` runs native KataGo's analysis engine on the same work from that report: the same
position and visits, and every turn of the same game at the same visits. It also reports how often each browser review
mode recommends native's move. It clears KataGo's evaluation cache before
each timed query, as the browser starts each search with an empty one. `--count-review` runs only the review and reports the network evaluations it made,
from the "NN rows" KataGo writes to its log as it exits (`docs/spikes/results/2026-11-b18-eval-count.md`).

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
WebGPU network on the golden positions in one batch (each variant: Winograd fused and unfused, implicit GEMM, fp32 with scale8, GEMM v2 fused and unfused, GEMM v3 fused, and fp16 fused and unfused with each GEMM where the device has `shader-f16`; `--variants <text>` or `?variants=` runs only those whose JSON contains the text), compares every output with the reference forward pass, and
compares the post-processed results with native KataGo's (symmetry 0) at the goldens test's tolerances. It also
evaluates the positions in three calls made at once (`maxInFlight` 2, so the third waits its turn) and requires
exactly the batched call's outputs (`concurrentCallsMatchBatched`; a mismatch fails the variant). `--bench`
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

**The GEMM benchmark** (the page's "GEMM benchmark" button) times GEMM v2 and seven GEMM v3 shapes on their own, at
b18's three GEMM shapes (the Winograd GEMM of a 3×3 192→192 convolution, and the 1×1 convolutions 384→192 and
192→384) and batches 16, 32 and 64, with GpuNetwork's parameters and dispatch. After a 2-second warm-up, each timing is
one compute pass of 30 identical dispatches (median of 7), so it is far longer than a coarse timestamp step. Every
kernel's output must equal GEMM v2's exactly and match a float64 product on the CPU; `perNetwork` sums each kernel's
time over b18's 62 + 18 + 18 such GEMMs.

**The transform benchmark** (the page's "Transform benchmark" button) times the Winograd input and output transforms
(v1 and v2) on their own at b18's 192 channels and batches 16, 32 and 64, the same way: the input transform fused
(batch norm and activation), the output transform adding the residual. Each version's output must equal v1's. It also
times a plain copy of the input transform's minimum bytes, as the device's practical ceiling, and reports each
transform's bytes per second against it. `?transformBatches=16,64` chooses the batches. `?transformLadder=1` adds v3
and the variants measured for it, and measures every version by stage (`WinoTransformStage`: without the transform
arithmetic, global reads only, global writes only). Each timing is the median of 3, against a stream kernel that reads
and writes the same bytes at the same buffer sizes. A final section times the whole kernels again, interleaved over 7
rounds with their stream kernel (`docs/spikes/results/2026-11-b18-transform-ladder.md`).

**Call trace:** `GpuNetwork.startTrace()` / `stopTrace()` record every network call's wall-clock times, the network
code's own JavaScript parts and its compute pass's GPU timestamps (carried back in the call's own readback, so tracing
adds no wait). The search benchmark reports each review mode's `trace`: GPU busy time, calls, inputs per call, and GPU
time by call size. With `reviewGame`'s `trace` option (measurement only), each search request's queued, sent and
answered times are recorded too, and `trace.offGpu` splits the GPU's idle time by cause: by what was happening at the
time, and along the path of the call that ends each gap (`docs/spikes/results/2026-11-b18-off-gpu.md`). `?traceRaw=1`
adds every call and request to the report.

**Search CPU benchmark:** `scripts/search-cpu-bench.mjs --model <b18 .bin.gz>` times the search's own JavaScript in
Node, with no GPU: the review's turns, one at a time at batch 16, with a stand-in network whose outputs are a fixed
function of its inputs, so every build that gives the same results grows the same trees. `--dist` points it at another
build to compare, and `--profile` writes a CPU profile of the timed searches.

**Kernel timing** in the GEMM, transform and dispatch benchmarks goes through `test/gpu/timing.js`: a warm-up of the
same work for at least 300 ms, then the timed passes, all submitted at once with no waiting between them, so the GPU
doesn't go idle and lower its clocks mid-measurement. The figure is the timed passes' whole GPU span divided by their
number. On the GEMM benchmark, `?gemmBatches=16,64` runs only those batches and `?gemmReverse=1` measures the v3
shapes in the opposite order. `?gemmShapes=small` measures narrower v3 tiles instead (`GemmV3Config.wx`, threads
along N), for small batches. `?gemmLadder=1` measures v3 shapes whole and by stage (the main loop from workgroup memory and
from registers, and the loads alone; `GemmV3Config.stage`), against the device's ceilings measured in the same session
(`test/gpu/ceilings.js`); `?gemmLadderShapes=` and `?gemmLadderVariants=vload,blocks,prefetch,double` choose the shapes
and the candidate fixes (`docs/spikes/results/2026-11-b18-gemm-ladder.md`). Where the device has subgroup matrices,
`?gemmMmaLadder=1` measures matrix GEMM shapes whole, at 8 × K, their multiply-accumulates alone at 8 × K, and without
their loop (`GemmMmaConfig.stage`, `kRepeat`), against the matrix peak and the read bandwidth from L2-sized and
DRAM-sized working sets measured in the same session; `?gemmMmaLadder=whole` measures them whole only, and
`?gemmMmaShapes=<JSON list of [fm, fn, ta, options]>` chooses them (`docs/spikes/results/2026-11-b18-matrix-ladder.md`).

**The budget benchmark** (the page's "Budget benchmark" button, `test/gpu/budget-bench.js`) measures what each kind
of work in the network costs on this device, against what the device can do. It measures the ceilings first: copy
bandwidth, vector FMA throughput in f32 and f16, matrix throughput where the device has subgroup matrices, and a
near-empty dispatch. Then, at each batch, it times the network's pass whole, without each kind of dispatch, and with
only that kind, alongside each kind's least work (`GpuNetwork.opWork`). It uses `GpuNetwork.recordPass`, which encodes
the plan (optionally leaving kinds out) into a pass for timing. `?budgetVariant=<options JSON>` and
`?budgetBatches=1,16` change the network and the batches. Results and the plan built on them:
`docs/spikes/results/2026-11-b18-budget.md`.

**The dispatch benchmark** (the page's "Dispatch benchmark" button) measures what a dispatch costs beyond its own
work: a near-empty kernel dispatched 500 times in one pass, each waiting for the one before (one buffer) or not (a buffer
each); GEMM v3 at b18's Winograd shape, dependent and independent, at batches 16, 32 and 64; and the b18 plan's single
pass at those batches, on a plan built for each batch and on one built for 64, each timed after 1 s of the
same pass (`?networkWarmUpMs=` changes it), because building a plan leaves the GPU idle and its clocks low.

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
