# Spike 0001: Phase 3, matrix hardware (b18) on Edge (NVIDIA): efficiency closed, selection adopted

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Browser: the separate test Edge with
`--enable-unsafe-webgpu`, as in #80 (the owner's one exception to the no-flags rule, for matrix hardware only). Code:
branch `spike-0001/matrix-auto`, on top of `spike-0001/transform-ladder` (#86). Every figure below is quoted from, or
computed from, the reports in this directory.

## Outcome

Phase 3 had two parts, as agreed with the owner:

1. **Use the matrix GEMM automatically, only where it measured faster.** Done: `GpuNetwork`'s new option `matrix`,
   default `'auto'`. Where the browser offers subgroup matrices (today only behind the switch) and storage is f16:
   - the Winograd GEMMs and 1×1 convolutions run on the matrix GEMM;
   - each call uses the shape measured fastest for its batch size.

   It was faster than GEMM v3 at every batch measured, 1 to 64, for all three of b18's GEMM shapes. Without the
   feature nothing changes: the check without the switch has no variant on matrix hardware.
2. **Raise the matrix GEMM's efficiency.** The goal was ≥ 50% of matrix peak at batch 16, with a stop if it stayed
   below 35% after fixing what the ladder showed. **Closed by its stop rule:** the best whole kernel at the batch-16
   Winograd shape reached 28% of the measured peak. The fixes along the way made the matrix GEMM 29–52% faster than
   #80's kernel at batch 16. They are what the selection uses.

**Change from prior, and against native.** "Off" is Phase 2's default (#86); "auto" is this phase's default. Each
pair was run one after the other, in fresh switched browsers.

| Measure | Matrix off | Matrix auto | Change | Native | Auto against native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Review, batch 16 two at once, `totalMs` | 30915.5 | 26015.4 | −15.8% | 16412.1–16596.1 | 1.57–1.59× (was 1.86–1.88×) |
| Review, batch 16 four at once, `totalMs` | 29550.1 | 23864.5 | −19.2% | same | 1.44–1.45× (was 1.78–1.80×) |
| Batch-16 pass (ms, budget benchmark, 2 runs each) | 23.3482, 23.2558 | 18.2076, 17.5385 | −22.0% to −24.9% | 15.54 at native's rate | 1.13–1.17× |
| Batch-16 evaluations/s | 685.3, 688.0 | 878.8, 912.3 | +28% to +33% | 1029.36 | 1.13–1.17× slower (was 1.50×) |

- **Phase 2's result** (#86): review 30832.9 ms and batch-16 pass 23.4525 ms. Today's "off" runs (30915.5 ms;
  23.3482 and 23.2558 ms) are within 0.3% and 0.9% of them.
- **Clocks:** the "off" runs ran at lower median SM clocks than the "auto" runs:
  - review: 1470 against 1605 MHz;
  - budget: 1380 and 1455 against 1530 and 1545 MHz.

  The GPU was at its power cap (`0x4`) in nearly every busy sample of every run. I haven't separated the clocks' share
  of the change.
- **Users:** users get this by default only once browsers ship the feature without a switch. Until then they can
  turn it on themselves with `--enable-unsafe-webgpu`, at their own risk (`packages/engine-webgpu/README.md`, "Browser
  flags for more speed", added in #88).

## How it was measured

- **Matrix ladder:** `?gemmMmaLadder=1` on the GEMM benchmark (`test/gpu/gemm-bench.js`), at b18's three GEMMs, batches
  16 and 64. Each matrix shape (`?gemmMmaShapes=`) is measured four ways:
  - **whole:** the kernel as the network runs it;
  - **loop with loads:** the whole kernel with its loop run over 8 × K (`kRepeat: 8`, addresses wrapping), ÷ 8;
  - **multiply-accumulates alone:** the loop with each step's matrices loaded once before it (`stage: 'mmaOnly'`),
    over 8 × K, ÷ 8;
  - **fixed:** no loop (`stage: 'fixed'`): the store and epilogue only.

  Percentages are of the matrix floor: flops ÷ the matrix peak measured in the same session. Every whole kernel's
  output is compared with GEMM v2's at the reference check's tolerance (`worstV2Ratio` 0 in every run).
- **The difference of 1 × and 8 × K wasn't used.** The first run took the loop's time as (8 × K − 1 × K) ÷ 7, as Phase 1
  did. In `peak-edge-nvidia` it gave multiply-accumulate rates of 110–153% of peak, which is impossible. Workgroups overlap one another's
  fixed part, so the subtraction underestimates the loop for this kernel. The ÷ 8 figures include 1/8 of the fixed
  part, so they slightly overstate the loop's time.
- **Ceilings, measured in the same sessions:**
  - **Matrix peak:** #80's kernel (1 subgroup per workgroup, 4 accumulators) gave 33.03–33.06 TFLOPS in the first
    three runs and 43.14–43.58 in the four after them; why it changed is not known. With 8 accumulators, or 4 or 8
    subgroups per workgroup, the same work reached 43.14–44.07 TFLOPS in every run. `measureMatrixPeak` now takes the
    best of five variants.
  - **L2 read bandwidth:** working sets of 1 and 2 MiB, read in 4 KB blocks spread over the whole set: 1340.4 and
    1287.4 GB/s. A 64 MiB set gave 450.6 GB/s. A first version of this kernel re-read the same addresses (7.4 TB/s, L1
    hits). It was fixed before any use.
  - **Copy:** 348–351 GB/s (budget benchmark).
- **Batch sweep:** `?gemmMmaLadder=whole` at batches 1, 2, 4, 8, 16, 32 and 64.
- **In the network:**
  - **The check page:** goldens, and exact equality across calls of 1, 4–5 and 14 positions. Run with and without
    the switch.
  - **The search benchmark:** `?searchVariant=`, off then auto.
  - **The budget benchmark:** `?budgetVariant=`, off, auto, auto, off.
- **Reports** (`2026-11-b18-matrix-ladder-…`, each with its clock log where one was taken):

  | Report | Time (UTC) | What |
  | --- | --- | --- |
  | `peak-edge-nvidia` | 18:10:04 | #80's shapes, with the corrected peak |
  | `tiles-edge-nvidia` | 18:13:36 | edge tiles, tile sizes, subgroups per workgroup |
  | `memory-edge-nvidia` | 18:17:24 | no copying path, fragment epilogue |
  | `prefetch-edge-nvidia` | 18:20:32 | `uk: 4`, and the fixed part |
  | `l2-edge-nvidia` | 18:25:03 | L2 read bandwidth |
  | `share-edge-nvidia` | 18:30:00 | sharing through workgroup memory, with and without register prefetch |
  | `batches-edge-nvidia` | 18:32:49 | batches 1–64, whole kernels |
  | `check-first` | 18:37:20 | the first check of the selection: `ok: false` (see below) |
  | `search-off`, `search-auto` | 18:40:00, 18:42:44 | review and live search |
  | `budget-off`, `budget-auto`, `budget-auto2`, `budget-off2` | 18:44:57, 18:46:26, 18:48:57, 18:50:35 | budget benchmark |
  | `check`, `check-noswitch` | 18:57:20, 18:58:24 | the final check: `ok: true`, with and without the switch |

- **Not kept:** three development runs.
  - The first ladder: its peak was #80's under-measure, and `peak-edge-nvidia` repeats its kernels.
  - A run whose L2 figure was the invalid first version.
  - A run of a first sharing kernel, with a branch per subgroup, since replaced.

## What the ladder showed, and what each fix did

Batch 16 unless noted. The floor is the matrix floor at 43.6–44.0 TFLOPS: 0.024 ms for the Winograd GEMM, 0.019 ms for
each 1×1.

**Start: #80's kernel** (2 × 1 matrices per subgroup, one subgroup per workgroup, A transposed; `peak-edge-nvidia`):

| Shape | Whole | Loop with loads | Multiply-accumulates | Fixed |
| --- | ---: | ---: | ---: | ---: |
| Winograd | 0.1241 (20%) | 0.1016 (24%) | 0.0256 (95%) | 0.0246 |
| 1×1 384→192 | 0.1754 (11%) | 0.0918 (21%) | 0.0272 (72%) | 0.1056 |
| 1×1 192→384 | 0.1470 (13%) | 0.0898 (22%) | 0.0238 (82%) | 0.0807 |

The multiply-accumulates weren't the limit. The loads were, and for the 1×1 shapes the fixed part was too.

1. **Edge tiles** (`overreadN`, `tiles-edge-nvidia`).
   - **The problem:** the 1×1 GEMMs have N = 361 per board. Each board's last tile is partial, so it took the copying
     path, element by element through workgroup memory.
   - **The fix:** those tiles also take the direct path. The columns past N are computed from whatever follows in B
     and never written. The caller pads B by 256 elements.
   - **Result:**
     - 1×1 384→192: fixed part 0.1061 → 0.0107 ms, whole kernel 0.1763 → 0.0981 ms (−44%);
     - 1×1 192→384: whole 0.1537 → 0.1038 ms.
2. **Tile sizes** (`tiles-edge-nvidia`). Larger tiles per subgroup reuse each load more.
   - Loop with loads at the Winograd shape: 2 × 1 24%, 2 × 2 31%, 4 × 2 38%.
   - But the fixed part grew: 0.0243 → 0.0306 → 0.0522 ms, and for 4 × 4 0.0799 ms.
   - More subgroups per workgroup made the loop slower: 2 × 1 with 2 × 2 subgroups, 19% against 24%.
3. **Workgroup memory** (`noCopy`, `epilogue: 'fragment'`, `memory-edge-nvidia`).
   - **The problem:** every workgroup declared the copying path's As and Bs, and a Cs for its whole tile, even on the
     direct path. That's 14 KB at 4 × 2 and 24 KB at 4 × 4.
   - **The fix:**
     - `noCopy` drops the copying path and As and Bs, where the caller guarantees whole tiles in M and K;
     - `'fragment'` stores one 16 × 16 accumulator at a time through 1 KB.
   - **Result for 4 × 4:**
     - Winograd: whole 0.1512 (no copying path alone) → 0.0921 ms, multiply-accumulates 73% → 81%, loop 33% → 44%;
     - 1×1 384→192: 0.1041 → 0.0642 ms.
   - **A measurement fault:** the fixed stage read 0 ms for `noCopy` kernels in this run. Without A and B in the shader,
     `layout: 'auto'` dropped their bindings and the bind group was invalid. The fixed stage now keeps a use of both.
4. **Prefetch depth** (`uk: 4`, `prefetch-edge-nvidia`): no consistent change in the whole kernel (4 × 4: 0.0887 →
   0.0868 ms; 2 × 4 at 1×1 384→192: 0.0724 → 0.0876 ms). Not adopted.
   - The fixed part, now measured, is 0.0215–0.0236 ms at the Winograd shape. Its 5.53 MB of output takes 0.0158 ms at
     349.7 GB/s, so the epilogue is at about 70% of write bandwidth.
5. **Loads against L2** (`l2-edge-nvidia` for the ceiling, `memory-edge-nvidia` for the loops, Winograd shape). Each
   tile re-reads A once per column of tiles and B once per row of tiles:

   | Tile | Traffic (MB) | Loop with loads (ms) | Rate (GB/s) | Of L2 ceiling |
   | --- | ---: | ---: | ---: | ---: |
   | 2 × 1 | 99.5 | 0.0944 | 1054 | 79–82% |
   | 2 × 2 | 67.7 | 0.0703 | 963 | 72–75% |
   | 4 × 2 | 51.1 | 0.0577 | 886 | 66–69% |
   | 4 × 4 | 35.2 | 0.0547 | 644 | 48–50% |

   - The small tiles are bound by L2 bandwidth.
   - 4 × 4 moves the least but at half the L2 rate. It holds 16 accumulators per subgroup; that registers limit its
     warps is my assumption, not measured.
6. **Sharing loads through workgroup memory** (`share`, `share-edge-nvidia`). Several subgroups per workgroup, each
   step's matrices loaded from L2 once per workgroup and staged in workgroup memory, with and without the next step
   loaded into registers first (`share: 'prefetch'`). Against the direct 4 × 4 (0.0884 ms Winograd, 0.0618 and 0.0761
   ms for the 1×1s):

   | Shape | Winograd | 1×1 384→192 | 1×1 192→384 |
   | --- | ---: | ---: | ---: |
   | 4 × 2, 3 × 2 subgroups, shared | 0.1022 | 0.0585 | 0.0715 |
   | the same, prefetch | 0.1000 | 0.0560 | 0.0725 |
   | 2 × 2, 2 × 2 subgroups, shared | 0.1016 | 0.0680 | 0.0772 |
   | the same, prefetch | 0.1088 | 0.0696 | 0.0820 |

   Sharing was slower at the Winograd shape and faster for the 1×1s. Prefetch made no consistent difference. This was
   the last fix the ladder pointed to for the loads. The best at the batch-16 Winograd shape stayed at 28% (direct
   4 × 4), below the 35% stop threshold.

**About the goal.** At the batch-16 Winograd shape, the compulsory traffic (A, B and C, 13.7 MB) takes 0.039 ms at copy
bandwidth. That's more than its matrix floor of 0.024 ms. So 50% of matrix peak (0.048 ms) was within 25% of the
memory floor. I set the goal without that figure.

## The selection

**Batch sweep** (`batches-edge-nvidia`, whole kernels, ms). Only shapes with A as `GpuNetwork` stores it for GEMM v3
are listed, so one copy of the weights serves both kernels. Against the same shapes with A transposed, they ranged from
26.5% faster to 24.9% slower, with a median of about 4% slower.

| Batch | Winograd: v3 | Winograd: best matrix | 1×1 384→192: v3 | best matrix | 1×1 192→384: v3 | best matrix |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0505 | 0.0131 (2 × 1) | 0.0433 | 0.0115 (2 × 1) | 0.0238 | 0.0093 (2 × 1) |
| 2 | 0.0512 | 0.0205 (2 × 2) | 0.0433 | 0.0145 (2 × 1) | 0.0252 | 0.0132 (2 × 2) |
| 4 | 0.0559 | 0.0374 (2 × 2) | 0.0450 | 0.0224 (2 × 2) | 0.0376 | 0.0201 (2 × 2) |
| 8 | 0.0981 | 0.0601 (2 × 2) | 0.0677 | 0.0349 (2 × 2) | 0.0642 | 0.0341 (shared) |
| 16 | 0.1757 | 0.1073 (shared) | 0.1166 | 0.0625 (shared) | 0.1222 | 0.0702 (shared) |
| 32 | 0.3085 | 0.1828 (shared) | 0.2292 | 0.1191 (shared) | 0.2367 | 0.1362 (shared) |
| 64 | 0.5655 | 0.3372 (shared) | 0.4484 | 0.2336 (shared) | 0.4629 | 0.2686 (shared) |

"Shared" is 4 × 2 with 3 × 2 subgroups, shared. The matrix GEMM was faster at every point, by 35–74%.
`MATRIX_GEMM_BY_BATCH` takes:

- 2 × 1 at batch 1;
- 2 × 2 at batches 2–8;
- shared above that.

Where that isn't the best measured (1×1 384→192 at batch 2, 1×1 192→384 at batch 8), the shape used is 12% and 15%
slower than the best. Batch sizes between the measured points follow their range; they weren't measured.

**In the network** (`GpuNetwork`, `matrix: 'auto'`):

- **When it applies:** f16 storage, `gemm` 'v2' or 'v3', and B an activation tensor. The tensors get 256 elements of
  padding when the plan uses matrix hardware.
- **Plan entries:** each fitting GEMM becomes one entry per batch range plus the GEMM v3 entry. Each entry runs only
  at its batch sizes (`when`). `recordPass`, `evaluate`, `profile`, `opWork` and `dispatchCount` use only the entries
  active at the call's batch size.
- **A GEMM runs on matrix hardware at every batch size or at none.** Where a range's shape doesn't fit, it takes the
  next smaller one that does. The first check (`check-first`) failed exact equality between calls of different sizes:
  - at batch 9 and up, 11 Winograd GEMMs whose M isn't a multiple of 192 had fallen back to GEMM v3;
  - at smaller batches they ran on matrix hardware.

  With the rule above, every check passes, including exact equality across calls of 1, 4–5 and 14 positions. So the
  three matrix shapes give identical results.
- **Coverage in b18** (`matrixUse`):
  - 77 of 78 Winograd GEMMs and 36 of 41 1×1 convolutions run on matrix hardware;
  - at batch 9 and up, 10 of the Winograd GEMMs use 2 × 2 because the shared shape doesn't fit.

## Network results

**Review** (`search-off`, `search-auto`, 250 turns × 100 visits). Native's `totalMs` is 16412.1 and 16596.1 (#80).

| Mode | Off `totalMs` | Auto `totalMs` | Change | Against native | Evaluations per GPU-busy second | Agrees with native (two runs) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| batch 64, 1 at once | 42225 | 33412.3 | −20.9% | 2.54–2.57× → 2.01–2.04× | 755.9 → 995.3 | 171/177 → 174/179 of 250 |
| batch 16, 1 at once | 40090.7 | 31456.3 | −21.5% | 2.42–2.44× → 1.90–1.92× | 655.0 → 819.0 | 180/180 → 181/187 |
| batch 16, 2 at once | 30915.5 | 26015.4 | −15.8% | 1.86–1.88× → 1.57–1.59× | 649.3 → 811.4 | 186/185 → 181/184 |
| batch 16, 4 at once | 29550.1 | 23864.5 | −19.2% | 1.78–1.80× → 1.44–1.45× | 728.5 → 925.0 | 187/189 → 186/184 |

- **The batch-16 × 1 "off" run** (40090.7 ms) was slower than this mode's other runs today: 35840.6 ms in #86's v3
  run.
- **Live `visitsPerSecond`:**

  | Batch | Off | Auto |
  | ---: | ---: | ---: |
  | 16 | 476.3 | 700.1 |
  | 32 | 655.9 | 638.8 |
  | 64 | 680.6 | 790 |

  Native's was 1128.9–1206.8.

**Budget benchmark** (each kind alone, ms):

| Kind | Batch 16, off (2 runs) | Batch 16, auto (2 runs) | Batch 64, off (2 runs) | Batch 64, auto (2 runs) |
| --- | --- | --- | --- | --- |
| Winograd GEMMs (78) | 14.9135\*, 12.5458 | 7.919, 50.0963\* | 40.8108, 40.7537 | 26.0394, 25.5807 |
| 1×1 convolutions (41) | 4.7237, 4.7217\* | 3.5098, 3.3728 | 17.7913, 17.7276 | 12.7339, 12.6738 |
| Whole pass | 23.3482, 23.2558 | 18.2076, 17.5385 | 80.8389, 89.617 | 62.0894, 61.2596 |

\* Repeats spread more than 15% (below). Without those, the batch-16 Winograd GEMMs went 12.5458 → 7.919 ms (−37%)
and the 1×1 convolutions 4.7237 → 3.5098 and 3.3728 ms. The second "off" run's batch-64 pass (89.617 ms) is 11% above
the first; in it, the input transforms alone took 12.636 ms against 8.88–8.90 in the other three runs.

At batch 64 the pass did 1030.8 and 1044.7 evaluations/s. Native's 1029.36 is at its own mean batch of 15.72, so
it isn't a like-for-like comparison.

**Unstable timings.**

- **Where:** all four budget runs had batch-16 sub-timings whose repeats spread more than 15%, at different kinds in
  each run. Examples:
  - off: 1×1 convolutions alone, 4.68–154.46 ms;
  - auto: Winograd GEMMs alone, 29.19–87.19 ms in `budget-auto2`;
  - auto: bnAct's full pass, 51.01–134.04 ms in `budget-auto`.
- **What it affects:** the "off" and "auto" runs alike, so it isn't specific to the matrix path. The affected figures
  above are marked \*. The whole-pass timings at each batch didn't spread more than 1.2% in any run.
- **The clock logs** show no drop in those windows. The cause is not known.

## What this means for the budget

- **Saved:** 5.1–5.8 ms of the batch-16 pass (23.26–23.35 → 17.54–18.21 ms). I predicted about −4 ms if the efficiency
  goal was met. The goal wasn't met, but the 1×1 convolutions, which the estimate left out, also moved: −1.2 to −1.35
  ms alone.
- **Still needed for parity:**
  - pass: 15.54 ms, −11% to −15% from here;
  - review: −36% to −37% at batch 16 two at once.

  The review's gap is larger than the pass's: the review also spends time off the GPU and waiting between calls. That
  belongs to Phase 4.
- **Where the batch-16 pass goes now** (each kind alone):
  - Winograd GEMMs: 7.9 ms;
  - 1×1 convolutions: 3.4–3.5 ms;
  - transforms: 2.7 + 2.5 ms;
  - batch norm: 1.3 ms.
- **For users today:** by default (no switch), none of this applies, and Phase 2's figures still stand: 1.86–1.88× native
  for the review. Users who start their browser with `--enable-unsafe-webgpu`, at their own risk, get the matrix
  figures above.

## Kept in the code

- **Defaults:** `GpuNetworkOptions.matrix` (`'auto'` | `'off'`, default `'auto'`), `MATRIX_GEMM_BY_BATCH` and
  `matrixGemmFits`. `GpuNetwork.matrix` and `matrixUse` report what the plan does.
- **New `GemmMmaConfig` options:**
  - used by the selection: `overreadN`, `noCopy`, `epilogue`, `sgm`, `sgn`, `share`;
  - measurement only: `stage`, `kRepeat`.

  Without them, the generated code is identical to before.
- **Measurement:**
  - `measureMatrixPeak` now measures five variants;
  - `measureCacheRead` is new (`test/gpu/ceilings.js`);
  - the GEMM benchmark has `?gemmMmaLadder` and `?gemmMmaShapes`, and pads B for `overreadN`.
- **Check page:** its f16 variants now say `matrix: 'off'`, and two `matrix: 'auto'` variants run where the device has
  the feature.
- **`winoGemm: 'mma'`** (#80) stays, unchanged.
