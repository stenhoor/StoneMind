# Spike 0001: Phase 2, Winograd transforms (b18) on Edge (NVIDIA): closed

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Code: branch
`spike-0001/transform-ladder`, on top of `spike-0001/gemm-ladder` (#85). Every figure below is quoted from, or computed
from, the reports in this directory.

## Outcome

**Closed by its stop rule, with an improvement kept.** Phase 2 of the plan (`2026-11-b18-budget.md`) aimed to get both
transforms to ≥ 80% of copy bandwidth at batch 16. It was to close if either stayed below 70% after fixing each limit
the ladder showed.

- **The input transform stayed below 70%:** 62% of its matched stream kernel at batch 16, and 61% of the copy floor in
  the network.
- **The output transform reached 78%**, and 75% in the network: above the stop threshold, below the goal.
- **Both new transforms ('v3') are now the default.** Their outputs are identical to v1's in the transform benchmark,
  and all 23 variants of the check page pass the goldens.

**Change from prior, and against native** (default against v3, measured one after the other in fresh browsers):

| Measure | Default (v2 in, v1 out) | v3 | Change | Native | v3 against native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Review, batch 16 two at once, `totalMs` | 32008.1 | 30832.9 | −3.7% | 16412.1–16596.1 | 1.86–1.88× (was 1.93–1.95×) |
| Batch-16 pass (ms, budget benchmark) | 25.1845 | 23.4525 | −6.9% | 15.54 at native's rate | 1.51× |
| Batch-16 evaluations/s | 635.3 | 682.2 | +7.4% | 1029.36 | 1.51× slower (was 1.62×) |

The prior reports (#81, #84) had 32279.6–33574.2 ms for the review and 26.2799 ms for the batch-16 pass. Today's
default runs (32008.1 ms, 25.1845 ms) fall at or below those, which is this setup's run-to-run variation.

## How it was measured

- **Transform benchmark ladder** (`?transformLadder=1&transformBatches=16,64` on the transform benchmark,
  `test/gpu/transform-bench.js`). It measures each version of the input and output transforms at b18's 192 channels,
  fused as in the network (input: batch norm and mish_scale8; output: adds the residual).
  - **Whole kernel**, and by stage (`WinoTransformStage`, measurement only):
    - **move:** everything except the transform arithmetic. The output transform's 16 values each sum the inputs
      congruent to them, so none of the 36 reads is dropped.
    - **reads:** only the global reads, then one write per thread.
    - **writes:** only the global writes, plus the output transform's residual read.
  - **Matched stream kernel:** every timing is compared with a kernel that reads and writes the same bytes,
    contiguously, at the same buffer sizes, timed right after it. At batch 16 these buffers are 2–8 MB, near this GPU's
    L2, so the 64 and 256 MiB copy figures don't apply.
  - **Repeats:** each figure is the median of 3 timings.
  - **Interleaved section:** the whole kernels and their stream kernel are timed again over 7 rounds, in an order that
    rotates each round, so clock drift affects all alike. The median, least and greatest are reported. Decisions were
    taken from this section; the stage timings varied by up to about 30% between runs.
  - **Correctness:** every whole kernel's output is compared with v1's (`diff 0` in all runs).
- **In the network:**
  - The budget benchmark (`?budgetBatches=16,64&budgetVariant=…`), default against v3.
  - The search benchmark (`?searchVariant=…`), default against v3.
  - The check page (all variants, goldens).
- **Reports:**
  - Final ladder: [`…-edge-nvidia.json`](2026-11-b18-transform-ladder-edge-nvidia.json) (2026-11-26T17:54:37Z), with
    [clock log](2026-11-b18-transform-ladder-edge-nvidia-clocks.csv). Median SM clock 1695 MHz in samples ≥ 90% busy;
    power cap (`0x4`) in 214 of 323.
  - Search: [default](2026-11-b18-transform-ladder-search-default.json) (17:43:36Z),
    [v3](2026-11-b18-transform-ladder-search-v3.json) (17:46:25Z), each with its clock log (median 1485 MHz, power cap
    in 216 and 223 busy samples).
  - Budget: [default](2026-11-b18-transform-ladder-budget-default.json) (17:49:18Z),
    [v3](2026-11-b18-transform-ladder-budget-v3.json) (17:50:49Z), each with its clock log (median 1425 and 1455 MHz).
  - Goldens: [`…-check.json`](2026-11-b18-transform-ladder-check.json) (17:53:18Z): `ok: true`, 23 variants.
- **Development runs:** sixteen earlier ladder runs found the limits and tried the fixes one at a time. Their code
  changed between runs, so only the final run, which has every version, is kept.

## Results (final ladder, interleaved medians)

| Transform, batch | Before (ms) | v3 (ms) | Change | v3, % of stream |
| --- | ---: | ---: | ---: | ---: |
| Input, 16 (v2 → v3) | 0.0471 | 0.0368 | −22% | 62% |
| Output, 16 (v1 → v3) | 0.0487 | 0.0362 | −26% | 78% |
| Input, 64 | 0.1729 | 0.1217 | −30% | 73% |
| Output, 64 | 0.1838 | 0.1333 | −27% | 84% |

**In the network** (budget benchmark, each kind alone, batch 16; floor = bytes ÷ 349.7 GB/s):

| Kind | Default (ms) | v3 (ms) | Floor (ms) | Efficiency |
| --- | ---: | ---: | ---: | --- |
| Input transform (78 dispatches) | 3.5858 | 2.7525 | 1.672 | 47% → 61% |
| Output transform (78) | 3.127 | 2.4997 | 1.868 | 60% → 75% |

At batch 64 the network's pass improved only 2.0% (88.1459 → 86.3577 ms). There, the input transform alone measured
slower with v3 (12.2927 → 13.0042 ms), although v3 was faster at batch 64 in the ladder (−30%) and in the batch-64
review mode (GPU busy 32010.1 → 29694.9 ms). These runs don't explain that.

## What the ladder showed, and what each fix did (batch 16)

**Input transform** (v2 before):
1. **Its reads were the limit:** whole ≈ reads + writes. Reads took 0.0222 ms at 11% of stream; writes were at 84%.
   The transform arithmetic cost nothing measurable (move ≈ whole).
2. **Unrolling the load, branch-free** (`unroll`): reads 0.0218 → 0.0193 ms. Small: the loop's latency was only part of
   it.
3. **The fusion was about 40% of the read time.** With fusion off, reads took 0.0131 ms. Every padded slot read the
   mask, scale, shift and pre-bias from global memory.
4. **The linear load** (`linear`, now in v3):
   - It reads only the board's points, as one contiguous span per workgroup, four at a time.
   - It stages the per-channel values and the mask (one bit per point) in workgroup memory.
   - It issues its board reads before that staging.

   Reads fell to 0.0153 ms and the whole kernel to 0.0356–0.0368 ms. The staging first used a serial 32-read loop;
   spreading it over all threads cut reads from 0.0181 to 0.0122 ms (consecutive development runs).
5. **Tried, and no better:** building the mask words without atomics (`maskBits: 'words'`), 4 channels per workgroup
   (`cpw: 4`), and 4 channels × 2 boards (`bpw: 2`). All measured 0.0368 ms at batch 16 in the final interleaved run,
   and v3 was best at batch 64.

**Output transform** (v1 before):
1. **Its writes were the limit:** a 4 × 4 block per thread, with the residual read the same way, at 30% of stream.
   Reads were at 80%.
2. **The linear write** (`WINO_OUT_V2` with `linear`):
   - It writes each workgroup's span of boards four at a time.
   - It issues the residual reads before any write.

   Writes reached 73–99% of stream.
3. **Two boards per workgroup** (`cpw: 4, bpw: 2`, now in v3). M holds every board's tiles contiguously, so each
   channel's reads are runs of 50 tiles instead of 25. Whole kernel 0.0378 → 0.0362 ms at batch 16, and
   0.1432 → 0.1333 ms at batch 64.

## What remains

- **Input transform, fusion:** reads with fusion take 0.0153 ms, with fusion off 0.0084 ms, and with an identity
  activation 0.0134 ms. So the staging and per-point fusion cost about 0.005 ms per dispatch, about 0.4 ms per batch-16
  pass. The mask's atomics are not the cause: plain words measured the same. What is, isn't isolated.
- **Input transform, phases:** within a workgroup the reads and writes run one after the other (0.0153 + 0.019 ms
  against 0.0356 whole), and at batch 16 there are only about 2 waves of workgroups to overlap them.
- **Output transform, reads of M:** 75–80% of stream.

## What this means for the budget

- **Saved:** the transforms went from 6.71 to 5.25 ms of the batch-16 pass (each alone, −1.46 ms), and the pass from
  25.18 to 23.45 ms. The plan predicted about −2 ms if both reached 80%.
- **Still needed for parity:** 15.54 ms, −34% from here.
- **Where the time is:** the two GEMM kinds are most of the pass, at 24–45% of the FMA floor in this run (each kind
  alone or by share). Phase 1 found no vector-unit fix that moved them past 42% (#85).
- **Next:** Phase 3, matrix hardware, is your decision.

## Kept in the code

- **Defaults:** `GpuNetworkOptions.winoTransforms`, `winoIn` and `winoOut` gain `'v3'`, now the default for both
  (`WINO_IN_V3`, `WINO_OUT_V3`).
- **Measurement options**, all off by default:
  - `WINO_IN_V2`'s `WinoInV2Options`: `stage`, `unroll`, `linear`, `maskBits`, `cpw`, `bpw`.
  - `WINO_OUT_V2`'s `WinoOutV2Options`: `stage`, `linear`, `cpw`, `bpw`.
  - `WINO_IN` and `WINO_OUT`'s `stage`.
- **Benchmark:** the transform benchmark's `?transformLadder` and `?transformBatches`.
