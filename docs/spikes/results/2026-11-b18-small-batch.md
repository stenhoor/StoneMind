# Spike 0001: GEMM tiles for small batches (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Code: branch
`spike-0001/small-batch`, on top of `spike-0001/review-time` (#81). All values below are quoted from the reports in this
directory.

## The question

The review trace (#81) showed a network call costs about 7.5–8 ms of GPU time with one input. The owner chose to try
cutting that per-call cost with GEMM shapes for small batches. Do narrower GEMM tiles make small calls cheaper, and how
much review time is there to win?

## What changed

- **`GemmV3Config.wx`** (new, optional): threads along N, 4, 8 or 16 (default 16, unchanged). The tile is
  `wy·rm × wx·rn`, so fewer threads along N give a narrower tile. No default changed.
- **GEMM benchmark:** `?gemmShapes=small` measures 8 narrower v3 shapes instead of the usual ones.

## Runs

Each run used a fresh separate Edge, one after another:

1. **GEMM benchmark, usual shapes**, at batches 1, 4, 8 and 16:
   [`…-gemm-edge-nvidia.json`](2026-11-b18-small-batch-gemm-edge-nvidia.json) (2026-11-26T15:42:43Z)
2. **GEMM benchmark, `?gemmShapes=small`**, at the same batches:
   [`…-gemm-small-edge-nvidia.json`](2026-11-b18-small-batch-gemm-small-edge-nvidia.json) (15:44:15Z)
3. **Search benchmark with traces**, `gemmV3Config {rm:4, rn:4, tk:16}`:
   [`…-search-rn4-edge-nvidia.json`](2026-11-b18-small-batch-search-rn4-edge-nvidia.json) (15:45:15Z)
4. **Search benchmark with traces**, `gemmV3Config {rm:4, rn:4, tk:16, wx:8}`:
   [`…-search-rn4wx8-edge-nvidia.json`](2026-11-b18-small-batch-search-rn4wx8-edge-nvidia.json) (15:48:14Z)
5. **Bench page** (every variant, per-dispatch profile at batches 1, 16 and 64):
   [`…-bench-edge-nvidia.json`](2026-11-b18-small-batch-bench-edge-nvidia.json) (15:51:51Z)

The page-default network is `{"winograd":true,"fuse":true,"precision":"f16"}` (GEMM v3, shape `{rm:4, rn:8, tk:16}`).
Its traced review is run 2 of #81
([`2026-11-b18-review-time-search-edge-nvidia-2.json`](2026-11-b18-review-time-search-edge-nvidia-2.json)).

## Results

**Correctness:**
- Every narrow shape gave output identical to GEMM v2 at every batch (`maxAbsDiffFromV2` 0).
- At batch 1 each one matched the CPU reference (`matchesReference` true).
- Both search runs report `ok: true`.

**Where a one-input pass goes** (run 5, page default):
- The GPU pass takes 7.602 ms.
- Each dispatch timed in its own pass sums to 9.372 ms:
  - Winograd GEMMs: 4.588 ms (78 dispatches)
  - 1×1 convolutions: 2.359 ms (41)
  - Winograd input transforms: 1.245 ms (78)
  - matvec: 0.524 ms (11)
  - Winograd output transforms: 0.459 ms (78)
  - batch norm: 0.197 ms (45)
- The timer's resolution on this browser is 65536 ns (`timerTickNs`), so these per-dispatch values are coarse.
- The dispatch benchmark measured about 0.0014 ms of fixed cost per dispatch (#78), about 0.5 ms for the pass's 338
  dispatches.

**The GEMM benchmark's sum for one network** (`perNetwork`, ms; run 2; the shape is rm.rn.tk.wy.wx):

| Batch | 4.8.16.16.16 (default) | 4.4.16.16.16 | 4.4.16.16.8 | 4.4.16.8.8 | 4.4.8.8.8 | 4.4.16.8.16 | 4.4.16.16.4 | 4.4.32.8.8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.977 | 3.054 | 2.642 | 2.949 | 2.925 | 3.132 | 3.636 | 3.513 |
| 4 | 5.15 | 4.95 | 5.311 | 6.943 | 6.636 | 5.889 | 7.439 | 7.131 |
| 8 | 8.628 | 8.86 | 9.031 | 10.822 | 10.063 | 10.093 | 11.413 | 13.084 |
| 16 | 16.675 | 16.047 | 16.515 | 20.149 | 19.419 | 18.013 | 20.164 | 21.56 |

**In the review** (traced search benchmark; GPU busy time in s; mean GPU pass in ms):

| Network | Review mode | `totalMs` | GPU busy | Pass, 1 input | Pass, 8 | Pass, 16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| default (#81 run 2) | batch 16, 1 at once | 42091 | 23.55 | 7.541 | 14.066 | 24.162 |
| default (#81 run 2) | batch 16, 2 at once | 33574.2 | 30.32 | 7.971 | 14.718 | 24.997 |
| 4.4.16.16.16 | batch 16, 1 at once | 42496.7 | 24.27 | 5.891 | 14.541 | 25.628 |
| 4.4.16.16.16 | batch 16, 2 at once | 35282.2 | 31.41 | 6.231 | 15.298 | 26.512 |
| 4.4.16.16.8 | batch 16, 1 at once | 44065.7 | 25.67 | 5.869 | 15.155 | 27.207 |
| 4.4.16.16.8 | batch 16, 2 at once | 37114.6 | 33.34 | 6.246 | 16.314 | 28.273 |

GPU busy time, all four review modes (s):

| Network | batch 64, 1 at once | batch 16, 1 at once | batch 16, 2 at once | batch 16, 4 at once |
| --- | ---: | ---: | ---: | ---: |
| default (#81 run 2) | 31.33 | 23.55 | 30.32 | 31.11 |
| 4.4.16.16.16 | 33.32 | 24.27 | 31.41 | 33.11 |
| 4.4.16.16.8 | 36.40 | 25.67 | 33.34 | 35.57 |

Review time was longer than #81 run 2 in 7 of the 8 comparisons. The exception was batch 64 one at a time with
4.4.16.16.16: 46949.5 against 46985.6 ms, with 1.99 s more GPU busy time. Review time varies between runs of the same
network: #81's runs 1 and 2 took 44259 and 46985.6 ms in that mode.

**GPU time by call size** (page default, batch 16 two at once, #81 run 2; calls × their mean pass):

| Inputs per call | 1 | 2–4 | 5–8 | 9–15 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calls | 267 | 51 | 55 | 112 | 989 |
| GPU s | 2.13 | 0.47 | 0.72 | 2.28 | 24.72 |

- GPU busy time was 30.32 s. Calls of 1–8 inputs took 3.32 s of it (11%), and 16-input calls 24.72 s (82%).
- In every mode of every traced run, the sum of the calls' pass times equals the GPU busy time (the union). The passes
  never overlapped, even with 2 or 4 calls in flight.

## What this shows

- **Narrow tiles help only the smallest calls:**
  - They cut the one-input pass from about 7.5–8.0 ms to 5.9–6.2 ms.
  - They made 16-input passes 1.5–3.3 ms slower, which outweighed the saving.
  - With both shapes, GPU busy time went up in every review mode.
- **What a per-batch tile could save:** using a narrow tile only for one-input calls, and the default otherwise, would
  have saved about 1.7 ms × 267 calls ≈ 0.46 s of GPU time. That is about 1.4% of the default 2-at-once review.
  This is an estimate from these figures, not measured.
- **The ceiling for small calls:** even if every call of 1–8 inputs took no GPU time, the 2-at-once review's GPU time
  would drop by 3.32 s (11%). The 16-input calls (82%) are where the time goes.
- **What I called a fixed cost per call in #81 isn't something small-batch kernels can remove.** It came from a
  straight-line fit of GPU time against inputs per call. The data:
  - The intercept (about 6 ms) appears in 16-, 32- and 64-input passes too: they take 24.2, 45.1 and 86.2 ms,
    about 1.1–1.3 ms per input above it.
  - At one input the GEMMs alone take 4.5–5.0 ms (`perNetwork`, default shape, runs 1 and 2).
  - So this cost is how efficiently the GEMMs run at these sizes, in every call, and not a cost only small calls pay.

## Not changed

No defaults. `wx` and `?gemmShapes=small` are measurement options.
