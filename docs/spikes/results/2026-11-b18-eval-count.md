# Spike 0001: Phase 4, part 2, the review's evaluation count (b18): goal met, no change needed

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Native: KataGo v1.18.2, CUDA,
`analysis_example.cfg` as it is. Browser figures: the review runs of part 1 (`2026-11-b18-off-gpu.md`). Every figure
below is quoted from, or computed from, the reports in this directory.

## Outcome

**Goal met.** The goal was for the browser's review to make within 2% of the network evaluations native KataGo makes
for the same work, or for the difference to be explained by KataGo's own behaviour. It was to close if native's count
couldn't be measured.

- **Native, measured:** 17806, 17678 and 17871 evaluations in three runs of the review, with mean batches of
  9.65–9.66.
- **The browser, at batch 16 two at once** (the review's default, as native's config): 18047 with the flag and 17957
  without. That's 1.0–2.1% and 0.5–1.6% above native's runs, and 1.5% and 1.0% above their mean.
- **The plan's estimate of about 7% more was wrong.** It assumed native evaluated at its batch-16 rate (1029.36/s)
  for its whole review, giving about 16890 evaluations. Native's review batches are smaller (mean 9.65), and it
  evaluated at 1083–1096/s over its review.
- **No code change is needed.** The count isn't a factor in the gap to native.

## How it was measured

- **Native:** `scripts/native-search-bench.mjs` has a new `--count-review` option. It runs only the review query
  (every turn of the same 250-move game at 100 visits, after `clear_cache`), with no warm-up and no live queries. It
  then reads the "NN rows" and "NN batches" that KataGo writes to its log as it exits. That needs no change to
  KataGo's config: the script already gives KataGo its log folder. Runs took 16249.6–16485.1 ms, with no warm-up, so
  they aren't timing runs.
- **Browser:** each review mode's network calls and inputs, from the trace (`trace.inputs`). The shared evaluation
  cache works as KataGo's does in the analysis engine, so these are evaluations that missed the cache.
- **Reports:** `2026-11-b18-eval-count-native-1.json` to `-3.json`, run one after another, with no browser running.

## The counts

| Review | Evaluations | Calls | Mean batch | Against native's mean (17785) |
| --- | ---: | ---: | ---: | ---: |
| Native, 2 analysis threads × 16 search threads | 17806, 17678, 17871 | 1844, 1831, 1852 | 9.65–9.66 | |
| Browser, batch 16 two at once, with the flag | 18047 | 1473 | 12.25 | 1.015 |
| Browser, batch 16 two at once, without the flag | 17957 | 1474 | 12.18 | 1.010 |
| Browser, batch 16 one at a time | 14221, 14222 | 1242, 1240 | 11.45–11.47 | 0.800 |
| Browser, batch 16 four at once | 20754, 20748 | 821, 819 | 25.28–25.33 | 1.167 |
| Browser, batch 64 one at a time | 21501, 21581 | 723 | 29.74–29.85 | 1.209–1.213 |

- **Identical across runs:** every browser count is the same in every run of part 1, before and after its changes.
- **With and without the flag:** the counts differ by 0.5% between the two, because the matrix and vector GEMMs
  round differently.
- **The other modes** are other search settings: more or fewer leaves in flight per search than native's 16 threads ×
  2 analyses. They are for comparison only.

## What this means for the budget

**The gap to native, revised** (batch 16 two at once, part 1's "after" runs). Native's rate here is from the counting
runs: 17678–17871 evaluations in 16249.6–16485.1 ms, 1083–1096/s.

| Factor | With the flag | Without the flag | Source |
| --- | ---: | ---: | --- |
| Network throughput: native's evaluations/s ÷ the browser's per GPU-busy second (795.1–866.3 with, 598.7–610.7 without) | 1.25–1.38 | 1.77–1.83 | measured |
| Time off the GPU: wall ÷ GPU busy | 1.134–1.158 | 1.073–1.080 | measured (part 1) |
| Evaluations: the browser's ÷ native's | 1.010–1.021 | 1.005–1.016 | measured |
| Product | 1.43–1.63 | 1.91–2.01 | |
| Measured review ratio (`totalMs` ÷ native's 16412.1–16596.1 from #80) | 1.45–1.57 | 1.90–1.97 | measured |

- **Part 2 closes with no change needed.** In the plan, the evaluation count stood for about 6% of the gap. It's
  1–2%.
- **What makes up the gap:** without the flag, most of it is network throughput on the GPU, which is what Phases 1–3
  worked on. With the flag, network throughput is still the larger factor, and time off the GPU is next.
- **Next:** part 3, batch norm, is the remaining Phase 4 work on network throughput.

## Kept in the code

- `scripts/native-search-bench.mjs --count-review`.
