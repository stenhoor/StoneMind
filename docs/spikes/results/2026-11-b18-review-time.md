# Spike 0001: where review time goes (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Code: branch `spike-0001/review-time`, on
top of `spike-0001/subgroup-matrix` (#80). All values below are quoted from the reports and the log in this directory.

## The question

With the Winograd GEMMs on matrix hardware (#80), the batch-16 GPU pass was 14% shorter, but review time improved only
3.6–3.7%. Native KataGo reviews the same 250 turns about twice as fast. Where does the browser's review time go?

## How it was measured

- **Trace:** `GpuNetwork.startTrace()` / `stopTrace()` (new, measurement only) record, for every network call:
  - wall-clock times on the JavaScript thread: start, submit, outputs readable, return;
  - the network code's own JavaScript parts (`CallTiming`);
  - the GPU timestamps of the call's compute pass. The two timestamps travel back in the call's own readback buffer, so
    tracing adds no wait.
- **Search benchmark:** traces each review mode and reports `trace` for each:
  - `gpuBusyMs` is the union of the calls' compute passes.
  - `callInFlightMs` is the union of the calls' wall-clock spans.
  - `networkJsMs` is the network code's JavaScript.
  - `byBatch` gives the calls by number of inputs, with their mean pass time.
- **Runs**, each in a fresh separate Edge, one after another:
  1. **Page default** (`{"winograd":true,"fuse":true,"precision":"f16"}`): [`…-search-edge-nvidia-1.json`](2026-11-b18-review-time-search-edge-nvidia-1.json) (2026-11-26T15:27:51.758Z)
  2. **Page default again**, with `byBatch`: [`…-search-edge-nvidia-2.json`](2026-11-b18-review-time-search-edge-nvidia-2.json) (2026-11-26T15:34:08.913Z)
  3. **With `winoGemm: 'mma'`**, in the Edge launched with `--enable-unsafe-webgpu` (the owner's exception, #80):
     [`…-search-edge-nvidia-mma.json`](2026-11-b18-review-time-search-edge-nvidia-mma.json) (2026-11-26T15:30:53.404Z)
- **Native:** KataGo's own `benchmark` command:
  - Run as `katago benchmark -model <b18> -config ~/katago/gtp_cuda.cfg -v 800 -n 10 -t 16,32 -no-server-thread-test`;
    the progress lines are removed from [`…-native-benchmark.txt`](2026-11-b18-review-time-native-benchmark.txt).
  - It reports the CUDA backend with `useFP16 = true useNHWC = true`.
  - `analysis_example.cfg` can't be used with `benchmark`, which needs `numSearchThreads`.

## Results

| Run | Review mode | `totalMs` | GPU busy (ms) | Calls | Inputs per call | Network JS (ms) | Evaluations per GPU-busy second |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | batch 64, 1 at once | 44259 | 31522.9 (71.2%) | 723 | 29.85 | 1355 | 685 |
| 1 | batch 16, 1 at once | 36731 | 23759.8 (64.7%) | 1240 | 11.47 | 1335.1 | 599 |
| 1 | batch 16, 2 at once | 32279.6 | 30513.8 (94.5%) | 1474 | 12.18 | 1615.4 | 588 |
| 1 | batch 16, 4 at once | 32188 | 31354.2 (97.4%) | 819 | 25.33 | 1369.3 | 662 |
| 2 | batch 64, 1 at once | 46985.6 | 31326.7 (66.7%) | 723 | 29.85 | 1652.7 | 689 |
| 2 | batch 16, 1 at once | 42091 | 23553.2 (56.0%) | 1240 | 11.47 | 1939.1 | 604 |
| 2 | batch 16, 2 at once | 33574.2 | 30321.6 (90.3%) | 1474 | 12.18 | 1950.7 | 592 |
| 2 | batch 16, 4 at once | 32071.7 | 31111.1 (97.0%) | 819 | 25.33 | 1555.6 | 667 |
| mma | batch 64, 1 at once | 43601.3 | 30787.2 (70.6%) | 722 | 29.94 | 1348.2 | 702 |
| mma | batch 16, 1 at once | 35868.2 | 22717.3 (63.3%) | 1245 | 11.5 | 1393.2 | 630 |
| mma | batch 16, 2 at once | 30303 | 29333.1 (96.8%) | 1485 | 12.19 | 1596.8 | 617 |
| mma | batch 16, 4 at once | 31239.4 | 30908.4 (98.9%) | 822 | 25.17 | 1321.2 | 669 |

**Facts:**
- **GPU busy time:** with 2 or 4 turns searched at once (the default is 2), the GPU was busy 90.3–98.9% of the review.
  One turn at a time, 56.0–71.2%.
- **Network JavaScript:** 1321.2–1950.7 ms in every mode, 3.6–5.8% of the review.
- **Matrix GEMM:** at 2 at once, it reduced GPU busy time from 30513.8 ms (run 1) to 29333.1 ms, and review time
  from 32279.6 to 30303 ms.

**GPU time per call by number of inputs** (run 2, calls one at a time, so each pass's interval is its own; mean pass ms,
with the number of calls in brackets):

| Mode | 1 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batch 16, 1 at once | 7.541 (265) | 9.153 (18) | 14.066 (16) | 24.162 (751) | — | — |
| batch 64, 1 at once | 8.158 (253) | 9.634 (2) | 14.565 (4) | 24.27 (3) | 45.105 (12) | 86.151 (226) |

In run 2, batch 64 one at a time made 253 calls with one input; batch 16 made 265 one at a time and 267 with 2 at once.
With 4 at once there were none. That's about one per turn; my assumption, not checked, is that each is a search's
first evaluation, its root.

**Native** (`benchmark`, 10 positions × 800 visits):

| `numSearchThreads` | visits/s | nnEvals/s | nnBatches/s | avgBatchSize |
| --- | ---: | ---: | ---: | ---: |
| 16 | 1399.65 | 1206.99 | 279.96 | 4.31 |
| 32 | 1435.83 | 1290.29 | 151.25 | 8.53 |
| 16 | 1424.24 | 1209.98 | 286.67 | 4.22 |
| 16 | 1388.78 | 1200.10 | 235.04 | 5.11 |

(The third row is KataGo's re-test at 16 threads. The fourth is its test with the batch capped at 8, "half the search
threads".)

## What this shows

- **The review is GPU-bound** when 2 or more turns are searched at once. The browser's search and the network's
  JavaScript are not where the time goes.
- **Native evaluates faster at smaller batches:**
  - Native: 1200.10–1290.29 evaluations per second, at average batches of 4.22–8.53.
  - Browser: 588–702 evaluations per GPU-busy second, at 11.47–29.94 inputs per call.
- **Browser per-call cost:** a call takes about 7.5–8 ms of GPU time even with one input. At 16 inputs it's about 24 ms,
  about 1.5 ms per input.
