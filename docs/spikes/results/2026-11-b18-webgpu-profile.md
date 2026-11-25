# Spike 0001: b18 WebGPU profile and correctness re-check on Edge (NVIDIA)

Date: 2026-11-25. Machine: the owner's Linux laptop. Task: issue #24. Code: `main` at `b232055` (#22 and #23 merged).
Report: [`2026-11-b18-webgpu-edge-nvidia-profile.json`](2026-11-b18-webgpu-edge-nvidia-profile.json). All values
below are quoted from it.

## How it was run

- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/gpu-check-edge-nvidia-profile.json`. The goldens' `modelSha256` is `9d7a6afe…e0cba51f1d`.
- **Browser:** the same separate Edge instance and launch as in #17 and #19:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  **Check and benchmark** was pressed over the DevTools Protocol, and the instance was closed afterwards.
- **Report header:** UA `… Chrome/158.0.0.0 Safari/537.36 Edg/158.0.0.0`,
  `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`,
  `features: ["shader-f16","subgroups","timestamp-query"]`, `date: 2026-11-25T19:33:33.571Z`, `goldensCompared: true`,
  `referenceCompared: false`, `ok: true`.
- **GPU check:** during the run, `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at
  up to 98% SM.

## Correctness

| Variant | `ok` | Positions with `matchesKataGo: true` | `batchOfOneMatchesBatched` | `failures` |
| --- | --- | ---: | --- | --- |
| Winograd fused | `true` | 10 / 10 | `true` | none |
| Winograd unfused | `true` | 10 / 10 | `true` | none |
| Implicit GEMM | `true` | 10 / 10 | `true` | none |

## Benchmark (`bench`)

Evaluations per second, with the median ms per batch in brackets:

| Batch | Winograd fused | Winograd unfused | Implicit GEMM |
| ---: | ---: | ---: | ---: |
| 1 | 59.2 (16.9) | 87.7 (11.4) | 26.5 (37.8) |
| 2 | 108.7 (18.4) | 157.5 (12.7) | 49.9 (40.1) |
| 4 | 205.1 (19.5) | 232.6 (17.2) | 83.5 (47.9) |
| 8 | 264.9 (30.2) | 292 (27.4) | 127 (63) |
| 16 | 315.6 (50.7) | 333.3 (48) | 142.7 (112.1) |
| 32 | 348.2 (91.9) | 360 (88.9) | 151.9 (210.7) |
| 64 | 376.5 (170) | 378 (169.3) | 163.7 (391) |

## Profile (`profile`)

For each variant: `timerTickNs`, then for batches 1, 16 and 64 the `wallMsMedian`, `gpuPassMsMedian`,
`perDispatch.dispatches`, `perDispatch.sumMs`, `perDispatch.byKind`, and the first 5 entries of `perDispatch.slowest`.
The report also has `perDispatch.wallMs` and the full `slowest` lists (10 entries each).

### Winograd fused

`timerTickNs`: 65536

| Batch | `wallMsMedian` | `gpuPassMsMedian` | `perDispatch.dispatches` | `perDispatch.sumMs` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.9 | 9.568 | 338 | 11.403 |
| 16 | 53.1 | 43.385 | 338 | 45.285 |
| 64 | 164.9 | 157.09 | 338 | 161.874 |

`perDispatch.byKind` (count, ms):

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.59 | 11, 0.59 | 11, 0.59 |
| `winoIn` | 78, 1.376 | 78, 5.702 | 78, 18.809 |
| `winoGemm` | 78, 4.391 | 78, 20.644 | 78, 75.17 |
| `winoOut` | 78, 1.114 | 78, 5.243 | 78, 18.809 |
| `bnAct` | 45, 0.459 | 45, 1.835 | 45, 6.226 |
| `conv` | 41, 3.473 | 41, 11.272 | 41, 41.812 |
| `gpool` | 7, 0 | 7, 0 | 7, 0.459 |

Top 5 of `perDispatch.slowest` (index, kind, ms):

| Rank | Batch 1 | Batch 16 | Batch 64 |
| ---: | --- | --- | --- |
| 1 | 21, `conv`, 0.131 | 10, `winoGemm`, 0.328 | 111, `conv`, 1.18 |
| 2 | 37, `conv`, 0.131 | 13, `winoGemm`, 0.328 | 145, `conv`, 1.18 |
| 3 | 75, `conv`, 0.131 | 16, `winoGemm`, 0.328 | 181, `conv`, 1.18 |
| 4 | 91, `conv`, 0.131 | 19, `conv`, 0.328 | 199, `conv`, 1.18 |
| 5 | 129, `conv`, 0.131 | 21, `conv`, 0.328 | 221, `conv`, 1.18 |

### Winograd unfused

`timerTickNs`: 65536

| Batch | `wallMsMedian` | `gpuPassMsMedian` | `perDispatch.dispatches` | `perDispatch.sumMs` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.5 | 9.241 | 410 | 11.272 |
| 16 | 48.8 | 44.892 | 410 | 46.399 |
| 64 | 178.5 | 160.104 | 410 | 164.692 |

`perDispatch.byKind` (count, ms):

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.524 | 11, 0.459 | 11, 0.786 |
| `winoIn` | 78, 0.983 | 78, 4.194 | 78, 15.008 |
| `winoGemm` | 78, 4.26 | 78, 19.988 | 78, 74.514 |
| `winoOut` | 78, 0.852 | 78, 5.702 | 78, 18.678 |
| `bnAct` | 117, 0.983 | 117, 4.522 | 117, 14.549 |
| `conv` | 41, 3.604 | 41, 11.469 | 41, 41.026 |
| `gpool` | 7, 0.066 | 7, 0.066 | 7, 0.131 |

Top 5 of `perDispatch.slowest` (index, kind, ms):

| Rank | Batch 1 | Batch 16 | Batch 64 |
| ---: | --- | --- | --- |
| 1 | 5, `conv`, 0.131 | 5, `conv`, 0.328 | 109, `conv`, 1.18 |
| 2 | 25, `conv`, 0.131 | 23, `conv`, 0.328 | 155, `conv`, 1.18 |
| 3 | 71, `conv`, 0.131 | 25, `conv`, 0.328 | 241, `conv`, 1.18 |
| 4 | 91, `conv`, 0.131 | 28, `winoGemm`, 0.328 | 333, `conv`, 1.18 |
| 5 | 137, `conv`, 0.131 | 32, `winoGemm`, 0.328 | 353, `conv`, 1.18 |

### Implicit GEMM

`timerTickNs`: 65536

| Batch | `wallMsMedian` | `gpuPassMsMedian` | `perDispatch.dispatches` | `perDispatch.sumMs` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 39.5 | 35.717 | 254 | 48.431 |
| 16 | 113.4 | 110.1 | 254 | 113.246 |
| 64 | 413 | 406.979 | 254 | 410.845 |

`perDispatch.byKind` (count, ms):

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.459 | 11, 0.59 | 11, 0.655 |
| `conv` | 119, 46.924 | 119, 108.593 | 119, 395.837 |
| `bnAct` | 117, 0.918 | 117, 3.932 | 117, 14.09 |
| `gpool` | 7, 0.131 | 7, 0.131 | 7, 0.262 |

Top 5 of `perDispatch.slowest` (index, kind, ms):

| Rank | Batch 1 | Batch 16 | Batch 64 |
| ---: | --- | --- | --- |
| 1 | 9, `conv`, 0.59 | 5, `conv`, 1.376 | 37, `conv`, 4.981 |
| 2 | 11, `conv`, 0.59 | 7, `conv`, 1.376 | 45, `conv`, 4.981 |
| 3 | 19, `conv`, 0.59 | 11, `conv`, 1.376 | 47, `conv`, 4.981 |
| 4 | 21, `conv`, 0.59 | 17, `conv`, 1.376 | 51, `conv`, 4.981 |
| 5 | 23, `conv`, 0.59 | 19, `conv`, 1.376 | 59, `conv`, 4.981 |
