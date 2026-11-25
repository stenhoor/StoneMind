# Spike 0001: b18 WebGPU GEMM v2 check and benchmark on Edge (NVIDIA)

Date: 2026-11-25. Machine: the owner's Linux laptop. Task: issue #33. Code: `main` at `6251855` (#32 merged).
Report: [`2026-11-b18-webgpu-edge-nvidia-gemm-v2.json`](2026-11-b18-webgpu-edge-nvidia-gemm-v2.json). All values
below are quoted from it.

## How it was run

- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/gpu-check-edge-nvidia-gemm-v2.json`.
- **Browser:** the separate Edge instance with the launch used in #17, #19, #24 and #27:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  **Check and benchmark** was pressed over the DevTools Protocol, and the instance was closed afterwards.
- **Report header:** `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`,
  `date: 2026-11-25T21:21:42.832Z`, `goldensCompared: true`, `ok: true`.
- **Variants:** ten, each with `ok: true`. None failed, so there is no "Failed at" line.
- **GPU check:** during the run, `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at
  up to 99% SM.
- **The v1 figures** below come from this same run, not from #27.

## Correctness

| Variant | `variant` | `ok` | Positions with `matchesKataGo: true` | `worstFp16LimitRatio` | `fp16LimitsPass` |
| --- | --- | --- | ---: | ---: | --- |
| Winograd fused | `{"winograd":true,"fuse":true}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| Winograd unfused | `{"winograd":true,"fuse":false}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| Implicit GEMM | `{"winograd":false}` | `true` | 10 / 10 | 0.0027317710772249543 | `true` |
| fp32 + scale8 | `{"winograd":true,"fuse":true,"scale8":true}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| GEMM v2 fp32 fused | `{"winograd":true,"fuse":true,"gemm":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| GEMM v2 fp32 unfused | `{"winograd":true,"fuse":false,"gemm":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp16 fused | `{"winograd":true,"fuse":true,"precision":"f16"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` |
| fp16 unfused | `{"winograd":true,"fuse":false,"precision":"f16"}` | `true` | 0 / 10 | 0.06510059568671471 | `true` |
| GEMM v2 fp16 fused | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v2"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` |
| GEMM v2 fp16 unfused | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v2"}` | `true` | 0 / 10 | 0.06510059568671471 | `true` |

## `bench`: evaluations per second

| Batch | Winograd fused | Winograd unfused | Implicit GEMM | fp32 + scale8 | GEMM v2 fp32 fused | GEMM v2 fp32 unfused | fp16 fused | fp16 unfused | GEMM v2 fp16 fused | GEMM v2 fp16 unfused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.7 | 84 | 23 | 80 | 51 | 78.1 | 58.8 | 31.1 | 26.7 | 30.3 |
| 2 | 156.2 | 156.3 | 47.7 | 120.5 | 102 | 110.5 | 115.6 | 117.6 | 120.5 | 114.3 |
| 4 | 236.7 | 228.6 | 85.1 | 195.1 | 212.8 | 212.8 | 224.7 | 215.1 | 261.4 | 248.4 |
| 8 | 261.4 | 280.7 | 126.4 | 246.2 | 299.6 | 268.5 | 297.4 | 292 | 362 | 377.4 |
| 16 | 308.9 | 316.2 | 140.2 | 298.5 | 373.8 | 361.2 | 366.1 | 354.8 | 408.2 | 413.4 |
| 32 | 332 | 340.4 | 147.7 | 332 | 413.4 | 385.1 | 383.2 | 379.1 | 473.4 | 463.8 |
| 64 | 358.5 | 366.1 | 148.2 | 367 | 427.8 | 414.2 | 396.5 | 404.3 | 506.3 | 495.7 |

## `profile`

Milliseconds:

| Variant | B1 `wallMsMedian` | B1 `gpuPassMsMedian` | B16 `wallMsMedian` | B16 `gpuPassMsMedian` | B64 `wallMsMedian` | B64 `gpuPassMsMedian` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Winograd fused | 13.2 | 9.437 | 50.2 | 45.548 | 170.9 | 164.037 |
| Winograd unfused | 12.6 | 9.503 | 51 | 46.727 | 175.1 | 167.707 |
| Implicit GEMM | 41.5 | 37.159 | 121.2 | 116.392 | 436.8 | 428.605 |
| fp32 + scale8 | 14.3 | 9.699 | 51 | 46.858 | 177.1 | 170.328 |
| GEMM v2 fp32 fused | 14 | 10.289 | 44.1 | 38.339 | 142.4 | 135.528 |
| GEMM v2 fp32 unfused | 17.4 | 9.961 | 47.6 | 37.683 | 152.2 | 137.232 |
| fp16 fused | 13.9 | 10.158 | 60.4 | 55.837 | 159.2 | 152.175 |
| fp16 unfused | 13.6 | 9.83 | 47.6 | 42.729 | 170.3 | 163.381 |
| GEMM v2 fp16 fused | 18.5 | 10.617 | 43.3 | 34.406 | 127.9 | 120.848 |
| GEMM v2 fp16 unfused | 14.4 | 11.469 | 41.6 | 37.88 | 138.1 | 132.514 |

`perDispatch.byKind` (count, ms) at batches 1, 16 and 64:

**Winograd fused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.852 | 11, 1.835 | 11, 1.966 |
| `winoIn` | 78, 1.376 | 78, 6.029 | 78, 20.185 |
| `winoGemm` | 78, 4.456 | 78, 22.217 | 78, 76.218 |
| `winoOut` | 78, 0.721 | 78, 5.374 | 78, 21.889 |
| `bnAct` | 45, 0.459 | 45, 2.163 | 45, 6.816 |
| `conv` | 41, 4.719 | 41, 11.076 | 41, 43.45 |
| `gpool` | 7, 0.131 | 7, 0.066 | 7, 0.131 |

**Winograd unfused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.59 | 11, 0.721 | 11, 0.655 |
| `winoIn` | 78, 1.18 | 78, 4.588 | 78, 15.008 |
| `winoGemm` | 78, 5.767 | 78, 20.84 | 78, 79.102 |
| `winoOut` | 78, 0.852 | 78, 5.177 | 78, 19.333 |
| `bnAct` | 117, 0.786 | 117, 3.801 | 117, 15.598 |
| `conv` | 41, 3.539 | 41, 11.272 | 41, 43.647 |
| `gpool` | 7, 0 | 7, 0.066 | 7, 0.197 |

**Implicit GEMM**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.59 | 11, 0.59 | 11, 0.459 |
| `conv` | 119, 36.307 | 119, 117.178 | 119, 414.908 |
| `bnAct` | 117, 1.245 | 117, 4.063 | 117, 14.09 |
| `gpool` | 7, 0.066 | 7, 0.131 | 7, 0.197 |

**fp32 + scale8**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.524 | 11, 0.655 | 11, 0.655 |
| `winoIn` | 78, 1.573 | 78, 5.833 | 78, 39.125 |
| `winoGemm` | 78, 4.653 | 78, 23.2 | 78, 76.218 |
| `winoOut` | 78, 1.049 | 78, 6.16 | 78, 19.726 |
| `bnAct` | 45, 0.197 | 45, 1.769 | 45, 6.357 |
| `conv` | 41, 3.539 | 41, 14.221 | 41, 48.955 |
| `gpool` | 7, 0.066 | 7, 0.066 | 7, 0.328 |

**GEMM v2 fp32 fused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.524 | 11, 0.655 | 11, 0.655 |
| `winoIn` | 78, 1.573 | 78, 5.702 | 78, 20.382 |
| `winoGemm` | 78, 5.964 | 78, 19.464 | 78, 65.405 |
| `winoOut` | 78, 0.852 | 78, 5.177 | 78, 20.578 |
| `bnAct` | 45, 0.262 | 45, 1.835 | 45, 7.799 |
| `conv1x1` | 41, 2.294 | 41, 6.881 | 41, 26.477 |
| `gpool` | 7, 0 | 7, 0 | 7, 0.197 |

**GEMM v2 fp32 unfused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.524 | 11, 0.59 | 11, 0.786 |
| `winoIn` | 78, 1.049 | 78, 4.26 | 78, 16.253 |
| `winoGemm` | 78, 6.357 | 78, 18.94 | 78, 62.783 |
| `winoOut` | 78, 0.786 | 78, 5.833 | 78, 21.103 |
| `bnAct` | 117, 1.245 | 117, 4.063 | 117, 15.335 |
| `conv1x1` | 41, 2.163 | 41, 6.881 | 41, 30.474 |
| `gpool` | 7, 0.066 | 7, 0 | 7, 0.131 |

**fp16 fused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.59 | 11, 0.59 | 11, 0.59 |
| `winoIn` | 78, 3.211 | 78, 17.17 | 78, 18.022 |
| `winoGemm` | 78, 4.522 | 78, 21.103 | 78, 75.301 |
| `winoOut` | 78, 1.114 | 78, 3.801 | 78, 11.731 |
| `bnAct` | 45, 0.393 | 45, 1.049 | 45, 6.16 |
| `conv` | 41, 3.473 | 41, 13.631 | 41, 44.63 |
| `gpool` | 7, 0 | 7, 0.066 | 7, 0.066 |

**fp16 unfused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.655 | 11, 0.524 | 11, 0.524 |
| `winoIn` | 78, 1.245 | 78, 4.325 | 78, 23.593 |
| `winoGemm` | 78, 4.522 | 78, 21.168 | 78, 76.481 |
| `winoOut` | 78, 0.655 | 78, 4.784 | 78, 11.796 |
| `bnAct` | 117, 0.852 | 117, 3.408 | 117, 12.386 |
| `conv` | 41, 3.539 | 41, 10.879 | 41, 42.402 |
| `gpool` | 7, 0.131 | 7, 0.197 | 7, 0.262 |

**GEMM v2 fp16 fused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.721 | 11, 0.655 | 11, 0.524 |
| `winoIn` | 78, 1.507 | 78, 4.784 | 78, 15.335 |
| `winoGemm` | 78, 6.291 | 78, 16.974 | 78, 63.373 |
| `winoOut` | 78, 0.918 | 78, 3.342 | 78, 11.6 |
| `bnAct` | 45, 0.197 | 45, 1.704 | 45, 5.374 |
| `conv1x1` | 41, 3.015 | 41, 6.816 | 41, 27.001 |
| `gpool` | 7, 0 | 7, 0 | 7, 0.197 |

**GEMM v2 fp16 unfused**

| Kind | Batch 1 | Batch 16 | Batch 64 |
| --- | ---: | ---: | ---: |
| `matvec` | 11, 0.524 | 11, 0.721 | 11, 0.59 |
| `winoIn` | 78, 1.376 | 78, 7.799 | 78, 12.976 |
| `winoGemm` | 78, 6.488 | 78, 18.416 | 78, 62.128 |
| `winoOut` | 78, 1.507 | 78, 5.898 | 78, 11.993 |
| `bnAct` | 117, 0.524 | 117, 3.146 | 117, 11.862 |
| `conv1x1` | 41, 2.753 | 41, 7.012 | 41, 27.46 |
| `gpool` | 7, 0.066 | 7, 0.131 | 7, 0.131 |

## GEMM v2 beside v1 and native `benchmarknn`

The v1 counterpart is the same variant without `gemm: "v2"`, from this run. The native columns are
`sumMedianNNEvalsPerSec` from [`2026-11-b18-native-benchmarknn.md`](2026-11-b18-native-benchmarknn.md), measured at
18:19–18:21 UTC. This run was at 21:21 UTC, so the two are not back to back. That file lists the measurement
differences between the two tools.

**GEMM v2 fp32 fused** beside **Winograd fused** (v1):

| Batch | GEMM v2 fp32 fused | Winograd fused | v2 ÷ v1 | Native FP32 | v2 ÷ native FP32 | Native FP16 | v2 ÷ native FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 51 | 66.7 | 0.76 | 207.5 | 0.25 | 359.9 | 0.14 |
| 2 | 102 | 156.2 | 0.65 | 376.4 | 0.27 | 719.3 | 0.14 |
| 4 | 212.8 | 236.7 | 0.90 | 461.4 | 0.46 | 937.8 | 0.23 |
| 8 | 299.6 | 261.4 | 1.15 | 536.5 | 0.56 | 996.7 | 0.30 |
| 16 | 373.8 | 308.9 | 1.21 | 525.7 | 0.71 | 1291.2 | 0.29 |
| 32 | 413.4 | 332 | 1.25 | 527.5 | 0.78 | 1114.4 | 0.37 |
| 64 | 427.8 | 358.5 | 1.19 | 543.2 | 0.79 | 1141.6 | 0.37 |

**GEMM v2 fp32 unfused** beside **Winograd unfused** (v1):

| Batch | GEMM v2 fp32 unfused | Winograd unfused | v2 ÷ v1 | Native FP32 | v2 ÷ native FP32 | Native FP16 | v2 ÷ native FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.1 | 84 | 0.93 | 207.5 | 0.38 | 359.9 | 0.22 |
| 2 | 110.5 | 156.3 | 0.71 | 376.4 | 0.29 | 719.3 | 0.15 |
| 4 | 212.8 | 228.6 | 0.93 | 461.4 | 0.46 | 937.8 | 0.23 |
| 8 | 268.5 | 280.7 | 0.96 | 536.5 | 0.50 | 996.7 | 0.27 |
| 16 | 361.2 | 316.2 | 1.14 | 525.7 | 0.69 | 1291.2 | 0.28 |
| 32 | 385.1 | 340.4 | 1.13 | 527.5 | 0.73 | 1114.4 | 0.35 |
| 64 | 414.2 | 366.1 | 1.13 | 543.2 | 0.76 | 1141.6 | 0.36 |

**GEMM v2 fp16 fused** beside **fp16 fused** (v1):

| Batch | GEMM v2 fp16 fused | fp16 fused | v2 ÷ v1 | Native FP32 | v2 ÷ native FP32 | Native FP16 | v2 ÷ native FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26.7 | 58.8 | 0.45 | 207.5 | 0.13 | 359.9 | 0.07 |
| 2 | 120.5 | 115.6 | 1.04 | 376.4 | 0.32 | 719.3 | 0.17 |
| 4 | 261.4 | 224.7 | 1.16 | 461.4 | 0.57 | 937.8 | 0.28 |
| 8 | 362 | 297.4 | 1.22 | 536.5 | 0.67 | 996.7 | 0.36 |
| 16 | 408.2 | 366.1 | 1.11 | 525.7 | 0.78 | 1291.2 | 0.32 |
| 32 | 473.4 | 383.2 | 1.24 | 527.5 | 0.90 | 1114.4 | 0.42 |
| 64 | 506.3 | 396.5 | 1.28 | 543.2 | 0.93 | 1141.6 | 0.44 |

**GEMM v2 fp16 unfused** beside **fp16 unfused** (v1):

| Batch | GEMM v2 fp16 unfused | fp16 unfused | v2 ÷ v1 | Native FP32 | v2 ÷ native FP32 | Native FP16 | v2 ÷ native FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30.3 | 31.1 | 0.97 | 207.5 | 0.15 | 359.9 | 0.08 |
| 2 | 114.3 | 117.6 | 0.97 | 376.4 | 0.30 | 719.3 | 0.16 |
| 4 | 248.4 | 215.1 | 1.15 | 461.4 | 0.54 | 937.8 | 0.26 |
| 8 | 377.4 | 292 | 1.29 | 536.5 | 0.70 | 996.7 | 0.38 |
| 16 | 413.4 | 354.8 | 1.17 | 525.7 | 0.79 | 1291.2 | 0.32 |
| 32 | 463.8 | 379.1 | 1.22 | 527.5 | 0.88 | 1114.4 | 0.42 |
| 64 | 495.7 | 404.3 | 1.23 | 543.2 | 0.91 | 1141.6 | 0.43 |
