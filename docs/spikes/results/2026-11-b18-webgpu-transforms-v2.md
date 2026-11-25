# Spike 0001: b18 WebGPU Winograd transforms v2 check and benchmark on Edge (NVIDIA)

Date: 2026-11-25. Machine: the owner's Linux laptop. Task: issue #44. Code: `main` at `5c5488a` (#43 merged).
Report: [`2026-11-b18-webgpu-edge-nvidia-transforms-v2.json`](2026-11-b18-webgpu-edge-nvidia-transforms-v2.json).
All values below are quoted from it.

## How it was run

- **Procedure:** the same as #33, with the output names changed.
- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/gpu-check-edge-nvidia-transforms-v2.json`.
- **Browser:** the separate Edge instance with the launch used in #17, #19, #24, #27, #33, #38 and #41:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  **Check and benchmark** was pressed over the DevTools Protocol (not Tune), and the instance was closed afterwards.
- **Report header:** `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`,
  `date: 2026-11-25T23:13:10.117Z`, `goldensCompared: true`, `ok: true`.
- **Variants:** 14, each with `ok: true`. None has a `failedAt` or `error`.
- **GPU check:** during the run, `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at
  up to 99% SM.

## Correctness

| Variant | `variant` | `ok` | Positions with `matchesKataGo: true` | `worstFp16LimitRatio` | `fp16LimitsPass` |
| --- | --- | --- | ---: | ---: | --- |
| fp32 fused, GEMM v1 | `{"winograd":true,"fuse":true,"gemm":"v1"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp32 unfused, GEMM v1 | `{"winograd":true,"fuse":false,"gemm":"v1"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp32 fused, GEMM v2 | `{"winograd":true,"fuse":true,"gemm":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp32 unfused, GEMM v2 | `{"winograd":true,"fuse":false,"gemm":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp32 fused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":true,"gemm":"v2","winoTransforms":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp32 unfused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":false,"gemm":"v2","winoTransforms":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp32 + scale8 fused, GEMM v2 | `{"winograd":true,"fuse":true,"scale8":true,"gemm":"v2"}` | `true` | 10 / 10 | 0.002731768973208893 | `true` |
| fp16 fused, GEMM v1 | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v1"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` |
| fp16 unfused, GEMM v1 | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v1"}` | `true` | 0 / 10 | 0.06510059568671471 | `true` |
| fp16 fused, GEMM v2 | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v2"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` |
| fp16 unfused, GEMM v2 | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v2"}` | `true` | 0 / 10 | 0.06510059568671471 | `true` |
| fp16 fused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v2","winoTransforms":"v2"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` |
| fp16 unfused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v2","winoTransforms":"v2"}` | `true` | 0 / 10 | 0.06510059568671471 | `true` |
| Implicit GEMM | `{"winograd":false,"gemm":"v1"}` | `true` | 10 / 10 | 0.0027317710772249543 | `true` |

## `bench`: evaluations per second

| Variant | B1 | B2 | B4 | B8 | B16 | B32 | B64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 fused, GEMM v1 | 68 | 129 | 202 | 268.5 | 314.3 | 352 | 394.3 |
| fp32 unfused, GEMM v1 | 89.3 | 157.5 | 229.9 | 299.6 | 336.1 | 366.1 | 381.6 |
| fp32 fused, GEMM v2 | 78.7 | 158.7 | 274 | 355.6 | 400 | 451.3 | 474.8 |
| fp32 unfused, GEMM v2 | 78.1 | 118.3 | 208.3 | 286.7 | 352.4 | 427.8 | 461.1 |
| fp32 fused, GEMM v2, transforms v2 | 66.7 | 123.5 | 227.3 | 306.5 | 363.6 | 416.7 | 479.4 |
| fp32 unfused, GEMM v2, transforms v2 | 64.9 | 119 | 261.4 | 305.3 | 348.6 | 397.5 | 453.6 |
| fp32 + scale8 fused, GEMM v2 | 77.5 | 123.5 | 222.2 | 337.6 | 357.9 | 444.4 | 474.1 |
| fp16 fused, GEMM v1 | 37.3 | 105.8 | 177 | 277.8 | 337.6 | 378.3 | 410.5 |
| fp16 unfused, GEMM v1 | 65.4 | 132.5 | 236.7 | 275.9 | 331.3 | 374.7 | 410 |
| fp16 fused, GEMM v2 | 62.5 | 126.6 | 272.1 | 372.1 | 444.4 | 492.3 | 530.2 |
| fp16 unfused, GEMM v2 | 29.9 | 122.7 | 229.9 | 386.5 | 426.7 | 485.6 | 526.3 |
| fp16 fused, GEMM v2, transforms v2 | 26.2 | 121.2 | 198 | 333.3 | 410.3 | 507.9 | 549.8 |
| fp16 unfused, GEMM v2, transforms v2 | 32.1 | 120.5 | 233.9 | 377.4 | 444.4 | 496.1 | 525.9 |
| Implicit GEMM | 26.1 | 53.3 | 88.7 | 132.2 | 147.5 | 152.1 | 154.6 |

## Side by side: with and without `winoTransforms: "v2"`

Each `winoTransforms: "v2"` variant beside the same variant without it, from this run. The ratio is evaluations per
second with v2 ÷ without.

**fp32 fused, GEMM v2, transforms v2** beside **fp32 fused, GEMM v2**:

| Batch | With transforms v2 | Without | With ÷ without |
| ---: | ---: | ---: | ---: |
| 1 | 66.7 | 78.7 | 0.85 |
| 2 | 123.5 | 158.7 | 0.78 |
| 4 | 227.3 | 274 | 0.83 |
| 8 | 306.5 | 355.6 | 0.86 |
| 16 | 363.6 | 400 | 0.91 |
| 32 | 416.7 | 451.3 | 0.92 |
| 64 | 479.4 | 474.8 | 1.01 |

**fp32 unfused, GEMM v2, transforms v2** beside **fp32 unfused, GEMM v2**:

| Batch | With transforms v2 | Without | With ÷ without |
| ---: | ---: | ---: | ---: |
| 1 | 64.9 | 78.1 | 0.83 |
| 2 | 119 | 118.3 | 1.01 |
| 4 | 261.4 | 208.3 | 1.25 |
| 8 | 305.3 | 286.7 | 1.06 |
| 16 | 348.6 | 352.4 | 0.99 |
| 32 | 397.5 | 427.8 | 0.93 |
| 64 | 453.6 | 461.1 | 0.98 |

**fp16 fused, GEMM v2, transforms v2** beside **fp16 fused, GEMM v2**:

| Batch | With transforms v2 | Without | With ÷ without |
| ---: | ---: | ---: | ---: |
| 1 | 26.2 | 62.5 | 0.42 |
| 2 | 121.2 | 126.6 | 0.96 |
| 4 | 198 | 272.1 | 0.73 |
| 8 | 333.3 | 372.1 | 0.90 |
| 16 | 410.3 | 444.4 | 0.92 |
| 32 | 507.9 | 492.3 | 1.03 |
| 64 | 549.8 | 530.2 | 1.04 |

**fp16 unfused, GEMM v2, transforms v2** beside **fp16 unfused, GEMM v2**:

| Batch | With transforms v2 | Without | With ÷ without |
| ---: | ---: | ---: | ---: |
| 1 | 32.1 | 29.9 | 1.07 |
| 2 | 120.5 | 122.7 | 0.98 |
| 4 | 233.9 | 229.9 | 1.02 |
| 8 | 377.4 | 386.5 | 0.98 |
| 16 | 444.4 | 426.7 | 1.04 |
| 32 | 496.1 | 485.6 | 1.02 |
| 64 | 525.9 | 526.3 | 1.00 |

## `profile`: `winoIn` and `winoOut` from `perDispatch.byKind`

The four GEMM v2 fused variants, fp32 and fp16, each with and without `winoTransforms: "v2"`:

| Variant | Kind | Batch 1 (count, ms) | Batch 16 (count, ms) | Batch 64 (count, ms) |
| --- | --- | ---: | ---: | ---: |
| fp32 fused, GEMM v2 | `winoIn` | 78, 1.704 | 78, 7.668 | 78, 30.605 |
| fp32 fused, GEMM v2 | `winoOut` | 78, 0.786 | 78, 5.177 | 78, 17.498 |
| fp32 fused, GEMM v2, transforms v2 | `winoIn` | 78, 1.442 | 78, 5.833 | 78, 18.285 |
| fp32 fused, GEMM v2, transforms v2 | `winoOut` | 78, 0.983 | 78, 5.177 | 78, 18.612 |
| fp16 fused, GEMM v2 | `winoIn` | 78, 0.918 | 78, 9.568 | 78, 15.925 |
| fp16 fused, GEMM v2 | `winoOut` | 78, 1.507 | 78, 3.67 | 78, 11.993 |
| fp16 fused, GEMM v2, transforms v2 | `winoIn` | 78, 0.721 | 78, 3.604 | 78, 12.059 |
| fp16 fused, GEMM v2, transforms v2 | `winoOut` | 78, 1.114 | 78, 3.604 | 78, 11.665 |

The same times, with transforms v2 ÷ without:

| Precision | Kind | Batch | With transforms v2 (ms) | Without (ms) | With ÷ without |
| --- | --- | ---: | ---: | ---: | ---: |
| fp32 fused | `winoIn` | 1 | 1.442 | 1.704 | 0.85 |
| fp32 fused | `winoIn` | 16 | 5.833 | 7.668 | 0.76 |
| fp32 fused | `winoIn` | 64 | 18.285 | 30.605 | 0.60 |
| fp32 fused | `winoOut` | 1 | 0.983 | 0.786 | 1.25 |
| fp32 fused | `winoOut` | 16 | 5.177 | 5.177 | 1.00 |
| fp32 fused | `winoOut` | 64 | 18.612 | 17.498 | 1.06 |
| fp16 fused | `winoIn` | 1 | 0.721 | 0.918 | 0.79 |
| fp16 fused | `winoIn` | 16 | 3.604 | 9.568 | 0.38 |
| fp16 fused | `winoIn` | 64 | 12.059 | 15.925 | 0.76 |
| fp16 fused | `winoOut` | 1 | 1.114 | 1.507 | 0.74 |
| fp16 fused | `winoOut` | 16 | 3.604 | 3.67 | 0.98 |
| fp16 fused | `winoOut` | 64 | 11.665 | 11.993 | 0.97 |
