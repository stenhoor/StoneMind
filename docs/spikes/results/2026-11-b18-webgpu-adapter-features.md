# Spike 0001: b18 WebGPU adapter features and check and benchmark on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Task: issue #48. Code: `main` at `8116e57` (#47 merged;
`packages/engine-webgpu/src/gpu/device.ts` has `adapterFeatures`).
Report: [`2026-11-b18-webgpu-edge-nvidia-features.json`](2026-11-b18-webgpu-edge-nvidia-features.json). All values
below are quoted from it.

## How it was run

- **Procedure:** the same as #44, with the output names changed. No browser flags were added.
- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/gpu-check-edge-nvidia-features.json`.
- **Browser:** the separate Edge instance with the launch used in #17, #19, #24, #27, #33, #38, #41 and #44:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  **Check and benchmark** was pressed over the DevTools Protocol (not Tune), and the instance was closed afterwards.
- **GPU check:** during the run, `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at
  up to 99% SM.

## Report header

- **`userAgent`:** `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36 Edg/158.0.0.0`
- **`adapter`:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}`
- **`features`:** `["shader-f16","subgroups","timestamp-query"]`
- **`adapterFeatures`:** 22 entries, listed below.
- **Contains `chromium-experimental-subgroup-matrix`:** no.
- **`ok`:** `true`
- **`date`:** `2026-11-26T00:20:13.762Z`
- **Variants:** 14, each with `ok: true`. None has a `failedAt` or `error`.

### `adapterFeatures`, as recorded

```json
["atomic-vec2u-min-max","bgra8unorm-storage","clip-distances","core-features-and-limits","depth-clip-control","depth32float-stencil8","dual-source-blending","float32-blendable","float32-filterable","indirect-first-instance","primitive-index","rg11b10ufloat-renderable","shader-f16","subgroup-size-control","subgroups","texture-component-swizzle","texture-compression-bc","texture-compression-bc-sliced-3d","texture-compression-unaligned","texture-formats-tier1","texture-formats-tier2","timestamp-query"]
```

| # | `adapterFeatures` entry |
| ---: | --- |
| 1 | `atomic-vec2u-min-max` |
| 2 | `bgra8unorm-storage` |
| 3 | `clip-distances` |
| 4 | `core-features-and-limits` |
| 5 | `depth-clip-control` |
| 6 | `depth32float-stencil8` |
| 7 | `dual-source-blending` |
| 8 | `float32-blendable` |
| 9 | `float32-filterable` |
| 10 | `indirect-first-instance` |
| 11 | `primitive-index` |
| 12 | `rg11b10ufloat-renderable` |
| 13 | `shader-f16` |
| 14 | `subgroup-size-control` |
| 15 | `subgroups` |
| 16 | `texture-component-swizzle` |
| 17 | `texture-compression-bc` |
| 18 | `texture-compression-bc-sliced-3d` |
| 19 | `texture-compression-unaligned` |
| 20 | `texture-formats-tier1` |
| 21 | `texture-formats-tier2` |
| 22 | `timestamp-query` |

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
| fp32 fused, GEMM v1 | 59.5 | 108.1 | 200 | 266.7 | 315.6 | 353.6 | 380.7 |
| fp32 unfused, GEMM v1 | 87 | 156.3 | 233.9 | 301.9 | 336.1 | 367 | 381.2 |
| fp32 fused, GEMM v2 | 80.6 | 150.4 | 236.7 | 303 | 404 | 451.3 | 463.8 |
| fp32 unfused, GEMM v2 | 63.7 | 157.5 | 213.9 | 292 | 350.9 | 435.4 | 456.5 |
| fp32 fused, GEMM v2, transforms v2 | 66.2 | 128.2 | 233.9 | 315 | 367.8 | 417.8 | 478.3 |
| fp32 unfused, GEMM v2, transforms v2 | 62.1 | 119 | 223.5 | 294.1 | 347.1 | 401 | 429.8 |
| fp32 + scale8 fused, GEMM v2 | 76.9 | 118.3 | 217.4 | 294.1 | 362 | 414 | 473.7 |
| fp16 fused, GEMM v1 | 28.8 | 137.9 | 251.6 | 322.6 | 375.6 | 406.1 | 431.3 |
| fp16 unfused, GEMM v1 | 55.6 | 135.1 | 251.6 | 323.9 | 368.7 | 399.5 | 425.8 |
| fp16 fused, GEMM v2 | 32.9 | 107 | 224.7 | 322.6 | 401 | 455.8 | 503.5 |
| fp16 unfused, GEMM v2 | 50.8 | 98.5 | 270.3 | 368.7 | 432.4 | 486.3 | 526.3 |
| fp16 fused, GEMM v2, transforms v2 | 65.4 | 125.8 | 285.7 | 394.1 | 454.5 | 509.6 | 547.5 |
| fp16 unfused, GEMM v2, transforms v2 | 29.2 | 108.7 | 208.3 | 313.7 | 387.4 | 450.1 | 499.6 |
| Implicit GEMM | 26.3 | 54.2 | 90.1 | 133.1 | 146.9 | 147.2 | 154.6 |
