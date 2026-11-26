# Spike 0001: onnxruntime-web baseline (b18) on Edge (NVIDIA), against the WebGPU engine

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Task: issue #57 (pass mark 1). Code: `main` at `32d1e09`
(#56 merged; `apps/engine-bench/ort-bench/` present).

Reports:
- [`2026-11-b18-ort-baseline-edge-nvidia.json`](2026-11-b18-ort-baseline-edge-nvidia.json): onnxruntime-web.
- [`2026-11-b18-webgpu-edge-nvidia-features.json`](2026-11-b18-webgpu-edge-nvidia-features.json): the WebGPU engine,
  from #48 / PR #49.

All values below are quoted from them.

## How it was run

- **ONNX graph:** fetched from `claude/spike-0001-b18-onnx-temp` (commit `9f3afbe`) into `/tmp/b18-onnx`, outside the
  working tree, and joined from its three parts. Its SHA-256 is
  `0f1e77308f993c1dec735984255851d32ba6c0aa97b5ab1bff013746cccb48e8`, as required. Size: 105514063 bytes.
- **Build:** `npm install`, then `npm run build` for `@stonemind/engine-webgpu` and `@stonemind/engine-bench`. The
  installed `onnxruntime-web` is 1.31.1. The page loads it from `dist/site/ort-bench/vendor/`, which the build copies
  from `node_modules`.
- **Server:** `REPORT_OUT=/tmp/ort-bench-edge-nvidia.json PORT=8080 node apps/engine-bench/scripts/serve.mjs`.
- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  - No flags were added.
  - `http://localhost:8080/ort-bench/` was opened over the DevTools Protocol.
  - The `.onnx` file and `~/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` were set on the two file
    inputs (`DOM.setFileInputFiles`), and **Check and benchmark** was pressed.
  - The run took from 02:30:07 to 02:31:04 UTC. The instance was closed and the server stopped afterwards.
- **GPU check:** `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0 during the WebGPU
  part, at up to 95% SM. Per the report's `steps`, the WebGPU bench ran from 02:30:09.722 to 02:30:14.353 UTC. No
  process showed SM use on the NVIDIA GPU during the WASM part (02:30:14–02:31:02 UTC).

## Report header

- **`userAgent`:** `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36 Edg/158.0.0.0`
- **`crossOriginIsolated`:** `true`
- **`hardwareConcurrency`:** `16`
- **`ok`:** `true`
- **`date`:** `2026-11-26T02:30:07.563Z`
- Also: `model: "kata1-b18c384nbt-s9996604416-d4316597426"`, `onnxBytes: 105514063`, `goldensCompared: true`.

## Providers

### `webgpu`

- **`executionProvider`:** `webgpu`
- **`ortVersion`:** `1.31.1`
- **`ok`:** `true`
- **`sessionCreateMs`:** 1171.2
- **`wasmNumThreads`:** 4
- **`error` / `failedAt`:** none
- **`benchPosition`:** `empty-19`

| Position | `matchesKataGo` | First `failures` |
| --- | --- | --- |
| `empty-19` | `true` | none |
| `opening-19` | `true` | none |
| `opening-19-japanese` | `true` | none |
| `fight-19` | `true` | none |
| `ladder-19` | `true` | none |
| `pass-19` | `true` | none |

| Batch | `medianMs` | `evalsPerSecond` |
| ---: | ---: | ---: |
| 1 | 24.62 | 40.6 |
| 2 | 25.67 | 77.9 |
| 4 | 29.78 | 134.3 |
| 8 | 51.24 | 156.1 |
| 16 | 94.53 | 169.2 |
| 32 | 177.17 | 180.6 |
| 64 | 341.68 | 187.3 |

### `wasm`

- **`executionProvider`:** `wasm`
- **`ortVersion`:** `1.31.1`
- **`ok`:** `true`
- **`sessionCreateMs`:** 342.8
- **`wasmNumThreads`:** 4
- **`error` / `failedAt`:** none
- **`benchPosition`:** `empty-19`

| Position | `matchesKataGo` | First `failures` |
| --- | --- | --- |
| `empty-19` | `true` | none |
| `opening-19` | `true` | none |
| `opening-19-japanese` | `true` | none |
| `fight-19` | `true` | none |
| `ladder-19` | `true` | none |
| `pass-19` | `true` | none |

| Batch | `medianMs` | `evalsPerSecond` |
| ---: | ---: | ---: |
| 1 | 258.78 | 3.9 |
| 2 | 767.35 | 2.6 |
| 4 | 940.98 | 4.3 |
| 8 | 1887.47 | 4.2 |
| 16 | 3793.88 | 4.2 |

## Side by side

**Engine figures:** from #49's `2026-11-b18-webgpu-edge-nvidia-features.json`, recorded at
`2026-11-26T00:20:13.762Z`. The two runs were about two hours 10 minutes apart, not back to back.

**Bench position:** both pages bench the first 19×19 golden position, `empty-19`, repeated to the batch size.

**Correctness differences:**
- onnxruntime-web is checked on the 6 19×19 goldens. The engine is checked on its own 10 positions, 5 of them 19×19.
- onnxruntime-web WASM was measured at batches 1–16 only, the page's default.

**"Fastest correct" at each batch**, two ways:
- **Fastest with `ok: true`:** all 14 engine variants have `ok: true` and `fp16LimitsPass: true`. This is the
  definition used in #38 and #41.
- **Fastest matching KataGo on all positions:** the 8 engine variants with `matchesKataGo: true` on all 10 positions.
  These are the fp32 variants and implicit GEMM.

onnxruntime-web matched KataGo on all 6 of its positions with both providers.

### Engine ÷ onnxruntime-web `webgpu`

| Batch | onnxruntime-web `webgpu` | Engine, fastest with `ok: true` | Variant | Engine ÷ ORT | Engine, fastest matching KataGo on all positions | Variant | Engine ÷ ORT |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 40.6 | 87 | fp32 unfused, GEMM v1 | 2.14 | 87 | fp32 unfused, GEMM v1 | 2.14 |
| 2 | 77.9 | 157.5 | fp32 unfused, GEMM v2 | 2.02 | 157.5 | fp32 unfused, GEMM v2 | 2.02 |
| 4 | 134.3 | 285.7 | fp16 fused, GEMM v2, transforms v2 | 2.13 | 236.7 | fp32 fused, GEMM v2 | 1.76 |
| 8 | 156.1 | 394.1 | fp16 fused, GEMM v2, transforms v2 | 2.52 | 315 | fp32 fused, GEMM v2, transforms v2 | 2.02 |
| 16 | 169.2 | 454.5 | fp16 fused, GEMM v2, transforms v2 | 2.69 | 404 | fp32 fused, GEMM v2 | 2.39 |
| 32 | 180.6 | 509.6 | fp16 fused, GEMM v2, transforms v2 | 2.82 | 451.3 | fp32 fused, GEMM v2 | 2.50 |
| 64 | 187.3 | 547.5 | fp16 fused, GEMM v2, transforms v2 | 2.92 | 478.3 | fp32 fused, GEMM v2, transforms v2 | 2.55 |

### Engine ÷ onnxruntime-web `wasm`

| Batch | onnxruntime-web `wasm` | Engine, fastest with `ok: true` | Variant | Engine ÷ ORT | Engine, fastest matching KataGo on all positions | Variant | Engine ÷ ORT |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 3.9 | 87 | fp32 unfused, GEMM v1 | 22.31 | 87 | fp32 unfused, GEMM v1 | 22.31 |
| 2 | 2.6 | 157.5 | fp32 unfused, GEMM v2 | 60.58 | 157.5 | fp32 unfused, GEMM v2 | 60.58 |
| 4 | 4.3 | 285.7 | fp16 fused, GEMM v2, transforms v2 | 66.44 | 236.7 | fp32 fused, GEMM v2 | 55.05 |
| 8 | 4.2 | 394.1 | fp16 fused, GEMM v2, transforms v2 | 93.83 | 315 | fp32 fused, GEMM v2, transforms v2 | 75.00 |
| 16 | 4.2 | 454.5 | fp16 fused, GEMM v2, transforms v2 | 108.21 | 404 | fp32 fused, GEMM v2 | 96.19 |

Engine variant names used above:

| Name used here | `variant` |
| --- | --- |
| fp32 fused, GEMM v1 | `{"winograd":true,"fuse":true,"gemm":"v1"}` |
| fp32 unfused, GEMM v1 | `{"winograd":true,"fuse":false,"gemm":"v1"}` |
| fp32 fused, GEMM v2 | `{"winograd":true,"fuse":true,"gemm":"v2"}` |
| fp32 unfused, GEMM v2 | `{"winograd":true,"fuse":false,"gemm":"v2"}` |
| fp32 fused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":true,"gemm":"v2","winoTransforms":"v2"}` |
| fp32 unfused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":false,"gemm":"v2","winoTransforms":"v2"}` |
| fp32 + scale8 fused, GEMM v2 | `{"winograd":true,"fuse":true,"scale8":true,"gemm":"v2"}` |
| fp16 fused, GEMM v1 | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v1"}` |
| fp16 unfused, GEMM v1 | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v1"}` |
| fp16 fused, GEMM v2 | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v2"}` |
| fp16 unfused, GEMM v2 | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v2"}` |
| fp16 fused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v2","winoTransforms":"v2"}` |
| fp16 unfused, GEMM v2, transforms v2 | `{"winograd":true,"fuse":false,"precision":"f16","gemm":"v2","winoTransforms":"v2"}` |
| Implicit GEMM | `{"winograd":false,"gemm":"v1"}` |
