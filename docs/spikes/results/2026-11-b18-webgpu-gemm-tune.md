# Spike 0001: b18 WebGPU GEMM v2 tuning on Edge (NVIDIA)

Date: 2026-11-25. Machine: the owner's Linux laptop. Task: issue #38. Code: `main` at `7302b44` (#37 merged).
Report: [`2026-11-b18-webgpu-edge-nvidia-tune.json`](2026-11-b18-webgpu-edge-nvidia-tune.json). All values below are
quoted from it.

## How it was run

- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/gpu-check-edge-nvidia-tune.json`.
- **Browser:** the separate Edge instance with the launch used in #17, #19, #24, #27 and #33:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  **Tune the GEMM** was pressed over the DevTools Protocol, and the instance was closed afterwards.
- **Report header:** `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`,
  `tuning: {"precision":"f16","batches":[16,64]}`, `date: 2026-11-25T22:27:23.131Z`, `goldensCompared: true`,
  `ok: true`.
- **Variants:** 16, covering `arith` f32 and f16 × `rm` 4/8 × `rn` 4/8 × `tk` 8/16.
  - Each also has `{"winograd":true,"fuse":true,"gemm":"v2","precision":"f16"}`.
  - None has a `failedAt` or `error`.
  - `profile` is `"skipped while tuning"` for every variant.
  - `batchOfOneMatchesBatched` is `true` for every variant.
- **GPU check:** during the run, `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at
  up to 90% SM.

## Variants

Sorted by evaluations per second at batch 64, fastest first:

| Rank | `gemmConfig` | `ok` | Positions with `matchesKataGo: true` | `worstFp16LimitRatio` | `fp16LimitsPass` | Eval/s batch 16 | Eval/s batch 64 |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | `{"rm":8,"rn":8,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 380 | 547.9 |
| 2 | `{"rm":8,"rn":8,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 368.7 | 545.1 |
| 3 | `{"rm":4,"rn":8,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 377.4 | 544.2 |
| 4 | `{"rm":4,"rn":8,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 364.5 | 540.5 |
| 5 | `{"rm":4,"rn":8,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 320.6 | 519.5 |
| 6 | `{"rm":8,"rn":8,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 358.7 | 518.6 |
| 7 | `{"rm":4,"rn":4,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 332.6 | 474.4 |
| 8 | `{"rm":4,"rn":4,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 300.2 | 464.8 |
| 9 | `{"rm":4,"rn":8,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 328.5 | 463.1 |
| 10 | `{"rm":8,"rn":4,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 320 | 462.4 |
| 11 | `{"rm":4,"rn":4,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 302.5 | 458.8 |
| 12 | `{"rm":8,"rn":4,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 319.4 | 458.5 |
| 13 | `{"rm":4,"rn":4,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 293 | 454.2 |
| 14 | `{"rm":8,"rn":4,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 323.9 | 444.1 |
| 15 | `{"rm":8,"rn":8,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 351.6 | 425.8 |
| 16 | `{"rm":8,"rn":4,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 303.6 | 425.5 |

All 16 variants have `ok: true` and `fp16LimitsPass: true`, and none has a position with `matchesKataGo: true`.

## Fastest correct variant beside native `benchmarknn`

Here "correct" means `ok: true` and `fp16LimitsPass: true`, which all 16 variants have. The native columns are
`sumMedianNNEvalsPerSec` from [`2026-11-b18-native-benchmarknn.md`](2026-11-b18-native-benchmarknn.md), measured at
18:19–18:21 UTC. This run was at 22:27 UTC, so the two are not back to back. That file lists the measurement
differences between the two tools.

Fastest variant with `ok: true` and `fp16LimitsPass: true`: `{"rm":8,"rn":8,"tk":16,"arith":"f32"}`.

| Batch | WebGPU eval/s | Native FP32 | WebGPU ÷ native FP32 | Native FP16 | WebGPU ÷ native FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 380 | 525.7 | 0.72 | 1291.2 | 0.29 |
| 64 | 547.9 | 543.2 | 1.01 | 1141.6 | 0.48 |
