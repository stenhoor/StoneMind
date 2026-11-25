# Spike 0001: b18 WebGPU GEMM v2 retune with packed f16 (`f16x2`) on Edge (NVIDIA)

Date: 2026-11-25. Machine: the owner's Linux laptop. Task: issue #41. Code: `main` at `a84ca02` (#40 merged).
Report: [`2026-11-b18-webgpu-edge-nvidia-tune-f16x2.json`](2026-11-b18-webgpu-edge-nvidia-tune-f16x2.json). All
values below are quoted from it.

## How it was run

- **Procedure:** the same as #38, with the output names changed.
- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/gpu-check-edge-nvidia-tune-f16x2.json`.
- **Browser:** the separate Edge instance with the launch used in #17, #19, #24, #27, #33 and #38:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  **Tune the GEMM** was pressed over the DevTools Protocol, and the instance was closed afterwards.
- **Report header:** `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`,
  `tuning: {"precision":"f16","batches":[16,64]}`, `date: 2026-11-25T22:45:46.618Z`, `goldensCompared: true`,
  `ok: true`.
- **Variants:** 24 distinct `gemmConfig`s, covering `arith` f32, f16 and f16x2 × `rm` 4/8 × `rn` 4/8 × `tk` 8/16.
  - Each also has `{"winograd":true,"fuse":true,"gemm":"v2","precision":"f16"}`.
  - None has a `failedAt` or `error`.
  - `profile` is `"skipped while tuning"` for every variant.
  - `batchOfOneMatchesBatched` is `true` for every variant.
- **GPU check:** during the run, `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at
  up to 96% SM.

## Variants

Sorted by evaluations per second at batch 64, fastest first:

| Rank | `gemmConfig` | `ok` | Positions with `matchesKataGo: true` | `worstFp16LimitRatio` | `fp16LimitsPass` | Eval/s batch 16 | Eval/s batch 64 |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | `{"rm":8,"rn":8,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 376.5 | 551.2 |
| 2 | `{"rm":8,"rn":8,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 426.7 | 546.1 |
| 3 | `{"rm":4,"rn":8,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 372.1 | 542.4 |
| 4 | `{"rm":4,"rn":8,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 365.3 | 541.5 |
| 5 | `{"rm":4,"rn":8,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 358.7 | 528.1 |
| 6 | `{"rm":4,"rn":8,"tk":16,"arith":"f16x2"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 360.4 | 527.6 |
| 7 | `{"rm":8,"rn":8,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 361.2 | 519.9 |
| 8 | `{"rm":8,"rn":8,"tk":16,"arith":"f16x2"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 325.9 | 519.1 |
| 9 | `{"rm":4,"rn":8,"tk":8,"arith":"f16x2"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 332.6 | 493.8 |
| 10 | `{"rm":4,"rn":8,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 347.1 | 492.7 |
| 11 | `{"rm":4,"rn":4,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 301.9 | 473.7 |
| 12 | `{"rm":4,"rn":4,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 329.9 | 472.7 |
| 13 | `{"rm":4,"rn":4,"tk":16,"arith":"f16x2"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 336.8 | 472.3 |
| 14 | `{"rm":8,"rn":4,"tk":16,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 328.5 | 466.5 |
| 15 | `{"rm":8,"rn":4,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 323.2 | 464.1 |
| 16 | `{"rm":8,"rn":4,"tk":16,"arith":"f16x2"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 320 | 462.1 |
| 17 | `{"rm":4,"rn":4,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 321.3 | 461.4 |
| 18 | `{"rm":4,"rn":4,"tk":8,"arith":"f16x2"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 318.1 | 460.1 |
| 19 | `{"rm":8,"rn":4,"tk":16,"arith":"f16"}` | `true` | 0 / 10 | 0.1030862351506369 | `true` | 321.3 | 456.8 |
| 20 | `{"rm":4,"rn":4,"tk":8,"arith":"f32"}` | `true` | 0 / 10 | 0.07957771519750967 | `true` | 374.7 | 440.2 |
| 21 | `{"rm":8,"rn":8,"tk":8,"arith":"f16x2"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 296.3 | 433.3 |
| 22 | `{"rm":8,"rn":8,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 292.5 | 427.8 |
| 23 | `{"rm":8,"rn":4,"tk":8,"arith":"f16"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 296.3 | 425.8 |
| 24 | `{"rm":8,"rn":4,"tk":8,"arith":"f16x2"}` | `true` | 0 / 10 | 0.09444647219756837 | `true` | 291.4 | 425 |

All 24 variants have `ok: true` and `fp16LimitsPass: true`, and none has a position with `matchesKataGo: true`.

## Fastest correct variant beside native `benchmarknn`

As in #38, "correct" here means `ok: true` and `fp16LimitsPass: true`. The native columns are `sumMedianNNEvalsPerSec`
from [`2026-11-b18-native-benchmarknn.md`](2026-11-b18-native-benchmarknn.md), measured at 18:19–18:21 UTC. This run
was at 22:45 UTC, so the two are not back to back. That file lists the measurement differences between the two tools.

Fastest variant with `ok: true` and `fp16LimitsPass: true`: `{"rm":8,"rn":8,"tk":16,"arith":"f32"}`.

| Batch | WebGPU eval/s | Native FP32 | WebGPU ÷ native FP32 | Native FP16 | WebGPU ÷ native FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 376.5 | 525.7 | 0.72 | 1291.2 | 0.29 |
| 64 | 551.2 | 543.2 | 1.01 | 1141.6 | 0.48 |
