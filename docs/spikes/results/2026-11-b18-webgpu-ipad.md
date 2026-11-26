# Spike 0001: b18 on the owner's iPad (Safari, Apple GPU)

Date: 2026-11-25. Device: the owner's iPad Pro (2026), Safari 26.6.1. Page: the published GPU check,
https://stenhoor.github.io/StoneMind/gpu-check/, with the b18c384nbt model picked from the iPad's files.

The owner sent these runs as page summaries in the conversation; no report files were attached. Values are copied
from those summaries. Evaluations per second are at batch 64.

## Device

The page reported `adapter: {"vendor":"apple","architecture":"apple","device":"apple","description":"apple"}`
and the features `shader-f16` and `timestamp-query`. It did not report `subgroups`. The user agent reads
`Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) … Version/26.6.1 Safari/605.1.15`, which can't distinguish an
iPad from a Mac; the device is the owner's iPad, on their word.

## Correctness (first run, 20:30 UTC)

All four fp32 variants then on the page matched native KataGo's goldens on 10 of 10 positions, and both fp16
variants passed KataGo's fp16 limits:
- worst ratio 0.077 fused, 0.080 unfused;
- the activation check within 1.6e-7.

## Check and benchmark: GEMM v1 against v2

Two runs on the same page build (before Winograd transforms v2), about 40 minutes apart:

| Variant | Run 1 | Run 2 |
| --- | ---: | ---: |
| GEMM v1, fp32, Winograd fused | 20.2 | 20.2 |
| GEMM v1, fp32, unfused | 20.1 | 20.0 |
| GEMM v2, fp32, fused | 101.1 | 105.3 |
| GEMM v2, fp32, unfused | 86.1 | 93.0 |
| GEMM v2, fp32 + scale8 | 91.7 | 89.5 |
| GEMM v1, fp16 storage, fused | 20.5 | 20.3 |
| GEMM v1, fp16 storage, unfused | 19.7 | 20.4 |
| GEMM v2, fp16 storage, fused | 90.8 | 99.4 |
| GEMM v2, fp16 storage, unfused | 91.4 | 98.0 |

In both runs, the implicit-GEMM variant (`winograd: false`, `gemm: 'v1'`), which runs last, lost the device:
`DEVICE LOST: {"reason":"unknown","message":""}`, then `OperationError: map async was not successful`. An earlier
run, before the page recorded device loss, failed at the same variant while profiling.

## Tune the GEMM (24 shapes, fp16 storage, Winograd fused)

All 24 passed (`ok: true`).

| `rm` × `rn`, `tk` | f32 arithmetic | f16 arithmetic | f16x2 arithmetic |
| --- | ---: | ---: | ---: |
| 4 × 4, 8 | 116.6 | **124.8** | 110.5 |
| 4 × 4, 16 | 113.5 | 123.3 | 102.7 |
| 4 × 8, 8 | 111.5 | 122.8 | 110.7 |
| 4 × 8, 16 (the default then) | 112.9 | 117.2 | 111.7 |
| 8 × 4, 8 | 99.1 | 103.2 | 96.5 |
| 8 × 4, 16 | 95.1 | 104.6 | 99.2 |
| 8 × 8, 8 | 82.4 | 93.0 | 88.5 |
| 8 × 8, 16 | 80.4 | 98.6 | 94.3 |

Each value comes from a single run.

## Check and benchmark with the per-device GEMM default (after PR #46)

These are evaluations per second at batch 64, all Winograd, from a single run. The default GEMM shape was used, which is
4 × 4, chunk 8.

| Variant | Fused | Unfused | `ok` |
| --- | ---: | ---: | --- |
| GEMM v1, f32 | 20.2 | 20.1 | true |
| GEMM v2, f32 | 111.7 | 108.3 | **false** |
| GEMM v2, f32, transforms v2 | 96.7 | 92.8 | **false** |
| GEMM v2, f32, scale8 | 95.8 | | **false** |
| GEMM v1, fp16 storage | 20.4 | 20.4 | true |
| GEMM v2, fp16 storage | **124.8** | 108.5 | true |
| GEMM v2, fp16 storage, transforms v2 | 113.9 | 103.9 | true |

- **The f32 failures are a bug in PR #46.** For f32 networks too, `defaultGemmConfig` chose f16 arithmetic whenever the
  device had `shader-f16`, so the f32 variants missed the f32 goldens tolerances. The fix gives f16 arithmetic only to
  fp16 storage. The f32 numbers above are therefore f16-arithmetic numbers.
- **Transforms v2 are slower on this device:** 113.9 against 124.8 fused, and 103.9 against 108.5 unfused, with fp16
  storage.
- **The implicit GEMM lost the device again,** as in the earlier runs.

## Not available

- **Native comparison:** there's no native KataGo number for this device. KataGo's command line can't run on iPadOS.
- **Profiles and report files:** the summaries sent don't include per-kernel profiles or the downloadable report.
