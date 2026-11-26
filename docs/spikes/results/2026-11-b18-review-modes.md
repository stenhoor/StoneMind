# Spike 0001 phase 5: b18 review modes, browser (Edge, NVIDIA) and native KataGo

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Task: issue #63. Code: `main` at `1e7cd3b` (#62 merged;
`packages/engine-webgpu/src/search/review.ts` present).

Reports:
- [`2026-11-b18-review-modes-edge-nvidia.json`](2026-11-b18-review-modes-edge-nvidia.json): the browser.
- [`2026-11-b18-review-modes-native.json`](2026-11-b18-review-modes-native.json): native KataGo.

All values below are quoted from them, or from the KataGo log where stated.

## How it was run

- **Procedure:** the same as #53, with new output names.
- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/review-edge-nvidia.json`.
- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  - No flags were added.
  - **Search benchmark** was pressed over the DevTools Protocol. The run took from 10:41:02 to 10:44:00 UTC.
  - The instance was closed and the server stopped afterwards.
  - **GPU check:** `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at up to 90% SM.
- **Native:** after the browser run, not at the same time.
  - **Command:**
    ```bash
    node packages/engine-webgpu/scripts/native-search-bench.mjs --katago ~/katago/katago --config ~/katago/analysis_example.cfg --model ~/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz --report /tmp/review-edge-nvidia.json --out /tmp/review-native.json
    ```
  - **Config:** the same as #53. Its uncommented lines are in [`2026-11-b18-search-bench.md`](2026-11-b18-search-bench.md).
  - **GPU check:** `nvidia-smi pmon` showed `katago` on GPU 0, at up to 96% SM.
  - The script's stderr was empty.

## Browser report header

- **`userAgent`:** `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36 Edg/158.0.0.0`
- **`adapter`:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}`
- **`variant`:** `{"winograd":true,"fuse":true,"gemm":"v2","precision":"f16"}`
- **`ok`:** `true`
- **`deviceLost`:** not present
- **`date`:** `2026-11-26T10:41:02.963Z`
- Also: `gemmConfig: {"rm":4,"rn":8,"tk":16,"arith":"f32"}`.

## Browser live

Position `fight-19`, `maxVisits` 2000:

| `batch` | `visitsPerSecond` | `meanBatch` | `nnMs` | `cpuMs` |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 358.3 | 15.9 | 5022.5 | 558.7 |
| 32 | 478.8 | 31.8 | 3721.8 | 455.3 |
| 64 | 520.7 | 62.3 | 3570.4 | 270.6 |

## Browser review modes

`review`: `moves: 250`, `visitsPerMove: 100`, `rules: "tromp-taylor"`, `komi: 7.5`.

| `batchSize` | `concurrency` | `visits` | `totalMs` | `visitsPerSecond` | `minutesFor250Moves` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 25000 | 57475.6 | 435 | 0.96 |
| 16 | 1 | 25000 | 51287.7 | 487.4 | 0.85 |
| 16 | 4 | 25000 | 50929.9 | 490.9 | 0.85 |

## Native

From the KataGo log, as printed:

```
Cuda backend thread 0: Model version 14 useFP16 = true useNHWC = true
Analyzing up to 2 positions at a time in parallel
```

### Native live (`live.runs`)

| Run | `visitsPerSecond` |
| ---: | ---: |
| 1 | 1224 |
| 2 | 1185.4 |
| 3 | 1238.5 |

### Native review

| `visits` | `totalMs` | `visitsPerSecond` | `minutesFor250Moves` |
| ---: | ---: | ---: | ---: |
| 27596 | 16285 | 1694.6 | 0.27 |

### `browserReviewModes`

| `batchSize` | `concurrency` | `minutesFor250Moves` | `agreesWithNative` / `turns` |
| ---: | ---: | ---: | ---: |
| 64 | 1 | 0.96 | 165 / 250 (66%) |
| 16 | 1 | 0.85 | 174 / 250 (70%) |
| 16 | 4 | 0.85 | 179 / 250 (72%) |

## Review modes against native

| Browser mode | Browser minutes for 250 moves | Native minutes for 250 moves | Browser ÷ native | Agrees with native |
| --- | ---: | ---: | ---: | ---: |
| batch 64, 1 at once | 0.96 | 0.27 | 3.56 | 165 / 250 (66%) |
| batch 16, 1 at once | 0.85 | 0.27 | 3.15 | 174 / 250 (70%) |
| batch 16, 4 at once | 0.85 | 0.27 | 3.15 | 179 / 250 (72%) |

The ratios are computed from the rounded `minutesFor250Moves`. From `totalMs` instead: batch 64, 1 at once: 57475.6 ÷ 16285 = 3.53; batch 16, 1 at once: 51287.7 ÷ 16285 = 3.15; batch 16, 4 at once: 50929.9 ÷ 16285 = 3.13.
