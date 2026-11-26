# Spike 0001 phase 5: b18 search benchmark, browser (Edge, NVIDIA) and native KataGo

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Task: issue #53. Code: `main` at `e7370ed` (#52 merged;
`packages/engine-webgpu/scripts/native-search-bench.mjs` present).

Reports:
- [`2026-11-b18-search-bench-edge-nvidia.json`](2026-11-b18-search-bench-edge-nvidia.json): the browser.
- [`2026-11-b18-search-bench-native.json`](2026-11-b18-search-bench-native.json): native KataGo.

All values below are quoted from them, or from the KataGo log where stated.

## How it was run

- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/search-bench-edge-nvidia.json`.
- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  - No flags were added.
  - **Search benchmark** was pressed over the DevTools Protocol. The run took 80 seconds, from 02:03:19 to
    02:04:39 UTC.
  - The instance was closed and the server stopped afterwards.
  - **GPU check:** `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at up to 94% SM.
- **Native:** after the browser run, not at the same time.
  - **Command:**
    ```bash
    node packages/engine-webgpu/scripts/native-search-bench.mjs --katago ~/katago/katago --config ~/katago/analysis_example.cfg --model ~/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz --report /tmp/search-bench-edge-nvidia.json --out /tmp/search-bench-native.json
    ```
  - **Paths used:** `/home/stenhoor/katago/katago` and `/home/stenhoor/katago/analysis_example.cfg`, as in the issue.
  - **GPU check:** `nvidia-smi pmon` showed `katago` on GPU 0, at up to 96% SM.

## Browser report header

- **`userAgent`:** `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36 Edg/158.0.0.0`
- **`adapter`:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}`
- **`features`:** `["shader-f16","subgroups","timestamp-query"]`
- **`variant`:** `{"winograd":true,"fuse":true,"gemm":"v2","precision":"f16"}`
- **`gemmConfig`:** `{"rm":4,"rn":8,"tk":16,"arith":"f32"}`
- **`ok`:** `true`
- **`deviceLost`:** not present
- **`date`:** `2026-11-26T02:03:19.474Z`

## Browser live

Position `fight-19` (`tromp-taylor`, komi 7.5), `maxVisits` 2000:

| `batch` | `visits` | `ms` | `visitsPerSecond` | `meanBatch` | `nnMs` | `cpuMs` | `cacheHits` | `bestMove` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 2000 | 4974.8 | 402 | 15.9 | 4251.4 | 723.4 | 166 | B7 |
| 32 | 2000 | 5193.6 | 385.1 | 31.8 | 4743.1 | 450.5 | 152 | B7 |
| 64 | 2000 | 4059.7 | 492.6 | 62.3 | 3670.1 | 389.6 | 67 | B7 |

## Browser review

| `moves` | `visits` | `totalMs` | `visitsPerSecond` | `minutesFor250Moves` |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 25000 | 61833.5 | 404.3 | 1.03 |

Also in the report: `rules: "tromp-taylor"`, `komi: 7.5`, `visitsPerMove: 100`, `batch: 64`.

## Native

- **`katagoVersion`:** `KataGo v1.18.2`
- **`config`:** `/home/stenhoor/katago/analysis_example.cfg`
- **`override`:** `null`

Uncommented lines of the analysis config used (`grep -v '^\s*#' ~/katago/analysis_example.cfg | grep -v '^\s*$'`):

```
logDir = analysis_logs  # Each run of KataGo will log to a separate file in this dir
reportAnalysisWinratesAs = BLACK
maxVisits = 500
numAnalysisThreads = 2
numSearchThreadsPerAnalysisThread = 16
nnMaxBatchSize = 64
nnCacheSizePowerOfTwo = 23
nnMutexPoolSizePowerOfTwo = 17
nnRandomize = true
```

The script adds `logDir=<temp dir>` as a command-line override. The KataGo log printed this config block at startup:

```
logDir = /tmp/katago-logs-te1wiC
maxVisits = 500
nnCacheSizePowerOfTwo = 23
nnMaxBatchSize = 64
nnMutexPoolSizePowerOfTwo = 17
nnRandomize = true
numAnalysisThreads = 2
numSearchThreads = 16
reportAnalysisWinratesAs = BLACK
```

### Native startup

From the KataGo log (`/tmp/katago-logs-te1wiC/20261125-200457-B89B29D1.log`), as printed:

```
2026-11-25 20:04:57-0600: After dedups: nnModelFile0 = /home/stenhoor/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz useFP16 auto
2026-11-25 20:04:58-0600: Cuda backend thread 0: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 20:04:58-0600: Cuda backend thread 0: Model version 14 useFP16 = true useNHWC = true
2026-11-25 20:04:58-0600: CUDA backend: running 1x1 NHWC convolutions as a GEMM
2026-11-25 20:04:58-0600: Analyzing up to 2 positions at a time in parallel
```

The script's stderr was empty.

### Native live

Position `fight-19`, `maxVisits` 2000, 3 runs:

| Run | `visits` | `ms` | `visitsPerSecond` | `bestMove` |
| ---: | ---: | ---: | ---: | --- |
| 1 | 2015 | 1729.6 | 1165 | B7 |
| 2 | 2010 | 1625.6 | 1236.5 | B7 |
| 3 | 2014 | 1635.2 | 1231.7 | B7 |

### Native review

| `moves` | `visits` | `totalMs` | `visitsPerSecond` | `minutesFor250Moves` |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 27565 | 16340.2 | 1686.9 | 0.27 |

Also in the report: `visitsPerMove: 100`.

The KataGo log at shutdown, as printed:

```
2026-11-25 20:05:21-0600: NN rows: 23662
2026-11-25 20:05:21-0600: NN batches: 2573
2026-11-25 20:05:21-0600: NN avg batch size: 9.19627
```

## Side by side

### Live

| Native run | Browser live, batch 64 (`visitsPerSecond`) | Native live (`visitsPerSecond`) | Browser ÷ native |
| ---: | ---: | ---: | ---: |
| 1 | 492.6 | 1165 | 0.42 |
| 2 | 492.6 | 1236.5 | 0.40 |
| 3 | 492.6 | 1231.7 | 0.40 |

### Review

| Browser review (`minutesFor250Moves`) | Native review (`minutesFor250Moves`) | Browser ÷ native |
| ---: | ---: | ---: |
| 1.03 | 0.27 | 3.81 |

From `totalMs` instead of the rounded minutes: 61833.5 ÷ 16340.2 = 3.78 (both reviews are 250 moves).

## Pass marks from #53

- **Live:** at least 200 visits/s with b18. The browser's live runs gave 385.1–492.6 visits/s.
- **Review:** a full review in under 5 minutes. The browser review's `minutesFor250Moves` is 1.03.
