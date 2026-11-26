# Spike 0001: two network calls in flight (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Code: branch `spike-0001/overlap-network-calls`, on top of
`main` at `da35a08`. This is speed step 2 from the handoff (#68): `GpuNetwork` allows `maxInFlight` calls at once
(default 2), and `SharedEvaluator` spreads requests across them. All values below are quoted from the reports in this
directory.

## How it was run

- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  No flags were added.
- **Three runs**, one after another in that instance, served by `scripts/gpu-check.mjs --serve` with the b18 model and
  goldens:
  1. **Check correctness:** [`2026-11-b18-inflight-check-edge-nvidia.json`](2026-11-b18-inflight-check-edge-nvidia.json),
     `date: 2026-11-26T12:16:51.404Z`.
  2. **Search benchmark, default (`maxInFlight` 2):**
     [`2026-11-b18-inflight-search-edge-nvidia.json`](2026-11-b18-inflight-search-edge-nvidia.json),
     `date: 2026-11-26T12:17:25.823Z`.
  3. **Search benchmark with `maxInFlight` 1:**
     [`2026-11-b18-inflight-search-edge-nvidia-maxinflight1.json`](2026-11-b18-inflight-search-edge-nvidia-maxinflight1.json),
     `date: 2026-11-26T12:21:07.068Z`. This used the page's
     `?searchVariant={"winograd":true,"fuse":true,"gemm":"v2","precision":"f16","maxInFlight":1}`, so its calls run one
     at a time, as before this change.
- **Each report:** `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`, `ok: true`,
  and no `deviceLost`.
- **GPU check:** `nvidia-smi pmon` showed Edge's GPU process on GPU 0 during the runs. `kwin_wayland` also showed SM use
  on GPU 0, as in #67.
- **Native:** `scripts/native-search-bench.mjs` with `~/katago/katago` and `~/katago/analysis_example.cfg`, run once
  against each search report after the browser runs. Reports:
  [`2026-11-b18-inflight-native.json`](2026-11-b18-inflight-native.json) and
  [`2026-11-b18-inflight-native-maxinflight1.json`](2026-11-b18-inflight-native-maxinflight1.json).
  `nvidia-smi pmon` showed `katago` on GPU 0 during the first native run; the second wasn't sampled.

## Correctness

Every one of the 14 variants has `ok: true`, `maxInFlight: 2` and `concurrentCallsMatchBatched: true`. The positions,
split across three calls made at once, gave exactly the batched call's outputs.

## Review

Review modes are 250 turns × 100 visits. The agreement column counts turns where the browser's move matches native's,
from the native run against that report.

| Review mode | `totalMs`, maxInFlight 2 | `totalMs`, maxInFlight 1 | 1 ÷ 2 | `minutesFor250Moves`, 2 / 1 | Agrees with native, 2 / 1 |
| --- | ---: | ---: | ---: | --- | --- |
| batch 64, 1 at once | 58703.9 | 59082 | 1.01 | 0.98 / 0.98 | 176/250 / 167/250 |
| batch 16, 1 at once | 52185.5 | 51905.6 | 0.99 | 0.87 / 0.87 | 182/250 / 174/250 |
| batch 16, 2 at once | 44666.2 | 50370.5 | 1.13 | 0.74 / 0.84 | 183/250 / 174/250 |
| batch 16, 4 at once | 43557.6 | 52451.9 | 1.20 | 0.73 / 0.87 | 187/250 / 178/250 |

The modes with one turn at a time have a single search, so their calls never overlap. Their `totalMs` differ by 1% or
less between the two runs.

## Live

The live search is one search, so its calls never overlap either.

| Live batch | `visitsPerSecond`, maxInFlight 2 | `visitsPerSecond`, maxInFlight 1 |
| ---: | ---: | ---: |
| 16 | 387.2 | 343 |
| 32 | 344.1 | 455.3 |
| 64 | 477.6 | 473.6 |

## Native

| Native run (against report) | `review.totalMs` | `minutesFor250Moves` | `live.runs` `visitsPerSecond` |
| --- | ---: | ---: | --- |
| maxInFlight 2 report | 17375.8 | 0.29 | 1136.5, 1185.8, 1107.2 |
| maxInFlight 1 report | 17298.9 | 0.29 | 1121.1, 1137.1, 1087.3 |
