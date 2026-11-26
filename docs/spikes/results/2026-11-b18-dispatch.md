# Spike 0001: what a dispatch costs (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Code: branch `spike-0001/dispatch-gaps`, on top of
`spike-0001/kernel-experiments` (#74). The question: how much of a b18 evaluation goes on gaps between its 338
dispatches, which would decide whether fusing kernels to cut the dispatch count is worth doing. All values below are
quoted from the reports in this directory.

## How it was run

- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  No flags were added. Each run used a fresh instance.
- **Page:** the GPU check page's "Dispatch benchmark" button (`test/gpu/dispatch-bench.js`), served by
  `scripts/gpu-check.mjs --serve` with the b18 model.
- **Adapter:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}`, with f16 storage.
- **Reports:**
  - [`2026-11-b18-dispatch-edge-nvidia.json`](2026-11-b18-dispatch-edge-nvidia.json), `date: 2026-11-26T13:51:57.685Z`.
  - [`2026-11-b18-dispatch-edge-nvidia-run2.json`](2026-11-b18-dispatch-edge-nvidia-run2.json), `date: 2026-11-26T13:51:15.588Z`,
    the run before it.
- **GPU check:** `nvidia-smi pmon` showed Edge's GPU process on GPU 0.
- **GPU clocks:** during the first report's run, a read-only
  `nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader -lms 250`
  was logged to [`2026-11-b18-dispatch-edge-nvidia-clocks.csv`](2026-11-b18-dispatch-edge-nvidia-clocks.csv). No GPU
  setting was changed.

## Fixed cost per dispatch

A near-empty kernel (one workgroup adding 1 to 64 values), dispatched 500 times in one pass. "Dependent" dispatches
all write one buffer, so each must wait for the one before it. "Independent" dispatches each write their own buffer.
Each figure is the median of 7 passes, divided by 500.

| Report | Dependent (ms per dispatch) | Independent (ms per dispatch) |
| --- | ---: | ---: |
| first | 0.00144 | 0.00144 |
| run 2 | 0.00131 | 0.00131 |

338 dispatches × 0.00144 ms = 0.49 ms per evaluation, at
any batch size.

## On a real kernel

GEMM v3 (the default shape) at b18's Winograd GEMM shape (3×3 192→192), 30 dispatches in one pass. Dependent
dispatches write one output; independent ones each write their own.

| Batch | Dependent (ms per dispatch) | Independent (ms per dispatch) |
| ---: | ---: | ---: |
| 16 | 0.1835 | 0.1813 |
| 32 | 0.3146 | 0.3124 |
| 64 | 0.5418 | 0.5461 |

## The network

The b18 plan's single pass, median of 7, for each batch. Each batch ran on a plan built for exactly that batch, and
on one built for batch 64 (as the search pages build it).

| Batch | Plan `maxBatch` | Dispatches | GPU pass (ms) |
| ---: | ---: | ---: | ---: |
| 16 | 16 | 338 | 30.802 |
| 16 | 64 | 338 | 32.637 |
| 32 | 32 | 338 | 51.118 |
| 32 | 64 | 338 | 49.611 |
| 64 | 64 | 338 | 92.144 |

These passes are longer than the 25.0 ms at batch 16 in `2026-11-b18-gemm-v3.md`'s profile.

## GPU clocks

Samples every 250 ms, 97 in all:
- **Busy:** in the 18 samples with `utilization.gpu` at least 50%, the SM clock was 420–1830 MHz.
- **Idle:** in the 49 samples whose `clocks_event_reasons.active` was `0x0000000000000001` (`gpu_idle` in
  `nvidia-smi --help-query-gpu`), the SM clock was 225–1800 MHz.
- **Other reasons:** `0x0000000000000004` (`sw_power_cap`) appeared in 4 samples, at 64.42–99.31 W.
  `0x0000000000000400` also appeared; `nvidia-smi --help-query-gpu` on this machine doesn't list that bit.
- **What this means for the measurements:** the benchmark pages wait for each pass before submitting the next, so the
  GPU goes idle between passes and its clocks change during a run.
