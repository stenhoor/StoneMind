# Spike 0001: the parts of a network call (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Task: issue #67. Code: `main` at `da35a08` (#66 merged;
`packages/engine-webgpu/src/gpu/network.ts` has `lastCallTiming`).
Report: [`2026-11-b18-call-timing-edge-nvidia.json`](2026-11-b18-call-timing-edge-nvidia.json). All values below are
quoted from it.

## How it was run

- **Procedure:** the same as #48, with a new output name.
- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/call-timing-edge-nvidia.json`.
- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  - No flags were added.
  - **Check and benchmark** was pressed over the DevTools Protocol.
  - The instance was closed and the server stopped afterwards.
- **Report header:** `adapter: {"vendor":"nvidia","architecture":"ampere","device":"","description":""}`,
  `date: 2026-11-26T12:01:04.263Z`, `ok: true`. There is no `deviceLost`.
- **GPU check:** `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at up to 99% SM.
  - The desktop compositor `kwin_wayland` also showed SM use on GPU 0 during this run, at up to 31%, in 63 one-second
    samples from 06:01:03 to 06:02:38 local time.
  - In the pmon logs of #57, #60 and #63, `kwin_wayland` showed no SM use on GPU 0.
- **How `callTiming` is recorded** (from `test/gpu/check.js` and `GpuNetwork.lastCallTiming`):
  - At each batch, after one warm-up call, each of the 5 timed calls records its parts.
  - Each `callTiming` field is the median of that field across the 5 calls, so the fields need not add up to `totalMs`.
  - `waitMs` is the wait until the outputs can be read: "GPU work plus the readback's latency", as `network.ts`
    documents it.

## Results

### fp16 fused, GEMM v2: `{"winograd":true,"fuse":true,"precision":"f16","gemm":"v2"}`

`ok`: `true`.

| Batch | `evalsPerSecond` | `medianMs` | `packMs` | `writeMs` | `encodeMs` | `submitMs` | `waitMs` | `readMs` | `unpackMs` | `totalMs` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.6 | 25.9 | 0 | 0 | 0 | 0 | 25.5 | 0.3 | 0 | 25.9 |
| 2 | 143.9 | 13.9 | 0 | 0 | 0.1 | 0 | 13.4 | 0.2 | 0 | 13.9 |
| 4 | 254.8 | 15.7 | 0.1 | 0 | 0.1 | 0 | 15.2 | 0.2 | 0 | 15.7 |
| 8 | 343.3 | 23.3 | 0.3 | 0.1 | 0.2 | 0 | 22.5 | 0.3 | 0.1 | 23.3 |
| 16 | 432.4 | 37 | 0.4 | 0.1 | 0.1 | 0 | 36.1 | 0.2 | 0.1 | 37 |
| 32 | 469.2 | 68.2 | 0.8 | 0.2 | 0.1 | 0 | 66.5 | 0.2 | 0.2 | 68.2 |
| 64 | 505.5 | 126.6 | 1.5 | 0.5 | 0.1 | 0 | 123.6 | 0.3 | 0.4 | 126.5 |

`profile.rows`:

| Batch | `wallMsMedian` | `gpuPassMsMedian` |
| ---: | ---: | ---: |
| 1 | 15.2 | 10.813 |
| 16 | 44.6 | 39.584 |
| 64 | 147.3 | 139.788 |

### fp32 fused, GEMM v2: `{"winograd":true,"fuse":true,"gemm":"v2"}`

`ok`: `true`.

| Batch | `evalsPerSecond` | `medianMs` | `packMs` | `writeMs` | `encodeMs` | `submitMs` | `waitMs` | `readMs` | `unpackMs` | `totalMs` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 61.7 | 16.2 | 0.1 | 0.1 | 0.4 | 0 | 15.4 | 0.1 | 0.1 | 16.2 |
| 2 | 117.6 | 17 | 0.2 | 0.1 | 0.3 | 0 | 16.1 | 0.4 | 0 | 17 |
| 4 | 210.5 | 19 | 0.3 | 0.1 | 0.3 | 0.1 | 18.1 | 0.3 | 0.1 | 19 |
| 8 | 295.2 | 27.1 | 0.6 | 0.2 | 0.2 | 0.1 | 25.3 | 0.3 | 0.2 | 27.1 |
| 16 | 350.9 | 45.6 | 1.3 | 0.4 | 0.3 | 0 | 42.8 | 0.4 | 0.3 | 45.6 |
| 32 | 389.3 | 82.2 | 2.7 | 0.8 | 0.3 | 0.1 | 77.1 | 0.3 | 0.6 | 82.1 |
| 64 | 448.8 | 142.6 | 1.7 | 0.5 | 0.2 | 0 | 137.6 | 0.1 | 0.4 | 142.6 |

`profile.rows`:

| Batch | `wallMsMedian` | `gpuPassMsMedian` |
| ---: | ---: | ---: |
| 1 | 15.4 | 10.289 |
| 16 | 43.9 | 38.797 |
| 64 | 141.7 | 134.873 |
