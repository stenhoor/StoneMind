# Spike 0001: kernel experiments after GEMM v3 (b18) on Edge (NVIDIA)

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Code: branch `spike-0001/wino-transforms`, on top of
`main` at `80305c8` (GEMM v3 merged). This is the follow-up to [`2026-11-b18-gemm-v3.md`](2026-11-b18-gemm-v3.md),
looking for more speed in the Winograd transforms and in GEMM v3. **None of the kernel changes tried here was faster
than what is on `main`, so none is made the default.** All values below are quoted from the reports in this directory.

## How it was run

- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  No flags were added. Each run used a fresh instance, closed afterwards with `Browser.close`.
- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and goldens. One button was pressed per run, over the
  DevTools Protocol.
- **Adapter:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}` in every report, with f16
  storage and GPU timestamps.
- **GPU check:** `nvidia-smi pmon` showed Edge's GPU process on GPU 0 during each run.
- **Timing:** each figure is one compute pass of 30 identical dispatches after a 2-second warm-up, median of 7, per
  dispatch.

## Winograd transforms

### The committed code

Report: [`2026-11-b18-kernel-experiments-transforms-edge-nvidia.json`](2026-11-b18-kernel-experiments-transforms-edge-nvidia.json),
`date: 2026-11-26T13:45:06.875Z`, `ok: true`. Every version's output equals v1's.

**The ceiling:** a plain copy of the input transform's minimum bytes (board + V).

| Batch | Bytes | ms | GB/s |
| ---: | ---: | ---: | ---: |
| 16 | 7747584 | 0.024 | 322.4 |
| 32 | 15495168 | 0.0459 | 337.8 |
| 64 | 30990336 | 0.0918 | 337.8 |

**v1 and v2**, at b18's 192 channels. GB/s counts each transform's minimum bytes: the input transform reads the
board and writes V; the output transform reads M and the residual, and writes the board.

| Batch | Transform | v1 ms (GB/s) | v2 ms (GB/s) |
| ---: | --- | ---: | ---: |
| 16 | in | 0.0524 (147.8) | 0.0437 (177.3) |
| 16 | out | 0.0481 (207.4) | 0.0502 (198.3) |
| 32 | in | 0.0961 (161.2) | 0.0786 (197) |
| 32 | out | 0.0939 (212.2) | 0.0961 (207.4) |
| 64 | in | 0.3058 (101.3) | 0.2512 (123.4) |
| 64 | out | 0.2512 (158.7) | 0.2228 (178.9) |

In this run, the input transform's v2 was faster than v1 at all three batches. In an earlier run of the same kernels
(below), v2 was slower at batch 32 (0.1201 against 0.0939 ms). So the default stays v1.

### Versions tried and not kept

Their code isn't on the branch; the reports record their timings.

- **Vector access** to V and M, with 2 or 4 tiles per thread (`v3 tpt 2`, `v3 tpt 4`):
  [`2026-11-b18-kernel-experiments-transforms-vector-edge-nvidia.json`](2026-11-b18-kernel-experiments-transforms-vector-edge-nvidia.json),
  `date: 2026-11-26T13:39:17.340Z`.

| Batch | Transform | v1 ms (GB/s) | v2 ms (GB/s) | v3 tpt 2 ms (GB/s) | v3 tpt 4 ms (GB/s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | in | 0.0524 (147.8) | 0.0459 (168.9) | 0.1005 (77.1) | 0.1223 (63.3) |
| 16 | out | 0.0481 (207.4) | 0.0502 (198.3) | 0.0612 (162.9) | 0.0961 (103.7) |
| 32 | in | 0.0939 (165) | 0.0765 (202.7) | 0.4129 (37.5) | 0.3801 (40.8) |
| 32 | out | 0.1136 (175.5) | 0.1049 (190.1) | 0.1595 (125) | 0.3189 (62.5) |
| 64 | in | 0.2927 (105.9) | 0.2294 (135.1) | 0.6772 (45.8) | 0.6248 (49.6) |
| 64 | out | 0.1988 (200.5) | 0.2053 (194.1) | 0.2512 (158.7) | 0.4172 (95.5) |

- **Fully unrolled** (every array index a compile-time constant), with 1 or 2 tiles per thread (`v3 tpt 1`,
  `v3 tpt 2`):
  [`2026-11-b18-kernel-experiments-transforms-unrolled-edge-nvidia.json`](2026-11-b18-kernel-experiments-transforms-unrolled-edge-nvidia.json),
  `date: 2026-11-26T13:40:52.108Z`.

| Batch | Transform | v1 ms (GB/s) | v2 ms (GB/s) | v3 tpt 1 ms (GB/s) | v3 tpt 2 ms (GB/s) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | in | 0.0524 (147.8) | 0.0481 (161.2) | 0.0415 (186.7) | 0.0655 (118.2) |
| 16 | out | 0.0481 (207.4) | 0.0502 (198.3) | 0.0481 (207.4) | 0.0612 (162.9) |
| 32 | in | 0.0939 (165) | 0.1201 (129) | 0.1245 (124.4) | 0.2469 (62.8) |
| 32 | out | 0.1398 (142.6) | 0.1267 (157.3) | 0.142 (140.4) | 0.2206 (90.3) |
| 64 | in | 0.5003 (61.9) | 0.2403 (129) | 0.2141 (144.8) | 0.4653 (66.6) |
| 64 | out | 0.2053 (194.1) | 0.1988 (200.5) | 0.2053 (194.1) | 0.2971 (134.2) |

## GEMM v3 options

These use the GEMM benchmark, as in [`2026-11-b18-gemm-v3.md`](2026-11-b18-gemm-v3.md): `perNetwork` is each
kernel's time summed over b18's 62 + 18 + 18 GEMMs. Every v3 output equals v2's exactly, in both reports
(`ok: true`). The device's `maxComputeWorkgroupStorageSize` is 49152.

**Double-buffered workgroup memory, and chunks of 32:**
[`2026-11-b18-kernel-experiments-gemm-double-edge-nvidia.json`](2026-11-b18-kernel-experiments-gemm-double-edge-nvidia.json),
`date: 2026-11-26T13:42:26.687Z`.

| Kernel | Batch 16 (ms) | Batch 32 (ms) | Batch 64 (ms) |
| --- | ---: | ---: | ---: |
| v2 `{"rm":4,"rn":8,"tk":16,"arith":"f32"}` | 23.01 | 65.332 | 91.559 |
| v3 `{"rm":4,"rn":8,"tk":16}` | 15.952 | 24.942 | 48.28 |
| v3 `{"rm":4,"rn":4,"tk":16}` | 16.387 | 26.386 | 53.909 |
| v3 `{"rm":4,"rn":8,"tk":8}` | 16.54 | 27.623 | 50.802 |
| v3 `{"rm":4,"rn":8,"tk":32}` | 19.736 | 29.07 | 55.721 |
| v3 `{"rm":4,"rn":8,"tk":16,"double":true}` | 16.551 | 26.87 | 52.273 |
| v3 `{"rm":4,"rn":4,"tk":16,"double":true}` | 18.072 | 28.051 | 57.027 |
| v3 `{"rm":4,"rn":8,"tk":8,"double":true}` | 18.042 | 29.607 | 53.901 |
| v3 `{"rm":4,"rn":8,"tk":32,"double":true}` | 25.807 | 32.658 | 59.799 |

**Threads along M (`wy`):**
[`2026-11-b18-kernel-experiments-gemm-wy-edge-nvidia.json`](2026-11-b18-kernel-experiments-gemm-wy-edge-nvidia.json),
`date: 2026-11-26T13:43:51.056Z`.

| Kernel | Batch 16 (ms) | Batch 32 (ms) | Batch 64 (ms) |
| --- | ---: | ---: | ---: |
| v2 `{"rm":4,"rn":8,"tk":16,"arith":"f32"}` | 22.697 | 60.101 | 93.283 |
| v3 `{"rm":4,"rn":8,"tk":16,"wy":16}` | 15.655 | 26.421 | 49.163 |
| v3 `{"rm":8,"rn":8,"tk":16,"wy":8}` | 18.354 | 29.343 | 51.947 |
| v3 `{"rm":8,"rn":8,"tk":8,"wy":8}` | 18.52 | 29.695 | 56.102 |
| v3 `{"rm":8,"rn":4,"tk":16,"wy":8}` | 15.237 | 27.586 | 54.006 |
| v3 `{"rm":8,"rn":8,"tk":32,"wy":8}` | 18.891 | 37.568 | 61.145 |
| v3 `{"rm":4,"rn":8,"tk":16,"wy":8}` | 18.117 | 30.725 | 57.524 |
| v3 `{"rm":8,"rn":8,"tk":16,"wy":16}` | 19.696 | 31.722 | 55.861 |

In both runs, GEMM v3's default `{"rm":4,"rn":8,"tk":16}` (16 threads along M, one buffer) was fastest at batches 32 and
64. At batch 16, `{"rm":8,"rn":4,"tk":16,"wy":8}` took 15.237 ms against the default's 15.655 ms.
