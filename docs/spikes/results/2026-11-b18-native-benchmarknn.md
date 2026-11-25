# Spike 0001: native KataGo forward-pass throughput by batch size (b18, `benchmarknn`)

Date: 2026-11-25. Machine: the owner's Linux laptop. Task: issue #20. Raw output:
[`2026-11-b18-native-benchmarknn.json`](2026-11-b18-native-benchmarknn.json), which holds all 14 JSON lines and the
`nvidia-smi` output.

## Setup

- **GPU:** NVIDIA GeForce RTX 3080 Laptop GPU, driver 620.73.04 (`nvidia-smi --query-gpu=name,driver_version --format=csv`).
- **KataGo:** v1.18.2, CUDA backend, compiled with CUDA 13.5.37. Every JSON line reports `revision` as
  `d91ea855110dae533f0aada947b2b7d78cc8a4e1-cuda`.
- **Network:** `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`. The config is `~/katago/gtp.cfg`, unchanged.
- **Command:** run from `~/katago`, once per batch size N in 1, 2, 4, 8, 16, 32 and 64, first with `useFP16=false` and
  then with `useFP16=true`:
  ```bash
  ./katago benchmarknn -model models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz -config gtp.cfg -batch-size <N> -boardsize 19 -json -override-config "cudaDeviceToUse=0, numNNServerThreadsPerModel=1, useFP16=<false|true>"
  ```
- **Iterations:** the defaults, 200 timed after 20 warm-up. Every JSON line reports `numIterations: 200`.
- **How the runs went:** one command at a time, from 12:19:48 to 12:21:29 local time (UTC−6), that is 18:19:48 to
  18:21:29 UTC. No WebGPU page was open, and `nvidia-smi` listed only `kwin_wayland` and Steam's helpers on GPU 0.
  All 14 commands exited 0.
- **Checks:** every line has `numThreads: 1`, `gpuIdxs: [0]` and `batchSize` equal to N. `usingFP16` is `false` for
  the FP32 runs and `true` for the FP16 runs. With `useFP16=true`, KataGo's log shows
  `Model version 14 useFP16 = true useNHWC = true`.

## Native results

`sumMedianNNEvalsPerSec`, and `perThreadMedianMs` for the single thread:

| Batch | FP32 evals/s | FP32 median ms | FP16 evals/s | FP16 median ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 207.5 | 4.82 | 359.9 | 2.78 |
| 2 | 376.4 | 5.31 | 719.3 | 2.78 |
| 4 | 461.4 | 8.67 | 937.8 | 4.27 |
| 8 | 536.5 | 14.91 | 996.7 | 8.03 |
| 16 | 525.7 | 30.43 | 1291.2 | 12.39 |
| 32 | 527.5 | 60.66 | 1114.4 | 28.71 |
| 64 | 543.2 | 117.82 | 1141.6 | 56.06 |

## Comparison with WebGPU on Edge 158 (NVIDIA)

The WebGPU figures are evaluations per second for Edge 158 on the NVIDIA GPU. They are copied from
[`2026-11-b18-webgpu.md`](2026-11-b18-webgpu.md), whose report is
[`2026-11-b18-webgpu-edge-nvidia.json`](2026-11-b18-webgpu-edge-nvidia.json). Each WebGPU cell gives the figure, then
in brackets WebGPU ÷ native FP32 and WebGPU ÷ native FP16. The native columns are `sumMedianNNEvalsPerSec`.

| Batch | Native FP32 | Native FP16 | Winograd fused (÷FP32, ÷FP16) | Winograd unfused (÷FP32, ÷FP16) | Implicit GEMM (÷FP32, ÷FP16) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 207.5 | 359.9 | 61 (0.29, 0.17) | 77.5 (0.37, 0.22) | 25.2 (0.12, 0.07) |
| 2 | 376.4 | 719.3 | 150.4 (0.40, 0.21) | 143.9 (0.38, 0.20) | 49.8 (0.13, 0.07) |
| 4 | 461.4 | 937.8 | 232.6 (0.50, 0.25) | 233.9 (0.51, 0.25) | 85.3 (0.18, 0.09) |
| 8 | 536.5 | 996.7 | 268.5 (0.50, 0.27) | 292 (0.54, 0.29) | 124.4 (0.23, 0.12) |
| 16 | 525.7 | 1291.2 | 313.7 (0.60, 0.24) | 315 (0.60, 0.24) | 135.9 (0.26, 0.11) |
| 32 | 527.5 | 1114.4 | 335.1 (0.64, 0.30) | 336.5 (0.64, 0.30) | 145.5 (0.28, 0.13) |
| 64 | 543.2 | 1141.6 | 365.1 (0.67, 0.32) | 365.9 (0.67, 0.32) | 155.2 (0.29, 0.14) |

## Measurement differences

These are what the sources show:

- **What is timed.**
  - **KataGo:** per its `benchmarknn -help`, "Timing includes host-device transfers and output postprocessing, excludes
    input feature generation and search."
  - **WebGPU:** the benchmark times `GpuNetwork.evaluate` (`packages/engine-webgpu/src/gpu/network.ts`). That covers
    packing the inputs, uploading them, running the network and reading the outputs back, including cropping the
    raw outputs per position. Post-processing (`postProcess`) is a separate call that the benchmark doesn't time
    (`test/gpu/check.js`).
- **Repetitions.** The WebGPU figures are medians of 5 timed runs after 1 warm-up (`runs = 5` in `test/gpu/check.js`).
  KataGo's are medians of 200 after 20.
- **Precision.** The WebGPU kernels are fp32: the header comment of `packages/engine-webgpu/src/gpu/kernels.ts` says
  "WGSL kernels for the network, fp32". The native table has both FP32 and FP16.
- **Timing of the runs.** The WebGPU Edge run was at 17:41:50 UTC, from its report's `date`, about 38 minutes before
  these native runs. This is not a back-to-back comparison.
