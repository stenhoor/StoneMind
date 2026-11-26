# Spike 0001 phase 5: b18 top-move agreement, browser (Edge, NVIDIA) and native KataGo

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop. Task: issue #60. Code: `main` at `69cb60d` (#59 merged;
`packages/engine-webgpu/scripts/native-agreement.mjs` present).

Reports:
- [`2026-11-b18-agreement-edge-nvidia.json`](2026-11-b18-agreement-edge-nvidia.json): the browser.
- [`2026-11-b18-agreement-native.json`](2026-11-b18-agreement-native.json): native KataGo, including each position's
  moves in `perTurn`.

All values below are quoted from them, or from the KataGo log where stated.

## How it was run

- **Server:** `scripts/gpu-check.mjs --serve` with the b18 model and the goldens on `main`, writing
  `--out /tmp/agreement-edge-nvidia.json`.
- **Browser:** the separate Edge instance with the launch used in #48:
  `msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
  - No flags were added.
  - **Agreement positions** was pressed over the DevTools Protocol. The run took from 09:42:13 to 09:45:00 UTC.
  - The instance was closed and the server stopped afterwards.
  - **GPU check:** `nvidia-smi pmon` showed Edge's GPU process (`msedge --type=gpu-process`) on GPU 0, at up to 94% SM.
- **Native:** after the browser run, not at the same time.
  - **Command:**
    ```bash
    node packages/engine-webgpu/scripts/native-agreement.mjs --katago ~/katago/katago --config ~/katago/analysis_example.cfg --model ~/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz --report /tmp/agreement-edge-nvidia.json --out /tmp/agreement-native.json
    ```
  - **Config:** the same analysis config as #53. Its uncommented lines are in
    [`2026-11-b18-search-bench.md`](2026-11-b18-search-bench.md).
  - **GPU check:** `nvidia-smi pmon` showed `katago` on GPU 0, at up to 98% SM.
  - The script's stderr was empty.

## Browser report header

- **`userAgent`:** `Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36 Edg/158.0.0.0`
- **`adapter`:** `{"vendor":"nvidia","architecture":"ampere","device":"","description":""}`
- **`variant`:** `{"winograd":true,"fuse":true,"gemm":"v2","precision":"f16"}`
- **`ok`:** `true`
- **`deviceLost`:** not present
- **`date`:** `2026-11-26T09:42:13.381Z`
- **`visits`:** `[100,500]`
- **`batches`:** `[16,64]`
- Also: `game: {"rules":"tromp-taylor","komi":7.5,"moves":250}`, `every: 5`, 50 entries in `positions`.

## Native

- **`katagoVersion`:** `KataGo v1.18.2`
- **`config`:** `/home/stenhoor/katago/analysis_example.cfg`
- **`referenceVisits`:** `4000`

From the KataGo log, as printed:

```
Cuda backend thread 0: Model version 14 useFP16 = true useNHWC = true
Analyzing up to 2 positions at a time in parallel
```

## Agreement

The table shows, for each `visits` × `browserBatch`, the count of positions where the two moves agree, out of
`positions`, with the percentage.
- **Native runs:** `a` and `b` are the two native runs at that visit count. The `nativeVsNative` and
  `nativeVsReference` columns come from the same native runs for both browser batches.
- **Reference:** native at `referenceVisits` (4000).

| `visits` | `browserBatch` | `positions` | `browserVsNative.a` | `browserVsNative.b` | `nativeVsNative` | `browserVsReference` | `nativeVsReference.a` | `nativeVsReference.b` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 16 | 50 | 38 (76%) | 35 (70%) | 41 (82%) | 29 (58%) | 32 (64%) | 31 (62%) |
| 100 | 64 | 50 | 28 (56%) | 28 (56%) | 41 (82%) | 26 (52%) | 32 (64%) | 31 (62%) |
| 500 | 16 | 50 | 33 (66%) | 30 (60%) | 43 (86%) | 31 (62%) | 32 (64%) | 35 (70%) |
| 500 | 64 | 50 | 25 (50%) | 25 (50%) | 43 (86%) | 24 (48%) | 32 (64%) | 35 (70%) |
