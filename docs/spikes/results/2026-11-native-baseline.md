# Spike 0001: native KataGo baseline and bridge check

Date: 2026-11-25. Machine: the owner's Linux laptop. Spike: [0001](../0001-browser-katago-webgpu.md), phase 1
([testing steps](../0001-phase1-testing.md)).

Everything here was measured or read off the machine in this session. Anything not confirmed is marked
**Assumption**.

## 1. Setup

| Item | Value | Source |
| --- | --- | --- |
| OS | CachyOS (rolling, Arch-based), kernel `7.3.5-2-cachyos` x86_64 | `/etc/os-release`, `uname -a` |
| CPU / RAM | Intel Core i9-11900H @ 2.50GHz, 31 GiB | `/proc/cpuinfo`, `free -g` |
| GPU used | GPU 0: NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB, compute capability 8.6, VBIOS 94.04.55.00.AA | `nvidia-smi`, KataGo log |
| Other GPU | GPU 1: Intel (integrated). Not used. **Assumption:** taken from the owner's brief. `nvidia-smi` lists NVIDIA devices only. | owner |
| NVIDIA driver | 620.73.04 (KMD 620.73.04, CUDA UMD 13.5) | `nvidia-smi` |
| KataGo | v1.18.2, git `d91ea855110dae533f0aada947b2b7d78cc8a4e1`, compiled 2026-11-22 05:12:25 | `katago version` |
| Backend | **CUDA**, compiled with CUDA 13.5.37. Not TensorRT. | `katago version` |
| Network | `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz` (97,898,094 bytes) | see below |
| Network SHA-256 | `9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d` | `sha256sum` |
| Network as KataGo reports it | model version 14, nbt convnet, 26,326,985 params | KataGo log |
| GTP config | `~/katago/gtp.cfg` (the brief said `gtp.config`; the file on disk is `gtp.cfg`) | `ls ~/katago` |
| Analysis config | `cpp/configs/analysis_example.cfg` at tag `v1.18.2`, unchanged, SHA-256 `44d3782d8802697f9d0aaf3f2c2cb8bbabed1f6796ed0858a32868cccae34889` | raw.githubusercontent.com/lightvector/KataGo/v1.18.2 |
| Node.js | v26.13.1 (npm 12.2.3) | `node --version` |
| StoneMind | `main` at `3add8f9` (phase 1, PR #5, is already merged into `main`) | `git log` |
| Browsers | Microsoft Edge 158.0.4370.33 (Chromium-based); Mozilla Firefox 160.0.1 (UA `rv:160.0`) | `msedge --version`, `firefox --version` |

How the network was chosen: `~/katago/models/` had no b18c384nbt file. It held only
`kata1-tf3-b11c768-s12382M-d6533M.bin.gz`. With the owner's approval, the newest kata1-b18c384nbt on
https://katagotraining.org/networks/ was downloaded. It was uploaded 2024-05-26 12:47:48 UTC, and no b18c384nbt row is
newer in the page's date-sorted table. Its URL is
`https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`.
`gzip -t` passes. This file is the spike's reference net.

`~/katago` had no `analysis_example.cfg`, so it was fetched from the KataGo repository at the tag matching
`katago version` (`v1.18.2`).

Other load on GPU 0 during the runs: the desktop compositor (`kwin_wayland`) and Steam's web helpers were resident
(about 235 MiB). `nvidia-smi` showed 20% utilisation before the benchmarks started.

<details><summary><code>nvidia-smi</code> (before the benchmarks)</summary>

```
Wed Nov 25 09:17:37 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 620.73.04              KMD Version: 620.73.04     CUDA UMD Version: 13.5     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   46C    P8             16W /   90W |     397MiB /  16384MiB |     20%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1394    C+G   /usr/bin/kwin_wayland                    17MiB |
|    0   N/A  N/A           17944      G   ...l/share/Steam/steamrt64/steam          6MiB |
|    0   N/A  N/A           17966      G   ./steamwebhelper                         96MiB |
|    0   N/A  N/A           17994      G   ...team/steamrt64/steamwebhelper        116MiB |
+-----------------------------------------------------------------------------------------+
```

</details>

## 2. Native benchmark (`katago benchmark`, 800 visits, GPU 0)

Commands, run from `~/katago` one after the other:

```bash
./katago benchmark -model models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz -config gtp.cfg -v 800 -override-config "cudaDeviceToUse=0"
./katago benchmark -model models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz -config gtp.cfg -v 800 -override-config "cudaDeviceToUse=0, useFP16=false"
```

The default run resolved `useFP16 auto` to `useFP16 = true useNHWC = true`. The forced run reported
`useFP16 = false useNHWC = false`. Both quotes come from KataGo's log.

| numSearchThreads | FP16 (default) visits/s | FP16 nnEvals/s | FP32 (forced) visits/s | FP32 nnEvals/s |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 945.29 | 814.43 | 481.66 | 414.74 |
| 8 | — | — | 522.65 | 443.57 |
| 10 | 1228.22 | 1061.50 | 577.12 | 482.01 |
| 12 | 1313.92 | 1107.82 | **604.52** | 500.07 |
| 16 | 1333.23 | 1142.00 | 589.58 | 503.72 |
| 20 | **1402.00** | 1219.49 | 599.03 | 519.32 |
| 24 | 1389.56 | 1231.74 | — | — |
| 32 | 1316.79 | 1175.40 | — | — |

KataGo recommended 20 threads for FP16 and 12 for FP32 (see the raw output). A dash means KataGo's thread search did
not test that count.

## 3. Bridge and bench page

The bridge ran from `main` (`3add8f9`). The testing doc's `git switch claude/spike-0001-phase1` step is out of date,
because that branch is merged.

```bash
npm install && npm run build
npm run serve -w @stonemind/engine-bench
node packages/katago-bridge/dist/cli.js --katago ~/katago/katago \
  --model ~/katago/models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
  --config ~/katago/analysis_example.cfg --page http://localhost:8080/engine-bench/
```

KataGo printed `Started, ready to begin handling requests` about 2 seconds after launch (09:26:57 → 09:26:59). It
used `useFP16 = true`, `nnMaxBatchSize = 64`, `numAnalysisThreads = 2`, `numSearchThreads = 16` (the config's
`numSearchThreadsPerAnalysisThread`) and `nnCacheSizePowerOfTwo = 23`.

The bridge was **restarted between the two browsers**. Each browser's first Opening search therefore started with an
empty KataGo NN cache (see the cache note below).

### How each browser was driven

- **Edge** was driven by Claude through the Claude in Chrome extension. The extension runs in Edge on this machine,
  and **Google Chrome is not installed**. With the owner's agreement, the Chromium row is Edge. **Chrome itself was not
  tested.**
- **Firefox** was the owner's normal profile, relaunched with `firefox --remote-debugging-port 9222`. A Node script
  then drove it over Firefox's built-in WebDriver BiDi. Controls were set through the DOM (`select.value`,
  `form.requestSubmit()`, `button.click()`), not with real mouse clicks.

### Checklist

| Step | Edge 158 | Firefox 160 |
| --- | --- | --- |
| 5.1 Header says **Connected to KataGo on this computer**; panel shows version and network | Pass: `1.18.2`, `kata1-b18c384nbt-s9996604416-d4316597426`, max batch 64, FP16 `auto` | Pass: same four values |
| 5.2 Address bar no longer shows `#bridge=…&token=…` | Pass: URL became `http://localhost:8080/engine-bench/` | Pass: `location.href` has no `token=`, URL `http://localhost:8080/engine-bench/` |
| 5.3 Opening, 2000 visits: table fills and the visit count climbs during the search | Pass: a mid-search screenshot showed 35 visits at 0.30 s with the table filled; it ended at 2,015 visits | Pass: visits seen as `99, 357, 634, 910, 1,186, 1,450, 1,734, 2,015` |
| 5.4 Stop during a 100000-visit search: stops within about a second, results stay | Pass: stopped at 7,327 visits / 5.59 s, unchanged 3 s later, table kept. The first capture after Stop already showed the final values. Exact latency not measured. | Pass: page showed stopped **108 ms** after Stop (7,753 → 8,024 visits), unchanged 2 s later, 10 table rows kept |
| Local-network permission prompt | None seen. **Limit:** screenshots cover the page only, not the browser's own UI. The connection succeeded with no input from me. | None seen. **Limit:** BiDi cannot see browser UI. The page reported Connected 0.3 s after navigation with no input. |
| 6. Wrong token refused | Pass: page showed "the local bridge refused the connection or is not running"; bridge logged `refused a connection from http://localhost:8080: wrong or missing pairing token` | Pass: same message and same bridge log line |
| 6. `http://127.0.0.1:6565/` | Pass: only `This is the StoneMind KataGo bridge. Connect with a WebSocket from an allowed page.` (HTTP 426, `text/plain`, also checked with curl) | Pass: same text |

### Visits/s: first 2000-visit "Opening: four star points" search

The page measures this from the engine's reported visits and the wall-clock time, including the bridge.

| Browser | Visits | Time | **Visits/s** |
| --- | ---: | ---: | ---: |
| Edge 158 | 2,015 | 2.15 s | **938** |
| Firefox 160 | 2,015 | 2.09 s | **965** |

For reference, the native benchmark gave 1333.23 visits/s at 16 threads with FP16. The two numbers are not measured
the same way:
- The benchmark spreads 800 visits over each of 10 positions and sweeps thread counts.
- The page runs one 2000-visit search through the bridge.

**NN cache affects repeat runs.** Repeating the same 2000-visit Opening search in Edge, without restarting the bridge,
gave **5,596 visits/s** (2,015 visits in 0.36 s). KataGo's NN cache holds results between queries, so only a
first search on a fresh bridge is a fair figure. The 100000-visit Stop runs above also began on a cached position.

## 4. Other things the bridge or setup showed

- **The token is not one-time in the code.** The bridge prints "The link holds a one-time token". But
  `packages/katago-bridge/src/bridge.ts` compares every connection against the same token for the bridge's lifetime, and
  allows several clients at once. This was checked in the code, not tested by reconnecting.
- **KataGo writes `analysis_logs/` into the bridge's working directory.** That is the StoneMind repo root when you
  follow the testing doc. `analysis_example.cfg` sets `logDir = analysis_logs`. The files are ignored by `.gitignore`'s
  `*.log` rule, so they don't show in `git status`.
- `npm install` warned that `esbuild@0.28.2`'s postinstall script was blocked (not covered by `allowScripts`).
  `npm run build` still succeeded.
- The bridge printed no errors. KataGo exited with code 0 both times it was stopped.

## 5. Raw benchmark output

KataGo redraws its progress lines with carriage returns (`\r`). They are shown here as line breaks. Nothing else was
changed.

### Run 1: default precision (`cudaDeviceToUse=0`)

<details><summary>Full output</summary>

```
2026-11-25 09:23:39-0600: Running with following config:
allowResignation = true
cudaDeviceToUse = 0
friendlyPassOk = true
hasButton = false
koRule = SITUATIONAL
lagBuffer = 1.0
logAllGTPCommunication = true
logDir = gtp_logs
logSearchInfo = true
logSearchInfoForChosenMove = false
logToStderr = false
multiStoneSuicideLegal = false
nnCacheSizePowerOfTwo = 20
nnMutexPoolSizePowerOfTwo = 16
numNNServerThreadsPerModel = 2
numSearchThreads = 12
ponderingEnabled = false
resignConsecTurns = 3
resignThreshold = -0.90
scoringRule = AREA
searchFactorAfterOnePass = 0.50
searchFactorAfterTwoPass = 0.25
searchFactorWhenWinning = 0.40
searchFactorWhenWinningThreshold = 0.95
taxRule = NONE
whiteHandicapBonus = N-1

2026-11-25 09:23:39-0600: Loading model and initializing benchmark...
2026-11-25 09:23:39-0600: Testing with default positions for board size: 19
2026-11-25 09:23:39-0600: nnRandSeed0 = 7009100423055246983
2026-11-25 09:23:39-0600: After dedups: nnModelFile0 = models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz useFP16 auto
2026-11-25 09:23:39-0600: Initializing neural net buffer to be size 19 * 19 exactly
2026-11-25 09:23:40-0600: Cuda backend thread 0: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:23:40-0600: Cuda backend thread 0: Model version 14 useFP16 = true useNHWC = true
2026-11-25 09:23:40-0600: Cuda backend thread 0: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)
2026-11-25 09:23:40-0600: Cuda backend thread 1: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:23:40-0600: Cuda backend thread 1: Model version 14 useFP16 = true useNHWC = true
2026-11-25 09:23:40-0600: Cuda backend thread 1: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)
2026-11-25 09:23:40-0600: CUDA backend: running 1x1 NHWC convolutions as a GEMM
2026-11-25 09:23:40-0600: CUDA backend: running 1x1 NHWC convolutions as a GEMM

2026-11-25 09:23:41-0600: Loaded config gtp.cfg and/or command-line and query overrides
2026-11-25 09:23:41-0600: Loaded model models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

Testing using 800 visits.
  If you have a good GPU, you might increase this using "-visits N" to get more accurate results.
  If you have a weak GPU and this is taking forever, you can decrease it instead to finish the benchmark faster.

Your GTP config is currently set to cudaUseFP16 = auto

Your GTP config is currently set to use numSearchThreads = 12
Automatically trying different numbers of threads to home in on the best (board size 19x19):

2026-11-25 09:23:41-0600: GPU 0 finishing, processed 1 rows 1 batches
2026-11-25 09:23:41-0600: GPU 0 finishing, processed 4 rows 4 batches
2026-11-25 09:23:41-0600: nnRandSeed0 = 4758837202314423253
2026-11-25 09:23:41-0600: After dedups: nnModelFile0 = models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz useFP16 auto
2026-11-25 09:23:41-0600: Initializing neural net buffer to be size 19 * 19 exactly
2026-11-25 09:23:42-0600: Cuda backend thread 0: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:23:42-0600: Cuda backend thread 0: Model version 14 useFP16 = true useNHWC = true
2026-11-25 09:23:42-0600: Cuda backend thread 0: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)
2026-11-25 09:23:42-0600: Cuda backend thread 1: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:23:42-0600: Cuda backend thread 1: Model version 14 useFP16 = true useNHWC = true
2026-11-25 09:23:42-0600: Cuda backend thread 1: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)
2026-11-25 09:23:42-0600: CUDA backend: running 1x1 NHWC convolutions as a GEMM
2026-11-25 09:23:42-0600: CUDA backend: running 1x1 NHWC convolutions as a GEMM


Possible numbers of threads to test: 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32,


numSearchThreads =  5: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads =  5: 1 / 10 positions, visits/s = 976.79 (0.8 secs)
numSearchThreads =  5: 2 / 10 positions, visits/s = 896.90 (1.8 secs)
numSearchThreads =  5: 3 / 10 positions, visits/s = 879.22 (2.7 secs)
numSearchThreads =  5: 4 / 10 positions, visits/s = 906.81 (3.5 secs)
numSearchThreads =  5: 5 / 10 positions, visits/s = 904.13 (4.4 secs)
numSearchThreads =  5: 6 / 10 positions, visits/s = 897.68 (5.4 secs)
numSearchThreads =  5: 7 / 10 positions, visits/s = 914.21 (6.2 secs)
numSearchThreads =  5: 8 / 10 positions, visits/s = 915.79 (7.0 secs)
numSearchThreads =  5: 9 / 10 positions, visits/s = 928.18 (7.8 secs)
numSearchThreads =  5: 10 / 10 positions, visits/s = 945.29 nnEvals/s = 814.43 nnBatches/s = 525.08 avgBatchSize = 1.55 (8.5 secs)

numSearchThreads = 12: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 12: 1 / 10 positions, visits/s = 1614.30 (0.5 secs)
numSearchThreads = 12: 2 / 10 positions, visits/s = 1299.78 (1.2 secs)
numSearchThreads = 12: 3 / 10 positions, visits/s = 1241.74 (2.0 secs)
numSearchThreads = 12: 4 / 10 positions, visits/s = 1250.76 (2.6 secs)
numSearchThreads = 12: 5 / 10 positions, visits/s = 1290.99 (3.1 secs)
numSearchThreads = 12: 6 / 10 positions, visits/s = 1269.63 (3.8 secs)
numSearchThreads = 12: 7 / 10 positions, visits/s = 1275.97 (4.4 secs)
numSearchThreads = 12: 8 / 10 positions, visits/s = 1288.25 (5.0 secs)
numSearchThreads = 12: 9 / 10 positions, visits/s = 1308.50 (5.6 secs)
numSearchThreads = 12: 10 / 10 positions, visits/s = 1313.92 nnEvals/s = 1107.82 nnBatches/s = 299.60 avgBatchSize = 3.70 (6.2 secs)

numSearchThreads = 10: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 10: 1 / 10 positions, visits/s = 1607.87 (0.5 secs)
numSearchThreads = 10: 2 / 10 positions, visits/s = 1284.90 (1.3 secs)
numSearchThreads = 10: 3 / 10 positions, visits/s = 1206.31 (2.0 secs)
numSearchThreads = 10: 4 / 10 positions, visits/s = 1219.28 (2.7 secs)
numSearchThreads = 10: 5 / 10 positions, visits/s = 1188.18 (3.4 secs)
numSearchThreads = 10: 6 / 10 positions, visits/s = 1178.25 (4.1 secs)
numSearchThreads = 10: 7 / 10 positions, visits/s = 1189.48 (4.8 secs)
numSearchThreads = 10: 8 / 10 positions, visits/s = 1188.95 (5.4 secs)
numSearchThreads = 10: 9 / 10 positions, visits/s = 1211.04 (6.0 secs)
numSearchThreads = 10: 10 / 10 positions, visits/s = 1228.22 nnEvals/s = 1061.50 nnBatches/s = 360.46 avgBatchSize = 2.94 (6.6 secs)

numSearchThreads = 20: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 20: 1 / 10 positions, visits/s = 1587.71 (0.5 secs)
numSearchThreads = 20: 2 / 10 positions, visits/s = 1402.88 (1.2 secs)
numSearchThreads = 20: 3 / 10 positions, visits/s = 1350.21 (1.8 secs)
numSearchThreads = 20: 4 / 10 positions, visits/s = 1354.10 (2.4 secs)
numSearchThreads = 20: 5 / 10 positions, visits/s = 1356.97 (3.0 secs)
numSearchThreads = 20: 6 / 10 positions, visits/s = 1341.08 (3.7 secs)
numSearchThreads = 20: 7 / 10 positions, visits/s = 1350.17 (4.2 secs)
numSearchThreads = 20: 8 / 10 positions, visits/s = 1361.99 (4.8 secs)
numSearchThreads = 20: 9 / 10 positions, visits/s = 1380.79 (5.3 secs)
numSearchThreads = 20: 10 / 10 positions, visits/s = 1402.00 nnEvals/s = 1219.49 nnBatches/s = 207.70 avgBatchSize = 5.87 (5.8 secs)

numSearchThreads = 16: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 16: 1 / 10 positions, visits/s = 1633.54 (0.5 secs)
numSearchThreads = 16: 2 / 10 positions, visits/s = 1370.26 (1.2 secs)
numSearchThreads = 16: 3 / 10 positions, visits/s = 1307.41 (1.9 secs)
numSearchThreads = 16: 4 / 10 positions, visits/s = 1302.02 (2.5 secs)
numSearchThreads = 16: 5 / 10 positions, visits/s = 1312.10 (3.1 secs)
numSearchThreads = 16: 6 / 10 positions, visits/s = 1278.76 (3.8 secs)
numSearchThreads = 16: 7 / 10 positions, visits/s = 1296.10 (4.4 secs)
numSearchThreads = 16: 8 / 10 positions, visits/s = 1300.35 (5.0 secs)
numSearchThreads = 16: 9 / 10 positions, visits/s = 1322.26 (5.5 secs)
numSearchThreads = 16: 10 / 10 positions, visits/s = 1333.23 nnEvals/s = 1142.00 nnBatches/s = 253.40 avgBatchSize = 4.51 (6.1 secs)

numSearchThreads = 24: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 24: 1 / 10 positions, visits/s = 1660.33 (0.5 secs)
numSearchThreads = 24: 2 / 10 positions, visits/s = 1421.42 (1.2 secs)
numSearchThreads = 24: 3 / 10 positions, visits/s = 1360.11 (1.8 secs)
numSearchThreads = 24: 4 / 10 positions, visits/s = 1348.56 (2.4 secs)
numSearchThreads = 24: 5 / 10 positions, visits/s = 1350.23 (3.0 secs)
numSearchThreads = 24: 6 / 10 positions, visits/s = 1326.72 (3.7 secs)
numSearchThreads = 24: 7 / 10 positions, visits/s = 1340.60 (4.3 secs)
numSearchThreads = 24: 8 / 10 positions, visits/s = 1341.35 (4.9 secs)
numSearchThreads = 24: 9 / 10 positions, visits/s = 1375.14 (5.4 secs)
numSearchThreads = 24: 10 / 10 positions, visits/s = 1389.56 nnEvals/s = 1231.74 nnBatches/s = 172.68 avgBatchSize = 7.13 (5.9 secs)

numSearchThreads = 32: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 32: 1 / 10 positions, visits/s = 1538.56 (0.5 secs)
numSearchThreads = 32: 2 / 10 positions, visits/s = 1311.34 (1.3 secs)
numSearchThreads = 32: 3 / 10 positions, visits/s = 1275.89 (2.0 secs)
numSearchThreads = 32: 4 / 10 positions, visits/s = 1259.70 (2.6 secs)
numSearchThreads = 32: 5 / 10 positions, visits/s = 1272.46 (3.3 secs)
numSearchThreads = 32: 6 / 10 positions, visits/s = 1258.75 (4.0 secs)
numSearchThreads = 32: 7 / 10 positions, visits/s = 1276.19 (4.6 secs)
numSearchThreads = 32: 8 / 10 positions, visits/s = 1281.22 (5.2 secs)
numSearchThreads = 32: 9 / 10 positions, visits/s = 1303.54 (5.7 secs)
numSearchThreads = 32: 10 / 10 positions, visits/s = 1316.79 nnEvals/s = 1175.40 nnBatches/s = 139.65 avgBatchSize = 8.42 (6.3 secs)


Ordered summary of results:

numSearchThreads =  5: 10 / 10 positions, visits/s = 945.29 nnEvals/s = 814.43 nnBatches/s = 525.08 avgBatchSize = 1.55 (8.5 secs) (EloDiff baseline)
numSearchThreads = 10: 10 / 10 positions, visits/s = 1228.22 nnEvals/s = 1061.50 nnBatches/s = 360.46 avgBatchSize = 2.94 (6.6 secs) (EloDiff +87)
numSearchThreads = 12: 10 / 10 positions, visits/s = 1313.92 nnEvals/s = 1107.82 nnBatches/s = 299.60 avgBatchSize = 3.70 (6.2 secs) (EloDiff +108)
numSearchThreads = 16: 10 / 10 positions, visits/s = 1333.23 nnEvals/s = 1142.00 nnBatches/s = 253.40 avgBatchSize = 4.51 (6.1 secs) (EloDiff +106)
numSearchThreads = 20: 10 / 10 positions, visits/s = 1402.00 nnEvals/s = 1219.49 nnBatches/s = 207.70 avgBatchSize = 5.87 (5.8 secs) (EloDiff +118)
numSearchThreads = 24: 10 / 10 positions, visits/s = 1389.56 nnEvals/s = 1231.74 nnBatches/s = 172.68 avgBatchSize = 7.13 (5.9 secs) (EloDiff +107)
numSearchThreads = 32: 10 / 10 positions, visits/s = 1316.79 nnEvals/s = 1175.40 nnBatches/s = 139.65 avgBatchSize = 8.42 (6.3 secs) (EloDiff +71)


Based on some test data, each speed doubling gains perhaps ~250 Elo by searching deeper.
Based on some test data, each thread costs perhaps 7 Elo if using 800 visits, and 2 Elo if using 5000 visits (by making MCTS worse).
So APPROXIMATELY based on this benchmark, if you intend to do a 5 second search:
numSearchThreads =  5: (baseline)
numSearchThreads = 10:   +87 Elo
numSearchThreads = 12:  +108 Elo
numSearchThreads = 16:  +106 Elo
numSearchThreads = 20:  +118 Elo (recommended)
numSearchThreads = 24:  +107 Elo
numSearchThreads = 32:   +71 Elo

If you care about performance, you may want to edit numSearchThreads in gtp.cfg and/or command-line and query overrides based on the above results!
If you intend to do much longer searches, configure the seconds per game move you expect with the '-time' flag and benchmark again.
If you intend to do short or fixed-visit searches, use lower numSearchThreads for better strength, high threads will weaken strength.
If interested see also other notes about performance and mem usage in the top of gtp.cfg and/or command-line and query overrides

Running additional tests of a few other settings at numSearchThreads = 20.
Re-measuring the current recommendation as a baseline:

numSearchThreads = 20: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 20: 1 / 10 positions, visits/s = 1600.23 (0.5 secs)
numSearchThreads = 20: 2 / 10 positions, visits/s = 1395.25 (1.2 secs)
numSearchThreads = 20: 3 / 10 positions, visits/s = 1336.94 (1.8 secs)
numSearchThreads = 20: 4 / 10 positions, visits/s = 1326.57 (2.5 secs)
numSearchThreads = 20: 5 / 10 positions, visits/s = 1325.55 (3.1 secs)
numSearchThreads = 20: 6 / 10 positions, visits/s = 1310.33 (3.7 secs)
numSearchThreads = 20: 7 / 10 positions, visits/s = 1332.39 (4.3 secs)
numSearchThreads = 20: 8 / 10 positions, visits/s = 1330.77 (4.9 secs)
numSearchThreads = 20: 9 / 10 positions, visits/s = 1359.24 (5.4 secs)
numSearchThreads = 20: 10 / 10 positions, visits/s = 1370.10 nnEvals/s = 1201.05 nnBatches/s = 207.37 avgBatchSize = 5.79 (6.0 secs)
Testing a max batch size of 10, half the search threads, which can pipeline better on some GPUs:

numSearchThreads = 20: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 20: 1 / 10 positions, visits/s = 1601.66 (0.5 secs)
numSearchThreads = 20: 2 / 10 positions, visits/s = 1389.90 (1.2 secs)
numSearchThreads = 20: 3 / 10 positions, visits/s = 1314.33 (1.9 secs)
numSearchThreads = 20: 4 / 10 positions, visits/s = 1308.79 (2.5 secs)
numSearchThreads = 20: 5 / 10 positions, visits/s = 1309.13 (3.1 secs)
numSearchThreads = 20: 6 / 10 positions, visits/s = 1284.78 (3.8 secs)
numSearchThreads = 20: 7 / 10 positions, visits/s = 1293.46 (4.4 secs)
numSearchThreads = 20: 8 / 10 positions, visits/s = 1290.13 (5.1 secs)
numSearchThreads = 20: 9 / 10 positions, visits/s = 1318.71 (5.6 secs)
numSearchThreads = 20: 10 / 10 positions, visits/s = 1330.16 nnEvals/s = 1169.15 nnBatches/s = 180.34 avgBatchSize = 6.48 (6.2 secs) (EloDiff -12)
Half batch size was not at least 3% faster (measured -2.9%), keeping the default.

2026-11-25 09:24:42-0600: GPU 0 finishing, processed 31994 rows 7850 batches
2026-11-25 09:24:42-0600: GPU 0 finishing, processed 31932 rows 7856 batches
```

</details>

### Run 2: forced FP32 (`cudaDeviceToUse=0, useFP16=false`)

<details><summary>Full output</summary>

```
2026-11-25 09:24:46-0600: Running with following config:
allowResignation = true
cudaDeviceToUse = 0
friendlyPassOk = true
hasButton = false
koRule = SITUATIONAL
lagBuffer = 1.0
logAllGTPCommunication = true
logDir = gtp_logs
logSearchInfo = true
logSearchInfoForChosenMove = false
logToStderr = false
multiStoneSuicideLegal = false
nnCacheSizePowerOfTwo = 20
nnMutexPoolSizePowerOfTwo = 16
numNNServerThreadsPerModel = 2
numSearchThreads = 12
ponderingEnabled = false
resignConsecTurns = 3
resignThreshold = -0.90
scoringRule = AREA
searchFactorAfterOnePass = 0.50
searchFactorAfterTwoPass = 0.25
searchFactorWhenWinning = 0.40
searchFactorWhenWinningThreshold = 0.95
taxRule = NONE
useFP16 = false
whiteHandicapBonus = N-1

2026-11-25 09:24:46-0600: Loading model and initializing benchmark...
2026-11-25 09:24:46-0600: Testing with default positions for board size: 19
2026-11-25 09:24:46-0600: nnRandSeed0 = 6176755316355380900
2026-11-25 09:24:46-0600: After dedups: nnModelFile0 = models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz useFP16 false
2026-11-25 09:24:46-0600: Initializing neural net buffer to be size 19 * 19 exactly
2026-11-25 09:24:47-0600: Cuda backend thread 0: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:24:47-0600: Cuda backend thread 0: Model version 14 useFP16 = false useNHWC = false
2026-11-25 09:24:47-0600: Cuda backend thread 0: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)
2026-11-25 09:24:47-0600: Cuda backend thread 1: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:24:47-0600: Cuda backend thread 1: Model version 14 useFP16 = false useNHWC = false
2026-11-25 09:24:47-0600: Cuda backend thread 1: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)

2026-11-25 09:24:49-0600: Loaded config gtp.cfg and/or command-line and query overrides
2026-11-25 09:24:49-0600: Loaded model models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

Testing using 800 visits.
  If you have a good GPU, you might increase this using "-visits N" to get more accurate results.
  If you have a weak GPU and this is taking forever, you can decrease it instead to finish the benchmark faster.

Your GTP config is currently set to cudaUseFP16 = false
If you have a strong GPU capable of FP16 tensor cores (e.g. RTX2080) setting this to true may give a large performance boost.

Your GTP config is currently set to use numSearchThreads = 12
Automatically trying different numbers of threads to home in on the best (board size 19x19):

2026-11-25 09:24:49-0600: GPU 0 finishing, processed 5 rows 5 batches
2026-11-25 09:24:49-0600: GPU 0 finishing, processed 0 rows 0 batches
2026-11-25 09:24:49-0600: nnRandSeed0 = 12274694743369963761
2026-11-25 09:24:49-0600: After dedups: nnModelFile0 = models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz useFP16 false
2026-11-25 09:24:49-0600: Initializing neural net buffer to be size 19 * 19 exactly
2026-11-25 09:24:50-0600: Cuda backend thread 0: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:24:50-0600: Cuda backend thread 0: Model version 14 useFP16 = false useNHWC = false
2026-11-25 09:24:50-0600: Cuda backend thread 1: Found GPU NVIDIA GeForce RTX 3080 Laptop GPU memory 16781344768 compute capability major 8 minor 6
2026-11-25 09:24:50-0600: Cuda backend thread 1: Model version 14 useFP16 = false useNHWC = false
2026-11-25 09:24:50-0600: Cuda backend thread 0: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)
2026-11-25 09:24:50-0600: Cuda backend thread 1: Model name: kata1-b18c384nbt-s9996604416-d4316597426 (nbt convnet, 26326985 params)


Possible numbers of threads to test: 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32,


numSearchThreads =  5: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads =  5: 1 / 10 positions, visits/s = 544.19 (1.5 secs)
numSearchThreads =  5: 2 / 10 positions, visits/s = 461.64 (3.5 secs)
numSearchThreads =  5: 3 / 10 positions, visits/s = 444.09 (5.4 secs)
numSearchThreads =  5: 4 / 10 positions, visits/s = 459.64 (7.0 secs)
numSearchThreads =  5: 5 / 10 positions, visits/s = 459.28 (8.8 secs)
numSearchThreads =  5: 6 / 10 positions, visits/s = 452.11 (10.7 secs)
numSearchThreads =  5: 7 / 10 positions, visits/s = 460.04 (12.2 secs)
numSearchThreads =  5: 8 / 10 positions, visits/s = 463.13 (13.9 secs)
numSearchThreads =  5: 9 / 10 positions, visits/s = 473.18 (15.3 secs)
numSearchThreads =  5: 10 / 10 positions, visits/s = 481.66 nnEvals/s = 414.74 nnBatches/s = 253.95 avgBatchSize = 1.63 (16.7 secs)

numSearchThreads = 12: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 12: 1 / 10 positions, visits/s = 805.57 (1.0 secs)
numSearchThreads = 12: 2 / 10 positions, visits/s = 626.58 (2.6 secs)
numSearchThreads = 12: 3 / 10 positions, visits/s = 585.53 (4.2 secs)
numSearchThreads = 12: 4 / 10 positions, visits/s = 587.49 (5.5 secs)
numSearchThreads = 12: 5 / 10 positions, visits/s = 599.16 (6.8 secs)
numSearchThreads = 12: 6 / 10 positions, visits/s = 585.42 (8.3 secs)
numSearchThreads = 12: 7 / 10 positions, visits/s = 583.14 (9.7 secs)
numSearchThreads = 12: 8 / 10 positions, visits/s = 589.05 (11.0 secs)
numSearchThreads = 12: 9 / 10 positions, visits/s = 598.17 (12.2 secs)
numSearchThreads = 12: 10 / 10 positions, visits/s = 604.52 nnEvals/s = 500.07 nnBatches/s = 135.60 avgBatchSize = 3.69 (13.4 secs)

numSearchThreads = 10: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 10: 1 / 10 positions, visits/s = 773.71 (1.0 secs)
numSearchThreads = 10: 2 / 10 positions, visits/s = 602.24 (2.7 secs)
numSearchThreads = 10: 3 / 10 positions, visits/s = 565.25 (4.3 secs)
numSearchThreads = 10: 4 / 10 positions, visits/s = 576.94 (5.6 secs)
numSearchThreads = 10: 5 / 10 positions, visits/s = 583.98 (6.9 secs)
numSearchThreads = 10: 6 / 10 positions, visits/s = 566.10 (8.6 secs)
numSearchThreads = 10: 7 / 10 positions, visits/s = 565.60 (10.0 secs)
numSearchThreads = 10: 8 / 10 positions, visits/s = 564.98 (11.5 secs)
numSearchThreads = 10: 9 / 10 positions, visits/s = 571.73 (12.7 secs)
numSearchThreads = 10: 10 / 10 positions, visits/s = 577.12 nnEvals/s = 482.01 nnBatches/s = 159.27 avgBatchSize = 3.03 (14.0 secs)

numSearchThreads = 20: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 20: 1 / 10 positions, visits/s = 675.92 (1.2 secs)
numSearchThreads = 20: 2 / 10 positions, visits/s = 590.35 (2.8 secs)
numSearchThreads = 20: 3 / 10 positions, visits/s = 568.96 (4.3 secs)
numSearchThreads = 20: 4 / 10 positions, visits/s = 578.63 (5.7 secs)
numSearchThreads = 20: 5 / 10 positions, visits/s = 579.92 (7.1 secs)
numSearchThreads = 20: 6 / 10 positions, visits/s = 566.03 (8.7 secs)
numSearchThreads = 20: 7 / 10 positions, visits/s = 577.40 (9.9 secs)
numSearchThreads = 20: 8 / 10 positions, visits/s = 582.05 (11.2 secs)
numSearchThreads = 20: 9 / 10 positions, visits/s = 593.21 (12.4 secs)
numSearchThreads = 20: 10 / 10 positions, visits/s = 599.03 nnEvals/s = 519.32 nnBatches/s = 87.76 avgBatchSize = 5.92 (13.7 secs)

numSearchThreads =  8: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads =  8: 1 / 10 positions, visits/s = 776.61 (1.0 secs)
numSearchThreads =  8: 2 / 10 positions, visits/s = 567.44 (2.8 secs)
numSearchThreads =  8: 3 / 10 positions, visits/s = 538.01 (4.5 secs)
numSearchThreads =  8: 4 / 10 positions, visits/s = 544.52 (5.9 secs)
numSearchThreads =  8: 5 / 10 positions, visits/s = 525.93 (7.7 secs)
numSearchThreads =  8: 6 / 10 positions, visits/s = 508.06 (9.5 secs)
numSearchThreads =  8: 7 / 10 positions, visits/s = 513.57 (11.0 secs)
numSearchThreads =  8: 8 / 10 positions, visits/s = 511.35 (12.6 secs)
numSearchThreads =  8: 9 / 10 positions, visits/s = 516.50 (14.1 secs)
numSearchThreads =  8: 10 / 10 positions, visits/s = 522.65 nnEvals/s = 443.57 nnBatches/s = 190.17 avgBatchSize = 2.33 (15.4 secs)

numSearchThreads = 16: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 16: 1 / 10 positions, visits/s = 689.49 (1.2 secs)
numSearchThreads = 16: 2 / 10 positions, visits/s = 579.81 (2.8 secs)
numSearchThreads = 16: 3 / 10 positions, visits/s = 554.70 (4.4 secs)
numSearchThreads = 16: 4 / 10 positions, visits/s = 566.53 (5.8 secs)
numSearchThreads = 16: 5 / 10 positions, visits/s = 573.65 (7.1 secs)
numSearchThreads = 16: 6 / 10 positions, visits/s = 559.48 (8.7 secs)
numSearchThreads = 16: 7 / 10 positions, visits/s = 573.10 (9.9 secs)
numSearchThreads = 16: 8 / 10 positions, visits/s = 577.08 (11.3 secs)
numSearchThreads = 16: 9 / 10 positions, visits/s = 586.69 (12.5 secs)
numSearchThreads = 16: 10 / 10 positions, visits/s = 589.58 nnEvals/s = 503.72 nnBatches/s = 100.92 avgBatchSize = 4.99 (13.8 secs)


Ordered summary of results:

numSearchThreads =  5: 10 / 10 positions, visits/s = 481.66 nnEvals/s = 414.74 nnBatches/s = 253.95 avgBatchSize = 1.63 (16.7 secs) (EloDiff baseline)
numSearchThreads =  8: 10 / 10 positions, visits/s = 522.65 nnEvals/s = 443.57 nnBatches/s = 190.17 avgBatchSize = 2.33 (15.4 secs) (EloDiff +19)
numSearchThreads = 10: 10 / 10 positions, visits/s = 577.12 nnEvals/s = 482.01 nnBatches/s = 159.27 avgBatchSize = 3.03 (14.0 secs) (EloDiff +50)
numSearchThreads = 12: 10 / 10 positions, visits/s = 604.52 nnEvals/s = 500.07 nnBatches/s = 135.60 avgBatchSize = 3.69 (13.4 secs) (EloDiff +61)
numSearchThreads = 16: 10 / 10 positions, visits/s = 589.58 nnEvals/s = 503.72 nnBatches/s = 100.92 avgBatchSize = 4.99 (13.8 secs) (EloDiff +38)
numSearchThreads = 20: 10 / 10 positions, visits/s = 599.03 nnEvals/s = 519.32 nnBatches/s = 87.76 avgBatchSize = 5.92 (13.7 secs) (EloDiff +31)


Based on some test data, each speed doubling gains perhaps ~250 Elo by searching deeper.
Based on some test data, each thread costs perhaps 7 Elo if using 800 visits, and 2 Elo if using 5000 visits (by making MCTS worse).
So APPROXIMATELY based on this benchmark, if you intend to do a 5 second search:
numSearchThreads =  5: (baseline)
numSearchThreads =  8:   +19 Elo
numSearchThreads = 10:   +50 Elo
numSearchThreads = 12:   +61 Elo (recommended)
numSearchThreads = 16:   +38 Elo
numSearchThreads = 20:   +31 Elo

If you care about performance, you may want to edit numSearchThreads in gtp.cfg and/or command-line and query overrides based on the above results!
If you intend to do much longer searches, configure the seconds per game move you expect with the '-time' flag and benchmark again.
If you intend to do short or fixed-visit searches, use lower numSearchThreads for better strength, high threads will weaken strength.
If interested see also other notes about performance and mem usage in the top of gtp.cfg and/or command-line and query overrides

Running additional tests of a few other settings at numSearchThreads = 12.
Re-measuring the current recommendation as a baseline:

numSearchThreads = 12: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 12: 1 / 10 positions, visits/s = 763.81 (1.1 secs)
numSearchThreads = 12: 2 / 10 positions, visits/s = 610.11 (2.7 secs)
numSearchThreads = 12: 3 / 10 positions, visits/s = 578.94 (4.2 secs)
numSearchThreads = 12: 4 / 10 positions, visits/s = 587.14 (5.5 secs)
numSearchThreads = 12: 5 / 10 positions, visits/s = 593.52 (6.8 secs)
numSearchThreads = 12: 6 / 10 positions, visits/s = 580.56 (8.4 secs)
numSearchThreads = 12: 7 / 10 positions, visits/s = 579.36 (9.8 secs)
numSearchThreads = 12: 8 / 10 positions, visits/s = 577.72 (11.2 secs)
numSearchThreads = 12: 9 / 10 positions, visits/s = 585.00 (12.5 secs)
numSearchThreads = 12: 10 / 10 positions, visits/s = 589.66 nnEvals/s = 500.81 nnBatches/s = 136.62 avgBatchSize = 3.67 (13.8 secs)
Testing a max batch size of 6, half the search threads, which can pipeline better on some GPUs:

numSearchThreads = 12: 0 / 10 positions, visits/s = -nan (0.0 secs)
numSearchThreads = 12: 1 / 10 positions, visits/s = 736.74 (1.1 secs)
numSearchThreads = 12: 2 / 10 positions, visits/s = 602.91 (2.7 secs)
numSearchThreads = 12: 3 / 10 positions, visits/s = 569.02 (4.3 secs)
numSearchThreads = 12: 4 / 10 positions, visits/s = 575.56 (5.6 secs)
numSearchThreads = 12: 5 / 10 positions, visits/s = 583.11 (7.0 secs)
numSearchThreads = 12: 6 / 10 positions, visits/s = 570.80 (8.5 secs)
numSearchThreads = 12: 7 / 10 positions, visits/s = 574.75 (9.9 secs)
numSearchThreads = 12: 8 / 10 positions, visits/s = 569.01 (11.4 secs)
numSearchThreads = 12: 9 / 10 positions, visits/s = 577.02 (12.6 secs)
numSearchThreads = 12: 10 / 10 positions, visits/s = 582.82 nnEvals/s = 501.52 nnBatches/s = 129.32 avgBatchSize = 3.88 (13.9 secs) (EloDiff -5)
Half batch size was not at least 3% faster (measured -1.2%), keeping the default.

2026-11-25 09:26:48-0600: GPU 0 finishing, processed 27602 rows 8750 batches
2026-11-25 09:26:48-0600: GPU 0 finishing, processed 27555 rows 8752 batches
```

</details>
