# Spike 0001: the review check after Part A of the KataGo equivalence plan (b18)

Date: 2026-11-27 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU, i9-11900H). Browser: the separate test
Edge (version 158), with a throwaway profile, run with and without `--enable-unsafe-webgpu` (the flag), the browser
engine at its defaults. Native: KataGo v1.18.2, CUDA, `analysis_example.cfg` as it is. Every figure below is quoted
from, or computed from, the reports in this directory with the prefix `2026-11-b18-part-a-review-`.

## What this is

The plan (`docs/spikes/0001-katago-parity-plan.md`) runs this check once Part A is done. Part A ported native's search
features to the browser engine, each matched exactly against native (#96 to #106). Until now, native's review ran
features the browser didn't have, so this is the first speed comparison where both engines run the same search.

The work is the same as in earlier reports: the 250 turns of `test/goldens/agreement-game.json`, 100 visits per turn,
with the b18 network.

## Outcome

- **Time.** In the main mode (batch 16, two turns at once), the browser's review took:
  - with the flag, 24095.4 and 25303.1 ms;
  - without the flag, 30939.7 and 30897.6 ms.

  Native's review took 17464.5 and 17525.2 ms.
- **Ratio to native** (against the mean of today's two native runs, 17494.9 ms):
  - with the flag, 1.38 and 1.45, a gap of 7204 ms on the means;
  - without the flag, 1.77 and 1.77, a gap of 13424 ms on the means.
- **Delta from prior** (the last report, `2026-11-b18-search-js`, "after" rows):
  - with the flag, mean 24699.3 ms against 24689.7 ms: +9.6 ms, +0.04%;
  - without the flag, mean 30918.7 ms against 30898.3 ms: +20.4 ms, +0.07%.

  The ported features didn't change the review's time measurably.
- **Agreement with native's best moves** (main mode): 187 to 206 of 250. Native agrees with itself on 195 to 205.
- **Network evaluations** (main mode): 17627 with the flag and 17549 without. That's 2.3% fewer than in the last
  report (18047 and 17957). Native made 17851.

## Time in each mode

Times are the review's wall time in ms (`totalMs`), each from a fresh browser. The runs were in the order with the
flag, without, with, without. The ratio is to today's native mean, 17494.9 ms.

- **With the flag:**
  - batch 64, one at a time: 32516.2 and 32127.0 (1.86, 1.84);
  - batch 16, one at a time: 28271.1 and 29904.3 (1.62, 1.71);
  - batch 16, two at once: 24095.4 and 25303.1 (1.38, 1.45);
  - batch 16, four at once: 24218.8 and 24998.7 (1.38, 1.43).
- **Without the flag:**
  - batch 64, one at a time: 42809.8 and 41174.5 (2.45, 2.35);
  - batch 16, one at a time: 37677.8 and 36431.2 (2.15, 2.08);
  - batch 16, two at once: 30939.7 and 30897.6 (1.77, 1.77);
  - batch 16, four at once: 31306.9 and 31305.6 (1.79, 1.79).

**GPU working time** in the main mode was 22543.5 and 23770.0 ms with the flag, and 29578.6 and 30018.0 ms without.
Wall time divided by GPU working time was 1.069 and 1.064 with the flag, and 1.046 and 1.029 without. Most of the review
is the GPU running the network.

## Native's time

**Today's native runs:** 17464.5 and 17525.2 ms. The last report quoted native at 16412.1 and 16596.1 ms (#80).
Native's config and build haven't changed since, so this is run-to-run variation of native alone: today's mean is 6.0%
higher. Native measured 17495.5 and 17647.8 ms in `2026-11-b18-wino-in-v2`.

**Native goes past 100 visits a turn:** 27591 and 27613 visits in all, about 110 a turn, against the browser's 25000. Its
config runs 16 search threads for each of 2 analysis threads, and each search stops only once the threads in flight
finish. So native does about 10% more search in its time. The browser stops at exactly 100 visits.

## Agreement with native

A turn agrees when both engines' move of order 0 is the same. Native's search runs 16 threads and isn't deterministic,
so its agreement with itself is the control. The three native runs (two timed, one counting evaluations) agree with each
other on 195, 199 and 205 of 250 turns.

- **The browser in the main mode, against each of the three native runs:**
  - with the flag, 187, 200 and 190;
  - without the flag, 206, 203 and 193.
- **The other modes:** 178 to 206 against the three native runs. Batch 64, one at a time, is the lowest, at 178 to 185.
- **The browser agrees with itself.** Its two runs with each setting chose the same move on all 250 turns. The browser's
  search is deterministic at a fixed seed, and the default seed is fixed. With and without the flag, the browser agrees
  with itself on 208 turns, because the two settings' networks round differently.
- **Delta from prior:** the latest recorded agreement for this mode was 182 and 179 of 250, against one native run each
  (`2026-11-b18-wino-in-v2`). Today's is 187 to 206.

## Network evaluations

Evaluations are the positions sent to the network: `trace.inputs` in the browser's reports, and the "NN rows" native
writes to its log, from its run with `--count-review`.

- **Main mode:** 17627 with the flag and 17549 without, in 1458 and 1453 calls. The last report had 18047 and 17957.
  Between the two reports, the only changes to the engine were Part A's eleven steps, so the difference comes from
  them. How much each step accounts for wasn't measured.
- **Native:** 17851, in 1851 batches, for its 27574 visits.

## Checks on the runs

- **The test Edge ran on the NVIDIA GPU.** In the last three runs, pmon showed the test Edge's GPU process, matched by
  its process ID, as the process using GPU 0. In the first run (with the flag) the process ID wasn't recorded. pmon
  showed a process on GPU 0 whose name, `a0e9f4`, comes from the test profile's path. The only other process using
  GPU 0 in every run was the desktop compositor, `kwin_wayland`.
- **Nothing else was running.** The runs were one after another: four browser runs, then three native runs.
- **The browser's network** was its default in every run: Winograd, fused, f16, GEMM v3 and Winograd transforms v3.
  With the flag it used the GPU's matrix units (`matrix: true`); without the flag it didn't.
- **The logs:** each browser run has its GPU clocks (`-clocks.csv`, every 250 ms), CPU frequencies (`-cpufreq.csv`) and
  pmon output (`-pmon.txt`) alongside its report.

## Files

- **Browser:** `-flag-1.json`, `-flag-2.json`, `-noflag-1.json` and `-noflag-2.json`, each with its `-clocks.csv`,
  `-cpufreq.csv` and `-pmon.txt`.
- **Native:**
  - `-native-1.json` and `-native-2.json` are the timed runs of `scripts/native-search-bench.mjs`, against
    `-noflag-1.json`;
  - `-native-count.json` is its `--count-review` run.

  The agreement figures in those files are against `-noflag-1.json` only. The others above were computed from the
  move lists in the reports.
