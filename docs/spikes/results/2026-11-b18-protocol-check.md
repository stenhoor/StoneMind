# Spike 0001: the protocol check against native through the bridge (b18)

Date: 2026-11-27 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU, i9-11900H). Browser: the separate test
Edge (version 158) with a throwaway profile, run as in earlier reports:
`msedge --user-data-dir=<temp dir> --remote-debugging-port=9223 --no-first-run --no-default-browser-check --ignore-gpu-blocklist --use-webgpu-power-preference=force-high-performance`.
`nvidia-smi pmon` showed its GPU process on GPU 0 (the NVIDIA GPU) during both runs. Network: b18c384nbt
(`kata1-b18c384nbt-s9996604416-d4316597426`). Every figure below is quoted from the two reports in this directory:
`2026-11-b18-protocol-check.json` (the check) and `2026-11-b18-protocol-check-sensitivity.json` (the sensitivity run).

## What this is

Step 12 of the parity plan (`docs/spikes/0001-katago-parity-plan.md`). The plan's protocol check: "The same queries go
to native through the bridge and to the browser engine through the same `AnalysisEngine` interface. Pass: every field
native returns is present, with values within the network goldens' tolerance."

The check is a page, `apps/engine-bench/src/protocol.html`. It sends 21 cases to both engines, one engine at a time:
- **native KataGo** through `@stonemind/katago-bridge` and `connectBridge`;
- **the browser engine** on the WebGPU network, through `createAnalysisEngine`.

Both are `@stonemind/engine`'s `AnalysisEngine`, as an application would use them.

**Native's setup, chosen by the owner on 2026-11-27.** Native ran deterministically, so values could be compared within
tolerance:
- the CPU (Eigen) build, the one all the engine's goldens come from;
- a config made by `apps/engine-bench/scripts/protocol-config.mjs`: the owner's `analysis_example.cfg`, unchanged, plus
  one analysis thread, one search thread, `nnRandomize = false`, `wideRootNoise = 0`, `rootNumSymmetriesToSample = 1`
  and `useFP16 = false`.

The browser engine got the same config keys. Native reported version 1.18.2, git revision `d91ea855` (a later master
commit than the v1.18.2 tag, as PR #111 recorded).

**What is compared:**
- **Final results:** the same fields as native's (none missing, none extra), at every level. Visits, move order,
  principal variations, hashes and the player to move must be exact. Values must be within the engine's search check's
  tolerances: the network goldens' 2e-4 for probabilities, policy and ownership; 2e-3 for scores; 1e-3 for utility;
  1e-3 relative for weights and play selection values; ownership standard deviations compared as variances.
- **Reports during the search, and results of a search stopped early:** their fields only. Each engine reaches its own
  point in the search.
- **Errors and warnings:** the same text and field.
- **Action replies:** `query_version` the same version; `query_models` the same internal name, model version, human SL
  flag and fp16 flag. The model's `name` (the path given) and `maxBatchSize` are each engine's own.

**The cases:**
- **Actions:** `query_version`, `query_models` and `clear_cache`.
- **Results on 19×19:**
  - turns 2 and 3 with every optional output (policy, ownership and its standard deviation, per move too, PV visits,
    no-result value), with values for Black: Black to move at turn 2, White at turn 3;
  - AGA rules with values for White, at turns 0 and 1;
  - Chinese rules with values for the side to move and a PV length of 4.
- **Results on 9×9:**
  - komi 7, where a draw is possible;
  - initial stones with White to move;
  - three handicap stones with `whiteHandicapBonus` N-1;
  - `avoidMoves`;
  - `allowMoves`;
  - `rootPolicyTemperature`, `rootFpuReductionMax` and `overrideSettings`.
- **A 7×9 board.**
- **Reports during the search,** every 0.1 s.
- **A search stopped at its first report:** two turns; one is stopped mid-search, the other is still queued.
- **Errors:** an invalid komi, a board too large, an illegal move, a turn number past the moves, and `allowMoves` with
  `avoidMoves`.
- **Warnings:** unused fields.

## Outcome

- **The check passes: all 21 cases match native** (`2026-11-b18-protocol-check.json`, `"ok": true`).
  - Every searched case has the same visits on both engines, and every field within tolerance.
  - Every error and warning has the same text.
  - Both engines wrote reports during the search where asked: native 322, the browser 11, for the same 100 visits. Each
    browser report had native's fields.
  - In the stopped case, both ended both turns, with `noResults` for the queued turn on both.
- **The comparison catches a wrong engine.** For the sensitivity run, the browser engine's config said
  `reportAnalysisWinratesAs = WHITE` while native's said BLACK.
  - Every searched case reported for Black failed: 10 cases. The values came out with their signs flipped, for example
    ownership 0.039784 against native's -0.039784.
  - The two cases that set the side in the query (`aga-white-19`, `side-to-move-19`) still passed, as did the actions,
    the errors and the stopped search.

## Earlier attempts, recorded so nothing is hidden

- **Run 1** (not kept, 77 MB) also passed every case. But it had two problems:
  - Its `avoid-allow-9` case put `avoidMoves` and `allowMoves` in one query, which KataGo refuses. So it compared only
    the error message. It is now three cases: `avoid-9`, `allow-9`, and the refusal as `error-avoid-and-allow`.
  - It kept all 1241 of native's reports during the search. The report now keeps the first and a count.
- **The first try at run 2 didn't start.** Run 1's bridge was still running, because the run script stopped the shell
  around the bridge, not the bridge. I stopped it and fixed the script, then ran run 2.

## Timings

The page records each engine's time per case. These are not a speed comparison. Native ran on the CPU with one search
thread, chosen to be deterministic, not fast. The totals were 205.7 s for native and 10.3 s for the browser engine. The
speed comparison with native at its own config is the review check (`2026-11-b18-part-a-review.md`): the browser took
1.38 to 1.45 times native's time.

## Not covered

- **Rules the browser engine doesn't support yet** (territory scoring, a group tax, the button). It refuses them, a
  known difference from KataGo, so no case uses them. Part C of the plan adds them.
- **Native at its own config** (CUDA, fp16, 2 × 16 threads, random symmetries). Its results vary from run to run, so it
  can't be compared within tolerance.
- **Other browsers and GPUs.** This is one laptop GPU in one browser.
