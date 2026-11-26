# Spike 0001: Phase 5, item 2, faster search code (b18): goal not met, at the stop line

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU, i9-11900H). Browser: the separate test
Edge, with and without `--enable-unsafe-webgpu` (the flag). Native: KataGo v1.18.2, CUDA, `analysis_example.cfg` as it
is. Every figure below is quoted from, or computed from, the reports in this directory (prefix
`2026-11-b18-search-js-`).

## Outcome

The goal was GPU waiting at no more than 6% of GPU working time with the flag (review wall time ÷ GPU-busy time ≤
1.06). The stop rule was to stop if it was still above 10% (1.10) after the changes.

- **Not met.** With the flag, wall ÷ busy went from 1.119 and 1.121 to 1.093 and 1.106. One run is below the stop
  line and one above it.
- **GPU waiting per review**, with the flag: from 2744 and 2825 ms to 2182 and 2267 ms, about 20% less.
- **Without the flag:** wall ÷ busy went from 1.074 and 1.093 to 1.070 and 1.066, and waiting from 2220 and 2561 ms to
  2018 and 1925 ms, 15% less.
- **The search's own JavaScript** (Node benchmark, no GPU) takes 19.6–19.9% less time.
- **Results are unchanged.** All 250 turns' search analyses are identical before and after, with the benchmark's
  stand-in network. The browser reviews make the same evaluations (18047 with the flag, 17957 without) and agree with
  native on the same moves.
- **Kept:** both changes. Both are ports of KataGo's own code.

## What changed

1. **Taking back a move in the ladder search** (`src/rules/board.ts`), ported from KataGo's `Board::playMoveRecorded`
   and `Board::undo`.
   - **Before:** each recorded move copied the board's five arrays, and undoing it copied them back.
   - **Now:** as in KataGo, the record holds only the move, the ko point and which neighbouring chains it captured. Undo
     fills those chains back in and takes the stone off.
   - **Ported with it, because KataGo's undo depends on them:** KataGo's `playMoveAssumeLegal`, `mergeChains`,
     `removeChain`, `changeSurroundingLiberties`, `addChain`, `rebuildChain` and `isLibertyOf`. The board's chains are
     now built in KataGo's order.
   - **Also matched to KataGo:** `findLiberties` and the capture helpers now walk a chain from the given stone, as KataGo
     does, not from its head.
   - **Setup stones** (`setStone`) follow KataGo's `setStoneFailIfNoLibs` test: they fail if they would capture or have
     no liberties. That's the same outcome as before. The ko point is kept as before.
2. **Territory counting** (`src/rules/area.ts`). The steps were already KataGo's (`calculateAreaForPla`). The data
   structures are now KataGo's too:
   - typed arrays, reused between calls;
   - a linked list through each region, in place of per-region arrays;
   - the vital-chain lists filtered in place, in place of a new array per point;
   - chains walked through `next_in_chain`.

   This needed one new board method, `nextInChain`.

## The search's own time (Node benchmark)

`scripts/search-cpu-bench.mjs`: the review's first 100 turns at 100 visits, batch 16, with a stand-in network. Median of
5 runs, alternated by build, two rounds (`-cpu-bench.jsonl`). The tree is the same in every build: 10000 evaluations in
700 calls.

| Build | Round 1 (ms) | Round 2 (ms) | Against before |
| --- | ---: | ---: | ---: |
| Before | 1595.9 | 1595.8 | |
| With 1 (undo) | 1412.1 | 1435.5 | −11.5%, −10.0% |
| With 1 and 2 (now) | 1282.5 | 1279.0 | −19.6%, −19.9% |

**Same analyses:** every turn of the 250-move game, with the stand-in network, gives the same `analysis()` before
and after, byte for byte.

## The review (batch 16, two at once)

The same 250-move game at 100 visits per move. The order was before, after, after, before with the flag, then the same
without. Each run used a fresh browser. Native's review took 16412.1 and 16596.1 ms (#80).

| | Wall (ms) | GPU busy (ms) | Wall ÷ busy | GPU waiting (ms) | Ratio to native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before, with the flag | 25709.2, 26137.0 | 22965.0, 23312.0 | 1.119, 1.121 | 2744, 2825 | 1.55–1.59 |
| After, with the flag | 25673.5, 23705.8 | 23492.0, 21439.3 | 1.093, 1.106 | 2182, 2267 | 1.43–1.56 |
| Before, without the flag | 32245.3, 30098.0 | 30025.1, 27537.0 | 1.074, 1.093 | 2220, 2561 | 1.81–1.96 |
| After, without the flag | 30734.4, 31062.1 | 28716.4, 29137.0 | 1.070, 1.066 | 2018, 1925 | 1.85–1.89 |

- **Waiting is the measure this item changes.** The mean saving is 560 ms with the flag and 419 ms without. That's about
  2.2% and 1.3% of the review (computed from the means; not measured directly).
- **Wall time** moved by −4.8% with the flag and −0.9% without (means). It isn't a clean measure of this change. GPU
  busy time, which this change doesn't touch, varied by up to 9.6% between runs of the same build (21439.3 against
  23492.0 ms).
- **My plan's estimate was too high.** It was 4–6% faster reviews with the flag. The measured waiting saving is about
  2%.

## What's left

- **GPU waiting with the flag** is now 9–11% of GPU working time, against the 6% goal.
- **The rest of the search's own time,** from the profile before these changes:
  - processing the network's results, about 13%;
  - board symmetries, about 10%;
  - choosing the next move to search, about 16%;
  - superko bookkeeping, about 7%.

  None of these was in this item's plan.
