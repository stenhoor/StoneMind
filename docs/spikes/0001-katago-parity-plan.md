# Spike 0001: plan for functionality equivalent to native KataGo

Status: **approved** 2026-11-26, in progress. Part A step 1 (LCB move selection): https://github.com/stenhoor/StoneMind/pull/96. Step 2 (value weighting): https://github.com/stenhoor/StoneMind/pull/97. Step 3 (uncertainty weighting): https://github.com/stenhoor/StoneMind/pull/98. Step 4 (subtree value bias): https://github.com/stenhoor/StoneMind/pull/99. Step 5 (noise pruning): https://github.com/stenhoor/StoneMind/pull/100. Step 6 (policy optimism): https://github.com/stenhoor/StoneMind/pull/101. Step 7 (passing hacks and root ending bonus): https://github.com/stenhoor/StoneMind/pull/102. Step 7b (root pruning of useless moves): https://github.com/stenhoor/StoneMind/pull/103. Step 8 (root symmetry pruning): https://github.com/stenhoor/StoneMind/pull/104. Step 9 (graph search): https://github.com/stenhoor/StoneMind/pull/105. Step 10 (wide root noise): https://github.com/stenhoor/StoneMind/pull/106. Review check after part A: `docs/spikes/results/2026-11-b18-part-a-review.md` (https://github.com/stenhoor/StoneMind/pull/107). Step 11a (the search's output fields): https://github.com/stenhoor/StoneMind/pull/108.

## The goal

The browser engine should answer the same Analysis Engine queries as native KataGo 1.18 and give the same answers,
within the error the network itself allows. "The same answers" means:
- the same search, with every search feature native's analysis engine uses by default;
- the same analysis output fields;
- the same rule sets.

What doesn't change: this is a port, not a redesign. Every part below is ported from KataGo's own code, and any
difference is named in its PR.

## Where things stand

Done and checked against native: the model loader, the reference network, the rules and input features, the WebGPU
network, and the core of the search.

Not done:

1. **Eleven search features native's analysis engine turns on by default.** Our search test turns all of them off on
   the native side, so the two searches match exactly (same moves, same visits). The features, with native's defaults
   from KataGo's `program/setup.cpp`:
   - LCB move selection (`useLcbForSelection`, on)
   - value weighting (`valueWeightExponent` 0.25)
   - uncertainty weighting (`useUncertainty`, on)
   - subtree value bias (`subtreeValueBiasFactor` 0.45)
   - noise pruning (`useNoisePruning`, on)
   - policy optimism (`policyOptimism` 1.0, `rootPolicyOptimism` 0.2)
   - graph search (`useGraphSearch`, on)
   - root symmetry pruning (`rootSymmetryPruning`, on)
   - passing hacks (`enablePassingHacks` and `enableMorePassingHacks`, on)
   - root ending bonus points (`rootEndingBonusPoints` 0.5)
   - root pruning of useless moves (`rootPruneUselessMoves`, on; added 2026-11-27, found while porting step 7)
   - wide root noise (`wideRootNoise` 0.04)
2. **The Analysis Engine protocol.** The engine has its own API. It doesn't yet implement `@stonemind/engine`'s
   `AnalysisEngine`, the contract native KataGo is already used through (via the bridge). Missing from the output:
   ownership and its standard deviation (per position and per move), the policy, `lcb`, `pvVisits` and `rootInfo`.
   Missing from the queries: `initialStones`, `allowMoves` and `avoidMoves`, `reportDuringSearchEvery`, `priority`,
   `analysisPVLen`, `rootFpuReductionMax`, `rootPolicyTemperature`, `overrideSettings`, and the actions
   (`terminate`, `clear_cache`, `query_version`).
3. **Rules.** The search scores by area only. The rules engine refuses territory scoring's encore, group-tax endings,
   no-result long cycles and rule sets with a button. So Japanese rules don't work yet.
4. **Transformer networks.** The model loader refuses networks with transformer blocks or RMSNorm trunks. This is
   phase 6 of the spike.

## How each step is checked

Every step uses the same checks, so "done" means the same thing every time:

- **Exact search check** (`test/search.test.mjs`, goldens from `scripts/capture-search-goldens.mjs`). Native KataGo
  runs with one search thread and symmetry 0, on the same positions.
  - **When a feature is ported,** it is switched on in both engines. That means removing its override from the goldens
    script, which is a change to the config native runs with in the test. Approving this plan approves those changes,
    and no others.
  - **Pass:** the same moves get exactly the same visits and principal variations, and win rates and leads are within
    the network goldens' tolerance.
- **The random feature** (wide root noise) can't match native move for move, because the two engines'
  random numbers differ. With the randomness switched off, they must pass the exact check. With it on, the visit
  spread over many runs must match native's within a tolerance fixed before the runs.
- **Protocol check.** The same queries go to native through the bridge and to the browser engine through the same
  `AnalysisEngine` interface. Pass: every field native returns is present, with values within the network goldens'
  tolerance.
- **Review check** after each part: the 250-move review, with native's config as it is and the browser at its
  defaults. It reports agreement with native's best moves and the time, with the delta from the prior run and the ratio
  to native.

**Stop rule, for every step:** if a ported feature can't pass its check, I stop that step, record exactly what
differs, and ask you before going on. The remaining steps don't depend on a stopped one unless the plan says so.

## The steps, in order

The order follows the code's dependencies: KataGo's later features read values the earlier ones produce.

### Part A: search features

Each feature is its own PR, with the exact check passing.

1. **LCB move selection**, which also gives the `lcb` output field.
2. **Value weighting.**
3. **Uncertainty weighting.** It uses the network's short-term error outputs, which the engine already computes.
4. **Subtree value bias.**
5. **Noise pruning.**
6. **Policy optimism**, at the root and below it.
7. **Passing hacks and root ending bonus points.**
   - 7b. **Root pruning of useless moves.** Added 2026-11-27 with the owner's approval: found while porting step 7,
     it wasn't in the list above.
8. **Root symmetry pruning.**
9. **Graph search.** This is the largest step: positions reached by different move orders share one node, which
   changes how the tree is stored. My reason for putting it after the others: each earlier feature is then checked once
   on the simpler tree, before it has to work on shared nodes.
10. **Wide root noise**, checked as described for the random feature.

**When part A is done:** run the review check with the browser at native's defaults. This is the first like-for-like
speed comparison, since until now native has run features the browser doesn't have.

### Part B: the Analysis Engine protocol

11. **`AnalysisEngine` for the browser engine,** the same contract native uses:
    - streaming results (`reportDuringSearchEvery`), `priority`, `terminate` and `clear_cache`;
    - every query field listed above;
    - every output field listed above.

    Split into four PRs on 2026-11-27, when Part B started, following how KataGo divides the code. Each is ported
    from KataGo's `v1.18.2` tag, the version the checks run against:
    - 11a. **The search's output fields** (KataGo's `getAnalysisJson`, less the hashes): `weight`, `edgeWeight`,
      `noResultValue`, `pvVisits` and `pvEdgeVisits` per move, `rootInfo`'s values, the policy, and ownership with its
      standard deviation, for the position and per move. Checked with the exact search check, which now compares every
      field native reports.
    - 11b. **The position hashes** in `rootInfo` (`thisHash`, `symHash`): KataGo's random number generator and hash
      tables. Checked exactly.
    - 11c. **The query fields that change the search:** `initialStones`, `initialPlayer`, `whiteHandicapBonus`,
      `allowMoves` and `avoidMoves`, `rootFpuReductionMax`, `rootPolicyTemperature`, and `overrideSettings` for the
      ported parameters. Checked with the exact search check.
    - 11d. **The engine:** KataGo's `analysis.cpp` behind `AnalysisEngine`, through `@stonemind/engine`'s
      `JsonLineSession`. That covers parsing with KataGo's errors and warnings, `analyzeTurns`, `priority`, reports
      during the search, which side the values are for, and the actions (`query_version`, `query_models`,
      `clear_cache`, `terminate`, `terminate_all`).

    **Not in scope:** focus moves (`focusMoves` and `set_focus`). They were added to KataGo after v1.18.2 and are in no
    release yet.
12. **The protocol check** against native through the bridge.

### Part C: rules

13. **Territory scoring in the search,** with KataGo's encore (`BoardHistory`), for Japanese and similar rules.
14. **The rest of what's refused today:** group-tax endings, no-result long cycles, button rule sets.

Checked with rules goldens from native on positions that exercise each rule, plus the exact search check under those
rules.

### Part D: transformer networks (spike phase 6)

15. **RMSNorm, RoPE, attention and SwiGLU** kernels, and the loader changes for tf2-b10c384.

Checked against native's `kata-raw-nn` outputs, as the convolutional networks were. Speed is reported against b18.

## Not in this plan

- **The human-style network** (`humanSLProfile`). Native supports it. It's a separate model and a separate decision.
- **Tree reuse between moves.** This is a GTP feature. Native's analysis engine starts each query's search fresh.
- **Further speed work.** Phase 5 closed it.

## Size

These are estimates, not measurements.
- **Part A:** steps 1–8 are each small to medium, mostly changes to how the search scores and chooses children.
  Graph search (step 9) is large.
- **Part B:** medium.
- **Part C:** large. KataGo's encore and its scoring are a sizeable part of `BoardHistory`.
- **Part D:** large: new kernels and a new network shape.
