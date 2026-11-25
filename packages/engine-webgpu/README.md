# @stonemind/engine-webgpu

The experimental browser engine from [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md). **Phases 2 and 3:**

- **Model loader** (`src/model/`): parses official KataGo `.bin.gz` / `.txt.gz` files (model versions 8–17;
  convolutional nets), using the platform `DecompressionStream`. It follows the reading order of KataGo's `desc.cpp`.
  Transformer blocks and RMSNorm trunks are refused clearly; they come in phase 6.
- **Reference network** (`src/reference/`): a plain fp32 TypeScript forward pass, post-processing and the 8 symmetries,
  following KataGo's CPU (Eigen) backend and `nneval.cpp`. It is slow on purpose: it is the correctness reference for
  the WGSL kernels in phase 4.
- **Rules engine** (`src/rules/`): a port of the parts of KataGo's `Board` and `BoardHistory` that the network's inputs
  depend on: captures, suicide, simple ko, positional and situational superko, the ladder search, pass-alive area, the
  named rule sets, territory scoring's per-stone adjustment and the white handicap bonus. Passes, the territory-scoring
  encore and rule sets with a button are refused with `UnsupportedPositionError` rather than approximated.
- **V7 input encoder** (`src/features/v7.ts`): `encodePositionV7` builds KataGo's 22 spatial planes and 19 globals from
  a position (setup stones and moves), following `NNInputs::fillRowV7`. The earlier simple encoder
  (`src/features/simple.ts`), for positions with no captures, ko, ladders or one-colour regions, is still exported.

## Verification

`test/tiny-board.test.mjs` reproduces KataGo's own `runnnontinyboardtest` (g170-b6c96, symmetry 3). The parameter
count, win/loss, score, lead, and every policy and ownership value match KataGo's recorded output
(`cpp/tests/results/runNNOnTinyBoardTest.txt` at commit d91ea85) to the printed precision. A second test checks all 8
symmetries exactly against mirrored boards.

`test/goldens.test.mjs` compares the reference network with outputs captured from native KataGo 1.18.2 (CPU/Eigen
build, fp32) on the positions in `test/goldens/positions.json`, under all 8 symmetries: win/loss, lead, score, every
policy and ownership value. Inputs come from `encodePositionV7`. All 10 positions match for g170-b6c96. The
b18c384nbt goldens (captured by the owner, `test/goldens/kata1-b18c384nbt-…json`) run only when
`STONEMIND_B18_MODEL` points at that model file, because the 98 MB model is not downloaded in CI; otherwise that test
is skipped by name.

`test/rules.test.mjs` checks the rules engine on hand-built positions: captures, suicide, simple ko, a triple-ko cycle
under positional, situational and simple ko, a ladder with and without a ladder breaker, and pass-alive area.

**Capturing goldens** for another network (for example b18c384nbt on a machine with KataGo):

```bash
node packages/engine-webgpu/scripts/capture-goldens.mjs --katago <katago> --config <gtp config> --model <net.bin.gz>
```

It writes `test/goldens/<net name>.json`, using `kata-raw-nn <symmetry> 0` with FP16 turned off, and records the
KataGo version and the network's SHA-256.

The test network (CC0, a g170-run net) is downloaded by `npm run fixtures` from KataGo's repository at that commit and
checked by SHA-256. It is not committed.

## Depends on

None.

## Used by

Nothing yet. Planned: `apps/engine-bench` (phase 4).

## Development

```bash
npm run check      # typecheck, build, fetch the test network, node --test
```

## Notes

- Written fresh in TypeScript. KataGo's code (MIT, © David J Wu) is the reference for the file format and the maths,
  as the comments in each file say.
- Not yet checked against b18c384nbt. That needs goldens from a native KataGo, which is the next step.
