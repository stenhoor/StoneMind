# @stonemind/engine-webgpu

The experimental browser engine from [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md). **Phase 2:**

- **Model loader** (`src/model/`): parses official KataGo `.bin.gz` / `.txt.gz` files (model versions 8–17;
  convolutional nets), using the platform `DecompressionStream`. It follows the reading order of KataGo's `desc.cpp`.
  Transformer blocks and RMSNorm trunks are refused clearly; they come in phase 6.
- **Reference network** (`src/reference/`): a plain fp32 TypeScript forward pass, post-processing and the 8 symmetries,
  following KataGo's CPU (Eigen) backend and `nneval.cpp`. It is slow on purpose: it is the correctness reference for
  the WGSL kernels in phase 4.
- **Simple input encoder** (`src/features/simple.ts`): KataGo's V7 features for positions with no captures, ko, ladders
  or one-colour regions. It throws on anything else rather than encoding it wrongly. The full rules engine is phase 3.

## Verification

`test/tiny-board.test.mjs` reproduces KataGo's own `runnnontinyboardtest` (g170-b6c96, symmetry 3). The parameter
count, win/loss, score, lead, and every policy and ownership value match KataGo's recorded output
(`cpp/tests/results/runNNOnTinyBoardTest.txt` at commit d91ea85) to the printed precision. A second test checks all 8
symmetries exactly against mirrored boards.

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
