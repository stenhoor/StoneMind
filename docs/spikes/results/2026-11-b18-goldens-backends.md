# Spike 0001: why b18 matched 0 of 10 goldens (KataGo's CUDA and Eigen backends differ)

Date: 2026-11-25. Follows [the b18 WebGPU results](2026-11-b18-webgpu.md) (0 of 10 positions matched) and
issue #19.

## Finding

There is no bug in the engine. The b18c384nbt goldens had been captured with KataGo's **CUDA** build. KataGo's own CUDA
and CPU (**Eigen**) builds disagree on this network by more than the goldens test's tolerances. No fp32 implementation
that matches one can match the other at those tolerances. The engine's reference follows the Eigen backend. Against
Eigen goldens, it matches.

## Evidence

All values are maximum absolute differences after post-processing (probabilities, ownership in [-1, 1], points for lead
and score), on the 10 positions in `packages/engine-webgpu/test/goldens/positions.json`. The goldens test's tolerances
are 2e-4 for win/loss, policy and ownership, and 2e-3 for lead and score.

**1. KataGo against itself.** These compare the Eigen goldens (captured in the cloud session with KataGo 1.18.2 built
with `USE_BACKEND=EIGEN`) with the owner's CUDA goldens (KataGo 1.18.2 CUDA, RTX 3080 Laptop GPU). Both use the same
model file (SHA-256 `9d7a6afe…e0cba51f1d`), both have `useFP16=false`, and the table covers all 8 symmetries:

| Position | win/loss | lead | score | policy | ownership |
| --- | --- | --- | --- | --- | --- |
| empty-19 | 6.6e-5 | 1.0e-3 | 1.0e-3 | 3.0e-4 | 5.7e-4 |
| opening-19 | 3.8e-5 | 1.0e-3 | 2.0e-3 | 3.8e-4 | 9.7e-4 |
| opening-19-japanese | 1.5e-4 | 3.0e-3 | 3.0e-3 | 3.6e-4 | 1.3e-3 |
| fight-19 | 1.1e-4 | 2.0e-3 | 2.0e-3 | 1.6e-4 | 1.3e-3 |
| ladder-19 | 3.0e-4 | 3.0e-3 | 3.0e-3 | 4.3e-4 | 1.9e-3 |
| capture-9 | 5.0e-6 | 2.0e-3 | 4.0e-3 | 3.6e-4 | 7.8e-4 |
| ko-9 | 3.9e-4 | 2.0e-3 | 3.0e-3 | 8.7e-4 | 6.5e-4 |
| area-9 | 2.2e-5 | 2.0e-3 | 1.0e-3 | 4.4e-4 | 5.6e-4 |
| opening-13 | 7.2e-5 | 1.0e-3 | 1.0e-3 | 4.6e-4 | 8.7e-4 |
| opening-9-japanese | 4.8e-4 | 1.0e-3 | 1.0e-3 | 3.2e-4 | 1.3e-3 |

**2. The engine's TypeScript reference against the Eigen goldens,** at symmetry 0:

| Position | win/loss | lead | score | policy | ownership |
| --- | --- | --- | --- | --- | --- |
| empty-19 | 1.8e-7 | 4.7e-4 | 4.3e-4 | 5.4e-7 | 3.0e-6 |
| opening-19 | 4.2e-7 | 3.3e-4 | 4.1e-5 | 7.1e-7 | 1.6e-6 |
| opening-19-japanese | 5.4e-7 | 2.2e-4 | 4.9e-4 | 5.3e-7 | 2.8e-6 |
| fight-19 | 3.5e-7 | 4.7e-4 | 3.6e-4 | 8.9e-7 | 2.2e-6 |
| ladder-19 | 5.6e-9 | 4.6e-4 | 3.1e-4 | 1.1e-6 | 5.8e-6 |
| capture-9 | 4.9e-7 | 1.1e-4 | 4.7e-4 | 8.9e-7 | 1.2e-6 |
| ko-9 | 9.8e-7 | 2.2e-4 | 4.0e-4 | 5.8e-7 | 1.4e-6 |
| area-9 | 5.7e-7 | 4.3e-4 | 4.9e-4 | 9.1e-7 | 1.5e-6 |
| opening-13 | 3.7e-7 | 3.4e-4 | 2.6e-4 | 2.0e-6 | 2.0e-6 |
| opening-9-japanese | 1.3e-6 | 4.0e-4 | 1.4e-4 | 5.2e-7 | 1.9e-6 |

KataGo prints lead and score to 3 decimals, so differences up to 5e-4 there are within its printing precision.

**3. The WebGPU network against the reference** is from issue #19, on the owner's machine, in Edge and Firefox on the
NVIDIA GPU, all three kernel variants. These are raw outputs before post-processing:
- policy logits: within 1.2e-4
- raw ownership: within 1.2e-5
- value logits: within 1.9e-5

The chain is therefore: WebGPU network ≈ TypeScript reference ≈ KataGo Eigen.

## What changed

- `test/goldens/kata1-b18c384nbt-s9996604416-d4316597426.json` now holds the Eigen capture.
- The owner's CUDA capture is kept as `…-cuda.json` for the record. Tests don't use it.
- Every goldens file records its `backend`.
- `scripts/capture-goldens.mjs` now requires `--backend` and says to use an Eigen build.

## Not yet re-run

- The GPU check's b18 comparison against the new goldens hasn't been re-run on the owner's machine. Step 3 above and
  the reference's match with Eigen suggest it will pass.
- Neither the WebGPU path nor native CUDA uses fp16 here. fp16 will need its own tolerances.
