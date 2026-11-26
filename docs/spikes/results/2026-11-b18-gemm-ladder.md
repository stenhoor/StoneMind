# Spike 0001: Phase 1, GEMM efficiency (b18) on Edge (NVIDIA): closed

Date: 2026-11-26 (UTC). Machine: the owner's Linux laptop (RTX 3080 Laptop GPU). Code: branch
`spike-0001/gemm-ladder`, on top of `spike-0001/budget` (#84). Every figure below is quoted from, or computed from,
the report in this directory.

## Outcome

**Closed by its stop rule.** Phase 1 of the plan (`2026-11-b18-budget.md`) aimed to get the GEMMs to ≥ 60% of
measured FMA throughput at batch 16, and to close the phase if they stayed below 45% after fixing each limit the
ladder showed.

- **Best whole kernel:** 41.7% of the FMA floor, in one case at batch 64. At the batch-16 Winograd shape, the best is
  33%.
- **No default changed.** The kernels the network uses generate exactly the same code as before this phase.
- **Change from prior:** none. Review time stays at 32279.6–33574.2 ms (#81, batch 16 two at once).
- **Against native:** still 1.95–2.05× native's 16412.1–16596.1 ms. At batch 16 the network still does 608.8
  evaluations/s against native's 1029.36 (#84).

## How it was measured

- **GEMM benchmark ladder:** `?gemmLadder=1` on the GEMM benchmark (`test/gpu/gemm-bench.js`).
  - The device's ceilings are measured first in the same session (`test/gpu/ceilings.js`).
  - Each GEMM v3 shape is measured whole and by stage (`GemmV3Config.stage`, measurement only):
    - **loop:** the main loop's arithmetic from workgroup memory (`stage: 'inner'` at K and 8 × K; the loop's own time
      is the difference ÷ 7, so the first chunk's load and the epilogue drop out);
    - **regs:** the same loop with its operands held in registers (`stage: 'innerRegs'`: the same barriers and FMAs,
      no workgroup-memory reads);
    - **loads:** the global loads into workgroup memory with no arithmetic (`stage: 'loads'`).
  - Each shape is also measured with four candidate fixes, each a new off-by-default `GemmV3Config` option:
    - **vload:** vec4 global loads where aligned (`vload`);
    - **blocks:** a different thread layout (`layout: 'blocks'`, each 32-thread warp covering an 8 × 4 block);
    - **prefetch:** each step's operands read one step ahead (`prefetch`);
    - **double:** double-buffered workgroup memory (`double`, which already existed).
  - Every whole-kernel output matched GEMM v2 exactly, and the CPU reference at batch 16 (`ok: true`).
- **Shapes:** b18's three GEMMs at batches 16 and 64:
  - the Winograd GEMM (M = 192, K = 192, N = 25 × batch, 36 matrices);
  - the 1×1 convolutions 384→192 and 192→384.
- **Report:** [`…-edge-nvidia.json`](2026-11-b18-gemm-ladder-edge-nvidia.json) (2026-11-26T17:00:58Z), with its
  nvidia-smi log [`…-edge-nvidia-clocks.csv`](2026-11-b18-gemm-ladder-edge-nvidia-clocks.csv).
  - The URL was `?gemmBatches=16,64&gemmLadder=1&gemmLadderShapes=[[4,8,16,16,16],[4,4,16,16,16],[8,8,16,16,16],[8,4,16,8,16],[8,8,16,8,8]]&gemmLadderVariants=vload,blocks,prefetch,double`.
  - Ceilings in this run: FMA 19.05 (f32) and 19.49 (f16) TFLOPS; copy 337.3–341.3 GB/s.
  - The GPU was at its power cap (`0x4`) in 592 of 621 samples with utilisation ≥ 90%; median SM clock 1470 MHz.
- **Earlier runs:** eight development runs found the same picture step by step (each stage, then each fix). Their
  code changed between runs, so only this consolidated run, which has every stage and fix, is kept.
- **Run-to-run variation:** in those runs the same kernel's time varied by up to about ±5–10% between runs.

## Results

Shapes are written rm.rn.tk.wy.wx (rows and columns per thread, chunk, threads along M and N). The default is
4.8.16.16.16, 256 threads. All figures are % of the FMA floor (flops ÷ 19.49 TFLOPS); higher is better.

**Batch 16, Winograd GEMM** (floor 0.0545 ms):

| Shape (threads) | Whole | vload | blocks | prefetch | double | double + vload | Loop | Loop, regs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4.8.16.16.16 (256), default | 33 | 31 | 30 | 26 | 29 | 27 | 43 | 71 |
| 4.4.16.16.16 (256) | 33 | 30 | 29 | 28 | 28 | 28 | 43 | 69 |
| 8.8.16.16.16 (256) | 25 | 23 | 23 | 26 | 24 | 24 | 40 | 51 |
| 8.4.16.8.16 (128) | 32 | 33 | 30 | 32 | 30 | 31 | 55 | 72 |
| 8.8.16.8.8 (64) | 29 | 25 | 28 | 26 | 24 | 24 | 63 | 65 |

The 1×1 shapes and batch 64 have the same pattern, in the report. Their best whole kernels: 39% (1×1 384→192, batch
16, default shape) and 41.7% (Winograd, batch 64, 8.4.16.8.16 with vload).

**Loads alone** (ms per dispatch, batch-16 Winograd GEMM): 0.0687–0.1239 depending on shape; the default's is 0.0733.
For comparison, the compulsory A, B and C bytes (13.7 MB) take 0.040 ms at the measured copy bandwidth.

## What the ladder showed, and what each fix did

1. **The loop's reads from workgroup memory are the largest limit for the default shape.**
   - With operands in registers the same loop reaches 71% (81% at batch 64). Reading them from workgroup memory, it
     reaches 43%.
   - With fewer threads per workgroup the reads cost less: 128 threads 55%, 64 threads 63%.
   - Tried, and no change in the loop: a different thread layout (`blocks`: 42% against 43%) and reading one step
     ahead (`prefetch`: 43%).
   - Tried, and slower: larger per-thread tiles (8.8.16.16.16: 40%).
   - Unexplained: why 256-thread workgroups pay more for these reads. The measurements don't settle it, and I have no
     profiler for the browser's GPU work.
2. **Loads aren't hidden behind the arithmetic.**
   - For each shape, whole ≈ loop + fixed per-workgroup cost + most of the loads. For example, 8.8.16.8.8 at batch 16
     (development run): 0.184 ≈ 0.085 + 0.024 + 0.075 ms.
   - Tried: vec4 loads (`vload`), which left the load stage unchanged (0.0733 → 0.0696 ms for the default), and double
     buffering (`double`), which was slower or no better in every case.
   - The shapes with the fastest loops (64 threads) have the slowest loads (0.1108–0.1376 ms), so their whole kernels
     end up no faster.
3. **Fixed cost per workgroup** (first chunk and epilogue): 0.011–0.041 ms per dispatch at batch 16, depending on
   shape (development run). Not worked on: it was the smallest of the three.

## What this means for the budget

The plan's route to native parity without matrix hardware needed the GEMMs at about 60% (`2026-11-b18-budget.md`).
These kernels stayed at 25–42% after every fix tried. With the GEMMs at the best whole-kernel rate seen at batch 16
(39%, the 1×1 shape), even transforms at 100% of copy bandwidth would leave the batch-16 pass at about 18.1 ms:
5.23 ÷ 0.39 + 3.60 + 0.75 + 0.32. That's an estimate from these figures, above the 15.54 ms that parity needs.

**So, on these measurements, the vector-unit path can't reach native's network throughput on this GPU.** Phase 2
(the transforms, predicted about −2 ms, about −8% of the pass) still applies. The matrix-hardware question (Phase 3)
is now the main one for parity, and it's yours to decide.

## Kept in the code

Measurement options only, all off by default: `GemmV3Config.stage` ('inner', 'innerRegs', 'loads'), `innerRepeat`,
`vload`, `layout`, `prefetch`. The GEMM benchmark has `?gemmLadder`, `?gemmLadderShapes` and `?gemmLadderVariants`.
The ceilings moved to `test/gpu/ceilings.js`, shared with the budget benchmark.
