# Spike 0001: KataGo in the browser on TypeScript + WebGPU

Status: **Planned**, not started.

Date: 2026-11-25

Owner decisions: recorded from an interview on 2026-11-25. They are marked ✅ below.

---

# Question

Can a browser-only KataGo engine, written in TypeScript with hand-written WebGPU (WGSL) kernels, run faster than the existing browser approaches (WASM, onnxruntime-web, TensorFlow.js)? Can it reach a level of analysis speed that makes coaching practical, with no install?

A local native KataGo must stay a first-class option. Nothing in this spike may block it.

---

# What already exists

I read the source of every project below, not just its README. All numbers are the projects' own, each from a single machine, so they are indicative and can't be compared row to row.

| Project | Licence | Search | Network inference | Reported speed |
|---|---|---|---|---|
| [Kaya](https://github.com/kaya-go/kaya) | AGPL-3.0 | Simplified TS PUCT (win rate only, Q=0 for unvisited moves); features miss ladders and pass-alive | onnxruntime-web: WebGPU EP, WASM fallback; ONNX from the PyTorch checkpoint | b18: ~10 evals/s on WebGPU, ~8 on 8-thread WASM (Radeon 8060S) |
| [yagoo](https://github.com/yet-another-ai/yagoo) | Apache-2.0 | Simplified Rust→WASM PUCT (no LCB, no subtree bias) | onnxruntime-web WebGPU; batch ≤ 4; JSON per eval | No throughput numbers; fp16 fails its accuracy check |
| [katago-web](https://github.com/JuneKim0712/katago-web) | MIT | Upstream C++ search via WASM + JSPI (can't be built from the repo), or a TS port without LCB | onnxruntime-web WebGPU, or TF.js with two custom WGSL kernels | b10c384: ORT WebGPU **3× slower** than TF.js WebGPU (Intel Arc 140V) |
| [web-katrain](https://github.com/Sir-Teo/web-katrain) | MIT | A careful 4.7k-line TS port of KataGo's search, single-threaded | Eager TF.js ops, fp32, nothing fused, several readbacks per batch; parses `.bin.gz` directly | No throughput numbers; its public site runs single-threaded WASM only |
| [katago-webgpu](https://github.com/saigo-online/katago-webgpu) | MIT | Upstream C++ search in WASM (Asyncify) | C++ → WGSL via Dawn; Winograd F(2,3) with a naive matmul | Native Dawn numbers only; ~250 visits/s in one browser log line |
| [katago-rs](https://github.com/Sagebati/katago-rs) | MIT | Faithful Rust port of KataGo search, async on one thread | burn/cubecl emits WGSL; Winograd F(4×4,3×3), fused BN+act, f16 | b18 f16: **972 evals/s at batch 64**, ahead of native OpenCL (465–597) on the same RX 6800 XT; 474–672 visits/s |
| [katahex-web](https://github.com/Phlogistique/katahex-web) (Hex) | AGPL-3.0 | KataHex C++ in WASM, several leaf evals per thread | **TypeScript + raw WGSL**: Winograd F(4×4,3×3), per-shape GEMM, f16 | b18: **224 evals/s at batch 64 vs native OpenCL 128**; ORT ~50 and TF.js ~26 (Iris Xe); Firefox at half Chrome's speed |

## What decides speed

1. **Hand-written kernels beat generic runtimes by 4–10×.** The two projects with their own WGSL beat native OpenCL at large batch sizes. onnxruntime-web and TF.js lose to dispatch overhead, ops they don't support, and unfused layers.
2. **The kernel recipe is known.** KataGo's OpenCL backend and both fast projects use:
   - Winograd F(4×4,3×3) for 3×3 convolutions.
   - A tiled GEMM generated for each shape.
   - Batch norm and activation fused into the transforms.
   - fp16 storage and maths where the adapter has `shader-f16`.
   - A pre-built plan per batch size, with one submit and one readback.
3. **Batch size is what unlocks the GPU.** The gains appear at batches of 32–64. The search must keep that many leaves in flight. katahex-web had to add several evals per thread to do so.
4. **After that, the search is the bottleneck.** katago-rs's single-threaded search reaches 0.73× native on small nets. A TypeScript search must be lean and fully asynchronous.

## Upstream KataGo (1.18.2)

- **Model format.** The `.bin.gz` format is a gzipped token stream with `@BIN@` float32 blocks, versions 8–17.
- **Transformers.** Version 17 adds transformer nets, which upstream calls the best for their cost. Examples are tf2-b10c384 (~9.6 GFLOP per eval) and tf3-b11c768.
- **Costs per 19×19 eval:** b18c384nbt is about 19 GFLOP and b28c512nbt about 52 GFLOP. This is my arithmetic from `docs/NetworkArchitectures.md`.
- **Input features** (`nninputs.cpp`) need full rules, ladder search, Benson pass-alive detection and 8 symmetries.
- **fp16.** KataGo's fp16 handling for convnets rescales Mish activations by 8 (`scale8`). A naive fp16 cast fails, as it did in yagoo.
- **Local engine.** The Analysis Engine speaks newline-delimited JSON over stdin/stdout. It streams partial results (`reportDuringSearchEvery`) and accepts `terminate`. KataGo has no network server, so a browser needs a small local bridge.
- **Licence.** KataGo's code is MIT, and porting it needs the copyright notice. Anything derived from CLBlast is Apache-2.0. The weights' licence is **unverified**: katagotraining.org was unreachable from the research session.

---

# Decisions

- ✅ **Architecture: pure TypeScript + WGSL.** The search is in TypeScript and the network runs in hand-written WGSL. There is no WebAssembly, which follows the smaltworks standard: no WebAssembly in an engine, and TypeScript writes the GPU code. No COOP/COEP headers are needed.
- ✅ **smaltkit/gpu: looked at first.**
  - What it has: one shared device, a pipeline cache, buffer-limit helpers and a naive fp32 matmul.
  - What it lacks: its shared device requests no optional features (`shader-f16`, `subgroups`, `timestamp-query`), and it isn't on npm yet.
  - **Decision:** the spike writes its own device setup with the same API shape. It then proposes a smaltkit change so callers can request features. Using smaltkit as a pinned dependency needs the owner's yes.
- ✅ **Code reuse: learn, don't copy.** MIT projects and KataGo's own kernels are references. StoneMind's code is written fresh, crediting designs it borrows. No AGPL code (Kaya, katahex-web) is read for anything beyond ideas.
- ✅ **Networks:** b18c384nbt first, then the transformer tf2-b10c384 in the same spike.
- ✅ **Pass mark** on the owner's Linux + NVIDIA machine:
  1. Beat onnxruntime-web (WebGPU and WASM) and TF.js on the same machine and browser.
  2. Live analysis at **≥ 200 visits/s** with b18.
  3. A full 19×19 game review, **~250 moves × 100 visits, in under 5 minutes**.

  Matching native KataGo is not a goal, but native is measured as a reference.
- ✅ **Native baseline:** KataGo with CUDA or TensorRT on the owner's Linux machine. It provides speed numbers, accuracy goldens (`kata-raw-nn`, analysis JSON) and the ONNX export for the onnxruntime-web baseline (`katago dumponnx`).
- ✅ **Weights:** the player supplies the `.bin.gz`, or the page downloads it directly from katagotraining.org if CORS allows. Nothing is redistributed until the weights' licence is checked.
- ✅ **Local KataGo:** one engine contract (the Analysis Engine JSON) shared by both engines, plus a tiny local bridge. The spike tests the bridge from an https page against the native engine.
- ✅ **Home:** new independent sub-projects in StoneMind, in their own PRs, marked experimental. The benchmark page is published on GitHub Pages so it can be run on any machine with only a browser (a Mac included).
- ✅ **Test rigs:** Linux + NVIDIA (primary), Windows + NVIDIA (reboot), and a Mac through the browser only, when available.

---

# Plan

## Sub-projects

| Package | What | Depends on |
|---|---|---|
| `@stonemind/engine` | The engine contract: Analysis Engine JSON types and an `AnalysisEngine` interface (query → async stream of responses; terminate). Both engines implement it. | domain |
| `@stonemind/engine-webgpu` | The browser engine: model loader, rules and features, WGSL network, TypeScript search. Experimental. | engine |
| `@stonemind/katago-bridge` | A local bridge: spawns `katago analysis` and relays JSON lines over a WebSocket on 127.0.0.1, with an Origin check and a session token. Node built-ins only. | engine (types) |
| `apps/engine-bench` | The benchmark page, published to GitHub Pages. It runs every engine on the same positions and records results as JSON. | engine, engine-webgpu |

New packages follow the smaltworks standards from the start: `node --test`, plain ES2022 modules that load without a bundler, and Node ≥ 22.18.

## Phases

1. **Contract and bridge.**
   - The `AnalysisEngine` interface and message types.
   - The bridge.
   - An https test page that drives the owner's native KataGo through the bridge, which proves the local path early.
   - Exit: a query from the GitHub Pages page reaches local KataGo and streams results back, with the browser's local-network permission behaviour documented for Chrome and Firefox.
2. **Model and reference forward pass.**
   - A `.bin.gz` parser (v8–17) using the platform `DecompressionStream`.
   - A plain fp32 TypeScript forward pass, slow but simple, as the correctness reference.
   - Exit: outputs match native `kata-raw-nn` within 1e-4 on golden positions.
3. **Rules and input features.**
   - Board, captures, ko and superko.
   - Ladders, Benson pass-alive areas, the V7 global features and the 8 symmetries.
   - Exit: NN outputs on 100+ positions, ladders and seki included, match native within tolerance.
4. **WebGPU network (b18).**
   - Device setup requesting `shader-f16`, `subgroups` and `timestamp-query` when present.
   - Winograd F(4×4,3×3), generated tiled GEMMs, fused BN+Mish, global pooling and heads.
   - Plans pre-built per batch size, with one submit and one readback.
   - fp32 first, then fp16 with scale8-style handling.
   - Exit: evals/s vs batch on the bench page beats onnxruntime-web WebGPU, using the ONNX from `katago dumponnx`, at every batch size. fp16 must stay within KataGo's own fp16 error bounds.
5. **Search.**
   - An asynchronous TypeScript search that keeps up to 64 leaves in flight with virtual loss.
   - Essentials first: log-cpuct PUCT, FPU, the NN cache, random symmetries, score utility and the analysis outputs.
   - Then each refinement, measured: LCB, subtree value bias, uncertainty weighting, optimistic policy, graph search.
   - Exit: ≥ 200 visits/s live with b18, a full review in under 5 minutes, and top-move agreement with native on a test set.
6. **Transformer (tf2-b10c384).**
   - RMSNorm, RoPE, attention and SwiGLU kernels.
   - Exit: correct outputs, with evals/s reported against b18.
7. **Report.**
   - Results from each rig, the pass/fail of each criterion, and what to keep.
   - A recommendation for StoneMind's engine.

## Measurement

- **The same positions, net and browser** for every engine. Warm-up runs are excluded, and each run reports the median of several.
- **What is recorded:** GPU timestamp queries where available, plus wall-clock evals/s per batch size and visits/s for search.
- **Baselines, run in the same page:**
  - onnxruntime-web with the WebGPU EP and the WASM EP.
  - TF.js with WebGPU, if its loading costs little.
  - Native KataGo through the bridge, as the reference.
- **Results** are saved as JSON in the repo, with GPU, driver, browser and OS recorded.

## Needs the owner's approval

- **Benchmark-only outside code.** onnxruntime-web (MIT) and TF.js (Apache-2.0), loaded only by `apps/engine-bench` and never by the product, to measure the alternatives fairly.
- **The weights' licence** must be checked on katagotraining.org before any network file is hosted.

## Risks

- **Browser and driver differences.** Firefox ran at half Chrome's speed in one study, and Intel/Mesa has a WGSL loop-variable bug.
- **fp16 accuracy.** It needs scale8-style handling, and transformers can't use scale8.
- **Rules and features are large to port**, though feature goldens against native limit the risk.
- **Local-network access is changing.** Chrome is replacing Private Network Access with a Local Network Access permission prompt. Phase 1 tests this first.
- **Single-threaded search.** The search can't run on several cores; batching leaves asynchronously is the mitigation.
