# Open Questions from the Initial Documentation Review

Version: 0.1

Collected while turning the specifications into code (project initialisation, 2026-11-25).
Each item links the code that currently carries the interim decision. Resolve items by
updating the relevant specification and then the code.

---

# Ontology

## 1. Dependency cycle: Direction of Play ↔ Whole Board Judgment

The Ontology Seed says Direction of Play *depends on* Whole Board Judgment, and Whole Board
Judgment *depends on* Direction of Play. Learning dependencies need to be acyclic to derive
learning paths. One of these should probably be `RELATED_TO` or `REINFORCES`.

Interim: recorded as a known cycle in `packages/ontology/test/ontology.test.ts`.

## 2. Concepts referenced but never defined

The seed refers to 21 names that are not concepts: **Shape** (used 8 times and listed as a
root dependency), **Fighting**, **Endgame**, Tactics, Opening Balance, Scoring, Future
Development, Weaknesses, Overconcentration, Strategic Pressure, Reduction Decisions,
Invasion Decisions, Shape Recognition, Endgame Decisions, Strategic Decisions,
Memorization, Experience, Improvement, Self-Awareness, Secure Profit, Tactical Variation.

Decide for each one: add a concept, alias it to an existing concept, or drop it.
"Shape", "Fighting" and "Endgame" are also seed *category* names, so they may need to be
concepts in their own right.

Interim: `resolveSeed()` reports them as `unresolved`, and the tests fail if a new one appears.

## 3. Aliases chosen during encoding

These seed names were mapped to existing concepts and need expert confirmation:
Framework Building → Framework Development; Corners → Corner Development;
Influence Evaluation → Influence; Whole Board Evaluation → Whole Board Judgment;
Weak Group Identification → Weak Groups; Sente/Gote Evaluation → Sente + Gote;
Frameworks → Framework; plural forms (Capturing Races, Invasions).

Interim: the `aliases` block in `packages/ontology/src/seed.json`.

## 4. Framework vs. Framework Development

"Framework" (territorial) and "Framework Development" (opening) overlap. Merge them, or
define how they differ.

## 5. Relationship taxonomy

- The Data Model lists `DEPENDS_ON` and `PREREQUISITE_FOR`. These are the same relation in
  opposite directions. Pick one canonical direction.
- The Ontology Design spec does not list `DEPENDS_ON`.
- The seed uses relations no spec defines: Related / Related To, Produces, Leads To,
  Critical Dependency For, Concerns.

Interim: `RELATED_TO`, `PRODUCES` and `LEADS_TO` were added to `RelationshipType`.
"Critical Dependency For" is encoded as `PREREQUISITE_FOR`, and "Concerns" as a text field.

## 6. Style concepts vs. style dimensions

The seed's Style Concepts (Aggression, Stability, ...) duplicate the Style Model
dimensions. Decide whether style lives in the ontology, in the Style Graph only, or in both
with explicit links.

# Style Model

## 7. Inconsistent dimension names and count

- Complexity ↔ Simplification (Style Model), Complexity ↔ Simplicity (Ontology Design),
  Complicated ↔ Simplifying (Data Model, Roadmap).
- The Data Model lists six dimensions and omits Initiative ↔ Stability.

Interim: `STYLE_DIMENSIONS` follows the Style Model's seven dimensions and names.

# Data Model

## 8. Fields added while typing the model

The Data Model sketches have no field types. Interim choices in `packages/domain`:
- Confidence uses `low | moderate | high` (from the Heuristics spec), not a number.
- Understanding and application use `[0, 1]`. A style value uses `[-1, +1]`.
- Fields added: `Goal.playerId`, `Game.playerId` / `playerColor`, `Recommendation.title` /
  `assumptions` / `createdDate` (needed for the explainability requirement),
  `Weakness.status` (from the Heuristics spec), `Weakness.limitsStyleDimensionId`,
  `Strength.amplifiesStyleDimensionId`, and the `Indicator` and `Hypothesis` entities
  (from the inference pipeline).

# Architecture and Providers

## 9. Smaltworks availability

Smaltworks is not available as a public package. Core code depends only on the
`@stonemind/storage` ports until its API and distribution are known.

## 10. Provider interface differences

The Technical Architecture `AIProvider` has `chat / summarize / explain`. The SDK version has
`summarize / explain / generateTeachingContent`. `EvidenceProvider` appears only in the
architecture document.

Interim: `@stonemind/provider-sdk` includes all of them, with `generateTeachingContent`
optional. It also adds `DataDisclosure` and consent enforcement (SDK Rule 4, Charter
"User Consent").

# Governance

## 11. Milestone 0 deliverables

The roadmap lists Governance as a Milestone 0 deliverable. There is no
CONTRIBUTING or governance document yet.

# Engines

## 12. Browser engine: passes and the end of the game (spike 0001)

The browser engine's rules engine follows KataGo's `BoardHistory`. The first version of the search supports area
scoring only (owner decision, 2026-11-26): passes, two passes (or simple ko's spight-like repeat) ending the game, and
area scoring with no group tax. Still refused with `UnsupportedPositionError`: the territory-scoring encore (Japanese
and Korean games past the main phase), ending a game with a group tax (stone-scoring, ancient-territory), no-result long
cycles, and rule sets with a button.

## 13. Browser engine: network calls in flight (spike 0001)

KataGo runs its search threads in parallel on the CPU while one neural-net server thread per model runs the GPU
(`numNNServerThreadsPerModel`, 1 by default in `analysis_example.cfg`). The browser engine runs on one JavaScript
thread, with no workers, so its search work never runs in parallel. While a network call runs on the GPU, the thread is
free (it waits with `await`), and letting more than one call be in flight lets it run another search's work in that
time. Decided in code, with no spec covering it:
- **`GpuNetwork` allows 2 calls in flight by default** (`maxInFlight`). Each has its own readback buffer, and calls
  are submitted in the order they're made.
- **`SharedEvaluator` spreads requests across calls.** When two calls can start, requests that arrived together are
  spread across them, each request whole, in arrival order. So two reviewed turns that asked together get two calls of
  16 rather than one of 32. KataGo's server takes everything queued, up to `nnMaxBatchSize`, when it is free.

Each search still has at most its batch of leaves in flight. Measured on the owner's laptop:
`docs/spikes/results/2026-11-b18-inflight.md`.

## 14. Browser engine: matrix hardware behind a browser switch (spike 0001)

Native KataGo's CUDA build can use the GPU's matrix hardware. In the browser it's reachable only through Chromium's
experimental `chromium-experimental-subgroup-matrix`, which Edge 158 on the owner's laptop offers only with
`--enable-unsafe-webgpu`. Decided in code, with no spec covering it:
- **The engine requests the feature whenever the adapter offers it**, and records the supported shapes.
- **`GpuNetwork` uses it automatically where it was measured faster** (`matrix: 'auto'`, the default; the owner's
  decision, 2026-11-26): the Winograd GEMMs and 1×1 convolutions, with f16 storage, in the shape measured fastest
  for the call's batch size on the owner's laptop (`MATRIX_GEMM_BY_BATCH`). Without the feature nothing changes.
- **A GEMM runs on matrix hardware at every batch size or at none.** The matrix and vector GEMMs round differently, so
  mixing them by batch size would make a position's result depend on how many positions shared its call; the three
  matrix shapes give identical results (the check page compares calls of 1, 4–5 and 14 exactly).

Open: whether the shapes and batch ranges measured on the owner's laptop suit other GPUs once browsers ship the
feature without a switch. Measured on the owner's laptop: `docs/spikes/results/2026-11-b18-matrix.md` and
`docs/spikes/results/2026-11-b18-matrix-ladder.md`.
