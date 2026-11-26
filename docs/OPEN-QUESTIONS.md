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
