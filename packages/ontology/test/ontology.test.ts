import { describe, expect, it } from "vitest";
import {
  findDependencyCycles,
  resolveSeed,
  SEED,
  STYLE_DIMENSIONS,
  validateOntology,
} from "../src/index.js";

const ontology = resolveSeed();

/**
 * Known open issues in Ontology Seed v0.1, recorded in docs/OPEN-QUESTIONS.md.
 * When the seed is fixed, shrink these lists. A new entry here must be a deliberate decision.
 */
const KNOWN_CYCLES = [["direction-of-play", "whole-board-judgment"]];
const KNOWN_UNRESOLVED_TARGETS = new Set([
  "Shape", "Fighting", "Endgame", "Opening Balance", "Scoring", "Tactics",
  "Future Development", "Weaknesses", "Overconcentration", "Strategic Pressure",
  "Reduction Decisions", "Invasion Decisions", "Shape Recognition", "Endgame Decisions",
  "Strategic Decisions", "Memorization", "Experience", "Improvement", "Self-Awareness",
  "Secure Profit", "Tactical Variation",
]);

describe("ontology seed", () => {
  it("covers every seed concept", () => {
    expect(ontology.concepts).toHaveLength(SEED.concepts.length);
    expect(ontology.concepts.length).toBeGreaterThan(50);
  });

  it("uses kebab-case ids", () => {
    for (const c of ontology.concepts) expect(c.id).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
  });

  it("has no structural issues other than known dependency cycles", () => {
    const issues = validateOntology(ontology).filter((i) => i.kind !== "dependency-cycle");
    expect(issues).toEqual([]);
  });

  it("has only the known dependency cycles", () => {
    expect(findDependencyCycles(ontology.relationships)).toEqual(KNOWN_CYCLES);
  });

  it("introduces no new unresolved concept references", () => {
    const fresh = ontology.unresolved.filter((u) => !KNOWN_UNRESOLVED_TARGETS.has(u.target));
    expect(fresh).toEqual([]);
  });

  it("aliases point at real concepts", () => {
    const ids = new Set(ontology.concepts.map((c) => c.id));
    for (const targets of Object.values(SEED.aliases)) {
      for (const id of targets) expect(ids).toContain(id);
    }
  });

  it("expands multi-target aliases (Sente/Gote Evaluation)", () => {
    const yose = ontology.relationships.filter(
      (r) => r.sourceConcept === "yose" && r.relationshipType === "DEPENDS_ON",
    );
    expect(yose.map((r) => r.targetConcept).sort()).toEqual(["counting", "gote", "sente"]);
  });
});

describe("dependency cycle detection", () => {
  const rel = (s: string, t: string, type: "DEPENDS_ON" | "PREREQUISITE_FOR" = "DEPENDS_ON") => ({
    id: `${s}-${t}`, sourceConcept: s, targetConcept: t, relationshipType: type,
  });

  it("treats DEPENDS_ON and PREREQUISITE_FOR as inverse directions", () => {
    // a depends on b, and a is a prerequisite for b => cycle
    expect(findDependencyCycles([rel("a", "b"), rel("a", "b", "PREREQUISITE_FOR")])).toEqual([["a", "b"]]);
  });

  it("accepts a DAG", () => {
    expect(findDependencyCycles([rel("c", "b"), rel("b", "a"), rel("c", "a")])).toEqual([]);
  });
});

describe("style dimensions", () => {
  it("defines the seven Style Model dimensions with unique ids", () => {
    expect(STYLE_DIMENSIONS).toHaveLength(7);
    expect(new Set(STYLE_DIMENSIONS.map((d) => d.id)).size).toBe(7);
  });
});
