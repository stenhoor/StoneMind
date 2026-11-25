import type { Id, Metadata, UnitInterval } from "./common.js";

/** Concept categories from docs/StoneMind-Ontology-Seed.md. */
export type ConceptCategory =
  | "meta"
  | "opening"
  | "strategic"
  | "shape"
  | "fighting"
  | "territorial"
  | "tactical"
  | "endgame"
  | "learning"
  | "style";

export type ConceptDifficulty = "beginner" | "intermediate" | "advanced" | "expert";

export interface Concept {
  id: Id;
  name: string;
  description?: string;
  category: ConceptCategory;
  difficulty?: ConceptDifficulty;
  /** Topics this concept is concerned with (seed "Concerns:" lists, mostly for meta concepts). */
  concerns?: string[];
  commonMisunderstandings?: string[];
  metadata?: Metadata;
}

/**
 * Relationship types. The first seven come from the Data Model / Ontology Design specs.
 * RELATED_TO, PRODUCES and LEADS_TO are used by the Ontology Seed but are not yet defined
 * in the Data Model spec (open question, see docs/OPEN-QUESTIONS.md).
 */
export type RelationshipType =
  | "SUPPORTS"
  | "INFLUENCES"
  | "PREREQUISITE_FOR"
  | "REINFORCES"
  | "EMERGES_FROM"
  | "CONTRASTS_WITH"
  | "DEPENDS_ON"
  | "RELATED_TO"
  | "PRODUCES"
  | "LEADS_TO";

/** Relationship types that express a learning dependency and must therefore be acyclic. */
export const DEPENDENCY_RELATIONSHIP_TYPES: readonly RelationshipType[] = [
  "DEPENDS_ON",
  "PREREQUISITE_FOR",
];

export interface ConceptRelationship {
  id: Id;
  sourceConcept: Id;
  targetConcept: Id;
  relationshipType: RelationshipType;
  strength?: UnitInterval;
}
