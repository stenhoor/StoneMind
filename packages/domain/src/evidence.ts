import type { ConfidenceLevel, Id, IsoDateTime, Metadata } from "./common.js";

export type EvidenceSourceType =
  | "human-game"
  | "teaching-game"
  | "tsumego"
  | "problem-set"
  | "professional-study"
  | "journal-entry"
  | "teacher-review"
  | "user-feedback";

/**
 * Atomic unit of evidence. Observations are facts: no interpretation happens here.
 * Pipeline: Evidence -> Observations -> Indicators -> Hypotheses -> Confidence -> Graph Updates -> Recommendations.
 */
export interface Observation {
  id: Id;
  source: EvidenceSourceType;
  /** Id of the source record (game, problem attempt, session, ...). */
  sourceRef?: Id;
  /** Machine-readable observation kind, e.g. "FailedInvasion". */
  type: string;
  timestamp: IsoDateTime;
  summary: string;
  associatedConcepts: Id[];
  confidence: ConfidenceLevel;
  metadata?: Metadata;
}

export interface EvidenceRecord {
  id: Id;
  playerId: Id;
  observationId: Id;
  concepts: Id[];
  weight: number;
  confidence: ConfidenceLevel;
}

/** A pattern grouping several observations, e.g. "Poor Invasion Performance". */
export interface Indicator {
  id: Id;
  playerId: Id;
  name: string;
  observationIds: Id[];
  concepts: Id[];
}

/** A provisional explanation of indicators, e.g. "Weak Invasion Judgment". */
export interface Hypothesis {
  id: Id;
  playerId: Id;
  statement: string;
  indicatorIds: Id[];
  concepts: Id[];
  confidence: ConfidenceLevel;
  status: "open" | "supported" | "refuted";
}
