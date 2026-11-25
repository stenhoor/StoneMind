import type { ConfidenceLevel, Id, IsoDateTime } from "./common.js";

export type RecommendationCategory =
  | "study"
  | "practice"
  | "review"
  | "reading"
  | "teaching-game"
  | "problem-set"
  | "tournament-preparation"
  | "opponent-preparation";

/**
 * Coaching output. Every recommendation must be explainable:
 * why it was made, what evidence supports it, what assumptions were used.
 */
export interface Recommendation {
  id: Id;
  playerId: Id;
  goalId?: Id;
  category: RecommendationCategory;
  title: string;
  rationale: string;
  supportingEvidence: Id[];
  assumptions: string[];
  confidence: ConfidenceLevel;
  /** Lower number = higher priority. */
  priority: number;
  createdDate: IsoDateTime;
}

export interface StudyPlan {
  id: Id;
  playerId: Id;
  goalId?: Id;
  createdDate: IsoDateTime;
  summary: string;
}

export interface StudyActivity {
  id: Id;
  studyPlanId: Id;
  type: RecommendationCategory;
  description: string;
  /** Estimated duration in minutes. */
  estimatedDuration?: number;
  linkedConcepts: Id[];
  linkedResources?: Id[];
}
