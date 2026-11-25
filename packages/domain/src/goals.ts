import type { Id, IsoDateTime } from "./common.js";

/** The player defines success. See docs/StoneMind-Goal-Management.md. */
export type GoalCategory =
  | "strength-improvement"
  | "tournament-preparation"
  | "opponent-preparation"
  | "concept-mastery"
  | "teaching"
  | "exploration"
  | "personal-enjoyment";

export type GoalStatus = "proposed" | "active" | "paused" | "achieved" | "abandoned" | "archived";

export interface Goal {
  id: Id;
  playerId: Id;
  title: string;
  description?: string;
  category: GoalCategory;
  status: GoalStatus;
  /** Lower number = higher priority. */
  priority: number;
  startDate?: IsoDateTime;
  targetDate?: IsoDateTime;
  successCriteria: string[];
}
