import type { Id, IsoDateTime } from "./common.js";

export interface Problem {
  id: Id;
  title: string;
  source: string;
  concepts: Id[];
  difficulty?: string;
}

export interface ProblemAttempt {
  id: Id;
  problemId: Id;
  playerId: Id;
  result: "solved" | "failed" | "skipped";
  attemptTime: IsoDateTime;
  /** Time spent, in seconds. */
  durationSeconds?: number;
  hintsUsed: number;
  notes?: string;
}
