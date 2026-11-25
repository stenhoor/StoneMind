import type { Id, IsoDateTime, Metadata } from "./common.js";

/** The central entity. Everything else exists to help understand and support the player. */
export interface Player {
  id: Id;
  displayName: string;
  createdDate: IsoDateTime;
  lastActiveDate: IsoDateTime;
  /** Rank as the player states it, e.g. "5k" or "1d". */
  rankClaimed?: string;
  /** Rank as StoneMind estimates it from evidence. Never authoritative (StoneMind is not a ranking authority). */
  rankEstimated?: string;
  preferences: Metadata;
  activeGoals: Id[];
  archivedGoals: Id[];
  profileMetadata: Metadata;
}

/**
 * Long-term, descriptive identity. This is NOT assessment information.
 */
export interface PlayerProfile {
  playerId: Id;
  preferences: Metadata;
  learningHistory: string[];
  milestones: string[];
  achievements: string[];
  studyHabits: string[];
}
