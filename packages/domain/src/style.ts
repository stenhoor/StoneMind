import type { ConfidenceLevel, Id, IsoDateTime, Trend } from "./common.js";

/** Style is separate from skill. See docs/StoneMind-Style-Model.md. */
export interface StyleDimension {
  id: Id;
  name: string;
  leftExtreme: string;
  rightExtreme: string;
  description?: string;
}

export interface StyleAssessment {
  playerId: Id;
  dimensionId: Id;
  /** Position on the dimension: -1 = leftExtreme, 0 = balanced, +1 = rightExtreme. */
  value: number;
  confidence: ConfidenceLevel;
  trend: Trend;
  evidenceCount: number;
  /** Stable style vs. a style the player is currently experimenting with. */
  stability?: "stable" | "experimental" | "unknown";
}

export interface StyleEvolutionEntry {
  playerId: Id;
  dimensionId: Id;
  observedAt: IsoDateTime;
  value: number;
  note?: string;
}
