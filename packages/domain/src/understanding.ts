import type { ConfidenceLevel, Id, IsoDateTime, Trend, UnitInterval } from "./common.js";

/**
 * The player's relationship with a concept (Player Understanding Graph).
 * Understanding (can they explain it?) and application (do they use it in games?)
 * are tracked separately on purpose.
 */
export interface ConceptAssessment {
  playerId: Id;
  conceptId: Id;
  understanding: UnitInterval;
  application: UnitInterval;
  /** How certain StoneMind is, not how good the player is. */
  confidence: ConfidenceLevel;
  trend: Trend;
  lastUpdated: IsoDateTime;
}

export interface Strength {
  id: Id;
  playerId: Id;
  conceptId: Id;
  evidenceWeight: number;
  confidence: ConfidenceLevel;
  impact: UnitInterval;
  /** Set when this strength amplifies the player's preferred style. */
  amplifiesStyleDimensionId?: Id;
}

export type WeaknessStatus = "not-a-weakness" | "potential" | "confirmed";

export interface Weakness {
  id: Id;
  playerId: Id;
  conceptId: Id;
  evidenceWeight: number;
  confidence: ConfidenceLevel;
  impact: UnitInterval;
  status: WeaknessStatus;
  /**
   * Set when this weakness prevents the player from expressing their preferred style.
   * Style-limiting weaknesses receive elevated coaching priority.
   */
  limitsStyleDimensionId?: Id;
}
