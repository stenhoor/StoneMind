/**
 * Shared primitives for the StoneMind data model.
 * See docs/StoneMind-Data-Model-Specification.md.
 */

/** Opaque identifier. Kept as a plain string so records stay portable (JSON export). */
export type Id = string;

/** ISO-8601 date or date-time string. Strings, not Date objects, so records round-trip through JSON. */
export type IsoDateTime = string;

/** A value normalised to the closed range [0, 1]. */
export type UnitInterval = number;

/**
 * How certain StoneMind is about a conclusion.
 * High confidence requires substantial, cross-validated evidence
 * (docs/StoneMind-Heuristics-and-Inference-Engine.md, "Confidence Framework").
 */
export type ConfidenceLevel = "low" | "moderate" | "high";

export type Trend = "improving" | "stable" | "declining" | "unknown";

/** Free-form, namespaced extension data. Lets plugins attach data without schema changes. */
export type Metadata = Record<string, unknown>;
