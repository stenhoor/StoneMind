import type { StyleDimension } from "@stonemind/domain";

/**
 * The seven style dimensions from docs/StoneMind-Style-Model.md.
 * Convention: assessment value -1 = leftExtreme, +1 = rightExtreme.
 */
export const STYLE_DIMENSIONS: readonly StyleDimension[] = [
  { id: "tactical-strategic", name: "Tactical ↔ Strategic", leftExtreme: "Tactical", rightExtreme: "Strategic" },
  { id: "aggressive-conservative", name: "Aggressive ↔ Conservative", leftExtreme: "Aggressive", rightExtreme: "Conservative" },
  { id: "territorial-influence", name: "Territorial ↔ Influence", leftExtreme: "Territorial", rightExtreme: "Influence" },
  { id: "complexity-simplification", name: "Complexity ↔ Simplification", leftExtreme: "Complexity", rightExtreme: "Simplification" },
  { id: "local-whole-board", name: "Local ↔ Whole Board", leftExtreme: "Local", rightExtreme: "Whole Board" },
  { id: "initiative-stability", name: "Initiative ↔ Stability", leftExtreme: "Initiative", rightExtreme: "Stability" },
  { id: "risk-seeking-risk-avoiding", name: "Risk Seeking ↔ Risk Avoiding", leftExtreme: "Risk Seeking", rightExtreme: "Risk Avoiding" },
];
