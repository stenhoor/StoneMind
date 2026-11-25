import type { Id, IsoDateTime, Metadata } from "./common.js";

export interface Game {
  id: Id;
  /** Location of the SGF record (file path, OPFS path, or URL). SGF stays the canonical game format. */
  sgfReference: string;
  date?: IsoDateTime;
  playerId: Id;
  playerColor?: "black" | "white";
  opponent?: string;
  /** Where the game came from, e.g. "ogs", "sgf-import". */
  source: string;
  /** SGF RE property value, e.g. "B+R", "W+3.5". */
  result?: string;
  metadata?: Metadata;
}

export interface GameAnalysis {
  id: Id;
  gameId: Id;
  /** Analysis provider id, e.g. "katago", "ogs-ai-review", "human-teacher". */
  provider: string;
  findings: Metadata[];
  concepts: Id[];
  summary?: string;
}
