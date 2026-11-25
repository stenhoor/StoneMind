import type { Id, IsoDateTime } from "./common.js";

export interface TeachingSession {
  id: Id;
  playerId: Id;
  provider: string;
  startDate: IsoDateTime;
  endDate?: IsoDateTime;
  objectives: string[];
  findings: string[];
}
