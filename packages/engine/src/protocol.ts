/**
 * KataGo's Analysis Engine protocol, as message types.
 *
 * Source: KataGo `docs/Analysis_Engine.md` (KataGo 1.18). Every StoneMind engine speaks this protocol:
 * a native KataGo through the local bridge, and the browser engine. Field names and meanings are KataGo's;
 * only the TypeScript shapes are ours. Fields KataGo marks optional are optional here.
 */

/** A player: "B" or "W". */
export type Player = 'B' | 'W';

/**
 * A board location as KataGo reads it: GTP coordinates such as "Q16" (columns skip "I"), "pass",
 * or explicit integer coordinates such as "(0,13)".
 */
export type Location = string;

/** A move or stone: [player, location]. */
export type Move = readonly [Player, Location];

/** Rules as a KataGo shorthand ("tromp-taylor", "chinese", "japanese", "korean", "aga", "new-zealand", ...) or a full rules object. */
export type Rules = string | RulesObject;

export interface RulesObject {
  koRule?: 'SIMPLE' | 'POSITIONAL' | 'SITUATIONAL';
  scoringRule?: 'AREA' | 'TERRITORY';
  taxRule?: 'NONE' | 'SEKI' | 'ALL';
  suicide?: boolean;
  hasButton?: boolean;
  whiteHandicapBonus?: '0' | 'N' | 'N-1';
  friendlyPassOk?: boolean;
}

export interface MoveRestriction {
  player: Player;
  moves: Location[];
  untilDepth: number;
}

/** An analysis query: one or more positions of one game to analyse. */
export interface AnalysisQuery {
  /** Required. Identifies every response to this query. */
  id: string;
  /** Required. The moves played, in order. Empty when the position has no move history. */
  moves: Move[];
  initialStones?: Move[];
  initialPlayer?: Player;
  /** Required. */
  rules: Rules;
  /** Optional but highly recommended. Half-integers only; within [-400, 400]. */
  komi?: number;
  whiteHandicapBonus?: '0' | 'N' | 'N-1';
  /** Required. */
  boardXSize: number;
  /** Required. */
  boardYSize: number;
  /** Turns to analyse: 0 is the initial position. Defaults to the last turn only. */
  analyzeTurns?: number[];
  maxVisits?: number;
  rootPolicyTemperature?: number;
  rootFpuReductionMax?: number;
  analysisPVLen?: number;
  includeOwnership?: boolean;
  includeOwnershipStdev?: boolean;
  includeMovesOwnership?: boolean;
  includeMovesOwnershipStdev?: boolean;
  includePolicy?: boolean;
  includePVVisits?: boolean;
  includeNoResultValue?: boolean;
  avoidMoves?: MoveRestriction[];
  allowMoves?: MoveRestriction[];
  focusMoves?: Location[];
  focusWeights?: number[];
  focusProb?: number;
  /** Per-query overrides of the engine's search parameters (cpuctExploration, humanSLProfile, ...). */
  overrideSettings?: Record<string, unknown>;
  /** Seconds between partial reports while a position is searched. */
  reportDuringSearchEvery?: number;
  priority?: number;
  priorities?: number[];
}

/** One candidate move in a result. */
export interface MoveInfo {
  move: Location;
  visits: number;
  edgeVisits?: number;
  winrate: number;
  scoreMean?: number;
  scoreStdev?: number;
  scoreLead: number;
  scoreSelfplay?: number;
  prior: number;
  noResultValue?: number;
  humanPrior?: number;
  utility?: number;
  lcb?: number;
  utilityLcb?: number;
  weight?: number;
  edgeWeight?: number;
  /** KataGo's ranking: 0 is the best. */
  order: number;
  playSelectionValue?: number;
  isSymmetryOf?: Location;
  pv: Location[];
  pvVisits?: number[];
  pvEdgeVisits?: number[];
  ownership?: number[];
  ownershipStdev?: number[];
}

/** Statistics for the analysed position itself. */
export interface RootInfo {
  winrate: number;
  scoreLead: number;
  scoreSelfplay?: number;
  utility?: number;
  visits: number;
  thisHash?: string;
  symHash?: string;
  currentPlayer: Player;
  rawWinrate?: number;
  rawLead?: number;
  rawScoreSelfplay?: number;
  rawScoreSelfplayStdev?: number;
  rawNoResultProb?: number;
  rawStWrError?: number;
  rawStScoreError?: number;
  rawVarTimeLeft?: number;
  humanWinrate?: number;
  humanScoreMean?: number;
  humanScoreStdev?: number;
  humanStWrError?: number;
  humanStScoreError?: number;
}

/**
 * The result for one turn of a query. Partial results (`isDuringSearch: true`) may come first; every turn ends with
 * exactly one result where `isDuringSearch` is false. A turn terminated before any search has `noResults: true` and
 * none of the data fields.
 */
export interface AnalysisResponse {
  id: string;
  turnNumber: number;
  isDuringSearch: boolean;
  noResults?: true;
  moveInfos?: MoveInfo[];
  rootInfo?: RootInfo;
  /** Row-major from the top-left (A19) to the bottom-right (T1); values in [-1, 1]. */
  ownership?: number[];
  ownershipStdev?: number[];
  /** boardXSize * boardYSize + 1 values (the last is pass); -1 marks an illegal move. */
  policy?: number[];
  humanPolicy?: number[];
}

/** An error. With an id and field when a query field failed to parse; a general error has neither. */
export interface ErrorResponse {
  error: string;
  id?: string;
  field?: string;
}

export interface WarningResponse {
  warning: string;
  field: string;
  id: string;
}

// ------------------------------------------------------------------ actions

export interface QueryVersionAction {
  id: string;
  action: 'query_version';
}

export interface ClearCacheAction {
  id: string;
  action: 'clear_cache';
}

export interface TerminateAction {
  id: string;
  action: 'terminate';
  terminateId: string;
  turnNumbers?: number[];
}

export interface TerminateAllAction {
  id: string;
  action: 'terminate_all';
  turnNumbers?: number[];
}

export interface SetFocusAction {
  id: string;
  action: 'set_focus';
  targetId: string;
  turnNumbers?: number[];
  focusMoves?: Location[];
  focusWeights?: number[];
  focusProb?: number;
}

export interface QueryModelsAction {
  id: string;
  action: 'query_models';
}

export type ActionQuery =
  | QueryVersionAction
  | ClearCacheAction
  | TerminateAction
  | TerminateAllAction
  | SetFocusAction
  | QueryModelsAction;

/** Any line a client sends. */
export type Query = AnalysisQuery | ActionQuery;

export interface ModelInfo {
  name: string;
  internalName: string;
  maxBatchSize: number;
  usesHumanSLProfile: boolean;
  version: number;
  usingFP16: string;
}

/** An action's reply: the action echoed back, plus any fields that action adds. */
export type ActionReply<A extends ActionQuery> = A &
  (A extends QueryVersionAction ? { version: string; git_hash: string } : unknown) &
  (A extends QueryModelsAction ? { models: ModelInfo[] } : unknown);

/** Any line an engine sends. */
export type EngineMessage = AnalysisResponse | ErrorResponse | WarningResponse | (ActionQuery & Record<string, unknown>);

// ------------------------------------------------------------- classifying

type Json = Record<string, unknown>;

const isObject = (v: unknown): v is Json => typeof v === 'object' && v !== null && !Array.isArray(v);

export function isActionQuery(q: unknown): q is ActionQuery {
  return isObject(q) && typeof q.id === 'string' && typeof q.action === 'string';
}

export function isErrorResponse(m: unknown): m is ErrorResponse {
  return isObject(m) && typeof m.error === 'string';
}

export function isWarningResponse(m: unknown): m is WarningResponse {
  return isObject(m) && typeof m.warning === 'string' && typeof m.id === 'string';
}

export function isAnalysisResponse(m: unknown): m is AnalysisResponse {
  return (
    isObject(m) &&
    typeof m.id === 'string' &&
    typeof m.turnNumber === 'number' &&
    typeof m.isDuringSearch === 'boolean' &&
    !('action' in m)
  );
}

/** The number of final responses a query produces: one per analysed turn. */
export function expectedFinals(query: Pick<AnalysisQuery, 'analyzeTurns'>): number {
  return query.analyzeTurns?.length ?? 1;
}
