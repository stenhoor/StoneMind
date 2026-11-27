/**
 * KataGo's analysis engine on this package's search, ported from KataGo v1.18.2's `cpp/command/analysis.cpp`: it reads
 * queries as JSON lines and writes results, errors and warnings as JSON lines, with KataGo's messages. As a
 * `LineTransport` it sits behind `@stonemind/engine`'s `JsonLineSession`, the same `AnalysisEngine` a native KataGo is
 * used through.
 *
 * Where it differs from KataGo:
 * - The searches run on one JavaScript thread. `numAnalysisThreads` searches run at once, their network calls combined
 *   (`SharedEvaluator`); each search's `numSearchThreads` is its batch of playouts per network call.
 * - Reports during a search are written at the first batch boundary after they fall due, where KataGo's report thread
 *   writes them at the time.
 * - Rules this engine's search doesn't support yet (territory scoring, a group tax, a button) are refused with an
 *   error, as is an initial stone on a stone of the other colour, and config settings for search features not ported.
 * - Messages that come from C++ libraries differ: JSON syntax errors, and config value errors (no file name).
 * - The evaluation cache isn't limited in size (KataGo's holds 2^`nnCacheSizePowerOfTwo` entries).
 */
import type { EngineDescriptor, LineTransport } from '@stonemind/engine';
import { Board, BLACK, WHITE, PASS_LOC, NULL_LOC, type Stone } from '../rules/board.js';
import { IllegalMoveError, UnsupportedPositionError, type GameHistory } from '../rules/history.js';
import {
  MAX_USER_KOMI, MIN_USER_KOMI, RulesParseError, komiIsIntOrHalfInt, parseKataGoRules, parseWhiteHandicapBonusRule,
  rulesToString, supportedRules, toLowerAscii, type KataGoRules, type Rules, type RulesObjectReader,
} from '../rules/rules.js';
import { analysisHistories, type QueryMove } from '../rules/analysis-position.js';
import type { ModelDesc } from '../model/types.js';
import type { Evaluation } from '../reference/postprocess.js';
import { Search, type NetEvaluator, type SearchParams } from '../search/search.js';
import { SharedEvaluator } from '../search/review.js';
import { Config, ConfigValueError, UnsupportedSettingError, avoidMoveUntilByLoc, searchParamsFromConfig, type AvoidMovesEntry } from '../search/query-settings.js';
import { addAnalysisJson, type ReportOptions } from './analysis-json.js';
import {
  JsonTypeError, dumpJson, getArray, getBool, getFloat, getInt, getInt64, getString, has, isInteger, isNumber, isObject,
  numberValue, parseJson, type JsonObject, type JsonValue,
} from './json.js';

/** The KataGo version this engine ports, reported by `query_version`. */
export const KATAGO_VERSION = '1.18.2';
/** The commit of KataGo's `v1.18.2` tag, reported by `query_version` as `git_hash`. */
export const KATAGO_GIT_HASH = 'fd0723fdbc0e9d82cf269c9630af8c27c57c07c4';

/** KataGo `Board::MAX_LEN` as KataGo's releases are built. */
const MAX_LEN = 19;

/** A config the engine can't start with (KataGo exits at startup). */
export class AnalysisConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AnalysisConfigError';
  }
}

export interface AnalysisEngineOptions {
  model: ModelDesc;
  /** The model's name, which `query_models` reports (KataGo reports the model file path it was given). */
  modelName: string;
  /** The network: the WebGPU network, or the reference forward pass. It must take boards up to 19×19. */
  net: NetEvaluator;
  /** Whether the network computes in 16-bit floats, which `query_models` reports. */
  usingFP16?: boolean;
  /**
   * KataGo's analysis config, key to value as text, as in KataGo's `analysis_example.cfg`. As in KataGo,
   * `numAnalysisThreads` and `numSearchThreadsPerAnalysisThread` (or `numSearchThreads`) are required.
   */
  config: Readonly<Record<string, string>>;
}

/**
 * Keys of a native KataGo config that set up its logging, network backend and cache. They mean nothing here and are
 * accepted without effect.
 */
const NATIVE_ONLY_KEYS = new Set([
  'logDir', 'logFile', 'logToStderr', 'logTimeStamp', 'logConfigContents', 'logAllRequests', 'logAllResponses',
  'logErrorsAndWarnings', 'logSearchInfo', 'nnMaxBatchSize', 'nnCacheSizePowerOfTwo', 'nnMutexPoolSizePowerOfTwo',
  'numNNServerThreadsPerModel', 'useFP16', 'useNHWC', 'homeDataDir',
]);

/** The analysis engine's own config keys, read at startup. */
const ENGINE_KEYS = new Set([
  'numAnalysisThreads', 'forDeterministicTesting', 'warnUnusedFields', 'analysisPVLen',
  'assumeMultipleStartingBlackMovesAreHandicap', 'preventCleanupPhase', 'nnRandomize',
]);

/** KataGo's `expectedKeys`: the fields of a query. */
const EXPECTED_KEYS = new Set([
  'id', 'action', 'terminateId', 'turnNumbers', 'boardXSize', 'boardYSize', 'initialStones', 'moves', 'initialPlayer',
  'analyzeTurns', 'priorities', 'rules', 'komi', 'whiteHandicapBonus', 'overrideSettings', 'maxVisits', 'analysisPVLen',
  'rootFpuReductionMax', 'rootPolicyTemperature', 'includeMovesOwnership', 'includeMovesOwnershipStdev',
  'includeOwnership', 'includeOwnershipStdev', 'includePolicy', 'includePVVisits', 'includeNoResultValue',
  'reportDuringSearchEvery', 'firstReportDuringSearchAfter', 'priority', 'allowMoves', 'avoidMoves',
]);

const RULES_MESSAGE = 'Must specify rules string, such as "chinese" or "tromp-taylor", or a JSON object with detailed rules parameters.';
const BOARD_MOVES_MESSAGE = 'Must be an array of pairs of the form: ["b" or "w", GTP board vertex]';
const AVOID_MESSAGE = "Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'";
const UNUSED_FIELD_MESSAGE = 'Unexpected or unused field, do you have a typo? (set warnUnusedFields=false in the config to disable this warning)';
/**
 * KataGo's priorities are from -(2^62 - 1) to 2^62 - 1. That bound isn't a JavaScript number (it rounds to 2^62), so the
 * checks below refuse 2^62 and beyond.
 */
const PRIORITY_LIMIT = 2 ** 62;

const QUEUED = 0;
const RUNNING = 1;
const TERMINATED = 2;

interface Request extends ReportOptions {
  internalId: number;
  id: string;
  turnNumber: number;
  hist: GameHistory;
  params: SearchParams;
  maxVisits: number;
  reportDuringSearch: boolean;
  reportDuringSearchEvery: number;
  firstReportDuringSearchAfter: number;
  priority: number;
  /** Order of arrival, for ties in priority. */
  order: number;
  avoid: { black: Int32Array | null; white: Int32Array | null };
  status: typeof QUEUED | typeof RUNNING | typeof TERMINATED;
  stop?: () => void;
}

const isTerminated = (r: Request): boolean => r.status === TERMINATED;

/** The search settings a config gives (KataGo's `loadParams` lambda). */
interface LoadedParams {
  params: SearchParams;
  maxVisits: number;
  perspective: Stone | null;
  /** Keys neither the search parameters nor the reporting side read. */
  unusedKeys: string[];
}

// KataGo Global::trim.
const trimAscii = (s: string): string => s.replace(/^[ \t\n\v\f\r]+|[ \t\n\v\f\r]+$/g, '');

/** KataGo `PlayerIO::tryParsePlayer`. */
function tryParsePlayer(s: string): Stone | null {
  const lower = toLowerAscii(s);
  if (lower === 'black' || lower === 'b') return BLACK;
  if (lower === 'white' || lower === 'w') return WHITE;
  return null;
}

/** KataGo `Global::tryStringToInt`: `istream >> int`, the whole string read. */
function tryStringToInt(str: string): number | null {
  const s = trimAscii(str);
  if (!/^[+-]?\d+$/.test(s)) return null;
  const x = Number(s);
  return x >= -(2 ** 31) && x < 2 ** 31 ? x : null;
}

/** KataGo `tryParseLetterCoordinate`: A to Z without I. */
function letterCoordinate(c: string): number | null {
  if (c >= 'A' && c <= 'H') return c.charCodeAt(0) - 65;
  if (c >= 'a' && c <= 'h') return c.charCodeAt(0) - 97;
  if (c >= 'J' && c <= 'Z') return c.charCodeAt(0) - 66;
  if (c >= 'j' && c <= 'z') return c.charCodeAt(0) - 98;
  return null;
}

/**
 * KataGo `Location::tryOfString`: a GTP vertex, `pass`, or `(x,y)`. KataGo doesn't check that an `(x,y)` is on the
 * board; here one that isn't is refused.
 */
function tryOfString(str: string, board: Board): number | null {
  const s = trimAscii(str);
  if (s.length < 2) return null;
  const lower = toLowerAscii(s);
  if (lower === 'pass' || lower === 'pss') return PASS_LOC;
  let x: number | null;
  let y: number | null;
  if (s[0] === '(') {
    if (s[s.length - 1] !== ')') return null;
    const inner = s.slice(1, -1);
    // KataGo Global::split reads pieces with getline, which drops a trailing empty one.
    const pieces = inner.length === 0 ? [] : inner.split(',');
    if (inner.endsWith(',')) pieces.pop();
    if (pieces.length !== 2) return null;
    x = tryStringToInt(pieces[0]!);
    y = tryStringToInt(pieces[1]!);
    if (x === null || y === null) return null;
  } else {
    x = letterCoordinate(s[0]!);
    if (x === null) return null;
    let rest: string;
    if (/[A-Za-z]/.test(s[1]!)) {
      const x1 = letterCoordinate(s[1]!);
      if (x1 === null) return null;
      x = (x + 1) * 25 + x1;
      rest = s.slice(2);
    } else rest = s.slice(1);
    const row = tryStringToInt(rest);
    if (row === null) return null;
    y = board.ySize - row;
  }
  if (x < 0 || y < 0 || x >= board.xSize || y >= board.ySize) return null;
  return board.loc(x, y);
}

/** KataGo `Setup::parseReportAnalysisWinrates`. */
function parseReportAnalysisWinrates(cfg: Config, defaultPerspective: Stone | null): Stone | null {
  if (!cfg.contains('reportAnalysisWinratesAs')) return defaultPerspective;
  const sOrig = cfg.getString('reportAnalysisWinratesAs');
  const s = toLowerAscii(sOrig);
  if (s === 'b' || s === 'black') return BLACK;
  if (s === 'w' || s === 'white') return WHITE;
  if (s === 'sidetomove') return null;
  throw new ConfigValueError(`Could not parse config value for reportAnalysisWinratesAs: ${sOrig}`);
}

/** A small seeded generator (mulberry32) for the searches' seeds. */
function seedSequence(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

/**
 * KataGo's analysis engine as a line transport: send it query lines, and it answers with result, error and warning
 * lines, as native KataGo's standard input and output do. Needs `loadPositionHashes()` first (`createAnalysisEngine`,
 * from `@stonemind/engine-webgpu/session`, does that).
 */
export class KataGoAnalysisLines implements LineTransport {
  readonly #model: ModelDesc;
  readonly #modelName: string;
  readonly #maxBatch: number;
  readonly #usingFP16: boolean;
  readonly #evaluator: SharedEvaluator;
  readonly #cache = new Map<string, Evaluation>();
  readonly #config: Readonly<Record<string, string>>;
  readonly #defaults: LoadedParams;
  readonly #analysisPVLen: number;
  readonly #assumeHandicap: boolean;
  readonly #warnUnusedFields: boolean;
  readonly #symmetry: number;
  /** Per analysis thread, the seeds of its searches. */
  readonly #seeds: (() => number)[];
  readonly #idleThreads: number[];
  /** Requests not finished, by internal id (KataGo `openRequests`). */
  readonly #openRequests = new Map<number, Request>();
  readonly #queue: Request[] = [];
  #internalIdCounter = 0;
  #numRequestsSoFar = 0;
  #receiver: ((line: string) => void) | undefined;
  #closeReceiver: ((reason: string) => void) | undefined;
  #pending: string[] = [];
  #flushScheduled = false;
  #closed = false;

  /**
   * Config keys that neither KataGo's analysis engine nor its search reads (native KataGo warns about these when it
   * starts).
   */
  readonly unusedConfigKeys: readonly string[];

  /**
   * @throws AnalysisConfigError for a config KataGo would refuse to start with, or one that sets search features not
   *   ported.
   */
  constructor(options: AnalysisEngineOptions) {
    this.#model = options.model;
    this.#modelName = options.modelName;
    this.#maxBatch = options.net.maxBatch;
    this.#usingFP16 = options.usingFP16 ?? false;
    this.#evaluator = new SharedEvaluator(options.net);
    const config: Record<string, string> = { ...options.config };
    // KataGo cfg.applyAlias.
    if (has(config, 'numSearchThreadsPerAnalysisThread')) {
      if (has(config, 'numSearchThreads')) {
        throw new AnalysisConfigError('Cannot specify both numSearchThreadsPerAnalysisThread and numSearchThreads in the same config');
      }
      config['numSearchThreads'] = config['numSearchThreadsPerAnalysisThread']!;
      delete config['numSearchThreadsPerAnalysisThread'];
    }
    this.#config = config;
    try {
      const cfg = new Config(new Map(Object.entries(config)));
      const numAnalysisThreads = cfg.getInt('numAnalysisThreads', 1, 16384);
      const forDeterministicTesting = cfg.contains('forDeterministicTesting') ? cfg.getBool('forDeterministicTesting') : false;
      this.#warnUnusedFields = cfg.contains('warnUnusedFields') ? cfg.getBool('warnUnusedFields') : true;
      this.#analysisPVLen = cfg.contains('analysisPVLen') ? cfg.getInt('analysisPVLen', 1, 100) : 15;
      this.#assumeHandicap = cfg.contains('assumeMultipleStartingBlackMovesAreHandicap') ? cfg.getBool('assumeMultipleStartingBlackMovesAreHandicap') : true;
      // KataGo's preventCleanupPhase only matters with territory scoring, which this engine refuses for now.
      if (cfg.contains('preventCleanupPhase')) cfg.getBool('preventCleanupPhase');
      // KataGo's nnRandomize: a random symmetry for each evaluation, or symmetry 0.
      this.#symmetry = (cfg.contains('nnRandomize') ? cfg.getBool('nnRandomize') : true) ? -1 : 0;
      this.#defaults = this.#loadParams(config, null);
      const seed = seedSequence(forDeterministicTesting ? 0x5eed : Math.floor(Math.random() * 2 ** 32));
      this.#seeds = Array.from({ length: numAnalysisThreads }, () => seedSequence(seed()));
      this.#idleThreads = Array.from({ length: numAnalysisThreads }, (_, i) => i);
    } catch (e) {
      if (e instanceof ConfigValueError || e instanceof UnsupportedSettingError) throw new AnalysisConfigError(e.message);
      throw e;
    }
    this.unusedConfigKeys = this.#defaults.unusedKeys.filter((k) => !ENGINE_KEYS.has(k) && !NATIVE_ONLY_KEYS.has(k));
  }

  /** KataGo's `loadParams` lambda: the search parameters and reporting side a config gives. */
  #loadParams(config: Readonly<Record<string, string>>, defaultPerspective: Stone | null): LoadedParams {
    const loaded = searchParamsFromConfig(config, { symmetry: this.#symmetry, analysisPVLen: this.#analysisPVLen });
    // KataGo requires numSearchThreads (getInt with no default).
    if (!has(config, 'numSearchThreads')) throw new ConfigValueError("Could not find key 'numSearchThreads'");
    const cfg = new Config(new Map(Object.entries(config)));
    const perspective = parseReportAnalysisWinrates(cfg, defaultPerspective);
    return {
      params: loaded.params,
      maxVisits: loaded.maxVisits ?? 2 ** 50,
      perspective,
      unusedKeys: loaded.unusedKeys.filter((k) => k !== 'reportAnalysisWinratesAs'),
    };
  }

  // LineTransport

  send(line: string): void {
    if (this.#closed) return;
    this.#handleLine(line);
  }

  onLine(receiver: (line: string) => void): void {
    this.#receiver = receiver;
  }

  onClose(receiver: (reason: string) => void): void {
    this.#closeReceiver = receiver;
  }

  /** Stops every search and writes nothing more (KataGo's `-quit-without-waiting`). */
  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#queue.length = 0;
    for (const r of this.#openRequests.values()) {
      r.status = TERMINATED;
      r.stop?.();
    }
    this.#openRequests.clear();
    this.#pending = [];
  }

  // Output

  #write(message: JsonObject): void {
    if (this.#closed) return;
    // KataGo writes from its own thread; here lines go out after the current task, in order.
    this.#pending.push(dumpJson(message));
    if (this.#flushScheduled) return;
    this.#flushScheduled = true;
    queueMicrotask(() => {
      this.#flushScheduled = false;
      const lines = this.#pending;
      this.#pending = [];
      for (const line of lines) this.#receiver?.(line);
    });
  }

  #reportError(s: string): void {
    this.#write({ error: s });
  }

  #reportErrorForId(id: string, field: string, s: string): void {
    this.#write({ id, field, error: s });
  }

  #reportWarningForId(id: string, field: string, s: string): void {
    this.#write({ id, field, warning: s });
  }

  #reportNoAnalysis(r: Request): void {
    this.#write({ id: r.id, turnNumber: r.turnNumber, isDuringSearch: false, noResults: true });
  }

  /** Writes the search's results; false, writing nothing, if the root has none yet. */
  #reportAnalysis(r: Request, search: Search, isDuringSearch: boolean): boolean {
    const ret: JsonObject = { id: r.id, turnNumber: r.turnNumber, isDuringSearch };
    if (!addAnalysisJson(search, r, ret)) return false;
    this.#write(ret);
    return true;
  }

  /** A search failed (for instance the GPU was lost): like native KataGo exiting, the engine closes. */
  #fatal(error: unknown): void {
    if (this.#closed) return;
    this.close();
    this.#closeReceiver?.(`engine failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  // The analysis threads

  /** Starts queued requests on idle threads, highest priority first, then earliest. */
  #startQueued(): void {
    while (this.#idleThreads.length > 0 && this.#queue.length > 0) {
      let best = 0;
      for (let i = 1; i < this.#queue.length; i++) {
        const a = this.#queue[i]!;
        const b = this.#queue[best]!;
        if (a.priority > b.priority || (a.priority === b.priority && a.order < b.order)) best = i;
      }
      const request = this.#queue.splice(best, 1)[0]!;
      // Terminated while queued: already reported.
      if (request.status === TERMINATED) {
        this.#openRequests.delete(request.internalId);
        continue;
      }
      const thread = this.#idleThreads.shift()!;
      this.#analyze(request, thread).then(
        () => {
          this.#idleThreads.push(thread);
          this.#idleThreads.sort((a, b) => a - b);
          this.#startQueued();
        },
        (e: unknown) => this.#fatal(e),
      );
    }
  }

  async #analyze(r: Request, thread: number): Promise<void> {
    r.status = RUNNING;
    const controller = new AbortController();
    r.stop = () => controller.abort();
    const search = new Search(this.#model, this.#evaluator, r.hist, { ...r.params, seed: this.#seeds[thread]!() }, this.#cache);
    if (r.avoid.black || r.avoid.white) search.setAvoidMoveUntilByLoc(r.avoid.black, r.avoid.white);

    // KataGo AsyncBot's report loop: the first report after firstReportDuringSearchAfter seconds, then one every
    // reportDuringSearchEvery seconds after the last; periods of 10^7 seconds or more mean never.
    let period = r.reportDuringSearch ? r.reportDuringSearchEvery : -1;
    let first = r.reportDuringSearch ? r.firstReportDuringSearchAfter : -1;
    if (period >= 1e7) period = -1;
    if (first >= 1e7) {
      first = -1;
      period = -1;
    }
    let due = first >= 0 ? performance.now() + first * 1000 : Infinity;
    const onBatch = (): void => {
      if (performance.now() < due || isTerminated(r)) return;
      this.#reportAnalysis(r, search, true);
      due = period >= 0 ? performance.now() + period * 1000 : Infinity;
    };
    await search.run({ maxVisits: r.maxVisits, signal: controller.signal, ...(first >= 0 ? { onBatch } : {}) });
    if (this.#closed) return;
    // (Read through a function: TypeScript would take r.status to be still RUNNING.)
    if (!this.#reportAnalysis(r, search, false) && isTerminated(r)) {
      // KataGo promises a final line for every turn, so a search stopped before it had results says so.
      this.#reportNoAnalysis(r);
    }
    this.#openRequests.delete(r.internalId);
  }

  #terminateRequest(r: Request): void {
    const prev = r.status;
    r.status = TERMINATED;
    if (prev === QUEUED) this.#reportNoAnalysis(r);
    else if (prev === RUNNING) r.stop?.();
  }

  // The request loop

  #handleLine(rawLine: string): void {
    const line = trimAscii(rawLine);
    if (line.length === 0) return;
    let input: JsonValue;
    try {
      input = parseJson(line);
    } catch (e) {
      this.#reportError(`${e instanceof Error ? e.message : String(e)} - could not parse input line as json request: ${line}`);
      return;
    }
    if (!isObject(input)) {
      this.#reportError(`Request line was valid json but was not an object, ignoring: ${dumpJson(input)}`);
      return;
    }
    const id = input['id'];
    if (typeof id !== 'string') {
      this.#reportError('Request must have a string "id" field');
      return;
    }
    const action = input['action'];
    if (typeof action === 'string') this.#handleAction(input, id, action);
    else this.#handleQuery(input, id);
  }

  #handleAction(input: JsonObject, id: string, action: string): void {
    const parseTurnNumbers = (): { ok: boolean; turns: Set<number> | null } => {
      if (!has(input, 'turnNumbers')) return { ok: true, turns: null };
      try {
        return { ok: true, turns: new Set(getArray(input['turnNumbers'], getInt)) };
      } catch (e) {
        if (!(e instanceof JsonTypeError)) throw e;
        this.#reportErrorForId(id, 'turnNumbers', 'If provided, must be an array of integers indicating turns to terminate');
        return { ok: false, turns: null };
      }
    };
    if (action === 'query_version') {
      this.#write({ ...input, version: KATAGO_VERSION, git_hash: KATAGO_GIT_HASH });
    } else if (action === 'query_models') {
      const model: JsonObject = {
        name: this.#modelName,
        internalName: this.#model.name,
        maxBatchSize: this.#maxBatch,
        usesHumanSLProfile: this.#model.metaEncoderVersion > 0,
        version: this.#model.modelVersion,
        usingFP16: this.#usingFP16 ? 'true' : 'false',
      };
      this.#write({ ...input, models: [model] });
    } else if (action === 'clear_cache') {
      this.#cache.clear();
      this.#write(input);
    } else if (action === 'terminate') {
      const terminateId = input['terminateId'];
      if (typeof terminateId !== 'string') {
        this.#reportErrorForId(id, 'terminateId', 'Requests for a terminate action must have a string "terminateId" field');
        return;
      }
      const { ok, turns } = parseTurnNumbers();
      if (!ok) return;
      for (const r of this.#openRequests.values()) {
        if (r.id === terminateId && (!turns || turns.has(r.turnNumber))) this.#terminateRequest(r);
      }
      this.#write(input);
    } else if (action === 'terminate_all') {
      const { ok, turns } = parseTurnNumbers();
      if (!ok) return;
      for (const r of this.#openRequests.values()) {
        if (!turns || turns.has(r.turnNumber)) this.#terminateRequest(r);
      }
      this.#write(input);
    } else {
      this.#reportError("'action' field must be 'query_version' or 'query_models' or 'clear_cache' or 'terminate' or 'terminate_all'");
    }
  }

  #handleQuery(input: JsonObject, id: string): void {
    const err = (field: string, message: string): false => {
      this.#reportErrorForId(id, field, message);
      return false;
    };
    // KataGo's parseInteger, parseDouble and parseBoolean: the value, or null having reported the error.
    const integer = (dict: JsonObject, field: string, min: number, max: number, message: string): number | null => {
      const v = dict[field];
      if (!isInteger(v) || v < min || v > max) return err(field, message) || null;
      return v;
    };
    const double = (dict: JsonObject, field: string, min: number, max: number, message: string): number | null => {
      const v = dict[field];
      if (!isNumber(v)) return err(field, message) || null;
      const x = numberValue(v);
      if (!Number.isFinite(x) || x < min || x > max) return err(field, message) || null;
      return x;
    };
    const boolean = (field: string): boolean | null => {
      const v = input[field];
      if (typeof v !== 'boolean') return err(field, 'Must be a boolean') || null;
      return v;
    };
    const player = (dict: JsonObject, field: string): Stone | null => {
      const v = dict[field];
      const pla = typeof v === 'string' ? tryParsePlayer(v) : null;
      if (pla === null) return err(field, 'Must be "b" or "w"') || null;
      return pla;
    };

    // Board size.
    const sizeMessage = `Must provide an integer from 2 to ${MAX_LEN}`;
    if (!has(input, 'boardXSize')) return void err('boardXSize', sizeMessage);
    if (!has(input, 'boardYSize')) return void err('boardYSize', sizeMessage);
    const xSize = integer(input, 'boardXSize', 2, MAX_LEN, sizeMessage);
    if (xSize === null) return;
    const ySize = integer(input, 'boardYSize', 2, MAX_LEN, sizeMessage);
    if (ySize === null) return;
    const blank = new Board(xSize, ySize);

    const boardLocs = (dict: JsonObject, field: string): number[] | null => {
      const v = dict[field];
      if (!Array.isArray(v)) return err(field, 'Must be an array of GTP board vertices') || null;
      const out: number[] = [];
      for (const elt of v) {
        if (typeof elt !== 'string') return err(field, 'Must be an array of GTP board vertices') || null;
        const loc = tryOfString(elt, blank);
        if (loc === null || loc === NULL_LOC) return err(field, `Could not parse board location: ${elt}`) || null;
        out.push(loc);
      }
      return out;
    };
    const boardMoves = (field: string, allowPass: boolean): QueryMove[] | null => {
      const v = input[field];
      if (!Array.isArray(v)) return err(field, BOARD_MOVES_MESSAGE) || null;
      const out: QueryMove[] = [];
      for (const elt of v) {
        if (!Array.isArray(elt) || elt.length !== 2 || typeof elt[0] !== 'string' || typeof elt[1] !== 'string') {
          return err(field, BOARD_MOVES_MESSAGE) || null;
        }
        const pla = tryParsePlayer(elt[0]);
        if (pla === null) return err(field, `Could not parse player: ${elt[0]}`) || null;
        const loc = tryOfString(elt[1], blank);
        if (loc === null || (!allowPass && loc === PASS_LOC)) return err(field, `Could not parse board location: ${elt[1]}`) || null;
        out.push([pla, loc]);
      }
      return out;
    };

    let placements: QueryMove[] = [];
    if (has(input, 'initialStones')) {
      const p = boardMoves('initialStones', false);
      if (!p) return;
      placements = p;
    }
    if (!has(input, 'moves')) return void err('moves', 'Must specify an array of [player,location] pairs');
    const moves = boardMoves('moves', true);
    if (!moves) return;
    let initialPlayer: Stone | undefined;
    if (has(input, 'initialPlayer')) {
      const p = player(input, 'initialPlayer');
      if (p === null) return;
      initialPlayer = p;
    }

    // Turns to analyse, and their priorities.
    const turns = new Set<number>();
    let analyzeTurns: number[] | null = null;
    if (has(input, 'analyzeTurns')) {
      try {
        analyzeTurns = getArray(input['analyzeTurns'], getInt);
      } catch (e) {
        if (!(e instanceof JsonTypeError)) throw e;
        return void err('analyzeTurns', 'Must specify an array of integers indicating turns to analyze');
      }
      for (const t of analyzeTurns) {
        if (t < 0 || t > moves.length) return void err('analyzeTurns', `Invalid turn number: ${t}`);
        turns.add(t);
      }
    } else turns.add(moves.length);
    const priorities = new Map<number, number>();
    if (has(input, 'priorities')) {
      let values: number[];
      try {
        values = getArray(input['priorities'], getInt64);
      } catch (e) {
        if (!(e instanceof JsonTypeError)) throw e;
        return void err('priorities', 'Must specify an array of integers indicating priorities');
      }
      if (!analyzeTurns) return void err('priorities', 'Can only specify when also specifying analyzeTurns');
      if (values.length !== analyzeTurns.length) return void err('priorities', 'Must be of matching length to analyzeTurns');
      for (let i = 0; i < values.length; i++) {
        const p = values[i]!;
        if (p <= -PRIORITY_LIMIT || p >= PRIORITY_LIMIT) return void err('priorities', `Invalid priority: ${p}`);
        priorities.set(analyzeTurns[i]!, p);
      }
    }

    // Rules and komi.
    if (!has(input, 'rules')) return void err('rules', RULES_MESSAGE);
    const rulesValue = input['rules'];
    let rulesText: string;
    let rulesObject: JsonValue | undefined;
    if (typeof rulesValue === 'string') rulesText = rulesValue;
    else if (isObject(rulesValue)) {
      rulesText = dumpJson(rulesValue);
      rulesObject = rulesValue;
    } else return void err('rules', RULES_MESSAGE);
    let rules: KataGoRules;
    try {
      // KataGo parses an object's text again; a string that starts with "{" is read as an object too.
      if (rulesObject === undefined && rulesText.startsWith('{')) rulesObject = parseJson(rulesText);
      rules = parseKataGoRules(rulesText, isObject(rulesObject) ? rulesReader(rulesObject) : undefined);
    } catch (e) {
      if (!(e instanceof RulesParseError || e instanceof JsonTypeError || e instanceof SyntaxError)) throw e;
      return void err('rules', `Could not parse rules: ${rulesText}`);
    }
    if (has(input, 'komi')) {
      const message = 'Must be a integer or half-integer from -400.0 to 400.0';
      const komi = double(input, 'komi', MIN_USER_KOMI, MAX_USER_KOMI, message);
      if (komi === null) return;
      rules.komi = Math.fround(komi);
      if (!komiIsIntOrHalfInt(rules.komi)) return void err('komi', message);
    }
    if (has(input, 'whiteHandicapBonus')) {
      const v = input['whiteHandicapBonus'];
      if (typeof v !== 'string') return void err('whiteHandicapBonus', 'Must be a string');
      try {
        rules.whiteHandicapBonusRule = parseWhiteHandicapBonusRule(v);
      } catch (e) {
        if (!(e instanceof RulesParseError)) throw e;
        return void err('whiteHandicapBonus', e.message);
      }
    }

    // Search settings.
    let { params, maxVisits, perspective } = this.#defaults;
    params = { ...params };
    if (has(input, 'overrideSettings')) {
      const settings = input['overrideSettings'];
      if (!isObject(settings)) return void err('overrideSettings', 'Must be an object');
      const overrides = Object.keys(settings).sort().map((k): [string, string] => {
        const v = settings[k]!;
        return [k, typeof v === 'string' ? v : dumpJson(v)];
      });
      if (overrides.length > 0) {
        // KataGo loads the parameters again from its config with the overrides set; an empty value removes a key.
        const config: Record<string, string> = { ...this.#config };
        for (const [k, v] of overrides) {
          if (v.length <= 0) delete config[k];
          else config[k] = v;
        }
        let loaded: LoadedParams;
        try {
          loaded = this.#loadParams(config, this.#defaults.perspective);
        } catch (e) {
          if (!(e instanceof ConfigValueError || e instanceof UnsupportedSettingError)) throw e;
          return void err('overrideSettings', `Could not set settings: ${e.message}`);
        }
        ({ params, maxVisits, perspective } = loaded);
        // Keys of the startup config count as used.
        const unused = loaded.unusedKeys.filter((k) => !has(this.#config, k)).sort();
        if (unused.length > 0) this.#reportWarningForId(id, 'overrideSettings', `Unknown config params: ${unused.join(',')}`);
      }
    }
    let analysisPVLen = this.#analysisPVLen;
    if (has(input, 'maxVisits')) {
      const v = integer(input, 'maxVisits', 1, 2 ** 50, 'Must be an integer from 1 to 2^50');
      if (v === null) return;
      maxVisits = v;
    }
    if (has(input, 'analysisPVLen')) {
      const v = integer(input, 'analysisPVLen', 1, 1000, 'Must be an integer from 1 to 1000');
      if (v === null) return;
      analysisPVLen = v;
    }
    params.analysisPVLen = analysisPVLen;
    if (has(input, 'rootFpuReductionMax')) {
      const v = double(input, 'rootFpuReductionMax', 0, 2, 'Must be a number from 0.0 to 2.0');
      if (v === null) return;
      params.rootFpuReductionMax = v;
    }
    if (has(input, 'rootPolicyTemperature')) {
      const v = double(input, 'rootPolicyTemperature', 0.01, 100, 'Must be a number from 0.01 to 100.0');
      if (v === null) return;
      params.rootPolicyTemperature = params.rootPolicyTemperatureEarly = v;
    }
    const report: ReportOptions = {
      perspective,
      includePolicy: false,
      includeOwnership: false,
      includeOwnershipStdev: false,
      includeMovesOwnership: false,
      includeMovesOwnershipStdev: false,
      includePVVisits: false,
      includeNoResultValue: false,
    };
    for (const field of ['includeMovesOwnership', 'includeMovesOwnershipStdev', 'includeOwnership', 'includeOwnershipStdev', 'includePolicy', 'includePVVisits', 'includeNoResultValue'] as const) {
      if (!has(input, field)) continue;
      const v = boolean(field);
      if (v === null) return;
      report[field] = v;
    }
    let reportDuringSearch = false;
    let reportDuringSearchEvery = 1e30;
    let firstReportDuringSearchAfter = 1e30;
    const secondsMessage = 'Must be number of seconds from 0.001 to 1000000.0';
    if (has(input, 'reportDuringSearchEvery')) {
      const v = double(input, 'reportDuringSearchEvery', 0.001, 1e6, secondsMessage);
      if (v === null) return;
      reportDuringSearch = true;
      reportDuringSearchEvery = firstReportDuringSearchAfter = v;
    }
    if (has(input, 'firstReportDuringSearchAfter')) {
      const v = double(input, 'firstReportDuringSearchAfter', 0.001, 1e6, secondsMessage);
      if (v === null) return;
      reportDuringSearch = true;
      firstReportDuringSearchAfter = v;
    }
    let priority = 0;
    if (has(input, 'priority')) {
      if (has(input, 'priorities')) return void err('priority', 'Cannot specify both priority and priorities');
      const message = 'Must be a number between -2^62 and 2^62';
      const v = integer(input, 'priority', -PRIORITY_LIMIT, PRIORITY_LIMIT, message);
      if (v === null) return;
      if (Math.abs(v) >= PRIORITY_LIMIT) return void err('priority', message);
      priority = v;
    }

    // allowMoves and avoidMoves.
    let avoid: Request['avoid'] = { black: null, white: null };
    const hasAllowMoves = has(input, 'allowMoves');
    if (hasAllowMoves || has(input, 'avoidMoves')) {
      if (hasAllowMoves && has(input, 'avoidMoves')) return void err('allowMoves', 'Cannot specify both allowMoves and avoidMoves');
      const field = hasAllowMoves ? 'allowMoves' : 'avoidMoves';
      const list = input[field];
      if (!Array.isArray(list)) return void err(field, AVOID_MESSAGE);
      if (hasAllowMoves && list.length > 2) return void err(field, 'Currently allowMoves only allows at most one entry per player');
      const entries: AvoidMovesEntry[] = [];
      const gotAllowMoves = new Set<Stone>();
      for (const entry of list) {
        if (!isObject(entry) || !has(entry, 'moves') || !has(entry, 'untilDepth') || !has(entry, 'player')) return void err(field, AVOID_MESSAGE);
        const pla = player(entry, 'player');
        if (pla === null) return;
        const locs = boardLocs(entry, 'moves');
        if (!locs) return;
        const untilDepth = integer(entry, 'untilDepth', 1, 1e9, 'Must be a positive integer');
        if (untilDepth === null) return;
        if (hasAllowMoves) {
          if (gotAllowMoves.has(pla)) return void err(field, 'Cannot specify allowMoves more than once for the same player');
          gotAllowMoves.add(pla);
        }
        entries.push({ player: pla, moves: locs, untilDepth });
      }
      avoid = avoidMoveUntilByLoc(blank.arraySize, entries, hasAllowMoves);
    }

    // The rules the network supports.
    const supported = supportedRules(rules);
    if (!supported.supported) {
      this.#reportWarningForId(id, 'rules', `Rules ${rulesToString(rules)} not supported by neural net, using ${rulesToString(supported.rules)} instead`);
      rules = supported.rules;
    }
    if (rules.scoringRule !== 'AREA' || rules.taxRule !== 'NONE' || rules.hasButton) {
      // Not yet ported (part C of the parity plan).
      return void err('rules', `Rules ${rulesToString(rules)} are not supported by this engine yet: it supports area scoring with no group tax and no button`);
    }

    if (this.#warnUnusedFields) {
      for (const key of Object.keys(input).sort()) {
        if (!EXPECTED_KEYS.has(key)) this.#reportWarningForId(id, key, UNUSED_FIELD_MESSAGE);
      }
    }

    // The position at each turn analysed.
    const { komi, ...ruleSet } = rules;
    let histories: Map<number, GameHistory>;
    try {
      histories = analysisHistories(
        {
          xSize,
          ySize,
          rules: ruleSet as Rules,
          komi,
          initialStones: placements,
          moves,
          ...(initialPlayer !== undefined ? { initialPlayer } : {}),
          assumeMultipleStartingBlackMovesAreHandicap: this.#assumeHandicap,
        },
        turns,
      );
    } catch (e) {
      if (e instanceof IllegalMoveError) return void err('moves', e.message);
      if (e instanceof UnsupportedPositionError) return void err('initialStones', e.message);
      throw e;
    }

    const turnOrder = [...turns].sort((a, b) => a - b);
    for (const turnNumber of turnOrder) {
      const request: Request = {
        ...report,
        internalId: this.#internalIdCounter++,
        id,
        turnNumber,
        hist: histories.get(turnNumber)!,
        params,
        maxVisits,
        reportDuringSearch,
        reportDuringSearchEvery,
        firstReportDuringSearchAfter,
        priority: priorities.size > 0 ? priorities.get(turnNumber)! : priority,
        order: this.#numRequestsSoFar++,
        avoid,
        status: QUEUED,
      };
      this.#openRequests.set(request.internalId, request);
      this.#queue.push(request);
    }
    this.#startQueued();
  }
}

/** A rules object read as KataGo reads its fields. */
function rulesReader(o: JsonObject): RulesObjectReader {
  return {
    keys: Object.keys(o).sort(),
    getString: (k) => getString(o[k]),
    getBool: (k) => getBool(o[k]),
    getFloat: (k) => getFloat(o[k]),
  };
}

/** The browser engine's descriptor. */
export const BROWSER_ENGINE_DESCRIPTOR: EngineDescriptor = { id: 'stonemind-webgpu', kind: 'browser', displayName: 'Browser engine' };
