/**
 * Reviewing a game: every turn searched at a fixed number of visits. Several turns are searched at once, each with a
 * small batch (which keeps each search's quality: in the owner's agreement test, batch 16 agreed with native KataGo's
 * 4000-visit reference about as often as native did, and batch 64 less often), while a shared evaluator combines their
 * leaves into one network call (which keeps the GPU busy). One evaluation cache is shared across the turns, as KataGo's
 * analysis engine shares one across the turns of a query.
 */
import type { Evaluation } from '../reference/postprocess.js';
import type { NetInput, RawOutputs } from '../reference/forward.js';
import type { ModelDesc } from '../model/types.js';
import type { GameHistory } from '../rules/history.js';
import { Search, type NetEvaluator, type SearchAnalysis, type SearchParams } from './search.js';

/**
 * A `NetEvaluator` for several concurrent searches: requests that arrive before the next task (the searches are
 * waiting on the same network call, so they resume and ask again together) are combined into one call of up to the
 * network's `maxBatch` inputs, and each caller gets its own outputs back.
 */
export class SharedEvaluator implements NetEvaluator {
  readonly maxBatch: number;
  readonly #net: NetEvaluator;
  #queue: { inputs: readonly NetInput[]; resolve: (o: RawOutputs[]) => void; reject: (e: unknown) => void }[] = [];
  #queued = 0;
  #scheduled = false;
  /** Network calls made, and the inputs they carried. */
  calls = 0;
  inputs = 0;

  constructor(net: NetEvaluator) {
    this.#net = net;
    this.maxBatch = net.maxBatch;
  }

  evaluate(inputs: readonly NetInput[]): Promise<RawOutputs[]> {
    if (inputs.length > this.maxBatch) return Promise.reject(new Error(`batch of ${inputs.length} is larger than ${this.maxBatch}`));
    return new Promise((resolve, reject) => {
      if (this.#queued + inputs.length > this.maxBatch) this.#flush();
      this.#queue.push({ inputs, resolve, reject });
      this.#queued += inputs.length;
      if (this.#queued >= this.maxBatch) this.#flush();
      else if (!this.#scheduled) {
        this.#scheduled = true;
        setTimeout(() => this.#flush(), 0);
      }
    });
  }

  #flush(): void {
    this.#scheduled = false;
    const queue = this.#queue;
    if (queue.length === 0) return;
    this.#queue = [];
    this.#queued = 0;
    const all = queue.flatMap((q) => q.inputs);
    this.calls += 1;
    this.inputs += all.length;
    this.#net.evaluate(all).then(
      (outputs) => {
        let i = 0;
        for (const q of queue) {
          q.resolve(outputs.slice(i, i + q.inputs.length));
          i += q.inputs.length;
        }
      },
      (e) => {
        for (const q of queue) q.reject(e);
      },
    );
  }
}

export interface ReviewOptions {
  /** Visits per turn (KataGo's maxVisits). */
  visits: number;
  /** Each search's batch (default 16, native KataGo's `numSearchThreadsPerAnalysisThread` in `analysis_example.cfg`). */
  batchSize?: number;
  /** Turns searched at once (default 2, native KataGo's `numAnalysisThreads` in its `analysis_example.cfg`). */
  concurrency?: number;
  /** Search parameters for every turn's search. */
  params?: Partial<SearchParams>;
  /** Called as each turn's search finishes. */
  onTurn?: (turn: number, analysis: SearchAnalysis) => void;
  signal?: AbortSignal;
}

/**
 * Review a game: search the position before each move, `histories[t]` being the position at turn t. Returns each turn's
 * analysis, by turn.
 */
export async function reviewGame(model: ModelDesc, net: NetEvaluator, histories: readonly GameHistory[], options: ReviewOptions): Promise<SearchAnalysis[]> {
  const batchSize = Math.max(1, Math.min(options.batchSize ?? 16, net.maxBatch));
  const concurrency = Math.max(1, options.concurrency ?? 2);
  const shared = new SharedEvaluator(net);
  const cache = new Map<string, Evaluation>();
  const results: SearchAnalysis[] = new Array(histories.length);
  let next = 0;
  const worker = async () => {
    while (next < histories.length && !options.signal?.aborted) {
      const turn = next++;
      const search = new Search(model, shared, histories[turn]!, { ...options.params, batchSize }, cache);
      await search.run({ maxVisits: options.visits, ...(options.signal ? { signal: options.signal } : {}) });
      results[turn] = search.analysis();
      options.onTurn?.(turn, results[turn]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, histories.length) }, worker));
  return results;
}
