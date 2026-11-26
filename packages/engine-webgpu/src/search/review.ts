/**
 * Reviewing a game: every turn searched at a fixed number of visits. Several turns are searched at once, each with a
 * small batch (which keeps each search's quality: in the owner's agreement test, batch 16 agreed with native KataGo's
 * 4000-visit reference about as often as native did, and batch 64 less often), while a shared evaluator combines their
 * leaves into network calls (which keeps the GPU busy). One evaluation cache is shared across the turns, as KataGo's
 * analysis engine shares one across the turns of a query.
 */
import type { Evaluation } from '../reference/postprocess.js';
import type { NetInput, RawOutputs } from '../reference/forward.js';
import type { ModelDesc } from '../model/types.js';
import type { GameHistory } from '../rules/history.js';
import { Search, type NetEvaluator, type SearchAnalysis, type SearchParams } from './search.js';

type Request = { inputs: readonly NetInput[]; resolve: (o: RawOutputs[]) => void; reject: (e: unknown) => void };

/**
 * A `NetEvaluator` for several concurrent searches, standing in for KataGo's neural-net server: the searches' requests
 * are collected until the end of the current task, then sent as network calls of up to the network's `maxBatch` inputs,
 * each caller getting its own outputs back.
 *
 * Up to the network's `maxInFlight` calls run at once. When several calls can start, the waiting requests are spread
 * across them (each request whole, in arrival order, to the call with the fewest inputs that has room), so searches that
 * asked together are answered separately: one search picks its next leaves while another's call runs on the GPU, as
 * KataGo's search threads run on the CPU while its server thread runs the GPU. Requests that don't fit wait for the next
 * free call. With `maxInFlight` 1, everything waiting goes into one call.
 */
export class SharedEvaluator implements NetEvaluator {
  readonly maxBatch: number;
  readonly maxInFlight: number;
  readonly #net: NetEvaluator;
  #queue: Request[] = [];
  #inFlight = 0;
  #scheduled = false;
  /** Network calls made, and the inputs they carried. */
  calls = 0;
  inputs = 0;

  constructor(net: NetEvaluator) {
    this.#net = net;
    this.maxBatch = net.maxBatch;
    this.maxInFlight = Math.max(1, net.maxInFlight ?? 1);
  }

  evaluate(inputs: readonly NetInput[]): Promise<RawOutputs[]> {
    if (inputs.length > this.maxBatch) return Promise.reject(new Error(`batch of ${inputs.length} is larger than ${this.maxBatch}`));
    return new Promise((resolve, reject) => {
      this.#queue.push({ inputs, resolve, reject });
      this.#schedule();
    });
  }

  #schedule(): void {
    if (this.#scheduled) return;
    this.#scheduled = true;
    setTimeout(() => {
      this.#scheduled = false;
      this.#dispatch();
    }, 0);
  }

  #dispatch(): void {
    const free = this.maxInFlight - this.#inFlight;
    if (free <= 0 || this.#queue.length === 0) return;
    const groups: { requests: Request[]; size: number }[] = Array.from({ length: Math.min(free, this.#queue.length) }, () => ({ requests: [], size: 0 }));
    const left: Request[] = [];
    for (const q of this.#queue) {
      let best: (typeof groups)[number] | undefined;
      for (const g of groups) if (g.size + q.inputs.length <= this.maxBatch && (!best || g.size < best.size)) best = g;
      if (best && left.length === 0) {
        best.requests.push(q);
        best.size += q.inputs.length;
      } else left.push(q);
    }
    this.#queue = left;
    for (const g of groups) if (g.requests.length > 0) this.#send(g.requests);
  }

  #send(requests: Request[]): void {
    const all = requests.flatMap((q) => q.inputs);
    this.calls += 1;
    this.inputs += all.length;
    this.#inFlight += 1;
    const done = () => {
      this.#inFlight -= 1;
      if (this.#queue.length > 0) this.#schedule();
    };
    this.#net.evaluate(all).then(
      (outputs) => {
        done();
        let i = 0;
        for (const q of requests) {
          q.resolve(outputs.slice(i, i + q.inputs.length));
          i += q.inputs.length;
        }
      },
      (e) => {
        done();
        for (const q of requests) q.reject(e);
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
