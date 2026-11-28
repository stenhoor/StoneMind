// A stand-in engine for the browser tests: it speaks KataGo's Analysis Engine protocol, in lines, without a network or a
// GPU, so the host's plumbing (tabs sharing one engine, replies reaching the right tab, a closed tab's queries stopped)
// can be tested in every browser. It is never offered to players. Its answers are fixed, not analysis:
// - an analysis query gets one final result per turn it asks for (`analyzeTurns`, else the last), with a single move
//   whose visits equal the query's `maxVisits`;
// - a query with `maxVisits` of 1,000,000 or more keeps searching until it is terminated, then ends with `noResults`;
// - `query_version` and `terminate` get KataGo's replies.
import type { EngineLines } from './host.js';

export const FAKE_VERSION = 'fake-for-tests';
export const ENDLESS_VISITS = 1_000_000;

export function fakeEngine(): EngineLines {
  let receiver: (line: string) => void = () => {};
  let closer: (reason: string) => void = () => {};
  let closed = false;
  const endless = new Map<string, number[]>();
  const out = (message: unknown) => queueMicrotask(() => !closed && receiver(JSON.stringify(message)));
  const final = (id: string, turnNumber: number, visits: number) => out({
    id, turnNumber, isDuringSearch: false,
    moveInfos: [{ move: 'Q16', visits, winrate: 0.5, scoreLead: 0, order: 0, pv: ['Q16'] }],
    rootInfo: { visits, winrate: 0.5, scoreLead: 0, currentPlayer: 'B' },
  });
  return {
    send(line) {
      if (closed) return;
      const q = JSON.parse(line) as { id?: string; action?: string; terminateId?: string; analyzeTurns?: number[]; moves?: unknown[]; maxVisits?: number };
      const id = q.id ?? '';
      if (q.action === 'query_version') return out({ id, action: 'query_version', version: FAKE_VERSION, git_hash: '' });
      if (q.action === 'terminate') {
        const turns = endless.get(q.terminateId ?? '');
        if (turns) {
          endless.delete(q.terminateId ?? '');
          for (const t of turns) out({ id: q.terminateId, turnNumber: t, isDuringSearch: false, noResults: true });
        }
        return out({ id, action: 'terminate', terminateId: q.terminateId });
      }
      if (q.action) return out({ id, error: `unknown action ${q.action}` });
      const turns = q.analyzeTurns ?? [q.moves?.length ?? 0];
      const visits = q.maxVisits ?? 500;
      if (visits >= ENDLESS_VISITS) return void endless.set(id, turns);
      for (const t of turns) final(id, t, visits);
    },
    onLine(r) {
      receiver = r;
    },
    onClose(r) {
      closer = r;
    },
    close() {
      if (closed) return;
      closed = true;
      closer('closed');
    },
  };
}
