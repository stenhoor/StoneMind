// A stand-in for `katago analysis` in tests: reads query lines on stdin and answers in KataGo's protocol.
// Each analysis turn gets one partial result (when reportDuringSearchEvery is set) and one final result, unless it is
// terminated first. Every line received is echoed to stderr as "got <json>", so tests can see what reached the engine.
const pendingTurns = new Map(); // id -> timers

const send = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');
const result = (id, turnNumber, isDuringSearch, visits) => ({
  id,
  turnNumber,
  isDuringSearch,
  moveInfos: [{ move: 'Q16', visits, winrate: 0.47, scoreLead: -0.3, prior: 0.21, order: 0, pv: ['Q16', 'D4'] }],
  rootInfo: { winrate: 0.47, scoreLead: -0.3, visits, currentPlayer: 'B' },
});

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let nl;
  while ((nl = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, nl);
    buffer = buffer.slice(nl + 1);
    if (!line.trim()) continue;
    process.stderr.write(`got ${line}\n`);
    let q;
    try {
      q = JSON.parse(line);
    } catch {
      send({ error: 'Could not parse input line as json request' });
      continue;
    }
    if (q.action === 'query_version') send({ ...q, version: '1.18.2', git_hash: 'fake' });
    else if (q.action === 'clear_cache' || q.action === 'set_focus') send(q);
    else if (q.action === 'query_models') send({ ...q, models: [{ name: 'fake.bin.gz', internalName: 'fake', maxBatchSize: 8, usesHumanSLProfile: false, version: 14, usingFP16: 'false' }] });
    else if (q.action === 'terminate') {
      send(q);
      for (const [key, timer] of pendingTurns) {
        const [id, turn] = key.split('#');
        if (id === q.terminateId && (!q.turnNumbers || q.turnNumbers.includes(Number(turn)))) {
          clearTimeout(timer);
          pendingTurns.delete(key);
          send({ id, turnNumber: Number(turn), isDuringSearch: false, noResults: true });
        }
      }
    } else if (q.action !== undefined) send({ error: `unknown action ${q.action}`, id: q.id, field: 'action' });
    else if (typeof q.komi === 'number' && Math.abs(q.komi) > 400) send({ error: 'komi out of range', field: 'komi', id: q.id });
    else {
      const turns = q.analyzeTurns ?? [q.moves?.length ?? 0];
      for (const turn of turns) {
        if (q.reportDuringSearchEvery) send(result(q.id, turn, true, 3));
        // A long "search" when maxVisits is large, so tests can terminate it.
        const delay = (q.maxVisits ?? 10) > 1000 ? 60_000 : 20;
        const key = `${q.id}#${turn}`;
        pendingTurns.set(
          key,
          setTimeout(() => {
            pendingTurns.delete(key);
            send(result(q.id, turn, false, q.maxVisits ?? 10));
          }, delay),
        );
      }
    }
  }
});
process.stdin.on('end', () => {
  for (const t of pendingTurns.values()) clearTimeout(t);
  process.exit(0);
});
