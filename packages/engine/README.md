# @stonemind/engine

The engine contract. Every StoneMind analysis engine implements it: a native KataGo on the player's computer (through
`@stonemind/katago-bridge`) and the browser engine planned in [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md).

- **`protocol.ts`:** KataGo's Analysis Engine protocol as TypeScript types: queries, results, errors, warnings and the
  action queries (`query_version`, `clear_cache`, `terminate`, `terminate_all`, `set_focus`, `query_models`). The source
  is KataGo's `docs/Analysis_Engine.md` (1.18).
- **`AnalysisEngine`:** `analyze(query)` returns an async stream of partial and final results, `action(...)` sends an
  action and waits for its reply, and `close()` releases the engine. Aborting a query terminates it, and the stream still
  ends with every turn's final result.
- **`JsonLineSession`:** an `AnalysisEngine` over any line transport that speaks the protocol. It routes replies by id.
- **`connectBridge(pairing)`:** connects to the local bridge over a WebSocket. **`parsePairingFragment(hash)`** reads the
  pairing from the link the bridge prints.

Specification: [spike 0001](../../docs/spikes/0001-browser-katago-webgpu.md), and KataGo's
[Analysis Engine docs](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md).

## Depends on

None.

## Used by

`@stonemind/katago-bridge` (the subprotocol name) and `apps/engine-bench`. Planned: `@stonemind/engine-webgpu` and
`@stonemind/provider-katago`.

## Deployment

Published as an npm package. Plain ES2022 modules: `apps/engine-bench` loads them in the browser through an import map,
with no bundler.

## Development

```bash
npm run check      # typecheck, build, then node --test
```

## Notes

- Tests use `node --test` on the built `dist/`, the smaltworks standard.
- The contract is KataGo's protocol, unchanged. A field KataGo adds can be added here without breaking anyone. Never
  rename a KataGo field.
