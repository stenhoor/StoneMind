# @stonemind/katago-bridge

Lets a StoneMind page use the **native KataGo on the player's own computer**. A browser can't start a program, and
KataGo has no network server, so this small bridge runs `katago analysis` and relays its JSON lines over a WebSocket on
127.0.0.1.

```bash
npm install && npm run build
```

```bash
node packages/katago-bridge/dist/cli.js --katago /path/to/katago --model kata1-b18c384nbt.bin.gz --config analysis.cfg
```

It prints a pairing link. Opening it pairs the engine bench page (on GitHub Pages) with the bridge. Run with `--help`
for every option.

## Security

- **Loopback only.** It listens on 127.0.0.1 and refuses a `Host` other than 127.0.0.1, localhost or [::1], which
  defends against DNS rebinding.
- **Allowed origins only.** By default these are `https://stenhoor.github.io`, `https://stonemind.app` and local
  development on `:8080`. Use `--origin` to change them.
- **A pairing token** (256 bits, new each time the bridge starts and valid until it stops; any number of pages may
  use it meanwhile) is required, sent in the WebSocket subprotocol list rather than the URL. The
  link carries it in the `#fragment`, which browsers never send to a server, and the page removes it from the address
  bar. Other pages on an allowed origin can't drive KataGo without it.
- **Pages are kept apart.** Each page's query ids are prefixed in the engine and restored on the way back. A page can
  terminate only its own queries: `terminate_all` becomes a terminate for each of that page's queries. Closing a page
  terminates its open queries.
- **Only KataGo's analysis actions are passed on.** Anything else is refused.

## Depends on

`@stonemind/engine`, for the subprotocol name. Node.js built-ins only: no outside packages, including type packages
(`src/node-shim.d.ts` declares the few Node APIs it uses).

## Used by

Players and developers running a native KataGo. `apps/engine-bench` connects to it.

## Deployment

Run locally from a checkout. It is not published to npm yet.

## Development

```bash
npm run check      # typecheck, build, then node --test against a fake KataGo (test/fake-katago.mjs)
```

## Notes

- KataGo is started with `-quit-without-waiting`, so stopping the bridge doesn't wait for searches to finish.
- **Untested here:** Chrome's Local Network Access permission prompt for public pages that reach localhost. Phase 1 of
  spike 0001 checks it on real browsers.
