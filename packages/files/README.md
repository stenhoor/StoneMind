# @stonemind/files

The file kit (build plan, Milestone 1, step 3): the `FileStore` port from `@stonemind/storage`, on the browser's
origin private file system (OPFS), plus browser helpers. It keeps the player's files (KataGo weights, configs, game
records) in the browser.

- **`OpfsFileStore`**: list, stat, read, write, mkdir, move and remove, by absolute paths such as
  `/weights/kata1-b18c384nbt.bin.gz`.
  - **Streamed writes:** data comes in as a stream, a blob, bytes or text, and is written in 1 MB chunks with progress,
    so a 100 MB weights file never sits whole in memory. An aborted write leaves no new file behind.
  - **Two write methods:** `createWritable` from the page where the browser has it, otherwise a dedicated worker with
    `createSyncAccessHandle` (`src/sync-writer.ts`). The capability reports (`docs/reports/capabilities/`) show both
    work in Edge and Firefox; Safari is not checked yet.
  - **Tabs never write the same file at once:** every write, move and remove holds a Web Lock on its paths.
  - **Change events:** after a change, a `change` event (`{ paths, remote }`) fires in this tab and, through a
    BroadcastChannel, in every other tab's store.
  - **Moves** use the browser's own move where it has one, and otherwise copy and remove.
- **Paths** (`src/paths.ts`): `normalize`, `join`, `parentOf`, `nameOf`, `isWithin`. Paths start with `/`; empty, `.`
  and `..` segments, `\` and control characters are refused.
- **Browser helpers** (`src/browser.ts`): `pickFiles` (the file picker), `filesFromDrop`, `importFiles` (streams files
  into a folder, with progress), `downloadBlob`, `storageInfo` (usage, quota, persistence) and `requestPersistence`.

## Commands

```bash
npm run check -w @stonemind/files    # typecheck and the path tests (node --test)
npm run e2e -w @stonemind/files      # the browser tests (Playwright), in Chromium
BROWSERS=chromium,firefox,webkit npm run e2e -w @stonemind/files   # as CI runs them
```

The browser tests drive `test/browser/index.html` and check each write method where the browser has it: reading,
listing, moving and removing; a 48 MB streamed write with progress; an aborted write; two tabs writing the same file at
once (the result is one write or the other, never a mix); change events between tabs; download; and storage info.
`STONEMIND_CHROMIUM` points them at another Chromium.

## Depends on

- `@stonemind/storage`, for the `FileStore` port and its types (types only; nothing is loaded at run time).
- Dev only: `playwright` 1.65.0 (Apache-2.0), for the browser tests; never shipped.

## Used by

- `apps/web`: the Files app (build plan step 4). Planned: the Engines app's weights (step 6).
