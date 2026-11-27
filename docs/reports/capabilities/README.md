# Capability reports

Reports from the app's capability page (`apps/web`, `/capabilities/`), one JSON file per browser and device, named
`<date>-<device>-<browser>.json`. Each is the page's report exactly as it was downloaded.

| File | Browser | Device | Notes |
|---|---|---|---|
| `2026-11-27-nvidia-laptop-edge-default.json` | Edge 158 | Owner's Linux laptop (RTX 3080 Laptop GPU, Intel iGPU) | Edge started in its default mode, not performance mode: the default adapter is the Intel iGPU (`gen-12lp`) |
| `2026-11-27-nvidia-laptop-firefox.json` | Firefox 161 | Same laptop | Firefox does not name the adapter (vendor and architecture are empty) |

What they show (2026-11-27):

- **WebGPU** has an adapter in the page, in a dedicated worker and in a shared worker, in both browsers. Both have
  `shader-f16`. The largest storage buffer is 4 GB in Edge and 2 GB in Firefox.
- **OPFS** writes work in both: `createWritable` in the page, `createSyncAccessHandle` in a dedicated worker. A shared
  worker has no `createSyncAccessHandle`, as the specification says (it is for dedicated workers).
- **Web Locks, BroadcastChannel,** `backdrop-filter`, SVG filters in it (as `CSS.supports` reports) and container
  queries are supported in both.
- **Storage:** 10 GB quota in both; persistence not granted.
