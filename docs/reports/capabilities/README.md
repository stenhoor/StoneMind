# Capability reports

Reports from the app's capability page (`apps/web`, `/capabilities/`), one JSON file per browser and device, named
`<date>-<device>-<browser>.json`. Each is the page's report exactly as it was downloaded.

| File | Browser | Device | Notes |
|---|---|---|---|
| `2026-11-27-nvidia-laptop-edge-default.json` | Edge 158 | Owner's Linux laptop (RTX 3080 Laptop GPU, Intel iGPU) | Edge started in its default mode, not performance mode: the default adapter is the Intel iGPU (`gen-12lp`) |
| `2026-11-27-nvidia-laptop-firefox.json` | Firefox 161 | Same laptop | Firefox does not name the adapter (vendor and architecture are empty) |
| `2026-11-27-nvidia-laptop-edge-high-performance-flag-not-applied.json` | Edge 158 | Same laptop | Meant to be Edge in high-performance mode, but the launch flag did not take effect (an Edge was most likely already running, so the new launch only opened a window in it). Both the default and the `high-performance` request gave the Intel iGPU |
| `2026-11-27-nvidia-laptop-edge-nvidia.json` | Edge 158 | Same laptop | Edge started so that WebGPU uses the NVIDIA GPU: the default and `high-performance` requests, the dedicated worker and the shared worker all got `nvidia` / `ampere` |
| `2026-11-27-nvidia-laptop-firefox-2.json` | Firefox 161 | Same laptop | With the high-performance row: Firefox gives an adapter for both requests and names neither, so the page cannot say which GPU it is. In spike 0001, `nvidia-smi` showed Firefox using the NVIDIA GPU with no flags |

What they show (2026-11-27):

- **WebGPU** has an adapter in the page, in a dedicated worker and in a shared worker, in both browsers. Both have
  `shader-f16`. The largest storage buffer is 4 GB in Edge and 2 GB in Firefox.
- **OPFS** writes work in both: `createWritable` in the page, `createSyncAccessHandle` in a dedicated worker. A shared
  worker has no `createSyncAccessHandle`, as the specification says (it is for dedicated workers).
- **Web Locks, BroadcastChannel,** `backdrop-filter`, SVG filters in it (as `CSS.supports` reports) and container
  queries are supported in both.
- **Storage:** 10 GB quota in both; persistence not granted.

Added later the same day, from the two further Edge reports:

- **On the NVIDIA GPU** (`nvidia-laptop-edge-nvidia.json`), a shared worker gets the NVIDIA adapter too. Edge 158 offers
  `chromium-experimental-subgroup-matrix` there, which spike 0001 found missing in Edge; it is not offered on the Intel
  iGPU. The largest storage buffer is 2 GB on the NVIDIA adapter (4 GB on the Intel iGPU).
- **Which GPU Edge uses is set by how Edge is started, not by the page.** Spike 0001 recorded the owner's usual Edge
  launch line with `--use-webgpu-power-preference=force-low-power` (`docs/spikes/results/2026-11-b18-webgpu.md`), a
  flag that overrides what a page asks for. The launch line behind the second Edge report was not checked. The app
  keeps asking for `high-performance` and shows the adapter it got.

