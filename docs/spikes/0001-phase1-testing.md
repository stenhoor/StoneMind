# Spike 0001, phase 1: testing the bridge with native KataGo

This is for the owner's Linux + NVIDIA machine. It takes about 15 minutes. The goal is to check phase 1's exit
criterion: a StoneMind page reaches the KataGo on your computer and streams results back, in Chrome and in Firefox.

## 1. What you need

- **Node.js 22.18 or newer:** `node --version`.
- **KataGo with the CUDA or TensorRT backend.** Check it starts with `katago version`.
- **A kata1 b18c384nbt network** from https://katagotraining.org/networks/. Note the exact file name: it becomes the
  spike's reference net.
- **An analysis config.** The bridge runs `katago analysis`, which takes an analysis config, not `gtp.config`. Use
  `cpp/configs/analysis_example.cfg` from KataGo's source, at https://github.com/lightvector/KataGo/tree/master/cpp/configs,
  unchanged. Assumption: a KataGo release archive may already include it, so check `~/katago` first.

## 2. Build StoneMind

```bash
git clone https://github.com/stenhoor/StoneMind.git && cd StoneMind
```

```bash
npm install && npm run build
```

## 3. Serve the bench page locally

GitHub Pages only publishes after this PR is merged, so use the local copy for now. Leave this running:

```bash
npm run serve -w @stonemind/engine-bench
```

It prints `http://localhost:8080/engine-bench/`.

## 4. Start the bridge

KataGo writes its logs to `analysis_logs/` in the folder you start the bridge from (`logDir` in the analysis config).
From the StoneMind folder that is ignored by git.

In a second terminal, from the StoneMind folder:

```bash
node packages/katago-bridge/dist/cli.js --katago "$(command -v katago)" --model /path/to/kata1-b18c384nbt-....bin.gz --config /path/to/analysis_example.cfg --page http://localhost:8080/engine-bench/
```

Wait for KataGo's log line `Started, ready to begin handling requests` (prefixed `[bridge] katago:`). The bridge
prints a **pairing link**.

## 5. Try it in each browser

Open the pairing link in **Chrome**, then in **Firefox**. For each, check:

1. The header says **Connected to KataGo on this computer**, and the panel shows the KataGo version and the network
   name.
2. The address bar no longer shows `#bridge=…&token=…`.
3. Choose **Opening: four star points**, set **Max visits** to **2000**, and press **Analyse**. The table fills in
   while it searches and the visit count climbs.
4. Press **Stop** during a long search (set max visits to 100000). It stops within about a second, and the last
   results stay on screen.

## 6. Security checks (optional, 2 minutes)

- Change one character of the token in the page's **Pairing token** field, then **Connect**. It should be refused.
  The bridge logs `wrong or missing pairing token`.
- Visit http://127.0.0.1:6565/ in a browser. It should answer with a plain-text notice and nothing else.

## 7. What to tell me

- [ ] Steps 5.1–5.4 in Chrome: pass or fail. Note any permission prompt (Chrome may ask about local network access).
- [ ] The same in Firefox.
- [ ] The **Visits/s** shown after a 2000-visit search on the opening position. This is the first native baseline.
- [ ] The network file name and KataGo version, from the panel.
- [ ] Anything the bridge printed that looked wrong.

After this PR merges and Pages is enabled (Settings → Pages → Source: **GitHub Actions**), repeat step 5 without
`--page`. The bridge's default link then opens https://stenhoor.github.io/StoneMind/engine-bench/, which tests the real
case: an https site reaching your computer.
