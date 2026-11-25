#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { DEFAULT_ALLOWED_ORIGINS, DEFAULT_PAGE_URL, startBridge } from './bridge.js';

const HELP = `StoneMind KataGo bridge: lets a StoneMind page use the KataGo on this computer.

Usage:
  katago-bridge --model <net.bin.gz> --config <analysis.cfg> [options]

Options:
  --katago <path>        The KataGo executable (default: katago, from PATH)
  --model <path>         The network file, passed to katago as -model
  --config <path>        The analysis config, passed to katago as -config
  --human-model <path>   Optional human SL network, passed as -human-model
  --port <n>             Port on 127.0.0.1 (default: 6565; 0 picks a free one)
  --origin <url>         A page origin allowed to connect; repeat for several
                         (default: ${DEFAULT_ALLOWED_ORIGINS.join(', ')})
  --page <url>           The page the pairing link opens (default: ${DEFAULT_PAGE_URL})
  --help                 Show this help

The bridge listens on 127.0.0.1 only. A page connects only if its origin is allowed and it presents the
pairing token in the link the bridge prints when it starts. The token is new each time the bridge
starts and stays valid until it stops.
`;

const { values } = parseArgs({
  options: {
    katago: { type: 'string', default: 'katago' },
    model: { type: 'string' },
    config: { type: 'string' },
    'human-model': { type: 'string' },
    port: { type: 'string', default: '6565' },
    origin: { type: 'string', multiple: true },
    page: { type: 'string' },
    help: { type: 'boolean', short: 'h' },
  },
});

if (values.help || typeof values.model !== 'string' || typeof values.config !== 'string') {
  process.stdout.write(HELP);
  process.exitCode = values.help ? 0 : 2;
} else {
  const args = ['analysis', '-model', values.model, '-config', values.config, '-quit-without-waiting'];
  if (typeof values['human-model'] === 'string') args.push('-human-model', values['human-model']);
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    process.stderr.write(`--port must be a number from 0 to 65535\n`);
    process.exitCode = 2;
  } else {
    const origins = Array.isArray(values.origin) ? values.origin : undefined;
    const bridge = await startBridge({
      command: String(values.katago),
      args,
      port,
      ...(origins ? { allowedOrigins: origins } : {}),
      ...(typeof values.page === 'string' ? { pageUrl: values.page } : {}),
      log: (m) => process.stderr.write(`[bridge] ${m}\n`),
    });
    process.stdout.write(`\nOpen this link to pair a page with KataGo on this computer:\n\n  ${bridge.pairingLink}\n\n`);
    process.stdout.write('The link holds a pairing token, valid until this bridge stops: share it with no one. Press Ctrl+C to stop.\n');
    const stop = () => void bridge.close().then(() => process.stderr.write('[bridge] stopped\n'));
    process.on('SIGINT', stop);
    process.on('SIGTERM', stop);
  }
}
