// Writes the KataGo config for the protocol check (src/protocol.ts): an analysis config with the settings that make
// native KataGo deterministic, as for the engine's goldens (packages/engine-webgpu/scripts/capture-protocol-goldens.mjs):
// one analysis thread, one search thread, no random symmetries or root noise, fp32. The analysis config given is read,
// not changed. Prints the bridge command that opens the check page paired, with the same settings for the browser engine.
//
//   node apps/engine-bench/scripts/protocol-config.mjs --config <analysis.cfg> --out <protocol.cfg> --katago <katago> --model <net.bin.gz>
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: { config: { type: 'string' }, out: { type: 'string' }, katago: { type: 'string' }, model: { type: 'string' }, page: { type: 'string', default: 'http://localhost:8080/engine-bench/protocol.html' } } });
if (!values.config || !values.out || !values.katago || !values.model) {
  console.error('usage: protocol-config.mjs --config <analysis.cfg> --out <protocol.cfg> --katago <katago> --model <net.bin.gz> [--page <url>]');
  process.exit(2);
}

const OVERRIDES = {
  numAnalysisThreads: '1',
  numSearchThreadsPerAnalysisThread: '1',
  nnRandomize: 'false',
  wideRootNoise: '0',
  rootNumSymmetriesToSample: '1',
  useFP16: 'false',
};

// KataGo's config format: `key = value`, `#` comments.
const settings = {};
for (const raw of readFileSync(values.config, 'utf8').split('\n')) {
  const line = raw.replace(/#.*/, '').trim();
  if (!line) continue;
  const eq = line.indexOf('=');
  if (eq <= 0) throw new Error(`Could not read config line: ${raw}`);
  settings[line.slice(0, eq).trim()] = line.slice(eq + 1).trim();
}
Object.assign(settings, OVERRIDES);
// A key read twice is an error to KataGo, so the overrides replace the analysis config's own lines.
const text = Object.entries(settings).map(([k, v]) => `${k} = ${v}`).join('\n') + '\n';
writeFileSync(values.out, `# The protocol check's KataGo config: ${values.config} with deterministic settings.\n${text}`);

const page = `${values.page}?${new URLSearchParams({ config: text })}`;
const bridge = ['node', 'packages/katago-bridge/dist/cli.js', '--katago', values.katago, '--model', values.model, '--config', resolve(values.out), '--page', page];
console.log(`Wrote ${values.out}. Serve the bench (npm run serve -w @stonemind/engine-bench), then start the bridge:\n`);
console.log(bridge.map((a) => (/^[\w./:=-]+$/.test(a) ? a : `'${a.replaceAll("'", "'\\''")}'`)).join(' '));
