// The engine host as a worker (build plan, Milestone 1, step 5). Started by engines/client.ts as a shared worker, so
// every tab of the app in this browser talks to the one engine; or, where a shared worker can't run WebGPU, as a
// dedicated worker for one tab. The logic is engines/host.ts; this file wires it to the worker and to the engines.
//
// Workers don't take a page's import map, so the app's build rewrites the `@stonemind/engine-webgpu` import below to
// the copy in vendor/ (scripts/assemble.mjs).
import { GpuNetwork, KataGoAnalysisLines, loadModel, loadPositionHashes, requestEngineDevice } from '@stonemind/engine-webgpu';
import { fakeEngine } from './fake.js';
import { type EngineFactory, EngineHost, type TabPort } from './host.js';
import type { FromHost, ToHost } from './protocol.js';

interface PortLike {
  postMessage(message: unknown): void;
  onmessage: ((event: MessageEvent) => void) | null;
  start?(): void;
}

const scope = self as unknown as { onconnect: ((event: MessageEvent) => void) | null; navigator: Navigator & { gpu?: { requestAdapter(): Promise<unknown> } } } & PortLike;
const isShared = typeof (globalThis as { SharedWorkerGlobalScope?: unknown }).SharedWorkerGlobalScope === 'function' && 'onconnect' in scope;

/** Reads a file from this browser's storage (OPFS), by the Files app's path. */
async function readWeights(path: string): Promise<File> {
  const parts = path.split('/').filter(Boolean);
  const name = parts.pop();
  if (!name) throw new Error('No weights file was chosen.');
  try {
    let dir = await navigator.storage.getDirectory();
    for (const part of parts) dir = await dir.getDirectoryHandle(part);
    return await (await dir.getFileHandle(name)).getFile();
  } catch {
    throw new Error(`There is no weights file at ${path} in this browser’s storage.`);
  }
}

const webgpu: EngineFactory = async (request, config) => {
  const file = await readWeights(request.weightsPath);
  const model = await loadModel(new Uint8Array(await file.arrayBuffer()));
  await loadPositionHashes();
  const device = await requestEngineDevice();
  // As the protocol check page does: a batch as large as the search threads, at most KataGo's nnMaxBatchSize.
  const threads = Number(config['numAnalysisThreads'] ?? 1) * Number(config['numSearchThreadsPerAnalysisThread'] ?? config['numSearchThreads'] ?? 1);
  const maxBatch = Math.max(1, Math.min(Number(config['nnMaxBatchSize'] ?? 64), threads));
  const net = await GpuNetwork.create(device, model, { maxBatch });
  const info = device.adapterInfo;
  return {
    lines: new KataGoAnalysisLines({ model, modelName: file.name, net, config }),
    info: {
      model: { name: model.name, version: model.modelVersion },
      adapter: {
        ...(info.vendor ? { vendor: info.vendor } : {}),
        ...(info.architecture ? { architecture: info.architecture } : {}),
        ...(info.description ? { description: info.description } : {}),
      },
    },
  };
};

const fake: EngineFactory = async (request) => {
  await readWeights(request.weightsPath);
  return { lines: fakeEngine(), info: { model: { name: 'fake', version: 0 } } };
};

async function hasWebGpu(): Promise<boolean> {
  try {
    return Boolean(await scope.navigator.gpu?.requestAdapter());
  } catch {
    return false;
  }
}

const host = hasWebGpu().then((gpu) => new EngineHost(isShared, gpu, { webgpu, fake }));

function attach(port: PortLike): void {
  const queued: ToHost[] = [];
  let receive: ((m: ToHost) => void) | null = null;
  port.onmessage = (e) => (receive ? receive(e.data as ToHost) : queued.push(e.data as ToHost));
  port.start?.();
  void host.then((h) => {
    const tab: TabPort = { post: (m: FromHost) => port.postMessage(m) };
    const connection = h.connect(tab);
    receive = connection.receive;
    for (const m of queued.splice(0)) receive(m);
  });
}

if (isShared) scope.onconnect = (e) => attach(e.ports[0] as unknown as PortLike);
else attach(scope);
