/**
 * The browser engine as an `AnalysisEngine` (`@stonemind/engine-webgpu/session`). A separate entry point from the
 * package's main one, which imports nothing outside this package, so that pages loading it without a bundler (the GPU
 * check) need no import map.
 */
import { JsonLineSession, type AnalysisEngine, type EngineDescriptor } from '@stonemind/engine';
import { loadPositionHashes } from './rules/hash.js';
import { BROWSER_ENGINE_DESCRIPTOR, KataGoAnalysisLines, type AnalysisEngineOptions } from './engine/analysis-engine.js';

/**
 * The browser engine as an `AnalysisEngine`: KataGo's analysis engine (`KataGoAnalysisLines`) behind a
 * `JsonLineSession`, as a native KataGo is used through the bridge.
 * @throws AnalysisConfigError, as `KataGoAnalysisLines`.
 */
export async function createAnalysisEngine(options: AnalysisEngineOptions, descriptor: EngineDescriptor = BROWSER_ENGINE_DESCRIPTOR): Promise<AnalysisEngine> {
  await loadPositionHashes();
  return new JsonLineSession(descriptor, new KataGoAnalysisLines(options));
}
