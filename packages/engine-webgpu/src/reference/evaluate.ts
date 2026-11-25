import type { ModelDesc } from '../model/types.js';
import { forward, type NetInput, type RawOutputs } from './forward.js';
import { postProcess, type Evaluation, type PostProcessOptions } from './postprocess.js';
import { copyWithSymmetry } from './symmetry.js';

/**
 * Evaluate a position as KataGo's NNEvaluator does with a fixed symmetry: transform the inputs, run the reference
 * network, map the spatial outputs back, then post-process.
 */
export function evaluateWithSymmetry(
  model: ModelDesc,
  input: NetInput,
  symmetry: number,
  options: PostProcessOptions,
): Evaluation {
  const { xSize, ySize } = input;
  const transposes = (symmetry & 4) !== 0 && xSize === ySize;
  if ((symmetry & 4) !== 0 && !transposes) throw new Error('transposing symmetries need a square board');
  const spatial = copyWithSymmetry(input.spatial, model.numInputChannels, ySize, xSize, symmetry, false);
  const raw = forward(model, { xSize, ySize, spatial, global: input.global });
  const back = (plane: Float32Array) => copyWithSymmetry(plane, 1, ySize, xSize, symmetry, true);
  const unrotated: RawOutputs = {
    ...raw,
    policy: raw.policy.map(back),
    ownership: back(raw.ownership),
  };
  return postProcess(model, unrotated, options);
}
