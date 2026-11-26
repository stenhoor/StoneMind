// Timing for the kernel benchmarks, with the GPU kept busy throughout.
//
// The GPU lowers its clocks when it goes idle, even briefly (on the owner's laptop, nvidia-smi showed the SM clock
// dropping between passes that were each awaited: docs/spikes/results/2026-11-b18-dispatch.md). So a measurement here
// is one submission sequence with no waiting in it: warm-up passes of the same work (at least `warmUpMs` of it), then
// the timed passes, each in its own command buffer with its own pair of timestamps, back to back. The page waits once,
// at the end. The result is the span from the first timed pass's start to the last one's end, divided by the number of
// timed passes: that holds whether or not one pass's start timestamp is written before the previous pass has finished
// (with passes back to back, per-pass intervals could then overlap; they are returned too, for comparison).

/**
 * Time `runs` passes of the same work, after a warm-up of the same work, all submitted without waiting.
 * @param device The GPUDevice (it must have timestamp-query).
 * @param record (pass) => void: encodes the work into a compute pass.
 * @returns `msPerPass`: the timed passes' whole GPU span divided by `runs`; `ms`: each pass's own interval, sorted;
 *   `warmUpPasses`.
 */
export async function timePasses(device, record, { runs = 7, warmUpMs = 300 } = {}) {
  const encodePass = (timestampWrites) => {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass(timestampWrites ? { timestampWrites } : {});
    record(p);
    p.end();
    return enc.finish();
  };
  // One pass on its own, to estimate how many make up the warm-up.
  const t0 = performance.now();
  device.queue.submit([encodePass(null)]);
  await device.queue.onSubmittedWorkDone();
  const estimate = Math.max(0.05, performance.now() - t0);
  const warm = Math.min(2000, Math.ceil(warmUpMs / estimate));
  const query = device.createQuerySet({ type: 'timestamp', count: 2 * runs });
  const resolve = device.createBuffer({ size: 16 * runs, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * runs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const buffers = [];
  for (let i = 0; i < warm; i++) buffers.push(encodePass(null));
  for (let r = 0; r < runs; r++) buffers.push(encodePass({ querySet: query, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 }));
  const enc = device.createCommandEncoder();
  enc.resolveQuerySet(query, 0, 2 * runs, resolve, 0);
  enc.copyBufferToBuffer(resolve, 0, read, 0, 16 * runs);
  buffers.push(enc.finish());
  device.queue.submit(buffers);
  await read.mapAsync(GPUMapMode.READ);
  const ts = new BigUint64Array(read.getMappedRange().slice(0));
  read.unmap();
  query.destroy();
  resolve.destroy();
  read.destroy();
  const ms = [];
  for (let r = 0; r < runs; r++) ms.push(Number(ts[2 * r + 1] - ts[2 * r]) / 1e6);
  const msPerPass = Number(ts[2 * runs - 1] - ts[0]) / 1e6 / runs;
  return { msPerPass, ms: ms.sort((a, b) => a - b), warmUpPasses: warm };
}

/** The median of a sorted list. */
export const median = (sorted) => sorted[Math.floor(sorted.length / 2)];
