// The capability page's probe inside a worker (build plan, Milestone 1, step 2). The same file runs as a dedicated
// worker and as a shared worker; it is a classic script (no imports), so every target browser can start it.
//
// It answers one message, { probe: true }, with what WebGPU and OPFS offer in this kind of worker.
/* eslint-disable */
'use strict';

function withTimeout(promise, ms, what) {
  return Promise.race([
    promise,
    new Promise(function (_, reject) {
      setTimeout(function () { reject(new Error(what + ' did not answer within ' + ms + ' ms')); }, ms);
    }),
  ]);
}

async function probe() {
  var out = { webgpu: { api: false, adapter: false }, opfs: { api: false, syncAccessHandle: false } };
  try {
    var gpu = self.navigator && self.navigator.gpu;
    out.webgpu.api = !!gpu;
    if (gpu) {
      var adapter = await withTimeout(gpu.requestAdapter(), 8000, 'requestAdapter');
      out.webgpu.adapter = !!adapter;
      if (adapter && adapter.info) out.webgpu.vendor = adapter.info.vendor || '';
    }
  } catch (e) {
    out.webgpu.error = String(e && e.message || e);
  }
  try {
    var storage = self.navigator && self.navigator.storage;
    out.opfs.api = !!(storage && storage.getDirectory);
    if (out.opfs.api) {
      var root = await withTimeout(storage.getDirectory(), 8000, 'getDirectory');
      var name = 'capability-probe-' + Math.random().toString(36).slice(2);
      var file = await root.getFileHandle(name, { create: true });
      if (typeof file.createSyncAccessHandle === 'function') {
        var handle = await file.createSyncAccessHandle();
        var data = new Uint8Array([1, 2, 3, 4]);
        handle.write(data, { at: 0 });
        var back = new Uint8Array(4);
        handle.read(back, { at: 0 });
        handle.close();
        out.opfs.syncAccessHandle = back[3] === 4;
      }
      await root.removeEntry(name);
    }
  } catch (e) {
    out.opfs.error = String(e && e.message || e);
  }
  return out;
}

function answer(port) {
  port.onmessage = function (event) {
    if (!event.data || !event.data.probe) return;
    probe().then(function (result) { port.postMessage(result); });
  };
  if (port.start) port.start();
}

if (typeof self.onconnect !== 'undefined' && typeof SharedWorkerGlobalScope !== 'undefined' && self instanceof SharedWorkerGlobalScope) {
  self.onconnect = function (event) { answer(event.ports[0]); };
} else {
  answer(self);
}
