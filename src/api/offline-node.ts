/// <reference types="node" />
import { Worker } from 'node:worker_threads';
import * as os from 'node:os';
import { readFile } from 'node:fs/promises';
import { FastEnhancerError } from './errors.js';
import { enhanceWithPool } from './offline-parallel.js';
import { OfflinePool, WorkerBudget, parallelLimit, type OfflineWorker } from './offline-pool.js';
import type { EnhanceOptions, PcmAudio } from './offline-pcm.js';

const runtime = (async () => {
  let concurrency = typeof os.availableParallelism === 'function' ? os.availableParallelism() : Math.max(1, os.cpus().length);
  if (process.platform === 'linux') {
    try {
      const [quota, period] = (await readFile('/sys/fs/cgroup/cpu.max', 'utf8')).trim().split(/\s+/);
      if (quota !== 'max' && Number(quota) > 0 && Number(period) > 0) concurrency = Math.min(concurrency, Math.max(1, Math.floor(Number(quota) / Number(period))));
    } catch { /* Affinity remains the portable fallback. */ }
  }
  return { concurrency, budget: new WorkerBudget(Math.max(1, concurrency - 1)) };
})();

function createWorker(): OfflineWorker {
  // Flags selecting stdin/eval input cannot be inherited by a file Worker.
  const execArgv = process.execArgv.filter((arg, i, args) => !arg.startsWith('--input-type') && !(i > 0 && args[i - 1] === '--input-type'));
  const worker = new Worker(new URL('./node-worker.js', import.meta.url), { execArgv });
  let stopped = false;
  return {
    send: (message, transfer) => worker.postMessage(message, transfer),
    listen(message, error) {
      worker.on('message', message);
      worker.on('messageerror', error);
      worker.on('error', error);
      worker.on('exit', code => { if (!stopped) error(new FastEnhancerError(`Audio Worker exited unexpectedly (${code})`)); });
    },
    stop() {
      stopped = true;
      worker.removeAllListeners();
      // Another worker may fail while termination is already in progress.
      worker.on('error', () => {});
      return worker.terminate();
    },
  };
}

export async function createNodePool(options: EnhanceOptions): Promise<OfflinePool> {
  const { concurrency, budget } = await runtime;
  return OfflinePool.create(parallelLimit(options, concurrency), createWorker, budget, options.signal);
}

export async function enhanceNodeParallel(audio: PcmAudio, options: EnhanceOptions): Promise<PcmAudio> {
  parallelLimit(options);
  const snapshot = { channels: audio.channels.map(c => c.slice()), sampleRate: audio.sampleRate };
  const pool = await createNodePool(options);
  try { return await enhanceWithPool(snapshot, options, pool); }
  finally { await pool.close(); }
}
