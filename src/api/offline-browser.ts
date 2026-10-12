import { FastEnhancerError, ValidationError } from './errors.js';
import { enhanceWithPool } from './offline-parallel.js';
import { OfflinePool, WorkerBudget, parallelLimit, type OfflineWorker } from './offline-pool.js';
import type { EnhanceOptions, PcmAudio } from './offline-pcm.js';
import { createTaskHandler } from './offline-worker-tasks.js';

const concurrency = typeof navigator === 'undefined' ? 2 : navigator.hardwareConcurrency || 2;
const budget = new WorkerBudget(Math.max(1, concurrency - 1));

function createWorker(): OfflineWorker {
  let worker: Worker;
  try { worker = new Worker(new URL('./parallel-worker.js', import.meta.url), { type: 'module' }); }
  catch (cause) { throw new FastEnhancerError('Could not start the audio Worker. Allow module workers in CSP, or omit parallel and set worker: false.', { cause }); }
  return {
    send: (message, transfer) => worker.postMessage(message, transfer),
    listen(message, error) {
      worker.onmessage = event => message(event.data);
      worker.onerror = event => { event.preventDefault(); error(new FastEnhancerError(`Audio Worker failed: ${event.message || 'check worker-src CSP and deployment'}`)); };
      worker.onmessageerror = () => error(new FastEnhancerError('Could not receive the audio Worker result'));
    },
    stop() { worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null; worker.terminate(); },
  };
}

const inlineBudget = new WorkerBudget(1);
function createInlineWorker(): OfflineWorker {
  const handle = createTaskHandler(true);
  let stopped = false, receive: (value: any) => void;
  return {
    listen(message) { receive = message; },
    send(message: any) {
      void handle(message.task).then(result => { if (!stopped) receive({ id: message.id, type: 'result', result }); }, error => {
        if (!stopped) receive({ id: message.id, type: 'error', error: { name: error?.name, message: String(error?.message ?? error), code: error?.code } });
      });
    },
    stop() { stopped = true; handle.dispose(); },
  };
}

export async function createBrowserPool(options: EnhanceOptions): Promise<OfflinePool> {
  const enabled = options.parallel !== undefined && options.parallel !== false;
  if (enabled) parallelLimit(options, concurrency);
  const available = typeof window !== 'undefined' && typeof Worker !== 'undefined';
  if ((enabled || options.worker === true) && !available) throw new ValidationError('Module Workers are unavailable');
  if (options.worker === false || !available) return OfflinePool.create(1, createInlineWorker, inlineBudget, options.signal);
  return OfflinePool.create(enabled ? parallelLimit(options, concurrency) : 1, createWorker, budget, options.signal);
}

export async function enhanceBrowserParallel(audio: PcmAudio, options: EnhanceOptions): Promise<PcmAudio> {
  parallelLimit(options, concurrency);
  // Snapshot before any asynchronous work, preserving caller ownership.
  const snapshot = { channels: audio.channels.map(c => c.slice()), sampleRate: audio.sampleRate };
  const pool = await createBrowserPool(options);
  try { return await enhanceWithPool(snapshot, options, pool); }
  finally { await pool.close(); }
}
