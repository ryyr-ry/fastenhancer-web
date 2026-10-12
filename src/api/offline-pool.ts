import { FastEnhancerError, ModelInitError, ValidationError, WasmLoadError } from './errors.js';
import { checkAbort, type EnhanceOptions } from './offline-pcm.js';

export interface OfflineWorker {
  send(message: unknown, transfer: ArrayBuffer[]): void;
  listen(message: (value: any) => void, error: (error: unknown) => void): void;
  stop(): void | Promise<unknown>;
}
export type WorkerFactory = () => OfflineWorker;

export function parallelLimit(options: EnhanceOptions, concurrency = 2): number {
  if (!options.parallel || typeof options.parallel !== 'object') throw new ValidationError('parallel must be false or an options object');
  if (options.worker === false) throw new ValidationError('parallel cannot be combined with worker: false');
  const maximum = options.parallel.maxWorkers;
  if (options.parallel.engine !== undefined && !['auto', 'channels', 'batch'].includes(options.parallel.engine)) throw new ValidationError('parallel.engine must be auto, channels or batch');
  if (maximum !== undefined && (!Number.isInteger(maximum) || maximum < 1 || maximum > 32)) {
    throw new ValidationError('parallel.maxWorkers must be an integer between 1 and 32');
  }
  return Math.min(maximum ?? 4, Math.max(1, concurrency - 1));
}

/** Atomic FIFO acquisition avoids deadlocks between jobs requesting several workers. */
export class WorkerBudget {
  readonly capacity: number;
  private used = 0;
  private queue: Array<{ count: number; signal?: AbortSignal; abort: () => void; resolve: (release: () => void) => void; reject: (reason: unknown) => void }> = [];
  constructor(capacity: number) { this.capacity = capacity; }
  acquire(count: number, signal?: AbortSignal): Promise<() => void> {
    checkAbort(signal);
    if (!Number.isInteger(count) || count < 1 || count > this.capacity) throw new ValidationError('Invalid Worker budget request');
    return new Promise((resolve, reject) => {
      const entry = { count, signal, resolve, reject, abort: () => {
        const index = this.queue.indexOf(entry);
        if (index >= 0) this.queue.splice(index, 1);
        reject(signal?.reason ?? new DOMException('Audio processing aborted', 'AbortError'));
        this.drain();
      } };
      this.queue.push(entry);
      signal?.addEventListener('abort', entry.abort, { once: true });
      this.drain();
    });
  }
  private drain(): void {
    while (this.queue.length && this.used + this.queue[0].count <= this.capacity) {
      const entry = this.queue.shift()!;
      entry.signal?.removeEventListener('abort', entry.abort);
      this.used += entry.count;
      let released = false;
      entry.resolve(() => {
        if (released) return;
        released = true;
        this.used -= entry.count;
        this.drain();
      });
    }
  }
}

function revive(detail: { name?: string; message: string; code?: string }): Error {
  const ErrorClass = detail.code === 'VALIDATION_ERROR' ? ValidationError
    : detail.code === 'MODEL_INIT_FAILED' ? ModelInitError
    : detail.code === 'WASM_LOAD_FAILED' ? WasmLoadError : FastEnhancerError;
  return new ErrorClass(detail.message);
}

export class OfflinePool {
  private release: () => void;
  private signal?: AbortSignal;
  private slots: Array<{ worker: OfflineWorker; pending?: { id: number; resolve: (result: any) => void; reject: (reason: unknown) => void } }> = [];
  private id = 0;
  private closed = false;
  private reason: unknown;
  private stopping?: Promise<void>;
  private abort = (): void => { void this.close(this.signal?.reason ?? new DOMException('Audio processing aborted', 'AbortError')); };
  private constructor(release: () => void, signal?: AbortSignal) { this.release = release; this.signal = signal; }
  static async create(count: number, factory: WorkerFactory, budget: WorkerBudget, signal?: AbortSignal): Promise<OfflinePool> {
    const release = await budget.acquire(Math.min(count, budget.capacity), signal);
    const pool = new OfflinePool(release, signal);
    try {
      checkAbort(signal);
      for (let i = 0; i < Math.min(count, budget.capacity); i++) {
        const slot: OfflinePool['slots'][number] = { worker: factory() };
        pool.slots.push(slot);
        slot.worker.listen(message => {
          const pending = slot.pending;
          if (!pending || pool.closed) return;
          if (message?.id !== pending.id || (message.type !== 'result' && message.type !== 'error')) {
            void pool.close(new FastEnhancerError('Invalid audio Worker response'));
            return;
          }
          slot.pending = undefined;
          if (message.type === 'error') {
            const error = typeof message.error?.message === 'string' ? revive(message.error) : new FastEnhancerError('Invalid audio Worker error');
            pending.reject(error);
            void pool.close(error);
          } else pending.resolve(message.result);
        }, error => { void pool.close(error instanceof Error ? error : new FastEnhancerError(String(error))); });
      }
      signal?.addEventListener('abort', pool.abort, { once: true });
      checkAbort(signal);
      return pool;
    } catch (error) { await pool.close(error); throw error; }
  }
  get size(): number { return this.slots.length; }
  async run<T>(index: number, task: unknown, transfer: ArrayBuffer[] = []): Promise<T> {
    if (this.closed) throw this.reason;
    checkAbort(this.signal);
    const slot = this.slots[index];
    if (!slot || slot.pending) throw new FastEnhancerError('Audio Worker is busy or unavailable');
    return new Promise<T>((resolve, reject) => {
      const id = ++this.id;
      slot.pending = { id, resolve, reject };
      try { slot.worker.send({ id, task }, transfer); }
      catch (error) { void this.close(error); }
    });
  }
  /** Each worker pulls one tile at a time; no unbounded queue or tile copies. */
  async map(count: number, task: (tile: number, worker: number) => Promise<void>): Promise<void> {
    let next = 0;
    await Promise.all(this.slots.map(async (_, worker) => {
      while (next < count) {
        if (this.closed) throw this.reason;
        checkAbort(this.signal);
        await task(next++, worker);
      }
    }));
  }
  close(reason: unknown = new FastEnhancerError('Audio Worker pool is closed')): Promise<void> {
    if (this.stopping) return this.stopping;
    this.closed = true;
    this.reason = reason;
    this.signal?.removeEventListener('abort', this.abort);
    for (const slot of this.slots) {
      slot.pending?.reject(reason);
      slot.pending = undefined;
    }
    this.stopping = Promise.allSettled(this.slots.map(slot => {
      try { return slot.worker.stop(); } catch { return undefined; }
    })).then(() => { this.release(); });
    return this.stopping;
  }
}
