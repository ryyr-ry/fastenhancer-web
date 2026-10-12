import { describe, it, expect } from 'vitest';
import { OfflinePool, WorkerBudget, parallelLimit, type OfflineWorker } from '../../../src/api/offline-pool.js';

class FakeWorker implements OfflineWorker {
  message!: (value: any) => void;
  error!: (error: unknown) => void;
  sent: any[] = [];
  stopped = false;
  stopPromise?: Promise<void>;
  listen(message: (value: any) => void, error: (error: unknown) => void) { this.message = message; this.error = error; }
  send(message: unknown) { this.sent.push(message); }
  reply(result: unknown = null) { this.message({ id: this.sent.at(-1).id, type: 'result', result }); }
  stop() { this.stopped = true; return this.stopPromise; }
}
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

describe('bounded offline Worker lifecycle', () => {
  it('acquires budgets atomically and lets an aborted waiter unblock the queue', async () => {
    const budget = new WorkerBudget(3), release = await budget.acquire(2), controller = new AbortController();
    const waiting = budget.acquire(2, controller.signal);
    const rejected = expect(waiting).rejects.toMatchObject({ name: 'AbortError' });
    let granted = false;
    const smaller = budget.acquire(1).then(release => { granted = true; return release; });
    await tick(); expect(granted).toBe(false);
    controller.abort(); await rejected;
    const releaseSmaller = await smaller; expect(granted).toBe(true);
    release(); releaseSmaller();
  });
  it('rejects every running request and releases the budget after abort', async () => {
    const workers: FakeWorker[] = [], budget = new WorkerBudget(2), controller = new AbortController();
    const pool = await OfflinePool.create(2, () => { const w = new FakeWorker(); workers.push(w); return w; }, budget, controller.signal);
    const reason = new Error('cancelled');
    const requests = [pool.run(0, 'left'), pool.run(1, 'right')];
    const settled = Promise.allSettled(requests);
    controller.abort(reason);
    expect((await settled).map(r => r.status === 'rejected' && r.reason)).toEqual([reason, reason]);
    await pool.close(); expect(workers.every(w => w.stopped)).toBe(true);
    (await budget.acquire(2))();
  });
  it('stops sibling workers after a protocol error and ignores late results', async () => {
    const workers: FakeWorker[] = [];
    const pool = await OfflinePool.create(2, () => { const w = new FakeWorker(); workers.push(w); return w; }, new WorkerBudget(2));
    const settled = Promise.allSettled([pool.run(0, 'a'), pool.run(1, 'b')]);
    workers[0].message({ id: -1, type: 'result' });
    expect((await settled).every(r => r.status === 'rejected')).toBe(true);
    workers[1].reply('late'); await pool.close();
    expect(workers.every(w => w.stopped)).toBe(true);
  });
  it('cleans up a partially constructed pool when worker startup fails', async () => {
    const first = new FakeWorker(), budget = new WorkerBudget(2); let count = 0;
    await expect(OfflinePool.create(2, () => { if (count++) throw new Error('CSP'); return first; }, budget)).rejects.toThrow('CSP');
    expect(first.stopped).toBe(true); (await budget.acquire(2))();
  });
  it('holds the shared budget until termination has completed', async () => {
    const worker = new FakeWorker(), budget = new WorkerBudget(1); let finish!: () => void;
    worker.stopPromise = new Promise(resolve => { finish = resolve; });
    const pool = await OfflinePool.create(1, () => worker, budget);
    const closed = pool.close(); let acquired = false;
    const waiting = budget.acquire(1).then(release => { acquired = true; return release; });
    await tick(); expect(acquired).toBe(false);
    finish(); await closed; (await waiting)();
  });
  it('runs multiple tiles concurrently without assigning two to one worker', async () => {
    let busy = 0, peak = 0, starts = 0;
    const pool = await OfflinePool.create(2, () => {
      const worker = new FakeWorker();
      worker.send = message => {
        worker.sent.push(message); starts++; busy++; peak = Math.max(peak, busy);
        setTimeout(() => { busy--; worker.reply(worker.sent.at(-1).task); }, 1);
      };
      return worker;
    }, new WorkerBudget(2));
    const results: number[] = [];
    await pool.map(12, async (tile, slot) => { results[tile] = await pool.run(slot, tile); });
    await pool.close(); expect(results).toEqual(Array.from({ length: 12 }, (_, i) => i));
    expect(starts).toBe(12); expect(peak).toBe(2);
  });
  it.each([0, -1, 1.5, 33, NaN])('rejects invalid parallel.maxWorkers %s', maximum => {
    expect(() => parallelLimit({ parallel: { maxWorkers: maximum } })).toThrow('maxWorkers');
  });
  it('respects worker: false and rejects non-object parallel options', () => {
    expect(() => parallelLimit({ worker: false, parallel: {} })).toThrow('worker: false');
    expect(() => parallelLimit({ parallel: true as any })).toThrow('options object');
  });
});
