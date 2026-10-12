import { parentPort } from 'node:worker_threads';
import { createTaskHandler, resultTransfers } from './offline-worker-tasks.js';
const handle = createTaskHandler();
parentPort.on('message', async ({ id, task }) => {
  try {
    const result = await handle(task);
    parentPort.postMessage({ id, type: 'result', result }, resultTransfers(result));
  } catch (error) {
    parentPort.postMessage({ id, type: 'error', error: { name: error?.name, message: String(error?.message ?? error), code: error?.code } });
  }
});
