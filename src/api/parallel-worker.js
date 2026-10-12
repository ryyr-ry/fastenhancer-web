import { createTaskHandler, resultTransfers } from './offline-worker-tasks.js';
const handle = createTaskHandler();
self.onmessage = async ({ data: { id, task } }) => {
  try {
    const result = await handle(task);
    self.postMessage({ id, type: 'result', result }, resultTransfers(result));
  } catch (error) {
    self.postMessage({ id, type: 'error', error: { name: error?.name, message: String(error?.message ?? error), code: error?.code } });
  }
};
