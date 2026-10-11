// A real module entry lets bundlers include the worker and its embedded models.
import { enhancePcmInline } from './offline-pcm.js';

self.onmessage = async ({ data }) => {
  try {
    const audio = await enhancePcmInline(data.audio, {
      ...data.options,
      onProgress: progress => self.postMessage({ type: 'progress', progress }),
    });
    self.postMessage({ type: 'result', audio }, audio.channels.map(c => c.buffer));
  } catch (error) {
    self.postMessage({ type: 'error', error: {
      name: error instanceof Error ? error.name : 'Error',
      message: error instanceof Error ? error.message : String(error),
      code: error?.code,
    } });
  }
};
