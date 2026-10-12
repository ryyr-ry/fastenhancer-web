import { loadModel } from './loader.js';
import { resampleTile, type ResampleTile } from './offline-resample.js';
import { DestroyedError, ValidationError } from './errors.js';
import type { Denoiser } from './index.js';
import type { EnhanceOptions } from './offline-pcm.js';
import { OfflineEngine } from './offline-engine.js';

export type ModelOptions = Pick<EnhanceOptions, 'model' | 'simd' | 'baseUrl'>;
export type OfflineTask =
  | { type: 'init'; channels: number[]; options: ModelOptions }
  | { type: 'infer'; channels: Array<{ key: number; samples: Float32Array }> }
  | { type: 'resample'; tile: ResampleTile };

export type BatchTask =
  | { type: 'init-batch'; channels: number[]; options: ModelOptions }
  | { type: 'batch-encode'; key: number; input: Float32Array; frames: number }
  | { type: 'batch-gru'; channels: Array<{ key: number; input: Float32Array }>; frames: number; block: number; first: number; end: number }
  | { type: 'batch-attention'; key: number; input: Float32Array; frames: number; block: number }
  | { type: 'batch-decode'; key: number; encoded: Float32Array; features: Float32Array; frames: number };

/** A worker owns each channel's state for the complete file. */
export function createTaskHandler(cooperative = false): ((task: OfflineTask | BatchTask) => Promise<unknown>) & { dispose(): void } {
  const denoisers = new Map<number, Denoiser>();
  const engines = new Map<number, OfflineEngine>();
  let hopSize = 512;
  let disposed = false;
  const ensureActive = (): void => { if (disposed) throw new DestroyedError('Audio processing session is closed'); };
  const clear = (): void => { denoisers.forEach(d => d.destroy()); denoisers.clear(); engines.forEach(e => e.destroy()); engines.clear(); };
  const handle = async (task: OfflineTask | BatchTask): Promise<unknown> => {
    ensureActive();
    if (task.type === 'resample') return resampleTile(task.tile);
    if (task.type === 'init-batch') {
      clear();
      try {
        for (const key of task.channels) {
          const engine = await OfflineEngine.create(task.options);
          if (disposed) { engine.destroy(); ensureActive(); }
          engines.set(key, engine);
        }
      } catch (error) { clear(); throw error; }
      return engines.values().next().value!.info;
    }
    if (task.type === 'batch-gru') return task.channels.map(c => ({ key: c.key, samples: engines.get(c.key)!.gru(c.input, task.frames, task.block, task.first, task.end) }));
    if (task.type === 'batch-encode') return engines.get(task.key)!.encode(task.input, task.frames);
    if (task.type === 'batch-attention') return engines.get(task.key)!.attention(task.input, task.frames, task.block);
    if (task.type === 'batch-decode') return engines.get(task.key)!.decode(task.encoded, task.features, task.frames);
    if (task.type === 'init') {
      clear();
      const model = await loadModel(task.options.model ?? 'small', { simd: task.options.simd, baseUrl: task.options.baseUrl });
      ensureActive();
      try {
        for (const key of task.channels) {
          const denoiser = await model.createDenoiser({ isolated: true });
          if (disposed) { denoiser.destroy(); ensureActive(); }
          denoisers.set(key, denoiser);
        }
      } catch (error) { denoisers.forEach(d => d.destroy()); denoisers.clear(); throw error; }
      hopSize = model.hopSize;
      return { sampleRate: model.sampleRate, hopSize, latency: model.nFft - hopSize };
    }
    if (task.type !== 'infer') throw new ValidationError('Unknown audio Worker task');
    const outputs: Array<{ key: number; samples: Float32Array }> = [];
    for (const { key, samples } of task.channels) {
      const denoiser = denoisers.get(key);
      if (!denoiser || samples.length % hopSize !== 0) throw new ValidationError('Invalid inference block');
      const output = new Float32Array(samples.length);
      let lastYield = performance.now();
      for (let i = 0; i < samples.length; i += hopSize) {
        ensureActive();
        const frame = denoiser.processFrame(samples.subarray(i, i + hopSize));
        if (!frame.every(Number.isFinite)) throw new ValidationError('Engine produced non-finite samples');
        output.set(frame, i);
        if (cooperative && performance.now() - lastYield > 16) {
          await new Promise(resolve => setTimeout(resolve, 0));
          lastYield = performance.now();
        }
      }
      outputs.push({ key, samples: output });
    }
    return outputs;
  };
  return Object.assign(handle, { dispose() { disposed = true; clear(); } });
}

export function resultTransfers(result: any): ArrayBuffer[] {
  if (result instanceof Float32Array) return [result.buffer as ArrayBuffer];
  if (!Array.isArray(result)) return [];
  return result.map(item => (item instanceof Float32Array ? item : item.samples).buffer as ArrayBuffer);
}
