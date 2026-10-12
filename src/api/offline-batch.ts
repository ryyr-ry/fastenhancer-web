import { ValidationError } from './errors.js';
import { OfflinePool } from './offline-pool.js';
import type { EnhanceOptions } from './offline-pcm.js';
import type { ModelOptions } from './offline-worker-tasks.js';
import type { BatchInfo } from './offline-engine.js';
import { detectSimdSupport } from './simd-detect.js';

export interface InferenceSession {
  info: { sampleRate: number; hopSize: number; latency: number };
  framesPerTile: number;
  process(channels: Float32Array[]): Promise<Float32Array[]>;
}

export async function initializeChannels(pool: OfflinePool, channels: number, options: ModelOptions): Promise<{ sampleRate: number; hopSize: number; latency: number }> {
  const results = await Promise.all(Array.from({ length: Math.min(channels, pool.size) }, (_, worker) =>
    pool.run<{ sampleRate: number; hopSize: number; latency: number }>(worker, { type: 'init', options,
      channels: Array.from({ length: channels }, (_, c) => c).filter(c => c % pool.size === worker) })));
  return results[0];
}

export async function createInferenceSession(pool: OfflinePool, count: number, options: EnhanceOptions, seconds: number): Promise<InferenceSession> {
  const config = options.parallel || undefined;
  // The scalar rebuild has different libc/compiler rounding. Preserve legacy
  // scalar samples exactly, including when batch was requested explicitly.
  const simd = options.simd ?? detectSimdSupport();
  const batch = simd && (config?.engine === 'batch' || (config?.engine !== 'channels' && config && !options.baseUrl && pool.size > 1 && (count === 1 || pool.size > 2) && seconds >= 4));
  if (batch) {
    const infos = await Promise.all(Array.from({ length: pool.size }, (_, worker) => pool.run<BatchInfo>(worker, {
      type: 'init-batch', options: { model: options.model, simd: options.simd, baseUrl: options.baseUrl }, channels: Array.from({ length: count }, (_, i) => i),
    })));
    return new BatchSession(pool, count, infos[0]);
  }
  const info = await initializeChannels(pool, count, { model: options.model, simd: options.simd, baseUrl: options.baseUrl });
  return { info, framesPerTile: 128, async process(channels) {
    const output: Float32Array[] = new Array(count);
    await Promise.all(Array.from({ length: Math.min(count, pool.size) }, async (_, worker) => {
      const payload = channels.flatMap((c, key) => key % pool.size === worker ? [{ key, samples: c.slice() }] : []);
      const result = await pool.run<Array<{ key: number; samples: Float32Array }>>(worker, { type: 'infer', channels: payload }, payload.map(c => c.samples.buffer));
      result.forEach(c => { output[c.key] = c.samples; });
    }));
    return output;
  } };
}

class BatchSession implements InferenceSession {
  private pool: OfflinePool;
  private count: number;
  readonly info: BatchInfo;
  readonly framesPerTile = 64;
  private previous: Float32Array[];
  private overlap: Float32Array[];
  constructor(pool: OfflinePool, count: number, info: BatchInfo) {
    this.pool = pool; this.count = count; this.info = info;
    this.previous = Array.from({ length: count }, () => new Float32Array(info.hopSize));
    this.overlap = this.previous.map(c => c.slice());
  }
  private async pieces(frames: number, task: (key: number, first: number, end: number, worker: number) => Promise<void>): Promise<void> {
    await this.pool.map(this.count * this.pool.size, async (piece, worker) => {
      const key = Math.floor(piece / this.pool.size), part = piece % this.pool.size;
      const first = Math.floor(frames * part / this.pool.size), end = Math.floor(frames * (part + 1) / this.pool.size);
      if (end > first) await task(key, first, end, worker);
    });
  }
  async process(channels: Float32Array[]): Promise<Float32Array[]> {
    const { hopSize: hop, encodedSize: stride, featureSize: features, featureChannels: width, frequencyBins: bins, blocks } = this.info;
    const frames = channels[0].length / hop;
    if (channels.length !== this.count || !Number.isInteger(frames) || frames < 1 || frames > this.framesPerTile
      || channels.some(c => c.length !== frames * hop || !c.every(Number.isFinite))) throw new ValidationError('Invalid offline batch input');
    const encoded = channels.map(() => new Float32Array(frames * stride));
    await this.pieces(frames, async (key, first, end, worker) => {
      const input = new Float32Array((end - first + 1) * hop);
      input.set(first === 0 ? this.previous[key] : channels[key].subarray((first - 1) * hop, first * hop));
      input.set(channels[key].subarray(first * hop, end * hop), hop);
      const result = await this.pool.run<Float32Array>(worker, { type: 'batch-encode', key, frames: end - first, input }, [input.buffer]);
      encoded[key].set(result, first * stride);
    });
    channels.forEach((c, key) => this.previous[key].set(c.subarray(c.length - hop)));
    let current = encoded.map(c => {
      const output = new Float32Array(frames * features);
      for (let t = 0; t < frames; t++) output.set(c.subarray((t + 1) * stride - features, (t + 1) * stride), t * features);
      return output;
    });
    for (let block = 0; block < blocks; block++) {
      const recurrent = current.map(() => new Float32Array(frames * features));
      // Ownership stays fixed across blocks and tiles for each GRU history.
      await Promise.all(Array.from({ length: this.pool.size }, async (_, worker) => {
        const first = Math.floor(bins * worker / this.pool.size), end = Math.floor(bins * (worker + 1) / this.pool.size);
        if (end === first) return;
        const payload = current.map((input, key) => ({ key, input: input.slice() }));
        const result = await this.pool.run<Array<{ key: number; samples: Float32Array }>>(worker,
          { type: 'batch-gru', channels: payload, frames, block, first, end }, payload.map(c => c.input.buffer));
        for (const c of result) for (let t = 0; t < frames; t++) recurrent[c.key].set(c.samples.subarray(t * (end - first) * width, (t + 1) * (end - first) * width), t * features + first * width);
      }));
      const attended = current.map(() => new Float32Array(frames * features));
      await this.pieces(frames, async (key, first, end, worker) => {
        const input = recurrent[key].slice(first * features, end * features);
        const result = await this.pool.run<Float32Array>(worker, { type: 'batch-attention', key, frames: end - first, block, input }, [input.buffer]);
        attended[key].set(result, first * features);
      });
      current = attended;
    }
    const raw = channels.map(() => new Float32Array(frames * hop * 2));
    await this.pieces(frames, async (key, first, end, worker) => {
      const data = encoded[key].slice(first * stride, end * stride), input = current[key].slice(first * features, end * features);
      const result = await this.pool.run<Float32Array>(worker, { type: 'batch-decode', key, frames: end - first, encoded: data, features: input }, [data.buffer, input.buffer]);
      raw[key].set(result, first * hop * 2);
    });
    return raw.map((c, key) => {
      const output = new Float32Array(frames * hop);
      for (let t = 0; t < frames; t++) for (let i = 0; i < hop; i++) {
        output[t * hop + i] = c[t * hop * 2 + i] + this.overlap[key][i];
        this.overlap[key][i] = c[t * hop * 2 + hop + i];
      }
      return output;
    });
  }
}
