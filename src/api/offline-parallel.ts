import { checkAbort, validatePcm, type EnhanceOptions, type PcmAudio } from './offline-pcm.js';
import { ValidationError } from './errors.js';
import { OfflinePool } from './offline-pool.js';
import { resampleInputRange } from './offline-resample.js';
import { createInferenceSession } from './offline-batch.js';

export const OFFLINE_TILE_SAMPLES = 65536;

export { initializeChannels } from './offline-batch.js';

export async function resampleParallel(audio: PcmAudio, rate: number, pool: OfflinePool, progress: (value: number) => void, length = Math.ceil(audio.channels[0].length * rate / audio.sampleRate)): Promise<PcmAudio> {
  if (audio.sampleRate === rate) { progress(1); return audio; }
  const output = audio.channels.map(() => new Float32Array(length));
  const tiles = Math.ceil(length / OFFLINE_TILE_SAMPLES);
  let completed = 0;
  await pool.map(tiles, async (tile, worker) => {
    const start = tile * OFFLINE_TILE_SAMPLES, end = Math.min(length, start + OFFLINE_TILE_SAMPLES);
    const [inputBase, inputEnd] = resampleInputRange(audio.sampleRate, rate, audio.channels[0].length, start, end);
    const channels = audio.channels.map(c => c.slice(inputBase, inputEnd));
    const result = await pool.run<Float32Array[]>(worker, { type: 'resample', tile: {
      channels, sourceRate: audio.sampleRate, targetRate: rate, sourceLength: audio.channels[0].length,
      inputBase, outputStart: start, outputEnd: end,
    } }, channels.map(c => c.buffer));
    result.forEach((c, i) => output[i].set(c, start));
    completed += end - start;
    progress(completed / length);
  });
  return { channels: output, sampleRate: rate };
}

/** Exact per-channel state and global resampling positions, with bounded transfers. */
export async function enhanceWithPool(audio: PcmAudio, options: EnhanceOptions, pool: OfflinePool): Promise<PcmAudio> {
  checkAbort(options.signal);
  validatePcm(audio);
  for (const channel of audio.channels) if (!channel.every(Number.isFinite)) throw new ValidationError('PCM contains non-finite samples');
  options.onProgress?.(0);
  const length = audio.channels[0].length;
  if (length === 0) { options.onProgress?.(1); checkAbort(options.signal); return { channels: audio.channels.map(() => new Float32Array(0)), sampleRate: audio.sampleRate }; }
  const session = await createInferenceSession(pool, audio.channels.length, options, length / audio.sampleRate);
  const model = session.info;
  const input = await resampleParallel(audio, model.sampleRate, pool, p => { options.onProgress?.(p * 0.1); checkAbort(options.signal); });
  const inputLength = input.channels[0].length;
  const output = input.channels.map(() => new Float32Array(inputLength));
  const frames = Math.ceil((inputLength + model.latency) / model.hopSize);
  const tileFrames = session.framesPerTile;
  for (let f = 0; f < frames; f += tileFrames) {
    checkAbort(options.signal);
    const start = f * model.hopSize, count = Math.min(tileFrames, frames - f) * model.hopSize;
    const padded = input.channels.map(c => { const samples = new Float32Array(count); if (start < inputLength) samples.set(c.subarray(start, start + count)); return samples; });
    const result = await session.process(padded);
    const outputStart = start - model.latency;
    const from = Math.max(0, -outputStart), to = Math.min(count, inputLength - outputStart);
    if (to > from) result.forEach((c, key) => output[key].set(c.subarray(from, to), outputStart + from));
    options.onProgress?.(0.1 + 0.8 * Math.min(frames, f + tileFrames) / frames);
  }
  checkAbort(options.signal);
  const result = await resampleParallel({ channels: output, sampleRate: model.sampleRate }, audio.sampleRate, pool, p => { options.onProgress?.(0.9 + p * 0.1); checkAbort(options.signal); }, length);
  checkAbort(options.signal);
  return result;
}
