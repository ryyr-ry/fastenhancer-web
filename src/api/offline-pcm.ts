import { loadModel } from './loader.js';
import { ValidationError } from './errors.js';
import type { ModelSize } from './wasm-binary.js';

/** Planar PCM: channels[0] is left/mono, channels[1] is right. */
export interface PcmAudio {
  channels: readonly Float32Array[];
  sampleRate: number;
}

export interface EnhanceOptions {
  /** Defaults to small. */
  model?: ModelSize;
  simd?: boolean;
  baseUrl?: string;
  signal?: AbortSignal;
  /** Fraction completed, from 0 to 1. */
  onProgress?: (progress: number) => void;
  /** Browser: use a module Worker by default. Set false for restricted CSP. */
  worker?: boolean;
  /** Opt-in exact parallel processing. The limit applies to the entire job. */
  parallel?: false | { maxWorkers?: number; engine?: 'auto' | 'channels' | 'batch' };
}

export function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw signal.reason ?? new DOMException('Audio processing aborted', 'AbortError');
}

export function validatePcm(audio: PcmAudio): void {
  if (!Number.isInteger(audio.sampleRate) || audio.sampleRate < 8000 || audio.sampleRate > 192000) {
    throw new ValidationError('sampleRate must be an integer between 8000 and 192000 Hz');
  }
  if (audio.channels.length !== 1 && audio.channels.length !== 2) {
    throw new ValidationError('Expected one mono channel or two stereo channels');
  }
  const length = audio.channels[0]?.length;
  for (const channel of audio.channels) {
    if (!(channel instanceof Float32Array) || channel.length !== length) {
      throw new ValidationError('PCM channels must be Float32Arrays of equal length');
    }
  }
}

const yieldTask = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0));

/** Zero-phase windowed-sinc resampling with an anti-aliasing low-pass filter. */
export async function resamplePcm(
  audio: PcmAudio,
  sampleRate: number,
  signal?: AbortSignal,
  onProgress?: (progress: number) => void,
  outputLength = Math.ceil(audio.channels[0].length * sampleRate / audio.sampleRate),
): Promise<PcmAudio> {
  checkAbort(signal);
  if (sampleRate === audio.sampleRate) {
    const channels = audio.channels.map(c => c.slice());
    onProgress?.(1);
    checkAbort(signal);
    return { channels, sampleRate };
  }
  const ratio = audio.sampleRate / sampleRate;
  const cutoff = Math.min(1, 1 / ratio) * 0.95;
  // Expand the kernel when downsampling so the transition band remains narrow.
  const radius = Math.ceil(32 / cutoff);
  const channels: Float32Array[] = [];
  let lastYield = performance.now();
  for (let c = 0; c < audio.channels.length; c++) {
    const input = audio.channels[c];
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i++) {
      const position = i * ratio;
      const start = Math.max(0, Math.ceil(position - radius));
      const end = Math.min(input.length - 1, Math.floor(position + radius));
      let sum = 0;
      let weightSum = 0;
      for (let j = start; j <= end; j++) {
        const distance = position - j;
        const x = Math.PI * cutoff * distance;
        const sinc = Math.abs(x) < 1e-12 ? 1 : Math.sin(x) / x;
        const window = 0.5 + 0.5 * Math.cos(Math.PI * distance / radius);
        const weight = cutoff * sinc * window;
        sum += input[j] * weight;
        weightSum += weight;
      }
      output[i] = weightSum === 0 ? 0 : sum / weightSum;
      if ((i & 1023) === 0) {
        checkAbort(signal);
        if (performance.now() - lastYield > 16) {
          onProgress?.((c + i / Math.max(1, outputLength)) / audio.channels.length);
          await yieldTask();
          checkAbort(signal);
          lastYield = performance.now();
        }
      }
    }
    channels.push(output);
  }
  onProgress?.(1);
  checkAbort(signal);
  return { channels, sampleRate };
}

/** Internal engine, shared by the browser Worker and non-browser runtimes. */
export async function enhancePcmInline(audio: PcmAudio, options: EnhanceOptions = {}): Promise<PcmAudio> {
  checkAbort(options.signal);
  validatePcm(audio);
  const length = audio.channels[0].length;
  for (const channel of audio.channels) {
    if (!channel.every(Number.isFinite)) throw new ValidationError('PCM contains non-finite samples');
  }
  options.onProgress?.(0);
  if (length === 0) {
    options.onProgress?.(1);
    checkAbort(options.signal);
    return { channels: audio.channels.map(() => new Float32Array(0)), sampleRate: audio.sampleRate };
  }
  // Resource caching is shared, but cancellation belongs to this job only.
  const model = await loadModel(options.model ?? 'small', { simd: options.simd, baseUrl: options.baseUrl });
  checkAbort(options.signal);
  const input = await resamplePcm(audio, model.sampleRate, options.signal, p => options.onProgress?.(p * 0.1));
  const denoiser = await model.createChannelDenoiser(audio.channels.length as 1 | 2);
  try {
    checkAbort(options.signal);
    const inputLength = input.channels[0].length;
    const output = input.channels.map(() => new Float32Array(inputLength));
    const frames = input.channels.map(() => new Float32Array(model.hopSize));
    const latency = denoiser.latencySamples;
    const totalFrames = Math.ceil((inputLength + latency) / model.hopSize);
    let lastYield = performance.now();
    for (let f = 0; f < totalFrames; f++) {
      checkAbort(options.signal);
      const start = f * model.hopSize;
      for (let c = 0; c < frames.length; c++) {
        frames[c].fill(0);
        if (start < inputLength) frames[c].set(input.channels[c].subarray(start, start + model.hopSize));
      }
      const processed = denoiser.processFrame(frames);
      // Drop the streaming delay and retain exactly the input's duration.
      const outputStart = start - latency;
      const from = Math.max(0, -outputStart);
      const to = Math.min(model.hopSize, inputLength - outputStart);
      if (to > from) {
        for (let c = 0; c < output.length; c++) {
          if (!processed[c].every(Number.isFinite)) throw new ValidationError('Engine produced non-finite samples');
          output[c].set(processed[c].subarray(from, to), outputStart + from);
        }
      }
      if (performance.now() - lastYield > 16) {
        options.onProgress?.(0.1 + 0.8 * (f + 1) / totalFrames);
        await yieldTask();
        lastYield = performance.now();
      }
    }
    checkAbort(options.signal);
    return await resamplePcm(
      { channels: output, sampleRate: model.sampleRate }, audio.sampleRate, options.signal,
      p => options.onProgress?.(0.9 + p * 0.1), length,
    );
  } finally {
    denoiser.destroy();
  }
}
