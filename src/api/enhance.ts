import {
  checkAbort, enhancePcmInline, validatePcm,
  type EnhanceOptions, type PcmAudio,
} from './offline-pcm.js';
import { decodeWav, encodeWav, isWav } from './wav.js';
import { FastEnhancerError, ModelInitError, ValidationError, WasmLoadError } from './errors.js';

export type { EnhanceOptions, PcmAudio } from './offline-pcm.js';

interface WorkerResponse {
  type: 'progress' | 'result' | 'error';
  progress?: number;
  audio?: PcmAudio;
  error?: { name: string; message: string; code?: string };
}

function enhanceInWorker(audio: PcmAudio, options: EnhanceOptions): Promise<PcmAudio> {
  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(new URL('./enhance-worker.js', import.meta.url), { type: 'module' });
    } catch (cause) {
      reject(new FastEnhancerError('Could not start the audio Worker. Allow module workers in CSP or set worker: false.', { cause }));
      return;
    }
    const cleanup = (): void => {
      worker.onmessage = null;
      worker.onerror = null;
      worker.onmessageerror = null;
      worker.terminate();
      options.signal?.removeEventListener('abort', abort);
    };
    const abort = (): void => {
      cleanup();
      reject(options.signal?.reason ?? new DOMException('Audio processing aborted', 'AbortError'));
    };
    options.signal?.addEventListener('abort', abort, { once: true });
    if (options.signal?.aborted) { abort(); return; }
    worker.onerror = event => {
      cleanup();
      reject(new FastEnhancerError(`Audio Worker failed: ${event.message || 'check worker-src CSP and worker asset deployment'}`));
    };
    worker.onmessageerror = () => {
      cleanup();
      reject(new FastEnhancerError('Could not receive the audio Worker result'));
    };
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.type === 'progress') {
        try { options.onProgress?.(message.progress!); }
        catch (error) { cleanup(); reject(error); }
      } else if (message.type === 'result') {
        cleanup();
        resolve(message.audio!);
      } else if (message.type === 'error') {
        cleanup();
        const detail = message.error!;
        const ErrorClass = detail.code === 'VALIDATION_ERROR' ? ValidationError
          : detail.code === 'MODEL_INIT_FAILED' ? ModelInitError
          : detail.code === 'WASM_LOAD_FAILED' ? WasmLoadError : FastEnhancerError;
        reject(new ErrorClass(detail.message));
      }
    };
    try {
      // Transfer copies: never detach or modify the caller's buffers.
      const channels = audio.channels.map(c => c.slice());
      worker.postMessage({
        audio: { channels, sampleRate: audio.sampleRate },
        options: { model: options.model, simd: options.simd, baseUrl: options.baseUrl },
      }, channels.map(c => c.buffer));
    } catch (error) { cleanup(); reject(error); }
  });
}

/** Enhance mono/stereo PCM, retaining channel order, sample rate and sample count. */
export async function enhancePcm(audio: PcmAudio, options: EnhanceOptions = {}): Promise<PcmAudio> {
  checkAbort(options.signal);
  validatePcm(audio);
  const hasWorker = typeof window !== 'undefined' && typeof Worker !== 'undefined';
  if (options.worker === true && !hasWorker) {
    throw new ValidationError('Module Workers are unavailable; omit worker or set worker: false');
  }
  if (options.worker !== false && hasWorker) return enhanceInWorker(audio, options);
  return enhancePcmInline(audio, options);
}

/** Enhance an AudioBuffer without playback or microphone permission. */
export async function enhanceBuffer(buffer: AudioBuffer, options: EnhanceOptions = {}): Promise<AudioBuffer> {
  const channels = Array.from({ length: buffer.numberOfChannels }, (_, c) => buffer.getChannelData(c));
  const clean = await enhancePcm({ channels, sampleRate: buffer.sampleRate }, options);
  const result = new AudioBuffer({
    numberOfChannels: clean.channels.length, length: clean.channels[0].length, sampleRate: clean.sampleRate,
  });
  clean.channels.forEach((channel, c) => result.copyToChannel(new Float32Array(channel), c));
  return result;
}

/**
 * Enhance a File/Blob and return a ready-to-save PCM16 WAV Blob.
 * WAV retains its sample rate. Other browser-decodable formats are decoded at 48 kHz.
 * Mono/stereo channel count and decoded duration are retained.
 */
export async function enhanceAudio(file: Blob, options: EnhanceOptions = {}): Promise<Blob> {
  checkAbort(options.signal);
  const bytes = await file.arrayBuffer();
  checkAbort(options.signal);
  let audio: PcmAudio;
  if (isWav(bytes)) {
    audio = decodeWav(bytes);
  } else {
    if (typeof OfflineAudioContext === 'undefined') {
      throw new ValidationError('This runtime supports WAV files only; browser decoding is required for other formats');
    }
    try {
      const context = new OfflineAudioContext(2, 1, 48000);
      const decoded = await context.decodeAudioData(bytes);
      audio = {
        channels: Array.from({ length: decoded.numberOfChannels }, (_, c) => decoded.getChannelData(c)),
        sampleRate: decoded.sampleRate,
      };
    } catch (cause) {
      throw new ValidationError('The browser could not decode this audio file', { cause });
    }
  }
  checkAbort(options.signal);
  const clean = await enhancePcm(audio, options);
  checkAbort(options.signal);
  return new Blob([encodeWav(clean)], { type: 'audio/wav' });
}
