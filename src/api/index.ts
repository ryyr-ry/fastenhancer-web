import { detectSimdSupport } from './simd-detect.js';

export { createDenoiser } from './frame-denoiser.js';
export type { Denoiser, DenoiserState, WasmInstance, Model, PerformanceStats } from './frame-denoiser.js';

interface SupportInfo {
  wasm: boolean;
  simd: boolean;
  audioWorklet: boolean;
}

export async function isSupported(): Promise<SupportInfo> {
  const wasmSupported =
    typeof WebAssembly !== 'undefined' &&
    typeof WebAssembly.instantiate === 'function';

  let simdSupported = false;
  if (wasmSupported) {
    try {
      simdSupported = detectSimdSupport();
    } catch {
      simdSupported = false;
    }
  }

  const audioWorkletSupported =
    typeof globalThis !== 'undefined' &&
    typeof (globalThis as unknown as Record<string, unknown>).AudioWorkletNode === 'function';

  return {
    wasm: wasmSupported,
    simd: simdSupported,
    audioWorklet: audioWorkletSupported,
  };
}

export { loadModel, clearModelCache, clearCachedModel } from './loader.js';
export { createChannelDenoiser } from './channel-denoiser.js';
export type { ChannelDenoiser, ChannelCount } from './channel-denoiser.js';
export { enhancePcm, enhanceBuffer, enhanceAudio, enhanceWavStream } from './enhance.js';
export type { PcmAudio, EnhanceOptions } from './enhance.js';
export type { LoadModelOptions, LoadedModel } from './loader.js';
export { createStreamDenoiser } from './stream-denoiser.js';
export type { StreamDenoiserOptions, StreamDenoiser, WorkletStateResponse } from './stream-denoiser.js';
export { getModels, recommendModel } from './models.js';
export type { ModelInfo, ModelPriority, ModelRecommendation, RecommendModelOptions } from './models.js';
export { diagnose } from './diagnose.js';
export type { DiagnoseResult } from './diagnose.js';
export {
  FastEnhancerError,
  WasmLoadError,
  ModelInitError,
  AudioContextError,
  WorkletError,
  ValidationError,
  DestroyedError,
} from './errors.js';
