import { DestroyedError, ModelInitError, ValidationError } from './errors.js';
import { detectSimdSupport } from './simd-detect.js';

export type DenoiserState = 'ready' | 'destroyed';
type EventType = 'statechange' | 'destroy' | 'error';
type EventHandler = (...args: unknown[]) => void;

export interface WasmInstance {
  _fe_init: (model_size: number, weight_data: number, weight_len: number) => number;
  _fe_process: (state: number, input: number, output: number) => number;
  _fe_process_inplace: (state: number) => number;
  _fe_destroy: (state: number) => void;
  _fe_get_input_ptr: (state: number) => number;
  _fe_get_output_ptr: (state: number) => number;
  _fe_get_hop_size: (state: number) => number;
  _fe_get_n_fft: (state: number) => number;
  _fe_set_agc: (state: number, enabled: number) => void;
  _fe_set_hpf: (state: number, enabled: number) => void;
  _fe_reset: (state: number) => void;
  HEAPF32: Float32Array;
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
}

export interface Model {
  size: 'tiny' | 'base' | 'small';
  sampleRate: number;
  nFft: number;
  hopSize: number;
  wasmFactory: () => Promise<WasmInstance>;
  _wasm?: WasmInstance;
}

const MODEL_SIZE_ID_MAP: Record<string, number> = {
  tiny: 0,
  base: 1,
  small: 2,
};

function resolveModelSizeId(model: Model, explicitId?: number): number {
  if (explicitId !== undefined) return explicitId;
  const id = MODEL_SIZE_ID_MAP[model.size];
  if (id === undefined) {
    throw new ValidationError(
      `Unknown model size: "${model.size}". Valid values: tiny, base, small`,
    );
  }
  return id;
}

/**
 * Windowed performance statistics over the most recent N frames (default: 2000).
 * Values are rolling — they reflect recent performance, not lifetime totals.
 * `totalFrames` is the lifetime count; all timing stats are windowed.
 */
export interface PerformanceStats {
  avgMs: number;
  p99Ms: number;
  droppedFrames: number;
  totalFrames: number;
}

export interface Denoiser {
  readonly state: DenoiserState;
  processFrame(input: Float32Array): Float32Array;
  bypass: boolean;
  agcEnabled: boolean;
  hpfEnabled: boolean;
  destroy(): void;
  on(event: EventType, handler: EventHandler): void;
  off(event: EventType, handler: EventHandler): void;
  once(event: EventType, handler: EventHandler): void;
  readonly performance: PerformanceStats;
  readonly isSwitching: boolean;
  switchModel(options: { model: Model; weightData: ArrayBuffer; modelSizeId?: number }): Promise<void>;
}

function createDenoiserInstance(initialWasm: WasmInstance, initialStatePtr: number, initialHopSize: number): Denoiser {
  let currentState: DenoiserState = 'ready';
  let bypassEnabled = false;
  let agcFlag = false;
  let hpfFlag = false;
  let switchCount = 0;
  let switchQueue: Promise<void> | null = null;

  let wasm = initialWasm;
  let statePtr = initialStatePtr;
  let hopSize = initialHopSize;

  function getValidatedPointerOffset(ptr: number, label: string): number {
    if (!ptr || ptr % 4 !== 0) {
      throw new ModelInitError(
        `Invalid WASM ${label} pointer: ${ptr} (must be non-zero and 4-byte aligned)`,
      );
    }
    return ptr / 4;
  }

  let inputOffset = getValidatedPointerOffset(wasm._fe_get_input_ptr(statePtr), 'input');
  let outputOffset = getValidatedPointerOffset(wasm._fe_get_output_ptr(statePtr), 'output');

  const listeners = new Map<EventType, Set<EventHandler>>();
  const onceListeners = new Map<EventType, Set<EventHandler>>();
  const maxTimingSamples = 2000;
  const processingTimes = new Float64Array(maxTimingSamples);
  let timingWriteIndex = 0;
  let timingCount = 0;
  let lifetimeFrames = 0;
  let droppedFrames = 0;

  function emit(event: EventType, ...args: unknown[]): void {
    const handlers = listeners.get(event);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(...args);
        } catch (e) {
          if (typeof console !== 'undefined') {
            console.error('[fastenhancer] Event handler error:', e);
          }
        }
      }
    }

    const onceHandlers = onceListeners.get(event);
    if (onceHandlers) {
      for (const handler of onceHandlers) {
        try {
          handler(...args);
        } catch (e) {
          if (typeof console !== 'undefined') {
            console.error('[fastenhancer] Event handler error:', e);
          }
        }
      }
      onceHandlers.clear();
    }
  }

  function ensureReady(operation: string): void {
    if (currentState === 'destroyed') {
      throw new DestroyedError(`Cannot ${operation}: denoiser is destroyed`);
    }
  }

  function isDestroyed(): boolean {
    return currentState === 'destroyed';
  }

  const denoiser: Denoiser & {
    [Symbol.dispose](): void;
    [Symbol.asyncDispose](): Promise<void>;
  } = {
    get state(): DenoiserState {
      return currentState;
    },

    processFrame(input: Float32Array): Float32Array {
      ensureReady('processFrame');

      if (input.length !== hopSize) {
        throw new ValidationError(
          `Expected ${hopSize} samples, got ${input.length}`,
        );
      }

      for (let i = 0; i < input.length; i++) {
        if (!Number.isFinite(input[i])) {
          throw new ValidationError(
            `Input contains non-finite value at index ${i}: ${input[i]}`,
          );
        }
      }

      if (bypassEnabled) {
        return new Float32Array(input);
      }

      const start = performance.now();
      try {
        wasm.HEAPF32.set(input, inputOffset);
        wasm._fe_process_inplace(statePtr);
        const output = new Float32Array(hopSize);
        output.set(wasm.HEAPF32.subarray(outputOffset, outputOffset + hopSize));
        const elapsed = performance.now() - start;
        processingTimes[timingWriteIndex % maxTimingSamples] = elapsed;
        timingWriteIndex++;
        timingCount = Math.min(timingCount + 1, maxTimingSamples);
        lifetimeFrames++;
        return output;
      } catch (err) {
        droppedFrames++;
        const errorEvent = {
          code: 'PROCESS_ERROR',
          message: err instanceof Error ? err.message : String(err),
          original: err,
        };
        emit('error', errorEvent);
        throw err;
      }
    },

    get bypass(): boolean {
      return bypassEnabled;
    },
    set bypass(value: boolean) {
      ensureReady('set bypass');
      bypassEnabled = value;
    },

    get agcEnabled(): boolean {
      return agcFlag;
    },
    set agcEnabled(value: boolean) {
      ensureReady('set agcEnabled');
      agcFlag = value;
      wasm._fe_set_agc(statePtr, value ? 1 : 0);
    },

    get hpfEnabled(): boolean {
      return hpfFlag;
    },
    set hpfEnabled(value: boolean) {
      ensureReady('set hpfEnabled');
      hpfFlag = value;
      wasm._fe_set_hpf(statePtr, value ? 1 : 0);
    },

    destroy(): void {
      if (currentState === 'destroyed') {
        return;
      }
      wasm._fe_destroy(statePtr);
      currentState = 'destroyed';
      emit('statechange', 'destroyed');
      emit('destroy');
      listeners.clear();
      onceListeners.clear();
    },

    on(event: EventType, handler: EventHandler): void {
      if (!listeners.has(event)) {
        listeners.set(event, new Set());
      }
      listeners.get(event)!.add(handler);
    },

    off(event: EventType, handler: EventHandler): void {
      listeners.get(event)?.delete(handler);
      onceListeners.get(event)?.delete(handler);
    },

    once(event: EventType, handler: EventHandler): void {
      if (!onceListeners.has(event)) {
        onceListeners.set(event, new Set());
      }
      onceListeners.get(event)!.add(handler);
    },

    get performance(): PerformanceStats {
      if (lifetimeFrames === 0) {
        return { avgMs: 0, p99Ms: 0, droppedFrames, totalFrames: 0 };
      }
      const samples: number[] = [];
      for (let i = 0; i < timingCount; i++) {
        samples.push(processingTimes[i]);
      }
      const sum = samples.reduce((a, b) => a + b, 0);
      const avgMs = sum / timingCount;
      samples.sort((a, b) => a - b);
      const p99Index = Math.min(
        Math.ceil(timingCount * 0.99) - 1,
        timingCount - 1,
      );
      const p99Ms = samples[p99Index];
      return { avgMs, p99Ms, droppedFrames, totalFrames: lifetimeFrames };
    },

    get isSwitching(): boolean {
      return switchCount > 0;
    },

    async switchModel(options: { model: Model; weightData: ArrayBuffer; modelSizeId?: number }): Promise<void> {
      ensureReady('switchModel');
      switchCount++;

      const doSwitch = async () => {
        try {
          const { model, weightData, modelSizeId: explicitId } = options;
          const sizeId = resolveModelSizeId(model, explicitId);
          const newWasm = model._wasm ?? (await model.wasmFactory());

          if (isDestroyed()) {
            return;
          }

          const weightBytes = new Uint8Array(weightData);
          const weightLen = weightBytes.byteLength;
          const weightPtr = newWasm._malloc(weightLen);
          if (weightPtr === 0) {
            throw new ModelInitError('Failed to allocate memory in the WASM heap');
          }

          let newStatePtr: number;
          try {
            new Uint8Array(newWasm.HEAPF32.buffer).set(weightBytes, weightPtr);
            newStatePtr = newWasm._fe_init(sizeId, weightPtr, weightLen);
          } finally {
            newWasm._free(weightPtr);
          }

          if (isDestroyed()) {
            if (newStatePtr !== 0) {
              newWasm._fe_destroy(newStatePtr);
            }
            return;
          }

          if (newStatePtr === 0) {
            throw new ModelInitError('Failed to initialize the new model');
          }

          const oldWasm = wasm;
          const oldStatePtr = statePtr;

          wasm = newWasm;
          statePtr = newStatePtr;
          hopSize = newWasm._fe_get_hop_size(newStatePtr);
          inputOffset = getValidatedPointerOffset(newWasm._fe_get_input_ptr(newStatePtr), 'input');
          outputOffset = getValidatedPointerOffset(newWasm._fe_get_output_ptr(newStatePtr), 'output');

          if (agcFlag) newWasm._fe_set_agc(newStatePtr, 1);
          if (hpfFlag) newWasm._fe_set_hpf(newStatePtr, 1);

          oldWasm._fe_destroy(oldStatePtr);

          timingWriteIndex = 0;
          timingCount = 0;
          lifetimeFrames = 0;
          droppedFrames = 0;
        } finally {
          switchCount--;
        }
      };

      switchQueue = (switchQueue ?? Promise.resolve()).then(doSwitch, doSwitch);
      await switchQueue;
    },

    [Symbol.dispose](): void {
      denoiser.destroy();
    },

    async [Symbol.asyncDispose](): Promise<void> {
      denoiser.destroy();
    },
  };

  return denoiser;
}

/**
 * Creates a frame-level denoiser.
 *
 * Multiple denoisers created from the same Model share the underlying
 * WASM module instance for memory efficiency. Each denoiser maintains
 * independent processing state (via separate fe_init calls).
 *
 * @param options - Model, weight data, and optional model size ID
 * @returns A Denoiser instance for frame-by-frame processing
 */
export async function createDenoiser(options: {
  model: Model;
  weightData: ArrayBuffer;
  modelSizeId?: number;
}): Promise<Denoiser> {
  const { model, weightData, modelSizeId: explicitId } = options;
  const sizeId = resolveModelSizeId(model, explicitId);
  const wasm = model._wasm ?? (await model.wasmFactory());

  const weightBytes = new Uint8Array(weightData);
  const weightLen = weightBytes.byteLength;
  const weightPtr = wasm._malloc(weightLen);
  if (weightPtr === 0) {
    throw new ModelInitError('Failed to allocate memory in the WASM heap');
  }

  let statePtr: number;
  try {
    wasm.HEAPF32.buffer; // ensure buffer is valid
    new Uint8Array(wasm.HEAPF32.buffer).set(weightBytes, weightPtr);
    statePtr = wasm._fe_init(sizeId, weightPtr, weightLen);
  } finally {
    wasm._free(weightPtr);
  }

  if (statePtr === 0) {
    throw new ModelInitError('Failed to initialize the denoiser engine');
  }

  const hopSize = wasm._fe_get_hop_size(statePtr);
  return createDenoiserInstance(wasm, statePtr, hopSize);
}

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
