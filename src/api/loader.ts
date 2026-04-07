/**
 * loader.ts — Unified resource loader (v2 + embedded)
 *
 * loadModel(modelSize, { baseUrl?, simd? }) loads WASM, weights, and export
 * map, returning a LoadedModel with createDenoiser/createStreamDenoiser.
 *
 * Two loading strategies:
 *   - baseUrl omitted → embedded JS modules (zero-config, works with all bundlers)
 *   - baseUrl provided → fetch() from URL (CDN / self-hosted deployment)
 *
 * Responsibility: resource loading, LoadedModel construction, and caching only.
 */

import { ValidationError, WasmLoadError } from './errors.js';
import { selectWasmVariant, type WasmVariant } from './wasm-loader.js';
import { detectSimdSupport } from './simd-detect.js';
import { instantiateWasm } from './wasm-instantiate.js';
import type { ModelSize } from './wasm-binary.js';
import { getModelConfig } from '../engine/model-config.js';
import {
  createStreamDenoiser as createStreamDenoiserImpl,
  type StreamDenoiser,
} from './stream-denoiser.js';
import type { Denoiser, WasmInstance } from './index.js';
import {
  loadEmbeddedWasm,
  loadEmbeddedWeights,
  loadEmbeddedExportMap,
  revokeProcessorBlobUrl,
} from './embedded-loader.js';

/** Options passed to loadModel */
export interface LoadModelOptions {
  /**
   * Base URL for all resource files.
   * If omitted: uses embedded JS modules (bundled assets loaded via dynamic import).
   * If provided: fetches WASM/weights/exportMap from this URL via fetch().
   */
  baseUrl?: string;
  /** Explicitly specify SIMD usage. If omitted, it is auto-detected. */
  simd?: boolean;
  /** AbortSignal for cancelling in-flight resource loading */
  signal?: AbortSignal;
}

/** Return type of loadModel. Provides createDenoiser / createStreamDenoiser methods. */
export interface LoadedModel {
  readonly size: ModelSize;
  readonly variant: WasmVariant;
  readonly sampleRate: number;
  readonly hopSize: number;
  readonly nFft: number;
  readonly modelSizeId: number;
  /** Raw WASM binary for AudioWorklet */
  readonly wasmBytes: ArrayBuffer;
  /** Weight binary data */
  readonly weightData: ArrayBuffer;
  /** WASM export name mapping (supports -O3 minified builds) */
  readonly exportMap: Record<string, string>;
  /** Emscripten SINGLE_FILE factory (lazy-loaded) */
  readonly wasmFactory: () => Promise<WasmInstance>;
  /** Layer 1: creates a frame-based denoiser */
  createDenoiser(): Promise<Denoiser>;
  /** Layer 2: creates a real-time stream denoiser through AudioWorklet */
  createStreamDenoiser(
    inputStream: MediaStream,
    options?: {
      workletUrl?: string;
      onWarning?: (message: string) => void;
      audioContext?: AudioContext;
      onAutoBypass?: (enabled: boolean) => void;
      onDestroy?: () => void;
      keepAliveInBackground?: boolean;
    },
  ): Promise<StreamDenoiser>;
}

const VALID_MODELS: ModelSize[] = ['tiny', 'base', 'small'];

const MODEL_SIZE_IDS: Record<ModelSize, number> = {
  tiny: 0,
  base: 1,
  small: 2,
};

const WEIGHT_FILENAMES: Record<ModelSize, string> = {
  tiny: 'fe_tiny_48k.bin',
  base: 'fe_base_48k.bin',
  small: 'fe_small_48k.bin',
};

function validateExportMap(data: unknown): Record<string, string> {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new WasmLoadError(
      'Export map JSON must be a plain object',
    );
  }
  const obj = data as Record<string, unknown>;
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value !== 'string') {
      throw new WasmLoadError(
        `Export map entry "${key}" must be a string, got ${typeof value}`,
      );
    }
  }
  return obj as Record<string, string>;
}

function normalizeUrl(base: string): string {
  return base.endsWith('/') ? base : `${base}/`;
}

function getCacheKey(
  modelSize: ModelSize,
  simd: boolean | undefined,
  baseUrl: string | undefined,
): string {
  const normalizedBaseUrl = baseUrl ? normalizeUrl(baseUrl) : undefined;
  return `${modelSize}:${simd ?? 'auto'}:${normalizedBaseUrl ?? 'default'}`;
}

async function detectSimd(): Promise<boolean> {
  try {
    return detectSimdSupport();
  } catch (e) {
    if (typeof console !== 'undefined') {
      console.warn('[fastenhancer] SIMD detection failed:', e);
    }
    return false;
  }
}

/* ================================================================
 * Cache
 * ================================================================ */

const modelCache = new Map<string, Promise<LoadedModel>>();

/**
 * Clears the entire model cache and revokes the cached processor.js Blob URL.
 * Does not destroy active denoiser instances — only removes cache entries.
 * After calling this, the next loadModel() call will re-fetch or re-decode all assets.
 */
export function clearModelCache(): void {
  modelCache.clear();
  revokeProcessorBlobUrl();
}

/**
 * Removes a specific model from the cache.
 * @param modelSize - The model size to evict
 * @param options - Must match the options used in the original loadModel() call
 */
export function clearCachedModel(
  modelSize: ModelSize,
  options?: LoadModelOptions,
): boolean {
  return modelCache.delete(getCacheKey(modelSize, options?.simd, options?.baseUrl));
}

/* ================================================================
 * loadModel — Main entry point
 * ================================================================ */

/**
 * Loads all model resources (WASM + weights + exportMap) at once.
 * Calls with the same arguments are cached and return the same Promise.
 *
 * @param modelSize - 'tiny' | 'base' | 'small'
 * @param options - baseUrl (auto-detected from import.meta.url if omitted), SIMD setting
 * @returns LoadedModel — with createDenoiser / createStreamDenoiser methods
 * @throws ValidationError If the model size is invalid
 * @throws WasmLoadError If fetching or parsing resources fails
 */
export function loadModel(
  modelSize: ModelSize,
  options?: LoadModelOptions,
): Promise<LoadedModel> {
  if (!VALID_MODELS.includes(modelSize)) {
    throw new ValidationError(
      `Invalid model size: "${modelSize}". Valid values: ${VALID_MODELS.join(', ')}`,
    );
  }

  const signal = options?.signal;
  if (signal?.aborted) {
    return Promise.reject(signal.reason ?? new Error('Aborted'));
  }

  const simd = options?.simd;
  const baseUrl = options?.baseUrl;
  const cacheKey = getCacheKey(modelSize, simd, baseUrl);

  const cached = modelCache.get(cacheKey);
  if (cached) return cached;

  const promise = loadModelImpl(modelSize, simd, baseUrl, signal);
  modelCache.set(cacheKey, promise);
  promise.catch(() => modelCache.delete(cacheKey));
  return promise;
}

/* ================================================================
 * loadModelImpl — Strategy dispatch
 * ================================================================ */

async function loadModelImpl(
  modelSize: ModelSize,
  simdOption: boolean | undefined,
  baseUrlOption: string | undefined,
  signal: AbortSignal | undefined,
): Promise<LoadedModel> {
  const simdSupported =
    simdOption !== undefined ? simdOption : await detectSimd();
  const variant = selectWasmVariant(simdSupported);

  if (baseUrlOption) {
    return loadModelViaFetch(modelSize, variant, baseUrlOption, signal);
  }
  return loadModelViaEmbed(modelSize, variant, signal);
}

/* ================================================================
 * loadModelViaEmbed — Zero-config path using embedded JS modules
 * ================================================================ */

async function loadModelViaEmbed(
  modelSize: ModelSize,
  variant: WasmVariant,
  signal: AbortSignal | undefined,
): Promise<LoadedModel> {
  let wasmBytes: ArrayBuffer;
  let weightData: ArrayBuffer;
  let exportMap: Record<string, string>;

  try {
    [wasmBytes, weightData, exportMap] = await Promise.all([
      loadEmbeddedWasm(modelSize, variant),
      loadEmbeddedWeights(modelSize),
      loadEmbeddedExportMap(modelSize, variant),
    ]);
  } catch (err) {
    if (err instanceof ValidationError) throw err;
    throw new WasmLoadError(
      `Failed to load embedded model resources: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (signal?.aborted) {
    throw signal.reason ?? new Error('Aborted');
  }

  return buildLoadedModel(modelSize, variant, wasmBytes, weightData, exportMap);
}

/* ================================================================
 * loadModelViaFetch — CDN/self-hosted path using fetch()
 * ================================================================ */

async function loadModelViaFetch(
  modelSize: ModelSize,
  variant: WasmVariant,
  baseUrlOption: string,
  signal: AbortSignal | undefined,
): Promise<LoadedModel> {
  const base = normalizeUrl(baseUrlOption);
  const prefix = `fastenhancer-${modelSize}-${variant}`;
  const wasmBinaryUrl = `${base}${prefix}.wasm`;
  const exportMapUrl = `${base}${prefix}-exports.json`;
  const weightUrl = `${base}${WEIGHT_FILENAMES[modelSize]}`;

  let wasmBytes: ArrayBuffer;
  let weightData: ArrayBuffer;
  let exportMap: Record<string, string>;

  try {
    const fetchOpts: RequestInit | undefined = signal ? { signal } : undefined;
    const [wasmRes, weightRes, mapRes] = await Promise.all([
      fetch(wasmBinaryUrl, fetchOpts),
      fetch(weightUrl, fetchOpts),
      fetch(exportMapUrl, fetchOpts),
    ]);

    if (!wasmRes.ok) {
      throw new WasmLoadError(
        `Failed to fetch WASM binary: ${wasmRes.status} ${wasmRes.statusText} (${wasmBinaryUrl})`,
      );
    }
    if (!weightRes.ok) {
      throw new WasmLoadError(
        `Failed to fetch weight file: ${weightRes.status} ${weightRes.statusText} (${weightUrl})`,
      );
    }
    if (!mapRes.ok) {
      throw new WasmLoadError(
        `Failed to fetch export map: ${mapRes.status} ${mapRes.statusText} (${exportMapUrl})`,
      );
    }

    [wasmBytes, weightData, exportMap] = await Promise.all([
      wasmRes.arrayBuffer(),
      weightRes.arrayBuffer(),
      mapRes.json().then(validateExportMap),
    ]);
  } catch (err) {
    if (err instanceof WasmLoadError) throw err;
    throw new WasmLoadError(
      `Failed to load model resources: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  return buildLoadedModel(modelSize, variant, wasmBytes, weightData, exportMap);
}

/* ================================================================
 * buildLoadedModel — Shared LoadedModel construction
 * ================================================================ */

function buildLoadedModel(
  modelSize: ModelSize,
  variant: WasmVariant,
  wasmBytes: ArrayBuffer,
  weightData: ArrayBuffer,
  exportMap: Record<string, string>,
): LoadedModel {
  const modelSizeId = MODEL_SIZE_IDS[modelSize];
  const modelConfig = getModelConfig(modelSize);

  let cachedWasmPromise: Promise<WasmInstance> | null = null;
  const wasmFactory = async (): Promise<WasmInstance> => {
    if (!cachedWasmPromise) {
      cachedWasmPromise = instantiateWasm(wasmBytes, exportMap);
      cachedWasmPromise.catch(() => {
        cachedWasmPromise = null;
      });
    }
    return cachedWasmPromise;
  };

  const frozenExportMap = Object.freeze({ ...exportMap });

  return {
    size: modelSize,
    variant,
    sampleRate: modelConfig.sampleRate,
    hopSize: modelConfig.hopSize,
    nFft: modelConfig.nFft,
    modelSizeId,
    // Defensive copy: each getter call returns a new ArrayBuffer copy.
    // Use sparingly — prefer storing the result if multiple accesses are needed.
    get wasmBytes() { return wasmBytes.slice(0); },
    get weightData() { return weightData.slice(0); },
    exportMap: frozenExportMap,
    wasmFactory,

    async createDenoiser(): Promise<Denoiser> {
      // Dynamic import intentionally avoids a runtime circular dependency with index.ts.
      const { createDenoiser: create } = await import('./index.js');
      return create({
        model: {
          size: modelSize,
          sampleRate: modelConfig.sampleRate,
          nFft: modelConfig.nFft,
          hopSize: modelConfig.hopSize,
          wasmFactory,
        },
        weightData,
        modelSizeId,
      });
    },

    async createStreamDenoiser(
      inputStream: MediaStream,
      opts?: {
        workletUrl?: string;
        onWarning?: (message: string) => void;
        audioContext?: AudioContext;
        onAutoBypass?: (enabled: boolean) => void;
        onDestroy?: () => void;
        keepAliveInBackground?: boolean;
      },
    ): Promise<StreamDenoiser> {
      return createStreamDenoiserImpl({
        inputStream,
        wasmBytes,
        weightBytes: weightData,
        exportMap,
        modelSize: modelSizeId,
        workletUrl: opts?.workletUrl,
        onWarning: opts?.onWarning,
        audioContext: opts?.audioContext,
        onAutoBypass: opts?.onAutoBypass,
        onDestroy: opts?.onDestroy,
        keepAliveInBackground: opts?.keepAliveInBackground,
      });
    },
  };
}
