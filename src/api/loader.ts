/**
 * loader.ts — 統合リソースローダー (v2)
 *
 * loadModel(modelSize, { baseUrl?, simd? }) で
 * WASM バイナリ(.wasm)、重みバイナリ(.bin)、エクスポートマップ(.json) を
 * 一括取得し、createDenoiser / createStreamDenoiser メソッド付きの
 * LoadedModel を返す。
 *
 * 責務: リソース取得 + LoadedModel構築 + キャッシュ のみ。推論処理は行わない。
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

/** loadModel に渡すオプション */
export interface LoadModelOptions {
  /**
   * 全リソースファイルのベースURL。
   * 省略時: import.meta.url から自動検出。
   * 指定時: このURL配下に WASM/重み/exportMap が存在する前提。
   */
  baseUrl?: string;
  /** SIMD使用を明示指定。省略時は自動検出。 */
  simd?: boolean;
}

/** loadModel の返却型。createDenoiser / createStreamDenoiser をメソッドとして持つ。 */
export interface LoadedModel {
  readonly size: ModelSize;
  readonly variant: WasmVariant;
  readonly sampleRate: number;
  readonly hopSize: number;
  readonly nFft: number;
  readonly modelSizeId: number;
  /** AudioWorklet 用の生 WASM バイナリ */
  readonly wasmBytes: ArrayBuffer;
  /** 重みバイナリデータ */
  readonly weightData: ArrayBuffer;
  /** WASM エクスポート名マッピング（-O3 ミニファイ対応） */
  readonly exportMap: Record<string, string>;
  /** Emscripten SINGLE_FILE factory（遅延ロード） */
  readonly wasmFactory: () => Promise<any>;
  /** Layer 1: フレーム単位デノイザーを生成 */
  createDenoiser(): Promise<any>;
  /** Layer 2: AudioWorklet 経由のリアルタイムストリームデノイザーを生成 */
  createStreamDenoiser(
    inputStream: MediaStream,
    options?: { workletUrl?: string; onWarning?: (message: string) => void },
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

function normalizeUrl(base: string): string {
  return base.endsWith('/') ? base : `${base}/`;
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
 * キャッシュ
 * ================================================================ */

const modelCache = new Map<string, Promise<LoadedModel>>();

/** テスト・リソース解放用: キャッシュを全消去する */
export function clearModelCache(): void {
  modelCache.clear();
}

/* ================================================================
 * loadModel — メインエントリーポイント
 * ================================================================ */

/**
 * モデルの全リソース（WASM + 重み + exportMap）を一括ロードする。
 * 同一引数の呼び出しはキャッシュされ、同一 Promise を返す。
 *
 * @param modelSize - 'tiny' | 'base' | 'small'
 * @param options - baseUrl（省略時は import.meta.url 自動検出）、simd 指定
 * @returns LoadedModel — createDenoiser / createStreamDenoiser メソッド付き
 * @throws ValidationError モデルサイズが不正な場合
 * @throws WasmLoadError fetch またはリソース解析に失敗した場合
 */
export function loadModel(
  modelSize: ModelSize,
  options?: LoadModelOptions,
): Promise<LoadedModel> {
  if (!VALID_MODELS.includes(modelSize)) {
    throw new ValidationError(
      `不正なモデルサイズ: "${modelSize}"。有効値: ${VALID_MODELS.join(', ')}`,
    );
  }

  const simd = options?.simd;
  const baseUrl = options?.baseUrl;
  const normalizedBaseUrl = baseUrl ? normalizeUrl(baseUrl) : undefined;
  const cacheKey = `${modelSize}:${simd ?? 'auto'}:${normalizedBaseUrl ?? 'default'}`;

  const cached = modelCache.get(cacheKey);
  if (cached) return cached;

  const promise = loadModelImpl(modelSize, simd, baseUrl);
  modelCache.set(cacheKey, promise);
  promise.catch(() => modelCache.delete(cacheKey));
  return promise;
}

/* ================================================================
 * loadModelImpl — 実リソース取得
 * ================================================================ */

async function loadModelImpl(
  modelSize: ModelSize,
  simdOption: boolean | undefined,
  baseUrlOption: string | undefined,
): Promise<LoadedModel> {
  const simdSupported =
    simdOption !== undefined ? simdOption : await detectSimd();
  const variant = selectWasmVariant(simdSupported);

  const wasmBase = baseUrlOption
    ? normalizeUrl(baseUrlOption)
    : new URL('../wasm/', import.meta.url).href;

  const weightBase = baseUrlOption
    ? normalizeUrl(baseUrlOption)
    : new URL('../../weights/', import.meta.url).href;

  const prefix = `fastenhancer-${modelSize}-${variant}`;
  const wasmBinaryUrl = `${wasmBase}${prefix}.wasm`;
  const exportMapUrl = `${wasmBase}${prefix}-exports.json`;
  const weightUrl = `${weightBase}${WEIGHT_FILENAMES[modelSize]}`;

  let wasmBytes: ArrayBuffer;
  let weightData: ArrayBuffer;
  let exportMap: Record<string, string>;

  try {
    const [wasmRes, weightRes, mapRes] = await Promise.all([
      fetch(wasmBinaryUrl),
      fetch(weightUrl),
      fetch(exportMapUrl),
    ]);

    if (!wasmRes.ok) {
      throw new WasmLoadError(
        `WASMバイナリの取得に失敗: ${wasmRes.status} ${wasmRes.statusText} (${wasmBinaryUrl})`,
      );
    }
    if (!weightRes.ok) {
      throw new WasmLoadError(
        `重みファイルの取得に失敗: ${weightRes.status} ${weightRes.statusText} (${weightUrl})`,
      );
    }
    if (!mapRes.ok) {
      throw new WasmLoadError(
        `エクスポートマップの取得に失敗: ${mapRes.status} ${mapRes.statusText} (${exportMapUrl})`,
      );
    }

    [wasmBytes, weightData, exportMap] = await Promise.all([
      wasmRes.arrayBuffer(),
      weightRes.arrayBuffer(),
      mapRes.json() as Promise<Record<string, string>>,
    ]);
  } catch (err) {
    if (err instanceof WasmLoadError) throw err;
    throw new WasmLoadError(
      `モデルリソースの読み込みに失敗しました: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  const modelSizeId = MODEL_SIZE_IDS[modelSize];
  const modelConfig = getModelConfig(modelSize);

  const wasmFactory = async (): Promise<any> => {
    return instantiateWasm(wasmBytes, exportMap);
  };

  const frozenExportMap = Object.freeze({ ...exportMap });

  return {
    size: modelSize,
    variant,
    sampleRate: modelConfig.sampleRate,
    hopSize: modelConfig.hopSize,
    nFft: modelConfig.nFft,
    modelSizeId,
    get wasmBytes() { return wasmBytes.slice(0); },
    get weightData() { return weightData.slice(0); },
    exportMap: frozenExportMap,
    wasmFactory,

    async createDenoiser() {
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
      opts?: { workletUrl?: string; onWarning?: (message: string) => void },
    ): Promise<StreamDenoiser> {
      return createStreamDenoiserImpl({
        inputStream,
        wasmBytes,
        weightBytes: weightData,
        exportMap,
        modelSize: modelSizeId,
        workletUrl: opts?.workletUrl,
        onWarning: opts?.onWarning,
      });
    },
  };
}
