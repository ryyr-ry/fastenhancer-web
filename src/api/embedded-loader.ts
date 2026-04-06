/**
 * embedded-loader.ts — 埋込バイナリ資産のルーター
 *
 * コード生成された src/embedded/*.ts モジュールをdynamic import()で読み込む。
 * switch文による静的パスはバンドラーのコード分割を正しく動作させる。
 *
 * 責務: 埋込モジュールの選択と読み込みのみ。
 * loader.tsから呼ばれ、baseUrl未指定時のゼロコンフィグパスを提供する。
 */

import { ValidationError } from './errors.js';
import type { ModelSize } from './wasm-binary.js';
import type { WasmVariant } from './wasm-loader.js';

const VALID_MODELS: readonly string[] = ['tiny', 'base', 'small'];
const VALID_VARIANTS: readonly string[] = ['scalar', 'simd'];

function validateModelVariant(model: string, variant: string): void {
  if (!VALID_MODELS.includes(model)) {
    throw new ValidationError(
      `Invalid model size for embedded loader: "${model}". Valid values: ${VALID_MODELS.join(', ')}`,
    );
  }
  if (!VALID_VARIANTS.includes(variant)) {
    throw new ValidationError(
      `Invalid variant for embedded loader: "${variant}". Valid values: ${VALID_VARIANTS.join(', ')}`,
    );
  }
}

function validateModel(model: string): void {
  if (!VALID_MODELS.includes(model)) {
    throw new ValidationError(
      `Invalid model size for embedded loader: "${model}". Valid values: ${VALID_MODELS.join(', ')}`,
    );
  }
}

/**
 * 埋込WASMバイナリを読み込む。
 * キャッシュ済みの同一ArrayBufferを返す。
 * Transferable安全のためにslice()が必要な場合は呼び出し側で行うこと。
 */
export async function loadEmbeddedWasm(
  model: ModelSize,
  variant: WasmVariant,
): Promise<ArrayBuffer> {
  validateModelVariant(model, variant);
  const key = `${model}-${variant}`;
  switch (key) {
    case 'tiny-scalar':
      return (await import('../embedded/wasm-tiny-scalar.js')).getWasmBytes();
    case 'tiny-simd':
      return (await import('../embedded/wasm-tiny-simd.js')).getWasmBytes();
    case 'base-scalar':
      return (await import('../embedded/wasm-base-scalar.js')).getWasmBytes();
    case 'base-simd':
      return (await import('../embedded/wasm-base-simd.js')).getWasmBytes();
    case 'small-scalar':
      return (await import('../embedded/wasm-small-scalar.js')).getWasmBytes();
    case 'small-simd':
      return (await import('../embedded/wasm-small-simd.js')).getWasmBytes();
    default:
      throw new ValidationError(`Unknown WASM variant: ${key}`);
  }
}

/**
 * 埋込重みバイナリを読み込む。
 * キャッシュ済みの同一ArrayBufferを返す。
 * Transferable安全のためにslice()が必要な場合は呼び出し側で行うこと。
 */
export async function loadEmbeddedWeights(
  model: ModelSize,
): Promise<ArrayBuffer> {
  validateModel(model);
  switch (model) {
    case 'tiny':
      return (await import('../embedded/weights-tiny.js')).getWeightData();
    case 'base':
      return (await import('../embedded/weights-base.js')).getWeightData();
    case 'small':
      return (await import('../embedded/weights-small.js')).getWeightData();
    default:
      throw new ValidationError(`Unknown model: ${model}`);
  }
}

/**
 * 埋込エクスポートマップを読み込む。
 * Object.freeze済みのRecordを返す。
 */
export async function loadEmbeddedExportMap(
  model: ModelSize,
  variant: WasmVariant,
): Promise<Record<string, string>> {
  validateModelVariant(model, variant);
  const key = `${model}-${variant}`;
  switch (key) {
    case 'tiny-scalar':
      return (await import('../embedded/exports-tiny-scalar.js')).exportMap;
    case 'tiny-simd':
      return (await import('../embedded/exports-tiny-simd.js')).exportMap;
    case 'base-scalar':
      return (await import('../embedded/exports-base-scalar.js')).exportMap;
    case 'base-simd':
      return (await import('../embedded/exports-base-simd.js')).exportMap;
    case 'small-scalar':
      return (await import('../embedded/exports-small-scalar.js')).exportMap;
    case 'small-simd':
      return (await import('../embedded/exports-small-simd.js')).exportMap;
    default:
      throw new ValidationError(`Unknown export map variant: ${key}`);
  }
}

/**
 * processor.jsのソースコードからBlob URLを生成する。
 * 結果はキャッシュされ、同一URLが返る。
 *
 * Node.js環境（vitest）ではURL.createObjectURLが存在しないため
 * data: URLにフォールバックする。
 */
let _blobUrl: string | null = null;
let _processorSourceCache: string | null = null;

/**
 * processor.jsのBlob URLを返す。
 * 初回呼び出し前にensureProcessorSourceLoaded()が完了している必要がある。
 * 結果はキャッシュされる。
 */
export function getProcessorBlobUrl(): string {
  if (_blobUrl) return _blobUrl;

  if (!_processorSourceCache) {
    throw new Error(
      'getProcessorBlobUrl() requires prior initialization. Call ensureProcessorSourceLoaded() first.',
    );
  }

  return _createBlobUrl(_processorSourceCache);
}

function _createBlobUrl(source: string): string {
  // ブラウザ環境: Blob + URL.createObjectURL
  if (typeof Blob !== 'undefined' && typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function') {
    const blob = new Blob([source], { type: 'application/javascript' });
    _blobUrl = URL.createObjectURL(blob);
    return _blobUrl;
  }

  // Node.js/テスト環境/Edge Runtime: data: URLフォールバック (Buffer不使用)
  const encoded = btoa(unescape(encodeURIComponent(source)));
  _blobUrl = `data:application/javascript;base64,${encoded}`;
  return _blobUrl;
}

/**
 * processor.jsのソースコードを非同期で読み込んでキャッシュする。
 * loadModel()の内部から1回呼ばれる。
 */
export async function ensureProcessorSourceLoaded(): Promise<void> {
  if (_processorSourceCache) return;
  const { processorSource } = await import('../embedded/processor-source.js');
  _processorSourceCache = processorSource;
}

/**
 * processor.jsのBlob URLを非同期で初期化して返す。
 * ensureProcessorSourceLoaded + getProcessorBlobUrl の一括ヘルパー。
 */
export async function initProcessorBlobUrl(): Promise<string> {
  if (_blobUrl) return _blobUrl;
  await ensureProcessorSourceLoaded();
  return getProcessorBlobUrl();
}

/**
 * キャッシュされたBlob URLを破棄する。テスト・クリーンアップ用。
 */
export function revokeProcessorBlobUrl(): void {
  if (_blobUrl) {
    if (_blobUrl.startsWith('blob:') && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(_blobUrl);
    }
    _blobUrl = null;
  }
}
