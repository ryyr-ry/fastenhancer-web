/**
 * wasm-binary.ts — WASMバイナリのパス生成
 *
 * AudioWorkletはEmscripten glueを実行できないため、
 * 生の.wasmバイナリを直接WebAssembly.instantiate()でロードする必要がある。
 * このモジュールはモデルサイズ・バリアントからWASMバイナリのURLを生成する。
 *
 * 責務: パス/URL生成のみ。実際のfetchはLayer 2が行う。
 */

import type { WasmVariant } from './wasm-loader';

/** 対応するモデルサイズ */
export type ModelSize = 'tiny' | 'base' | 'small';

/**
 * WASMバイナリファイルのパスを生成する。
 *
 * @param modelSize - モデルサイズ ('tiny' | 'base' | 'small')
 * @param variant - WASMバリアント ('scalar' | 'simd')
 * @param baseUrl - ベースURL。省略時はファイル名のみ返す。
 * @returns WASMバイナリファイルのパスまたはURL
 */
export function getWasmBinaryPath(
  modelSize: ModelSize,
  variant: WasmVariant,
  baseUrl?: string,
): string {
  const filename = `fastenhancer-${modelSize}-${variant}.wasm`;

  if (!baseUrl) {
    return filename;
  }

  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  return `${normalizedBase}${filename}`;
}
