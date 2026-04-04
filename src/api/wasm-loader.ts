/**
 * wasm-loader.ts — WASMバリアント選択
 *
 * SIMD検出結果に基づき、scalar / simd のどちらの
 * WASMビルドをロードするか決定する。
 *
 * 責務: バリアント選択のみ。実際のWASMロードはLayer 1/2が行う。
 */

/** WASMビルドバリアント */
export type WasmVariant = 'scalar' | 'simd';

/**
 * SIMD対応状況に応じてWASMバリアントを選択する。
 *
 * @param simdSupported - SIMD検出結果（detectSimdSupport()の戻り値）
 * @returns ロードすべきWASMバリアント
 */
export function selectWasmVariant(simdSupported: boolean): WasmVariant {
  return simdSupported ? 'simd' : 'scalar';
}
