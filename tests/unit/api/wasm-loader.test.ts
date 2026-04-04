/**
 * wasm-loader.test.ts — WASMバリアント選択ローダーのテスト
 *
 * A3レビュー指摘対応: SIMD検出結果に基づいて適切なWASMビルドを選択し、
 * ロードする仕組みのテスト。
 */
import { describe, it, expect } from 'vitest';
import { selectWasmVariant, type WasmVariant } from '../../../src/api/wasm-loader';

describe('selectWasmVariant', () => {
  it('SIMD対応環境では "simd" を返す', () => {
    const variant = selectWasmVariant(true);
    expect(variant).toBe('simd');
  });

  it('SIMD非対応環境では "scalar" を返す', () => {
    const variant = selectWasmVariant(false);
    expect(variant).toBe('scalar');
  });

  it('戻り値はWasmVariant型に適合する', () => {
    const variant: WasmVariant = selectWasmVariant(true);
    expect(['scalar', 'simd']).toContain(variant);
  });
});
