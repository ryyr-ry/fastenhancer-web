/**
 * wasm-binary.test.ts — WASM バイナリURL生成とローダーのテスト
 *
 * A2レビュー指摘対応: AudioWorkletはEmscripten glueを使えないため、
 * 生の.wasmバイナリを直接ロードする仕組みが必要。
 * このモジュールはWASMバイナリのURL生成とフェッチを担当する。
 */
import { describe, it, expect } from 'vitest';
import {
  getWasmBinaryPath,
  type ModelSize,
} from '../../../src/api/wasm-binary';

describe('getWasmBinaryPath', () => {
  it('tiny + simd の正しいパスを返す', () => {
    const path = getWasmBinaryPath('tiny', 'simd');
    expect(path).toBe('fastenhancer-tiny-simd.wasm');
  });

  it('tiny + scalar の正しいパスを返す', () => {
    const path = getWasmBinaryPath('tiny', 'scalar');
    expect(path).toBe('fastenhancer-tiny-scalar.wasm');
  });

  it('base + simd の正しいパスを返す', () => {
    const path = getWasmBinaryPath('base', 'simd');
    expect(path).toBe('fastenhancer-base-simd.wasm');
  });

  it('small + scalar の正しいパスを返す', () => {
    const path = getWasmBinaryPath('small', 'scalar');
    expect(path).toBe('fastenhancer-small-scalar.wasm');
  });

  it('baseUrlを指定した場合、プレフィックスとして付加される', () => {
    const path = getWasmBinaryPath('tiny', 'simd', '/assets/wasm/');
    expect(path).toBe('/assets/wasm/fastenhancer-tiny-simd.wasm');
  });

  it('baseUrlが末尾スラッシュなしでも正しく結合する', () => {
    const path = getWasmBinaryPath('tiny', 'simd', '/assets/wasm');
    expect(path).toBe('/assets/wasm/fastenhancer-tiny-simd.wasm');
  });

  it('baseUrlがhttpsの場合も正しく動作する', () => {
    const path = getWasmBinaryPath('base', 'scalar', 'https://cdn.example.com/wasm/');
    expect(path).toBe('https://cdn.example.com/wasm/fastenhancer-base-scalar.wasm');
  });

  it('全モデルサイズに対応し、命名規則が一貫している', () => {
    const sizes: ModelSize[] = ['tiny', 'base', 'small'];
    const variants = ['simd', 'scalar'] as const;
    for (const size of sizes) {
      for (const variant of variants) {
        const result = getWasmBinaryPath(size, variant);
        expect(result).toBe(`fastenhancer-${size}-${variant}.wasm`);
      }
    }
  });
});
