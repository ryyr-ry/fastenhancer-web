/**
 * simd-detect.test.ts — WASM SIMDランタイム検出のテスト
 *
 * A3レビュー指摘対応: SIMDサポートをランタイムで検出し、
 * 非対応ブラウザではscalar版にフォールバックする仕組みのテスト。
 */
import { describe, it, expect } from 'vitest';
import { detectSimdSupport, getSimdTestBytes } from '../../../src/api/simd-detect';

describe('detectSimdSupport', () => {
  it('boolean を返す', () => {
    const result = detectSimdSupport();
    expect(typeof result).toBe('boolean');
  });

  it('Node.js環境ではWebAssembly.validateが存在すれば動作する', () => {
    const result = detectSimdSupport();
    const hasValidate = typeof WebAssembly !== 'undefined' && typeof WebAssembly.validate === 'function';
    if (hasValidate) {
      const expected = WebAssembly.validate(getSimdTestBytes());
      expect(result).toBe(expected);
    } else {
      expect(result).toBe(false);
    }
  });

  it('WebAssembly.validateが例外を投げた場合はfalseを返す', () => {
    const originalValidate = WebAssembly.validate;
    try {
      (WebAssembly as any).validate = () => { throw new Error('not supported'); };
      expect(detectSimdSupport()).toBe(false);
    } finally {
      WebAssembly.validate = originalValidate;
    }
  });

  it('WebAssembly.validateがfalseを返した場合はfalseを返す', () => {
    const originalValidate = WebAssembly.validate;
    try {
      (WebAssembly as any).validate = () => false;
      expect(detectSimdSupport()).toBe(false);
    } finally {
      WebAssembly.validate = originalValidate;
    }
  });

  it('WebAssemblyオブジェクトが存在しない場合はfalseを返す', () => {
    const originalWasm = globalThis.WebAssembly;
    try {
      (globalThis as any).WebAssembly = undefined;
      expect(detectSimdSupport()).toBe(false);
    } finally {
      globalThis.WebAssembly = originalWasm;
    }
  });
});

describe('getSimdTestBytes', () => {
  it('Uint8Arrayを返す', () => {
    const bytes = getSimdTestBytes();
    expect(bytes).toBeInstanceOf(Uint8Array);
  });

  it('有効なWASMモジュールバイナリである', () => {
    const bytes = getSimdTestBytes();
    // WASMマジックナンバー: \0asm
    expect(bytes[0]).toBe(0x00);
    expect(bytes[1]).toBe(0x61); // 'a'
    expect(bytes[2]).toBe(0x73); // 's'
    expect(bytes[3]).toBe(0x6d); // 'm'
  });

  it('呼び出しごとに新しいインスタンスを返す', () => {
    const a = getSimdTestBytes();
    const b = getSimdTestBytes();
    expect(a).not.toBe(b);
    expect(a).toEqual(b);
  });
});
