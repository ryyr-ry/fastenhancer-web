/**
 * simd-detect.test.ts — tests for WASM SIMD runtime detection
 *
 * Covers the A3 review feedback: detect SIMD support at runtime and
 * fall back to the scalar build in unsupported browsers.
 */
import { describe, it, expect } from 'vitest';
import { detectSimdSupport, getSimdTestBytes } from '../../../src/api/simd-detect';

describe('detectSimdSupport', () => {
  it('returns a boolean', () => {
    const result = detectSimdSupport();
    expect(typeof result).toBe('boolean');
  });

  it('works in Node.js when WebAssembly.validate exists', () => {
    const result = detectSimdSupport();
    const hasValidate = typeof WebAssembly !== 'undefined' && typeof WebAssembly.validate === 'function';
    if (hasValidate) {
      const expected = WebAssembly.validate(getSimdTestBytes());
      expect(result).toBe(expected);
    } else {
      expect(result).toBe(false);
    }
  });

  it('returns false when WebAssembly.validate throws', () => {
    const originalValidate = WebAssembly.validate;
    try {
      (WebAssembly as any).validate = () => { throw new Error('not supported'); };
      expect(detectSimdSupport()).toBe(false);
    } finally {
      WebAssembly.validate = originalValidate;
    }
  });

  it('returns false when WebAssembly.validate returns false', () => {
    const originalValidate = WebAssembly.validate;
    try {
      (WebAssembly as any).validate = () => false;
      expect(detectSimdSupport()).toBe(false);
    } finally {
      WebAssembly.validate = originalValidate;
    }
  });

  it('returns false when the WebAssembly object does not exist', () => {
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
  it('returns a Uint8Array', () => {
    const bytes = getSimdTestBytes();
    expect(bytes).toBeInstanceOf(Uint8Array);
  });

  it('is a valid WASM module binary', () => {
    const bytes = getSimdTestBytes();
    // WASM magic number: \0asm
    expect(bytes[0]).toBe(0x00);
    expect(bytes[1]).toBe(0x61); // 'a'
    expect(bytes[2]).toBe(0x73); // 's'
    expect(bytes[3]).toBe(0x6d); // 'm'
  });

  it('returns a new instance on every call', () => {
    const a = getSimdTestBytes();
    const b = getSimdTestBytes();
    expect(a).not.toBe(b);
    expect(a).toEqual(b);
  });
});
