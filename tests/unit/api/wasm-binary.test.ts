/**
 * wasm-binary.test.ts — tests for WASM binary URL generation and loading
 *
 * Covers the A2 review feedback: AudioWorklet cannot use Emscripten glue,
 * so it needs a mechanism to load raw .wasm binaries directly.
 * This module is responsible for generating WASM binary URLs and fetching them.
 */
import { describe, it, expect } from 'vitest';
import {
  getWasmBinaryPath,
  type ModelSize,
} from '../../../src/api/wasm-binary';

describe('getWasmBinaryPath', () => {
  it('returns the correct path for tiny + simd', () => {
    const path = getWasmBinaryPath('tiny', 'simd');
    expect(path).toBe('fastenhancer-tiny-simd.wasm');
  });

  it('returns the correct path for tiny + scalar', () => {
    const path = getWasmBinaryPath('tiny', 'scalar');
    expect(path).toBe('fastenhancer-tiny-scalar.wasm');
  });

  it('returns the correct path for base + simd', () => {
    const path = getWasmBinaryPath('base', 'simd');
    expect(path).toBe('fastenhancer-base-simd.wasm');
  });

  it('returns the correct path for small + scalar', () => {
    const path = getWasmBinaryPath('small', 'scalar');
    expect(path).toBe('fastenhancer-small-scalar.wasm');
  });

  it('adds baseUrl as a prefix when specified', () => {
    const path = getWasmBinaryPath('tiny', 'simd', '/assets/wasm/');
    expect(path).toBe('/assets/wasm/fastenhancer-tiny-simd.wasm');
  });

  it('joins baseUrl correctly even without a trailing slash', () => {
    const path = getWasmBinaryPath('tiny', 'simd', '/assets/wasm');
    expect(path).toBe('/assets/wasm/fastenhancer-tiny-simd.wasm');
  });

  it('works correctly when baseUrl uses https', () => {
    const path = getWasmBinaryPath('base', 'scalar', 'https://cdn.example.com/wasm/');
    expect(path).toBe('https://cdn.example.com/wasm/fastenhancer-base-scalar.wasm');
  });

  it('supports all model sizes with consistent naming', () => {
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
