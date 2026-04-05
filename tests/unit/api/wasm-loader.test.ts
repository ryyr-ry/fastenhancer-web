/**
 * wasm-loader.test.ts — tests for the WASM variant-selection loader
 *
 * Covers the A3 review feedback: select and load the appropriate WASM
 * build based on SIMD detection results.
 */
import { describe, it, expect } from 'vitest';
import { selectWasmVariant, type WasmVariant } from '../../../src/api/wasm-loader';

describe('selectWasmVariant', () => {
  it('returns "simd" in SIMD-capable environments', () => {
    const variant = selectWasmVariant(true);
    expect(variant).toBe('simd');
  });

  it('returns "scalar" in environments without SIMD support', () => {
    const variant = selectWasmVariant(false);
    expect(variant).toBe('scalar');
  });

  it('always returns one of the two valid WasmVariant values across repeated calls', () => {
    const valid: WasmVariant[] = ['scalar', 'simd'];
    for (let i = 0; i < 10; i++) {
      expect(valid).toContain(selectWasmVariant(i % 2 === 0));
    }
  });
});
