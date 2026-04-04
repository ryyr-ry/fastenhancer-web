import { describe, it, expect } from 'vitest';
import { powerCompress, powerDecompress, applyComplexMask } from '../../../src/engine/compression.js';

/**
 * T-17 support: full-quadrant validation for powerCompress
 *
 * This test verifies the TypeScript reference implementation
 * (src/engine/compression.ts).
 * Compression correctness in the WASM/C implementation is verified by:
 *   - C native: tests/engine/test_compression.c (mag^0.3 compression / small-value stability)
 *   - WASM: tests/wasm/golden-vectors.test.ts (full pipeline matches PyTorch)
 *   - WASM: tests/wasm/wasm-properties.test.ts (silence / max amplitude / DC offset)
 */
describe('powerCompress across all quadrants', () => {
  it('first quadrant: (3, 4) → phase ≈ 0.927', () => {
    const result = powerCompress(3, 4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(4, 3), 10);
  });

  it('second quadrant: (-3, 4) → phase ≈ 2.214', () => {
    const result = powerCompress(-3, 4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(4, -3), 10);
  });

  it('third quadrant: (-3, -4) → phase ≈ -2.214', () => {
    const result = powerCompress(-3, -4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(-4, -3), 10);
  });

  it('fourth quadrant: (3, -4) → phase ≈ -0.927', () => {
    const result = powerCompress(3, -4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(-4, 3), 10);
  });

  it('negative real axis: (-5, 0) → phase = π', () => {
    const result = powerCompress(-5, 0);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.PI, 10);
  });

  it('negative imaginary axis: (0, -7) → phase = -π/2', () => {
    const result = powerCompress(0, -7);
    expect(result.mag).toBeCloseTo(7 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(-Math.PI / 2, 10);
  });

  it('round-trips across all quadrants', () => {
    const pairs: [number, number][] = [
      [3, 4],     // first quadrant
      [-3, 4],    // second quadrant
      [-3, -4],   // third quadrant
      [3, -4],    // fourth quadrant
      [-5, 0],    // negative real axis
      [0, -7],    // negative imaginary axis
      [0.001, -0.001], // very small values
      [100, 200], // large values
    ];
    for (const [re, im] of pairs) {
      const compressed = powerCompress(re, im);
      const restored = powerDecompress(compressed.mag, compressed.phase);
      expect(restored.real).toBeCloseTo(re, 4);
      expect(restored.imag).toBeCloseTo(im, 4);
    }
  });
});

/**
 * Additional validation for applyComplexMask
 */
describe('applyComplexMask additional cases', () => {
  it('inverts phase with a negative real mask', () => {
    const result = applyComplexMask(3, 4, -1, 0);
    expect(result.real).toBeCloseTo(-3, 10);
    expect(result.imag).toBeCloseTo(-4, 10);
  });

  it('rotates by -90 degrees with a purely imaginary -i mask', () => {
    const result = applyComplexMask(3, 4, 0, -1);
    expect(result.real).toBeCloseTo(4, 10);
    expect(result.imag).toBeCloseTo(-3, 10);
  });
});
