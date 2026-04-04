import { describe, it, expect } from 'vitest';
import { hannWindow } from '../../../src/engine/fft.js';

/**
 * T-14 support: COLA (Constant Overlap-Add) condition tests
 *
 * This test verifies hannWindow in the TypeScript reference implementation
 * (src/engine/fft.ts).
 * COLA correctness in the WASM/C implementation is verified by:
 *   - C native: tests/engine/test_stft.c (Hann window / overlap-add)
 *   - WASM: tests/wasm/golden-vectors.test.ts (full pipeline matches PyTorch)
 *   - WASM: tests/wasm/wasm-properties.test.ts (frame-boundary discontinuity check)
 */
describe('COLA condition', () => {
  it('keeps the overlap-add sum at 1.0 at every position for a periodic Hann window + hop=N/2', () => {
    const N = 1024;
    const hop = N / 2;
    const w = hannWindow(N);
    const numFrames = 10;
    const totalLen = hop * (numFrames - 1) + N;

    const sumBuffer = new Float32Array(totalLen);
    for (let frame = 0; frame < numFrames; frame++) {
      const offset = frame * hop;
      for (let i = 0; i < N; i++) {
        sumBuffer[offset + i] += w[i];
      }
    }

    // The sum should be 1.0 in the steady region (the center excluding the first and last hop)
    const startSteady = hop;
    const endSteady = totalLen - hop;
    for (let i = startSteady; i < endSteady; i++) {
      expect(sumBuffer[i]).toBeCloseTo(1.0, 6);
    }
  });

  it('satisfies neighboring-frame window sum: w[n] + w[n + hop] = 1.0', () => {
    const N = 1024;
    const hop = N / 2;
    const w = hannWindow(N);

    for (let n = 0; n < hop; n++) {
      expect(w[n] + w[n + hop]).toBeCloseTo(1.0, 6);
    }
  });

  it('still satisfies COLA with a small window size: N=256, hop=128', () => {
    const N = 256;
    const hop = N / 2;
    const w = hannWindow(N);

    for (let n = 0; n < hop; n++) {
      expect(w[n] + w[n + hop]).toBeCloseTo(1.0, 6);
    }
  });
});
