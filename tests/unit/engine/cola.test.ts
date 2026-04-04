import { describe, it, expect } from 'vitest';
import { hannWindow } from '../../../src/engine/fft.js';

/**
 * T-14対応: COLA(Constant Overlap-Add)条件テスト
 *
 * 本テストはTypeScript参照実装 (src/engine/fft.ts) の hannWindow を検証する。
 * WASM/C実装のCOLA正しさは以下で検証される:
 *   - C native: tests/engine/test_stft.c (Hann窓・overlap-add)
 *   - WASM: tests/wasm/golden-vectors.test.ts (全パイプライン PyTorch一致)
 *   - WASM: tests/wasm/wasm-properties.test.ts (フレーム境界不連続検査)
 */
describe('COLA条件', () => {
  it('periodic Hann窓 + hop=N/2 で overlap-add 合計が全位置で 1.0', () => {
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

    // 定常領域(最初と最後のhopを除く中央部分)で合計が1.0であること
    const startSteady = hop;
    const endSteady = totalLen - hop;
    for (let i = startSteady; i < endSteady; i++) {
      expect(sumBuffer[i]).toBeCloseTo(1.0, 6);
    }
  });

  it('窓関数の隣接フレーム合計: w[n] + w[n + hop] = 1.0', () => {
    const N = 1024;
    const hop = N / 2;
    const w = hannWindow(N);

    for (let n = 0; n < hop; n++) {
      expect(w[n] + w[n + hop]).toBeCloseTo(1.0, 6);
    }
  });

  it('小さいウィンドウサイズでもCOLA成立: N=256, hop=128', () => {
    const N = 256;
    const hop = N / 2;
    const w = hannWindow(N);

    for (let n = 0; n < hop; n++) {
      expect(w[n] + w[n + hop]).toBeCloseTo(1.0, 6);
    }
  });
});
