import { describe, it, expect } from 'vitest';
import { powerCompress, powerDecompress, applyComplexMask } from '../../../src/engine/compression.js';

/**
 * T-17対応: powerCompress 全象限検証
 *
 * 本テストはTypeScript参照実装 (src/engine/compression.ts) を検証する。
 * WASM/C実装の圧縮正しさは以下で検証される:
 *   - C native: tests/engine/test_compression.c (mag^0.3 圧縮・微小値安定性)
 *   - WASM: tests/wasm/golden-vectors.test.ts (全パイプライン PyTorch一致)
 *   - WASM: tests/wasm/wasm-properties.test.ts (無音/最大振幅/DC offset)
 */
describe('powerCompress 全象限', () => {
  it('第1象限: (3, 4) → 位相 ≈ 0.927', () => {
    const result = powerCompress(3, 4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(4, 3), 10);
  });

  it('第2象限: (-3, 4) → 位相 ≈ 2.214', () => {
    const result = powerCompress(-3, 4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(4, -3), 10);
  });

  it('第3象限: (-3, -4) → 位相 ≈ -2.214', () => {
    const result = powerCompress(-3, -4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(-4, -3), 10);
  });

  it('第4象限: (3, -4) → 位相 ≈ -0.927', () => {
    const result = powerCompress(3, -4);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.atan2(-4, 3), 10);
  });

  it('負の実軸: (-5, 0) → 位相 = π', () => {
    const result = powerCompress(-5, 0);
    expect(result.mag).toBeCloseTo(5 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(Math.PI, 10);
  });

  it('負の虚軸: (0, -7) → 位相 = -π/2', () => {
    const result = powerCompress(0, -7);
    expect(result.mag).toBeCloseTo(7 ** 0.3, 5);
    expect(result.phase).toBeCloseTo(-Math.PI / 2, 10);
  });

  it('全象限のラウンドトリップ', () => {
    const pairs: [number, number][] = [
      [3, 4],     // 第1象限
      [-3, 4],    // 第2象限
      [-3, -4],   // 第3象限
      [3, -4],    // 第4象限
      [-5, 0],    // 負の実軸
      [0, -7],    // 負の虚軸
      [0.001, -0.001], // 微小値
      [100, 200], // 大きな値
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
 * applyComplexMask 追加検証
 */
describe('applyComplexMask 追加', () => {
  it('負の実数マスクで位相反転', () => {
    const result = applyComplexMask(3, 4, -1, 0);
    expect(result.real).toBeCloseTo(-3, 10);
    expect(result.imag).toBeCloseTo(-4, 10);
  });

  it('純虚数マスク -i で -90度回転', () => {
    const result = applyComplexMask(3, 4, 0, -1);
    expect(result.real).toBeCloseTo(4, 10);
    expect(result.imag).toBeCloseTo(-3, 10);
  });
});
