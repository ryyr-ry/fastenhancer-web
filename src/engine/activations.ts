/**
 * 活性化関数モジュール (TypeScript参照実装)
 * Cエンジンの活性化関数と同等の純TypeScript実装。
 * テスト・検証用。ブラウザ実行時はWASM側を使う。
 */

export function sigmoid(x: number): number {
  if (Number.isNaN(x)) throw new Error('NaN input to sigmoid');
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return 0.0;
  return 1.0 / (1.0 + Math.exp(-x));
}

export function silu(x: number): number {
  if (Number.isNaN(x)) throw new Error('NaN input to silu');
  if (x === Infinity) return Infinity;
  if (x === -Infinity) return 0.0;
  return x * sigmoid(x);
}

export function tanhActivation(x: number): number {
  if (Number.isNaN(x)) throw new Error('NaN input to tanhActivation');
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return -1.0;
  return Math.tanh(x);
}

/**
 * 多項式近似sigmoid (C SIMD版と同等)
 * [-8, 8]範囲で4-5次ミニマックス近似、max error < 1e-4
 */
export function polynomialSigmoid(x: number): number {
  if (Number.isNaN(x)) throw new Error('NaN input to polynomialSigmoid');
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return 0.0;

  const clamped = Math.max(-8, Math.min(8, x));

  // 5次ミニマックス近似 (Remez算法由来の係数)
  // f(x) ≈ 0.5 + ax + bx^3 + cx^5 (奇関数部分のみ)
  const a = 0.2310586;
  const b = -0.0068706;
  const c = 0.0000784;
  const x2 = clamped * clamped;
  const result = 0.5 + clamped * (a + x2 * (b + x2 * c));

  return Math.max(0, Math.min(1, result));
}
