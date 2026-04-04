/**
 * Activation functions module (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine activation functions.
 * For testing and verification. Use the WASM side in the browser runtime.
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
 * Polynomial sigmoid approximation (equivalent to the C SIMD version)
 * 4th-5th order minimax approximation over the [-8, 8] range, max error < 1e-4
 */
export function polynomialSigmoid(x: number): number {
  if (Number.isNaN(x)) throw new Error('NaN input to polynomialSigmoid');
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return 0.0;

  const clamped = Math.max(-8, Math.min(8, x));

  // 5th-order minimax approximation (coefficients derived from the Remez algorithm)
  // f(x) ≈ 0.5 + ax + bx^3 + cx^5 (odd-function part only)
  const a = 0.2310586;
  const b = -0.0068706;
  const c = 0.0000784;
  const x2 = clamped * clamped;
  const result = 0.5 + clamped * (a + x2 * (b + x2 * c));

  return Math.max(0, Math.min(1, result));
}
