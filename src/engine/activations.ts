/**
 * Activation functions module (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine activation functions.
 * For testing and verification. Use the WASM side in the browser runtime.
 */

export function sigmoid(x: number): number {
  if (x !== x) return 0.5;
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return 0.0;
  return 1.0 / (1.0 + Math.exp(-x));
}

export function silu(x: number): number {
  if (x !== x) return 0.0;
  if (x === Infinity) return 3.4028235e+38;
  if (x === -Infinity) return 0.0;
  return x * sigmoid(x);
}

export function tanhActivation(x: number): number {
  if (x !== x) return 0.0;
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return -1.0;
  return Math.tanh(x);
}

/**
 * Polynomial sigmoid approximation using the C SIMD fast-exp coefficients.
 */
export function polynomialSigmoid(x: number): number {
  if (x !== x) return 0.5;
  if (x === Infinity) return 1.0;
  if (x === -Infinity) return 0.0;
  if (x >= 16.0) return 1.0;
  if (x <= -16.0) return 0.0;
  return 1.0 / (1.0 + fastExp(-x));
}

const FE_EXP_OVERFLOW = 88.0;
const FE_LOG2E = 1.4426950408889634;

function fastExp(x: number): number {
  if (x !== x) return 0.0;
  if (x < -FE_EXP_OVERFLOW) return 0.0;
  if (x > FE_EXP_OVERFLOW) return Infinity;

  const t = x * FE_LOG2E;
  const n = Math.floor(t);
  const f = t - n;
  const p = 1.0 + f * (0.6931472 + f * (0.2402265 + f * (0.0554953 + f * 0.0096838)));

  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setInt32(0, (n + 127) << 23, true);
  return p * view.getFloat32(0, true);
}
