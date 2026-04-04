/**
 * 1024-point Radix-2 DIT FFT / iFFT + Hann window
 * (TypeScript reference implementation)
 * Equivalent to the C engine's fft.c / stft.c. For testing and verification.
 */

export interface FftResult {
  real: Float32Array;
  imag: Float32Array;
}

/**
 * Generate a periodic Hann window: w[n] = 0.5 * (1 - cos(2πn/N))
 */
export function hannWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  const twoPiOverN = (2 * Math.PI) / n;
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos(twoPiOverN * i));
  }
  return w;
}

/**
 * Radix-2 DIT FFT (in-place style, returns new arrays)
 */
export function fft(realIn: Float32Array, imagIn: Float32Array): FftResult {
  const n = realIn.length;
  if (n === 0 || (n & (n - 1)) !== 0) {
    throw new Error(`FFT size must be power of 2, got ${n}`);
  }

  const real = new Float32Array(realIn);
  const imag = new Float32Array(imagIn);

  bitReverse(real, imag, n);

  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const angleStep = (-2 * Math.PI) / size;

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const angle = angleStep * j;
        const twRe = Math.cos(angle);
        const twIm = Math.sin(angle);

        const evenIdx = i + j;
        const oddIdx = i + j + halfSize;

        const tRe = twRe * real[oddIdx] - twIm * imag[oddIdx];
        const tIm = twRe * imag[oddIdx] + twIm * real[oddIdx];

        real[oddIdx] = real[evenIdx] - tRe;
        imag[oddIdx] = imag[evenIdx] - tIm;
        real[evenIdx] = real[evenIdx] + tRe;
        imag[evenIdx] = imag[evenIdx] + tIm;
      }
    }
  }

  return { real, imag };
}

/**
 * Inverse FFT: iFFT(X) = conj(FFT(conj(X))) / N
 */
export function ifft(realIn: Float32Array, imagIn: Float32Array): FftResult {
  const n = realIn.length;

  const conjImag = new Float32Array(n);
  for (let i = 0; i < n; i++) conjImag[i] = -imagIn[i];

  const result = fft(realIn, conjImag);

  const invN = 1.0 / n;
  for (let i = 0; i < n; i++) {
    result.real[i] *= invN;
    result.imag[i] = -result.imag[i] * invN;
  }

  return result;
}

function bitReverse(real: Float32Array, imag: Float32Array, n: number): void {
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    const j = reverseBits(i, bits);
    if (j > i) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }
}

function reverseBits(x: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}
