/**
 * 1D convolution operations module (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine's conv.c.
 */

/**
 * 1D convolution (single channel, single filter)
 * output[i] = sum_k input[i*stride + k] * kernel[k]  (for valid positions)
 */
export function conv1d(
  input: Float32Array,
  kernel: Float32Array,
  inChannels: number,
  outChannels: number,
  padding: number,
): Float32Array {
  const inputLen = input.length;
  const kernelLen = kernel.length;
  const outputLen = inputLen - kernelLen + 1 + 2 * padding;

  const padded = padding > 0 ? padArray(input, padding) : input;
  const paddedLen = padded.length;
  const outLen = paddedLen - kernelLen + 1;

  const output = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    let sum = 0;
    for (let k = 0; k < kernelLen; k++) {
      sum += padded[i + k] * kernel[k];
    }
    output[i] = sum;
  }
  return output;
}

/**
 * Strided 1D convolution
 */
export function stridedConv(
  input: Float32Array,
  kernel: Float32Array,
  inChannels: number,
  outChannels: number,
  stride: number,
): Float32Array {
  const inputLen = input.length;
  const kernelLen = kernel.length;
  const outputLen = Math.floor((inputLen - kernelLen) / stride) + 1;

  const output = new Float32Array(outputLen);
  for (let i = 0; i < outputLen; i++) {
    let sum = 0;
    const start = i * stride;
    for (let k = 0; k < kernelLen; k++) {
      sum += input[start + k] * kernel[k];
    }
    output[i] = sum;
  }
  return output;
}

/**
 * Transposed 1D convolution
 * output_length = (inputLen - 1) * stride + kernelLen
 */
export function convTranspose1d(
  input: Float32Array,
  kernel: Float32Array,
  inChannels: number,
  outChannels: number,
  stride: number,
): Float32Array {
  const inputLen = input.length;
  const kernelLen = kernel.length;
  const outputLen = (inputLen - 1) * stride + kernelLen;

  const output = new Float32Array(outputLen);
  for (let i = 0; i < inputLen; i++) {
    const outStart = i * stride;
    for (let k = 0; k < kernelLen; k++) {
      output[outStart + k] += input[i] * kernel[k];
    }
  }
  return output;
}

function padArray(input: Float32Array, padding: number): Float32Array {
  const padded = new Float32Array(input.length + 2 * padding);
  padded.set(input, padding);
  return padded;
}
