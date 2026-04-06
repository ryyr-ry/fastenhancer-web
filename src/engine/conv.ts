/**
 * 1D convolution operations module (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine's conv.c.
 */

export function conv1d(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | null,
  inLen: number,
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  stride: number,
  padding: number,
): Float32Array {
  if (stride <= 0) return new Float32Array(0);

  if (input.length < inChannels * inLen) {
    throw new RangeError(
      `input length (${input.length}) must be >= inChannels*inLen (${inChannels * inLen})`,
    );
  }
  const expectedWeight = outChannels * inChannels * kernelSize;
  if (weight.length < expectedWeight) {
    throw new RangeError(
      `weight length (${weight.length}) must be >= outChannels*inChannels*kernelSize (${expectedWeight})`,
    );
  }
  if (bias && bias.length < outChannels) {
    throw new RangeError(
      `bias length (${bias.length}) must be >= outChannels (${outChannels})`,
    );
  }

  const outLen = Math.floor((inLen + 2 * padding - kernelSize) / stride) + 1;
  if (outLen <= 0) return new Float32Array(0);

  const output = new Float32Array(outChannels * outLen);

  for (let oc = 0; oc < outChannels; oc++) {
    for (let p = 0; p < outLen; p++) {
      let sum = bias ? bias[oc] : 0.0;

      for (let ic = 0; ic < inChannels; ic++) {
        for (let k = 0; k < kernelSize; k++) {
          const inPos = p * stride + k - padding;
          if (inPos >= 0 && inPos < inLen) {
            const wIdx = oc * inChannels * kernelSize + ic * kernelSize + k;
            sum += weight[wIdx] * input[ic * inLen + inPos];
          }
        }
      }

      output[oc * outLen + p] = sum;
    }
  }

  return output;
}

export function stridedConv(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | null,
  inLen: number,
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  stride: number,
): Float32Array {
  if (stride <= 0) return new Float32Array(0);

  const outLen = Math.floor((inLen - kernelSize) / stride) + 1;
  if (outLen <= 0) return new Float32Array(0);

  return conv1d(
    input,
    weight,
    bias,
    inLen,
    inChannels,
    outChannels,
    kernelSize,
    stride,
    0,
  );
}

export function convTranspose1d(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array | null,
  inLen: number,
  inChannels: number,
  outChannels: number,
  kernelSize: number,
  stride: number,
  padding: number,
): Float32Array {
  if (stride <= 0) return new Float32Array(0);

  if (input.length < inChannels * inLen) {
    throw new RangeError(
      `input length (${input.length}) must be >= inChannels*inLen (${inChannels * inLen})`,
    );
  }
  const expectedWeight = inChannels * outChannels * kernelSize;
  if (weight.length < expectedWeight) {
    throw new RangeError(
      `weight length (${weight.length}) must be >= inChannels*outChannels*kernelSize (${expectedWeight})`,
    );
  }
  if (bias && bias.length < outChannels) {
    throw new RangeError(
      `bias length (${bias.length}) must be >= outChannels (${outChannels})`,
    );
  }

  const fullLen = (inLen - 1) * stride + kernelSize;
  const outLen = fullLen - 2 * padding;
  if (outLen <= 0) return new Float32Array(0);

  const output = new Float32Array(outChannels * outLen);

  for (let oc = 0; oc < outChannels; oc++) {
    const biasValue = bias ? bias[oc] : 0.0;
    for (let p = 0; p < outLen; p++) {
      output[oc * outLen + p] = biasValue;
    }
  }

  for (let oc = 0; oc < outChannels; oc++) {
    for (let ic = 0; ic < inChannels; ic++) {
      for (let i = 0; i < inLen; i++) {
        const inVal = input[ic * inLen + i];
        const fullStart = i * stride;

        for (let k = 0; k < kernelSize; k++) {
          const fullPos = fullStart + k;
          const outPos = fullPos - padding;

          if (outPos >= 0 && outPos < outLen) {
            const wIdx = ic * outChannels * kernelSize + oc * kernelSize + k;
            output[oc * outLen + outPos] += weight[wIdx] * inVal;
          }
        }
      }
    }
  }

  return output;
}
