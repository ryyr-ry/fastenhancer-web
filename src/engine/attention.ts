/**
 * Multi-head attention (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine's attention.c.
 */

export interface MhaWeights {
  W_qkv: Float32Array; // [dim, dim * 3] — fused Q/K/V projection
  b_qkv: Float32Array; // [dim * 3]
  W_out: Float32Array;  // [dim, dim] — output projection
  b_out: Float32Array;  // [dim]
}

/**
 * Multi-head attention
 * input: [seqLen * dim] (row-major: each row is one step of the sequence)
 * output: [seqLen * dim]
 */
export function multiHeadAttention(
  input: Float32Array,
  weights: MhaWeights,
  nHeads: number,
  seqLen: number,
  dim: number,
): Float32Array {
  const headDim = dim / nHeads;
  const dim3 = dim * 3;

  // QKV projection: [seqLen, dim] × [dim, 3*dim] + bias → [seqLen, 3*dim]
  const qkv = new Float32Array(seqLen * dim3);
  for (let s = 0; s < seqLen; s++) {
    for (let j = 0; j < dim3; j++) {
      let val = weights.b_qkv[j];
      for (let k = 0; k < dim; k++) {
        val += input[s * dim + k] * weights.W_qkv[k * dim3 + j];
      }
      qkv[s * dim3 + j] = val;
    }
  }

  // Split Q, K, and V (each [seqLen, dim])
  const q = new Float32Array(seqLen * dim);
  const k = new Float32Array(seqLen * dim);
  const v = new Float32Array(seqLen * dim);
  for (let s = 0; s < seqLen; s++) {
    for (let d = 0; d < dim; d++) {
      q[s * dim + d] = qkv[s * dim3 + d];
      k[s * dim + d] = qkv[s * dim3 + dim + d];
      v[s * dim + d] = qkv[s * dim3 + 2 * dim + d];
    }
  }

  const attnOut = new Float32Array(seqLen * dim);

  // Per-head attention
  for (let h = 0; h < nHeads; h++) {
    const offset = h * headDim;

    // Attention scores: [seqLen, seqLen]
    const scores = new Float32Array(seqLen * seqLen);
    const scale = 1.0 / Math.sqrt(headDim);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += q[i * dim + offset + d] * k[j * dim + offset + d];
        }
        scores[i * seqLen + j] = dot * scale;
      }
    }

    // Softmax (per row)
    for (let i = 0; i < seqLen; i++) {
      let maxVal = -Infinity;
      for (let j = 0; j < seqLen; j++) {
        if (scores[i * seqLen + j] > maxVal) maxVal = scores[i * seqLen + j];
      }

      let sumExp = 0;
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] = Math.exp(scores[i * seqLen + j] - maxVal);
        sumExp += scores[i * seqLen + j];
      }

      if (sumExp === 0) sumExp = 1e-12;
      for (let j = 0; j < seqLen; j++) {
        scores[i * seqLen + j] /= sumExp;
      }
    }

    // Attention output: scores × V
    for (let i = 0; i < seqLen; i++) {
      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let j = 0; j < seqLen; j++) {
          val += scores[i * seqLen + j] * v[j * dim + offset + d];
        }
        attnOut[i * dim + offset + d] = val;
      }
    }
  }

  // Output projection: [seqLen, dim] × [dim, dim] + bias → [seqLen, dim]
  const output = new Float32Array(seqLen * dim);
  for (let s = 0; s < seqLen; s++) {
    for (let j = 0; j < dim; j++) {
      let val = weights.b_out[j];
      for (let k2 = 0; k2 < dim; k2++) {
        val += attnOut[s * dim + k2] * weights.W_out[k2 * dim + j];
      }
      output[s * dim + j] = val;
    }
  }

  return output;
}
