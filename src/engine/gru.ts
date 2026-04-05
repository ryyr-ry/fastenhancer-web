/**
 * GRU cell (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine's gru.c.
 * Split-weight format (W_z/W_r/W_h, U_z/U_r/U_h, b_z/b_r/b_in_n/b_hn_n)
 */

export interface GruWeights {
  W_z: Float32Array; // [hiddenSize, inputSize] — update gate input weights
  U_z: Float32Array; // [hiddenSize, hiddenSize] — update gate hidden weights
  b_z: Float32Array; // [hiddenSize] — update gate bias
  W_r: Float32Array; // [hiddenSize, inputSize] — reset gate input weights
  U_r: Float32Array; // [hiddenSize, hiddenSize] — reset gate hidden weights
  b_r: Float32Array; // [hiddenSize] — reset gate bias
  W_h: Float32Array; // [hiddenSize, inputSize] — candidate hidden input weights
  U_h: Float32Array; // [hiddenSize, hiddenSize] — candidate hidden hidden weights
  b_in_n: Float32Array; // [hiddenSize] — candidate hidden input bias
  b_hn_n: Float32Array; // [hiddenSize] — candidate hidden hidden bias
}

/**
 * Execute one GRU step
 * z = sigmoid(W_z·x + U_z·h + b_z)
 * r = sigmoid(W_r·x + U_r·h + b_r)
 * n = tanh(W_h·x + b_in_n + r * (U_h·h + b_hn_n))
 * h_new = (1-z)*n + z*h
 */
export function gruStep(
  input: Float32Array,
  hidden: Float32Array,
  weights: GruWeights,
): Float32Array {
  const inputSize = input.length;
  const hiddenSize = hidden.length;

  if (weights.W_z.length !== hiddenSize * inputSize ||
      weights.W_r.length !== hiddenSize * inputSize ||
      weights.W_h.length !== hiddenSize * inputSize) {
    throw new RangeError(
      `Input weight matrices must have length hiddenSize*inputSize (${hiddenSize}*${inputSize}=${hiddenSize * inputSize})`,
    );
  }
  if (weights.U_z.length !== hiddenSize * hiddenSize ||
      weights.U_r.length !== hiddenSize * hiddenSize ||
      weights.U_h.length !== hiddenSize * hiddenSize) {
    throw new RangeError(
      `Hidden weight matrices must have length hiddenSize*hiddenSize (${hiddenSize}*${hiddenSize}=${hiddenSize * hiddenSize})`,
    );
  }
  if (weights.b_z.length !== hiddenSize ||
      weights.b_r.length !== hiddenSize ||
      weights.b_in_n.length !== hiddenSize ||
      weights.b_hn_n.length !== hiddenSize) {
    throw new RangeError(
      `Bias vectors must have length hiddenSize (${hiddenSize})`,
    );
  }

  const newHidden = new Float32Array(hiddenSize);

  for (let i = 0; i < hiddenSize; i++) {
    // Update gate z
    let zVal = weights.b_z[i];
    for (let j = 0; j < inputSize; j++) {
      zVal += weights.W_z[i * inputSize + j] * input[j];
    }
    for (let j = 0; j < hiddenSize; j++) {
      zVal += weights.U_z[i * hiddenSize + j] * hidden[j];
    }
    const z = sigmoid(zVal);

    // Reset gate r
    let rVal = weights.b_r[i];
    for (let j = 0; j < inputSize; j++) {
      rVal += weights.W_r[i * inputSize + j] * input[j];
    }
    for (let j = 0; j < hiddenSize; j++) {
      rVal += weights.U_r[i * hiddenSize + j] * hidden[j];
    }
    const r = sigmoid(rVal);

    // Candidate hidden state n
    let nVal = weights.b_in_n[i];
    for (let j = 0; j < inputSize; j++) {
      nVal += weights.W_h[i * inputSize + j] * input[j];
    }
    let uhVal = weights.b_hn_n[i];
    for (let j = 0; j < hiddenSize; j++) {
      uhVal += weights.U_h[i * hiddenSize + j] * hidden[j];
    }
    nVal += r * uhVal;
    const n = Math.tanh(nVal);

    newHidden[i] = (1 - z) * n + z * hidden[i];
  }

  return newHidden;
}

function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}
