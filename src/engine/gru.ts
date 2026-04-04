/**
 * GRUセル (TypeScript参照実装)
 * Cエンジンの gru.c と同等の純TypeScript実装。
 * 分離重み方式 (W_z/W_r/W_h, U_z/U_r/U_h, b_z/b_r/b_h)
 */

export interface GruWeights {
  W_z: Float32Array; // [hiddenSize, inputSize] — 更新ゲート入力重み
  U_z: Float32Array; // [hiddenSize, hiddenSize] — 更新ゲート隠れ重み
  b_z: Float32Array; // [hiddenSize] — 更新ゲートバイアス
  W_r: Float32Array; // [hiddenSize, inputSize] — リセットゲート入力重み
  U_r: Float32Array; // [hiddenSize, hiddenSize] — リセットゲート隠れ重み
  b_r: Float32Array; // [hiddenSize] — リセットゲートバイアス
  W_h: Float32Array; // [hiddenSize, inputSize] — 候補隠れ入力重み
  U_h: Float32Array; // [hiddenSize, hiddenSize] — 候補隠れ隠れ重み
  b_h: Float32Array; // [hiddenSize] — 候補隠れバイアス
}

/**
 * GRUの1ステップを実行
 * z = sigmoid(W_z·x + U_z·h + b_z)
 * r = sigmoid(W_r·x + U_r·h + b_r)
 * n = tanh(W_h·x + r * (U_h·h) + b_h)
 * h_new = (1-z)*n + z*h
 */
export function gruStep(
  input: Float32Array,
  hidden: Float32Array,
  weights: GruWeights,
): Float32Array {
  const inputSize = input.length;
  const hiddenSize = hidden.length;
  const newHidden = new Float32Array(hiddenSize);

  for (let i = 0; i < hiddenSize; i++) {
    // 更新ゲート z
    let zVal = weights.b_z[i];
    for (let j = 0; j < inputSize; j++) {
      zVal += weights.W_z[i * inputSize + j] * input[j];
    }
    for (let j = 0; j < hiddenSize; j++) {
      zVal += weights.U_z[i * hiddenSize + j] * hidden[j];
    }
    const z = sigmoid(zVal);

    // リセットゲート r
    let rVal = weights.b_r[i];
    for (let j = 0; j < inputSize; j++) {
      rVal += weights.W_r[i * inputSize + j] * input[j];
    }
    for (let j = 0; j < hiddenSize; j++) {
      rVal += weights.U_r[i * hiddenSize + j] * hidden[j];
    }
    const r = sigmoid(rVal);

    // 候補隠れ状態 n
    let nVal = weights.b_h[i];
    for (let j = 0; j < inputSize; j++) {
      nVal += weights.W_h[i * inputSize + j] * input[j];
    }
    let uhVal = 0;
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
