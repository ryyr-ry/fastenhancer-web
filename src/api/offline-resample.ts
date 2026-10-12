import { ValidationError } from './errors.js';

/** A tile uses absolute file positions, including the actual file boundaries. */
export interface ResampleTile {
  channels: readonly Float32Array[];
  sourceRate: number;
  targetRate: number;
  sourceLength: number;
  inputBase: number;
  outputStart: number;
  outputEnd: number;
}

export function resampleRadius(sourceRate: number, targetRate: number): number {
  return Math.ceil(32 / (Math.min(1, 1 / (sourceRate / targetRate)) * 0.95));
}

export function resampleInputRange(sourceRate: number, targetRate: number, length: number, start: number, end: number): [number, number] {
  if (sourceRate === targetRate) return [start, Math.min(length, end)];
  const ratio = sourceRate / targetRate;
  const radius = resampleRadius(sourceRate, targetRate);
  return [Math.max(0, Math.ceil(start * ratio - radius)), Math.min(length, Math.floor((end - 1) * ratio + radius) + 1)];
}

/** Preserve the old filter and accumulation order; share coefficients across channels. */
export function resampleTile(tile: ResampleTile): Float32Array[] {
  const { channels, sourceRate, targetRate, sourceLength, inputBase, outputStart, outputEnd } = tile;
  if (channels.length !== 1 && channels.length !== 2) throw new ValidationError('Expected mono or stereo resampling input');
  if (sourceRate === targetRate) return channels.map(c => c.slice(outputStart - inputBase, outputEnd - inputBase));
  const ratio = sourceRate / targetRate;
  const cutoff = Math.min(1, 1 / ratio) * 0.95;
  const radius = Math.ceil(32 / cutoff);
  const outputs = channels.map(() => new Float32Array(outputEnd - outputStart));
  for (let i = outputStart; i < outputEnd; i++) {
    const position = i * ratio;
    const start = Math.max(0, Math.ceil(position - radius));
    const end = Math.min(sourceLength - 1, Math.floor(position + radius));
    let sum0 = 0, sum1 = 0, weightSum = 0;
    for (let j = start; j <= end; j++) {
      const distance = position - j;
      const x = Math.PI * cutoff * distance;
      const sinc = Math.abs(x) < 1e-12 ? 1 : Math.sin(x) / x;
      const window = 0.5 + 0.5 * Math.cos(Math.PI * distance / radius);
      const weight = cutoff * sinc * window;
      sum0 += channels[0][j - inputBase] * weight;
      if (channels.length === 2) sum1 += channels[1][j - inputBase] * weight;
      weightSum += weight;
    }
    outputs[0][i - outputStart] = weightSum === 0 ? 0 : sum0 / weightSum;
    if (channels.length === 2) outputs[1][i - outputStart] = weightSum === 0 ? 0 : sum1 / weightSum;
  }
  return outputs;
}
