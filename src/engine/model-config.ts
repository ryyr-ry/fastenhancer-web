interface ModelConfig {
  nFft: number;
  sampleRate: number;
  nHeads: number;
  freq: number;
  stride: number;
  channels: number;
  hopSize: number;
  rfBlocks: number;
  kernelSize: number[];
  C2: number;
  F2: number;
  headDim: number;
  encoderBlocks: number;
}

type ModelSize = 'tiny' | 'base' | 'small';

const COMMON = {
  nFft: 1024,
  sampleRate: 48000,
  nHeads: 4,
  freq: 128,
  stride: 4,
} as const;

export const MODEL_CONFIGS: Readonly<Record<ModelSize, Readonly<ModelConfig>>> = Object.freeze({
  tiny: Object.freeze({
    ...COMMON,
    channels: 24,
    hopSize: 512,
    rfBlocks: 2,
    kernelSize: Object.freeze([8, 3, 3]) as readonly number[] as number[],
    C2: 20,
    F2: 24,
    headDim: 5,
    encoderBlocks: 2,
  }),
  base: Object.freeze({
    ...COMMON,
    channels: 48,
    hopSize: 512,
    rfBlocks: 3,
    kernelSize: Object.freeze([8, 3, 3]) as readonly number[] as number[],
    C2: 36,
    F2: 36,
    headDim: 9,
    encoderBlocks: 2,
  }),
  small: Object.freeze({
    ...COMMON,
    channels: 64,
    hopSize: 512,
    rfBlocks: 3,
    kernelSize: Object.freeze([8, 3, 3, 3]) as readonly number[] as number[],
    C2: 48,
    F2: 48,
    headDim: 12,
    encoderBlocks: 3,
  }),
});

export function getModelConfig(size: ModelSize): Readonly<ModelConfig> {
  const config = MODEL_CONFIGS[size];
  if (!config) {
    throw new Error(`Unknown model size: '${size}'`);
  }
  return config;
}
