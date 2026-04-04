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

export const MODEL_CONFIGS: Record<ModelSize, ModelConfig> = {
  tiny: {
    ...COMMON,
    channels: 24,
    hopSize: 512,
    rfBlocks: 2,
    kernelSize: [8, 3, 3],
    C2: 20,
    F2: 24,
    headDim: 5,
    encoderBlocks: 2,
  },
  base: {
    ...COMMON,
    channels: 48,
    hopSize: 512,
    rfBlocks: 3,
    kernelSize: [8, 3, 3],
    C2: 36,
    F2: 36,
    headDim: 9,
    encoderBlocks: 2,
  },
  small: {
    ...COMMON,
    channels: 64,
    hopSize: 512,
    rfBlocks: 3,
    kernelSize: [8, 3, 3, 3],
    C2: 48,
    F2: 48,
    headDim: 12,
    encoderBlocks: 3,
  },
};

export function getModelConfig(size: ModelSize): ModelConfig {
  const config = MODEL_CONFIGS[size];
  if (!config) {
    throw new Error(`Unknown model size: '${size}'`);
  }
  return config;
}
