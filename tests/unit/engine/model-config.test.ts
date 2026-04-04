import { describe, it, expect } from 'vitest';
import { MODEL_CONFIGS, getModelConfig } from '../../../src/engine/model-config.js';

const SIZES = ['tiny', 'base', 'small'] as const;

describe('MODEL_CONFIGS', () => {
  it('3つのモデルサイズが定義されている', () => {
    const keys = Object.keys(MODEL_CONFIGS);
    expect(keys).toHaveLength(3);
    for (const size of SIZES) {
      expect(MODEL_CONFIGS).toHaveProperty(size);
    }
  });

  it('全モデル共通: nFft=1024, sampleRate=48000, nHeads=4, freq=128, stride=4', () => {
    for (const size of SIZES) {
      const c = MODEL_CONFIGS[size];
      expect(c.nFft).toBe(1024);
      expect(c.sampleRate).toBe(48000);
      expect(c.nHeads).toBe(4);
      expect(c.freq).toBe(128);
      expect(c.stride).toBe(4);
    }
  });

  it('Tiny: channels=24, hopSize=512, rfBlocks=2, kernelSize=[8,3,3], C2=20, F2=24, headDim=5, encoderBlocks=2', () => {
    const c = MODEL_CONFIGS.tiny;
    expect(c.channels).toBe(24);
    expect(c.hopSize).toBe(512);
    expect(c.rfBlocks).toBe(2);
    expect(c.kernelSize).toEqual([8, 3, 3]);
    expect(c.C2).toBe(20);
    expect(c.F2).toBe(24);
    expect(c.headDim).toBe(5);
    expect(c.encoderBlocks).toBe(2);
  });

  it('Base: channels=48, hopSize=512, rfBlocks=3, kernelSize=[8,3,3], C2=36, F2=36, headDim=9, encoderBlocks=2', () => {
    const c = MODEL_CONFIGS.base;
    expect(c.channels).toBe(48);
    expect(c.hopSize).toBe(512);
    expect(c.rfBlocks).toBe(3);
    expect(c.kernelSize).toEqual([8, 3, 3]);
    expect(c.C2).toBe(36);
    expect(c.F2).toBe(36);
    expect(c.headDim).toBe(9);
    expect(c.encoderBlocks).toBe(2);
  });

  it('Small: channels=64, hopSize=512, rfBlocks=3, kernelSize=[8,3,3,3], C2=48, F2=48, headDim=12, encoderBlocks=3', () => {
    const c = MODEL_CONFIGS.small;
    expect(c.channels).toBe(64);
    expect(c.hopSize).toBe(512);
    expect(c.rfBlocks).toBe(3);
    expect(c.kernelSize).toEqual([8, 3, 3, 3]);
    expect(c.C2).toBe(48);
    expect(c.F2).toBe(48);
    expect(c.headDim).toBe(12);
    expect(c.encoderBlocks).toBe(3);
  });
});

describe('getModelConfig', () => {
  it('有効なサイズで対応する設定を返す', () => {
    for (const size of SIZES) {
      const config = getModelConfig(size);
      expect(config).toEqual(MODEL_CONFIGS[size]);
    }
  });

  it('無効なサイズでthrow', () => {
    expect(() => getModelConfig('huge' as any)).toThrow();
    expect(() => getModelConfig('' as any)).toThrow();
  });
});
