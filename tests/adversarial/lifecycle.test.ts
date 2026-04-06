import { describe, it, expect, beforeAll } from 'vitest';
import { createDenoiser } from '../../src/api/index.js';
import type { WasmInstance, Model } from '../../src/api/index.js';
import { DestroyedError, ModelInitError } from '../../src/api/errors.js';
import { loadRealWasm, loadRealWeightData, createRealModel } from '../helpers/real-model.js';

let sharedWasm: WasmInstance;
let sharedWeightData: ArrayBuffer;

beforeAll(async () => {
  sharedWasm = await loadRealWasm('tiny', 'simd');
  sharedWeightData = loadRealWeightData('tiny');
});

function freshModel(): Model {
  return createRealModel(sharedWasm, 'tiny');
}

describe('Lifecycle robustness (real WASM)', () => {
  it('does not throw on double destroy', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();
    expect(() => d.destroy()).not.toThrow();
    expect(d.state).toBe('destroyed');
  });

  it('throws on every operation after destroy', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();
    expect(() => d.processFrame(new Float32Array(512))).toThrow(DestroyedError);
    expect(() => { d.bypass = true; }).toThrow(DestroyedError);
    expect(() => { d.agcEnabled = false; }).toThrow(DestroyedError);
    expect(() => { d.hpfEnabled = false; }).toThrow(DestroyedError);
  });

  it('survives rapid create/destroy cycles (10 times)', async () => {
    for (let i = 0; i < 10; i++) {
      const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
      expect(d.state).toBe('ready');
      d.destroy();
      expect(d.state).toBe('destroyed');
    }
  });

  it('survives create → process → destroy cycles (5 times)', async () => {
    for (let i = 0; i < 5; i++) {
      const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
      const input = new Float32Array(512).fill(0.1 * (i + 1));
      const output = d.processFrame(input);
      expect(output).toHaveLength(512);
      for (let j = 0; j < 512; j++) {
        expect(Number.isFinite(output[j])).toBe(true);
      }
      d.destroy();
      expect(d.state).toBe('destroyed');
    }
  });

  it('fires both destroy and statechange events', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    let destroyFired = false;
    const stateChanges: string[] = [];
    d.on('destroy', () => { destroyFired = true; });
    d.on('statechange', (s: string) => stateChanges.push(s));
    d.destroy();
    expect(destroyFired).toBe(true);
    expect(stateChanges).toContain('destroyed');
  });

  it('sets state=destroyed after cleanup via Symbol.dispose', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d[Symbol.dispose]();
    expect(d.state).toBe('destroyed');
    expect(() => d.processFrame(new Float32Array(512))).toThrow(DestroyedError);
  });

  it('throws ModelInitError for corrupted weightData', async () => {
    const corrupted = new ArrayBuffer(24);
    await expect(
      createDenoiser({ model: freshModel(), weightData: corrupted }),
    ).rejects.toThrow(ModelInitError);
  });

  it('throws ModelInitError for weightData with invalid CRC32', async () => {
    const valid = new Uint8Array(sharedWeightData);
    const tampered = new Uint8Array(valid.length);
    tampered.set(valid);
    tampered[valid.length - 1] ^= 0xff;
    await expect(
      createDenoiser({ model: freshModel(), weightData: tampered.buffer }),
    ).rejects.toThrow(ModelInitError);
  });

  it('throws ModelInitError for weightData that exceeds heap size', async () => {
    const huge = new ArrayBuffer(2 * 1024 * 1024);
    await expect(
      createDenoiser({ model: freshModel(), weightData: huge }),
    ).rejects.toThrow(ModelInitError);
  });
});

