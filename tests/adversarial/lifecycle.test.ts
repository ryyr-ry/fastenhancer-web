import { describe, it, expect, beforeAll } from 'vitest';
import { createDenoiser } from '../../src/api/index.js';
import type { WasmInstance, Model } from '../../src/api/index.js';
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

describe('ライフサイクル堅牢性（実WASM）', () => {
  it('二重destroyでthrowしない', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();
    expect(() => d.destroy()).not.toThrow();
    expect(d.state).toBe('destroyed');
  });

  it('destroy後の全操作でthrow', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();
    expect(() => d.processFrame(new Float32Array(512))).toThrow();
    expect(() => { d.bypass = true; }).toThrow();
    expect(() => { d.agcEnabled = false; }).toThrow();
    expect(() => { d.hpfEnabled = false; }).toThrow();
  });

  it('高速生成・破棄サイクル(10回)', async () => {
    for (let i = 0; i < 10; i++) {
      const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
      expect(d.state).toBe('ready');
      d.destroy();
      expect(d.state).toBe('destroyed');
    }
  });

  it('生成→処理→破棄サイクル(5回)', async () => {
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

  it('destroyイベントとstatechangeイベントが両方発火', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    let destroyFired = false;
    const stateChanges: string[] = [];
    d.on('destroy', () => { destroyFired = true; });
    d.on('statechange', (s: string) => stateChanges.push(s));
    d.destroy();
    expect(destroyFired).toBe(true);
    expect(stateChanges).toContain('destroyed');
  });

  it('Symbol.disposeによるリソース解放後にstate=destroyed', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d[Symbol.dispose]();
    expect(d.state).toBe('destroyed');
    expect(() => d.processFrame(new Float32Array(512))).toThrow();
  });

  it('破損weightDataでModelInitErrorがthrow', async () => {
    const corrupted = new ArrayBuffer(24);
    await expect(
      createDenoiser({ model: freshModel(), weightData: corrupted }),
    ).rejects.toThrow();
  });

  it('不正なCRC32のweightDataでModelInitErrorがthrow', async () => {
    const valid = new Uint8Array(sharedWeightData);
    const tampered = new Uint8Array(valid.length);
    tampered.set(valid);
    tampered[valid.length - 1] ^= 0xff;
    await expect(
      createDenoiser({ model: freshModel(), weightData: tampered.buffer }),
    ).rejects.toThrow();
  });

  it('ヒープ超過サイズのweightDataでModelInitErrorがthrow', async () => {
    const huge = new ArrayBuffer(2 * 1024 * 1024);
    await expect(
      createDenoiser({ model: freshModel(), weightData: huge }),
    ).rejects.toThrow();
  });
});

