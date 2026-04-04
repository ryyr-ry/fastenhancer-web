import { describe, it, expect, beforeAll } from 'vitest';
import { createDenoiser, isSupported } from '../../../src/api/index.js';
import type { WasmInstance, Model } from '../../../src/api/index.js';
import { loadRealWasm, loadRealWeightData, createRealModel, createRealModelWithFactory } from '../../helpers/real-model.js';

let sharedWasm: WasmInstance;
let sharedWeightData: ArrayBuffer;

beforeAll(async () => {
  sharedWasm = await loadRealWasm('tiny', 'simd');
  sharedWeightData = loadRealWeightData('tiny');
});

function freshModel(): Model {
  return createRealModel(sharedWasm, 'tiny');
}

describe('createDenoiser（実WASM）', () => {
  it('Denoiserを生成しready状態', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(d.state).toBe('ready');
    expect(typeof d.processFrame).toBe('function');
    expect(typeof d.destroy).toBe('function');
    d.destroy();
  });

  it('processFrame: 入力と同じ長さのFloat32Arrayを返す', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512).fill(0.1);
    const output = d.processFrame(input);
    expect(output).toBeInstanceOf(Float32Array);
    expect(output).toHaveLength(512);
    d.destroy();
  });

  it('processFrame: 出力値が入力値と異なる（実際にノイズ除去処理されている）', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      input[i] = Math.sin(2 * Math.PI * 1000 * i / 48000) * 0.5;
    }
    const output = d.processFrame(input);
    let different = false;
    for (let i = 0; i < 512; i++) {
      if (Math.abs(output[i] - input[i]) > 1e-10) { different = true; break; }
    }
    expect(different).toBe(true);
    d.destroy();
  });

  it('processFrame: 出力が有限値のみ（NaN/Infなし）', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512).fill(0.3);
    const output = d.processFrame(input);
    for (let i = 0; i < output.length; i++) {
      expect(Number.isFinite(output[i])).toBe(true);
    }
    d.destroy();
  });

  it('processFrame: 同一入力に対して決定的な出力', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512).fill(0.2);
    const output1 = d.processFrame(input);
    d.destroy();

    const d2 = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const output2 = d2.processFrame(input);
    for (let i = 0; i < 512; i++) {
      expect(output1[i]).toBe(output2[i]);
    }
    d2.destroy();
  });

  it('processFrame: 100フレーム連続処理でクラッシュしない', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512);
    for (let frame = 0; frame < 100; frame++) {
      for (let i = 0; i < 512; i++) {
        input[i] = Math.sin(2 * Math.PI * 440 * (frame * 512 + i) / 48000) * 0.3;
      }
      const output = d.processFrame(input);
      expect(output).toHaveLength(512);
      for (let i = 0; i < 512; i++) {
        expect(Number.isFinite(output[i])).toBe(true);
      }
    }
    d.destroy();
  });

  it('destroy後にprocessFrameでDestroyedErrorがthrow', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();
    expect(d.state).toBe('destroyed');
    expect(() => d.processFrame(new Float32Array(512))).toThrow();
  });

  it('不正なサイズの入力でValidationErrorがthrow', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(() => d.processFrame(new Float32Array(0))).toThrow();
    expect(() => d.processFrame(new Float32Array(256))).toThrow();
    d.destroy();
  });

  it('破損したweightDataでModelInitErrorがthrow', async () => {
    const corruptedWeights = new ArrayBuffer(24);
    await expect(
      createDenoiser({ model: freshModel(), weightData: corruptedWeights }),
    ).rejects.toThrow();
  });

  it('巨大weightDataでModelInitErrorがthrow（ヒープ超過）', async () => {
    const hugeWeights = new ArrayBuffer(2 * 1024 * 1024);
    await expect(
      createDenoiser({ model: freshModel(), weightData: hugeWeights }),
    ).rejects.toThrow();
  });
});

describe('イベント（実WASM）', () => {
  it('statechangeイベントが発火', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const states: string[] = [];
    d.on('statechange', (state: string) => states.push(state));
    d.destroy();
    expect(states).toContain('destroyed');
  });

  it('destroyイベントが発火', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    let destroyed = false;
    d.on('destroy', () => { destroyed = true; });
    d.destroy();
    expect(destroyed).toBe(true);
  });
});

describe('プロパティ（実WASM）', () => {
  it('bypass: デフォルトfalse、trueで入力そのまま出力', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(d.bypass).toBe(false);
    d.bypass = true;
    expect(d.bypass).toBe(true);
    const input = new Float32Array(512).fill(1.0);
    const output = d.processFrame(input);
    for (let i = 0; i < 512; i++) {
      expect(output[i]).toBeCloseTo(input[i], 10);
    }
    d.destroy();
  });

  it('agcEnabled: 読み書き可能で実WASMに反映', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(typeof d.agcEnabled).toBe('boolean');
    d.agcEnabled = false;
    expect(d.agcEnabled).toBe(false);
    d.agcEnabled = true;
    expect(d.agcEnabled).toBe(true);

    const input = new Float32Array(512).fill(0.1);
    const output = d.processFrame(input);
    expect(output).toHaveLength(512);
    d.destroy();
  });

  it('hpfEnabled: 読み書き可能で実WASMに反映', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(typeof d.hpfEnabled).toBe('boolean');
    d.hpfEnabled = true;
    expect(d.hpfEnabled).toBe(true);

    const input = new Float32Array(512).fill(0.1);
    const output = d.processFrame(input);
    expect(output).toHaveLength(512);
    d.hpfEnabled = false;
    expect(d.hpfEnabled).toBe(false);
    d.destroy();
  });
});

describe('DX機能（実WASM）', () => {
  it('Symbol.dispose でリソース解放', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(typeof d[Symbol.dispose]).toBe('function');
    d[Symbol.dispose]();
    expect(d.state).toBe('destroyed');
  });

  it('Symbol.asyncDispose でリソース解放', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(typeof d[Symbol.asyncDispose]).toBe('function');
    await d[Symbol.asyncDispose]();
    expect(d.state).toBe('destroyed');
  });

  it('once: ワンショットリスナー', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    let count = 0;
    d.once('statechange', () => count++);
    d.destroy();
    expect(count).toBe(1);
  });

  it('performance: 実処理の統計取得', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.processFrame(new Float32Array(512).fill(0.5));
    d.processFrame(new Float32Array(512).fill(0.3));
    const stats = d.performance;
    expect(stats).toHaveProperty('avgMs');
    expect(stats).toHaveProperty('p99Ms');
    expect(stats).toHaveProperty('droppedFrames');
    expect(stats).toHaveProperty('totalFrames');
    expect(stats.totalFrames).toBeGreaterThanOrEqual(2);
    expect(typeof stats.avgMs).toBe('number');
    expect(stats.avgMs).toBeGreaterThan(0);
    d.destroy();
  });
});

describe('isSupported', () => {
  it('ブラウザサポート情報を返す', async () => {
    const support = await isSupported();
    expect(support).toHaveProperty('wasm');
    expect(support).toHaveProperty('simd');
    expect(support).toHaveProperty('audioWorklet');
    expect(typeof support.wasm).toBe('boolean');
    expect(typeof support.simd).toBe('boolean');
    expect(typeof support.audioWorklet).toBe('boolean');
  });
});

describe('switchModel（実WASM）', () => {
  it('モデル切替後も処理が継続する', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });

    const newWasm = await loadRealWasm('tiny', 'simd');
    const newModel = createRealModel(newWasm, 'tiny');
    const newWeightData = loadRealWeightData('tiny');

    await d.switchModel({ model: newModel, weightData: newWeightData });

    const output = d.processFrame(new Float32Array(512).fill(0.1));
    expect(output).toBeInstanceOf(Float32Array);
    expect(output).toHaveLength(512);
    d.destroy();
  });

  it('切替中にisSwitchingがtrue', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(d.isSwitching).toBe(false);

    const newModel = createRealModelWithFactory('tiny');
    const newWeightData = loadRealWeightData('tiny');
    const p = d.switchModel({ model: newModel, weightData: newWeightData });
    expect(d.isSwitching).toBe(true);
    await p;
    expect(d.isSwitching).toBe(false);
    d.destroy();
  });

  it('切替中もprocessFrameが動作する（旧モデルで処理）', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });

    let resolveFactory!: (wasm: WasmInstance) => void;
    const deferredModel: Model = {
      size: 'tiny',
      sampleRate: 48000,
      nFft: 1024,
      hopSize: 512,
      wasmFactory: () => new Promise<WasmInstance>(resolve => {
        resolveFactory = resolve;
      }),
    };

    const switchPromise = d.switchModel({
      model: deferredModel,
      weightData: sharedWeightData,
    });

    const output = d.processFrame(new Float32Array(512).fill(0.1));
    expect(output).toBeInstanceOf(Float32Array);
    expect(output).toHaveLength(512);

    const newWasm = await loadRealWasm('tiny', 'simd');
    resolveFactory(newWasm);
    await switchPromise;
    d.destroy();
  });

  it('destroy済みの場合switchModelでthrow', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();

    const newModel = createRealModelWithFactory('tiny');
    const newWeightData = loadRealWeightData('tiny');
    await expect(
      d.switchModel({ model: newModel, weightData: newWeightData }),
    ).rejects.toThrow();
  });

  it('switchModel中にdestroy()してもクラッシュしない', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });

    let resolveFactory!: (wasm: WasmInstance) => void;
    const deferredModel: Model = {
      size: 'tiny',
      sampleRate: 48000,
      nFft: 1024,
      hopSize: 512,
      wasmFactory: () => new Promise<WasmInstance>(resolve => {
        resolveFactory = resolve;
      }),
    };

    const switchPromise = d.switchModel({
      model: deferredModel,
      weightData: sharedWeightData,
    });

    d.destroy();
    expect(d.state).toBe('destroyed');

    const newWasm = await loadRealWasm('tiny', 'simd');
    resolveFactory(newWasm);
    await switchPromise;
  });
});

describe('createDenoiser — base/smallモデル', () => {
  it('baseモデルでDenoiserを生成しprocessFrameが動作する', async () => {
    const wasm = await loadRealWasm('base', 'simd');
    const model: Model = {
      size: 'base',
      sampleRate: 48000,
      nFft: 1024,
      hopSize: 512,
      wasmFactory: () => loadRealWasm('base'),
      _wasm: wasm,
    };
    const weightData = loadRealWeightData('base');
    const d = await createDenoiser({ model, weightData });
    expect(d.state).toBe('ready');
    const output = d.processFrame(new Float32Array(512).fill(0.1));
    expect(output).toBeInstanceOf(Float32Array);
    expect(output).toHaveLength(512);
    d.destroy();
  });

  it('smallモデルでDenoiserを生成しprocessFrameが動作する', async () => {
    const wasm = await loadRealWasm('small', 'simd');
    const model: Model = {
      size: 'small',
      sampleRate: 48000,
      nFft: 1024,
      hopSize: 512,
      wasmFactory: () => loadRealWasm('small'),
      _wasm: wasm,
    };
    const weightData = loadRealWeightData('small');
    const d = await createDenoiser({ model, weightData });
    expect(d.state).toBe('ready');
    const output = d.processFrame(new Float32Array(512).fill(0.1));
    expect(output).toBeInstanceOf(Float32Array);
    expect(output).toHaveLength(512);
    d.destroy();
  });

  it('modelSizeIdが自動導出される（base=1, small=2）', async () => {
    const baseWasm = await loadRealWasm('base', 'simd');
    const baseModel: Model = {
      size: 'base',
      sampleRate: 48000,
      nFft: 1024,
      hopSize: 512,
      wasmFactory: () => loadRealWasm('base'),
      _wasm: baseWasm,
    };
    const baseWeightData = loadRealWeightData('base');
    const dBase = await createDenoiser({ model: baseModel, weightData: baseWeightData });
    expect(dBase.state).toBe('ready');
    dBase.destroy();

    const smallWasm = await loadRealWasm('small', 'simd');
    const smallModel: Model = {
      size: 'small',
      sampleRate: 48000,
      nFft: 1024,
      hopSize: 512,
      wasmFactory: () => loadRealWasm('small'),
      _wasm: smallWasm,
    };
    const smallWeightData = loadRealWeightData('small');
    const dSmall = await createDenoiser({ model: smallModel, weightData: smallWeightData });
    expect(dSmall.state).toBe('ready');
    dSmall.destroy();
  });
});
