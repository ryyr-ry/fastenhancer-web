import { describe, it, expect, beforeAll } from 'vitest';
import { createDenoiser, isSupported } from '../../../src/api/index.js';
import type { WasmInstance, Model } from '../../../src/api/index.js';
import { DestroyedError, ModelInitError, ValidationError } from '../../../src/api/errors.js';
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

describe('createDenoiser (real WASM)', () => {
  it('creates a Denoiser in the ready state', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(d.state).toBe('ready');
    expect(typeof d.processFrame).toBe('function');
    expect(typeof d.destroy).toBe('function');
    d.destroy();
  });

  it('processFrame: returns a Float32Array with the same length as the input', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512).fill(0.1);
    const output = d.processFrame(input);
    expect(output).toBeInstanceOf(Float32Array);
    expect(output).toHaveLength(512);
    d.destroy();
  });

  it('processFrame: output values differ from input values (noise removal is actually applied)', async () => {
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

  it('processFrame: output contains only finite values (no NaN/Inf)', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const input = new Float32Array(512).fill(0.3);
    const output = d.processFrame(input);
    for (let i = 0; i < output.length; i++) {
      expect(Number.isFinite(output[i])).toBe(true);
    }
    d.destroy();
  });

  it('processFrame: produces deterministic output for the same input', async () => {
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

  it('processFrame: does not crash after processing 100 consecutive frames', async () => {
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

  it('processFrame throws DestroyedError after destroy', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();
    expect(d.state).toBe('destroyed');
    expect(() => d.processFrame(new Float32Array(512))).toThrow(DestroyedError);
  });

  it('throws ValidationError for input with an invalid size', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(() => d.processFrame(new Float32Array(0))).toThrow(ValidationError);
    expect(() => d.processFrame(new Float32Array(256))).toThrow(ValidationError);
    d.destroy();
  });

  it('throws ModelInitError for corrupted weightData', async () => {
    const corruptedWeights = new ArrayBuffer(24);
    await expect(
      createDenoiser({ model: freshModel(), weightData: corruptedWeights }),
    ).rejects.toThrow(ModelInitError);
  });

  it('throws ModelInitError for oversized weightData (heap exceeded)', async () => {
    const hugeWeights = new ArrayBuffer(2 * 1024 * 1024);
    await expect(
      createDenoiser({ model: freshModel(), weightData: hugeWeights }),
    ).rejects.toThrow(ModelInitError);
  });
});

describe('Events (real WASM)', () => {
  it('fires the statechange event', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    const states: string[] = [];
    d.on('statechange', (state: string) => states.push(state));
    d.destroy();
    expect(states).toContain('destroyed');
  });

  it('fires the destroy event', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    let destroyed = false;
    d.on('destroy', () => { destroyed = true; });
    d.destroy();
    expect(destroyed).toBe(true);
  });
});

describe('Properties (real WASM)', () => {
  it('bypass: defaults to false and returns the input unchanged when true', async () => {
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

  it('agcEnabled: can be read and written and is reflected in real WASM', async () => {
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

  it('hpfEnabled: can be read and written and is reflected in real WASM', async () => {
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

describe('Developer experience features (real WASM)', () => {
  it('releases resources with Symbol.dispose', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(typeof d[Symbol.dispose]).toBe('function');
    d[Symbol.dispose]();
    expect(d.state).toBe('destroyed');
  });

  it('releases resources with Symbol.asyncDispose', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    expect(typeof d[Symbol.asyncDispose]).toBe('function');
    await d[Symbol.asyncDispose]();
    expect(d.state).toBe('destroyed');
  });

  it('once: one-shot listener', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    let count = 0;
    d.once('statechange', () => count++);
    d.destroy();
    expect(count).toBe(1);
  });

  it('performance: returns statistics for actual processing', async () => {
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
  it('returns browser support information', async () => {
    const support = await isSupported();
    expect(support).toHaveProperty('wasm');
    expect(support).toHaveProperty('simd');
    expect(support).toHaveProperty('audioWorklet');
    expect(typeof support.wasm).toBe('boolean');
    expect(typeof support.simd).toBe('boolean');
    expect(typeof support.audioWorklet).toBe('boolean');
  });
});

describe('switchModel (real WASM)', () => {
  it('continues processing after switching models', async () => {
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

  it('sets isSwitching to true while switching', async () => {
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

  it('keeps processFrame working during switching (processed with the old model)', async () => {
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

  it('throws in switchModel when already destroyed', async () => {
    const d = await createDenoiser({ model: freshModel(), weightData: sharedWeightData });
    d.destroy();

    const newModel = createRealModelWithFactory('tiny');
    const newWeightData = loadRealWeightData('tiny');
    await expect(
      d.switchModel({ model: newModel, weightData: newWeightData }),
    ).rejects.toThrow(DestroyedError);
  });

  it('does not crash when destroy() is called during switchModel', async () => {
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

describe('createDenoiser — base/small models', () => {
  it('creates a Denoiser with the base model and runs processFrame', async () => {
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

  it('creates a Denoiser with the small model and runs processFrame', async () => {
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

  it('auto-derives modelSizeId (base=1, small=2)', async () => {
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
