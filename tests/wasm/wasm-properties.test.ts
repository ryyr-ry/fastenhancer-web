/**
 * wasm-properties.test.ts — WASM経由のCOLA/圧縮プロパティ検証
 *
 * tests/unit/engine/cola.test.ts と compression-extended.test.ts は
 * TypeScript参照実装のみを検証する。本テストは同等のプロパティを
 * 実際のWASMモジュール経由で検証し、C実装の正しさを保証する。
 *
 * 検証チェーン全体:
 *   C native: test_stft.c (COLA), test_compression.c (圧縮)
 *   WASM: golden-vectors.test.ts (全パイプライン PyTorch一致)
 *   WASM: wasm-differential.test.ts (C native ↔ WASM差分)
 *   WASM: 本テスト (COLA/圧縮の単体プロパティ)
 */
import { describe, it, expect } from 'vitest';
import {
  loadWasmModule,
  loadWeights,
  processAllFrames,
  type EmscriptenModule,
} from './wasm-helpers.js';

function initTinyScalar(): Promise<{
  mod: EmscriptenModule;
  state: number;
  hopSize: number;
  weightsPtr: number;
}> {
  return (async () => {
    const mod = await loadWasmModule('tiny', 'scalar');
    const weights = loadWeights(mod, 'tiny');
    const state = mod._fe_init(0, weights.ptr, weights.len);
    if (state === 0) throw new Error('fe_init failed');
    const hopSize = mod._fe_get_hop_size(state);
    return { mod, state, hopSize, weightsPtr: weights.ptr };
  })();
}

function cleanup(mod: EmscriptenModule, state: number, weightsPtr: number) {
  mod._fe_destroy(state);
  mod._free(weightsPtr);
}

describe('WASM COLA プロパティ', () => {
  it('無音入力でフレーム境界に不連続なし', async () => {
    const { mod, state, hopSize, weightsPtr } = await initTinyScalar();

    const nFrames = 10;
    const silence = new Float32Array(nFrames * hopSize);
    const output = processAllFrames(mod, state, silence, hopSize);

    for (let i = 0; i < output.length; i++) {
      expect(Number.isFinite(output[i])).toBe(true);
    }

    for (let f = 2; f < nFrames - 1; f++) {
      const boundary = f * hopSize;
      const diff = Math.abs(output[boundary] - output[boundary - 1]);
      expect(diff).toBeLessThan(0.1);
    }

    cleanup(mod, state, weightsPtr);
  });

  it('正弦波入力でフレーム境界の振幅変調なし', async () => {
    const { mod, state, hopSize, weightsPtr } = await initTinyScalar();

    const nFrames = 20;
    const freq = 1000;
    const sr = 48000;
    const input = new Float32Array(nFrames * hopSize);
    for (let i = 0; i < input.length; i++) {
      input[i] = 0.5 * Math.sin((2 * Math.PI * freq * i) / sr);
    }

    const output = processAllFrames(mod, state, input, hopSize);

    const startFrame = 5;
    for (let i = startFrame * hopSize; i < output.length; i++) {
      expect(Number.isFinite(output[i])).toBe(true);
    }

    let boundaryRmsSum = 0;
    let midframeRmsSum = 0;
    let boundaryCount = 0;
    let midframeCount = 0;

    for (let f = startFrame; f < nFrames - 1; f++) {
      const boundary = f * hopSize;
      for (let j = -2; j < 2; j++) {
        const idx = boundary + j;
        if (idx >= 0 && idx < output.length) {
          boundaryRmsSum += output[idx] * output[idx];
          boundaryCount++;
        }
      }
      const mid = boundary + Math.floor(hopSize / 2);
      for (let j = -2; j < 2; j++) {
        const idx = mid + j;
        if (idx >= 0 && idx < output.length) {
          midframeRmsSum += output[idx] * output[idx];
          midframeCount++;
        }
      }
    }

    const boundaryRms = Math.sqrt(boundaryRmsSum / boundaryCount);
    const midframeRms = Math.sqrt(midframeRmsSum / midframeCount);

    expect(midframeRms).toBeGreaterThan(0.0001);
    const ratio = boundaryRms / midframeRms;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);

    cleanup(mod, state, weightsPtr);
  });
});

describe('WASM 圧縮プロパティ', () => {
  it('無音入力で出力が有限かつ低振幅', async () => {
    const { mod, state, hopSize, weightsPtr } = await initTinyScalar();

    const silence = new Float32Array(hopSize);
    const inPtr = mod._fe_get_input_ptr(state);
    const outPtr = mod._fe_get_output_ptr(state);

    for (let f = 0; f < 5; f++) {
      mod.HEAPF32.set(silence, inPtr / 4);
      mod._fe_process(state, inPtr, outPtr);
      const out = mod.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);

      for (let i = 0; i < out.length; i++) {
        expect(Number.isFinite(out[i])).toBe(true);
      }

      let rms = 0;
      for (let i = 0; i < out.length; i++) rms += out[i] * out[i];
      rms = Math.sqrt(rms / out.length);
      expect(rms).toBeLessThan(1.0);
    }

    cleanup(mod, state, weightsPtr);
  });

  it('最大振幅入力で出力にNaN/Infなし', async () => {
    const { mod, state, hopSize, weightsPtr } = await initTinyScalar();

    const loud = new Float32Array(hopSize);
    for (let i = 0; i < hopSize; i++) {
      loud[i] = i % 2 === 0 ? 1.0 : -1.0;
    }

    const inPtr = mod._fe_get_input_ptr(state);
    const outPtr = mod._fe_get_output_ptr(state);

    for (let f = 0; f < 5; f++) {
      mod.HEAPF32.set(loud, inPtr / 4);
      mod._fe_process(state, inPtr, outPtr);
      const out = mod.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);

      for (let i = 0; i < out.length; i++) {
        expect(Number.isFinite(out[i])).toBe(true);
      }
    }

    cleanup(mod, state, weightsPtr);
  });

  it('DC offset入力で出力が発散しない', async () => {
    const { mod, state, hopSize, weightsPtr } = await initTinyScalar();

    const dc = new Float32Array(hopSize);
    dc.fill(0.5);

    const inPtr = mod._fe_get_input_ptr(state);
    const outPtr = mod._fe_get_output_ptr(state);

    for (let f = 0; f < 10; f++) {
      mod.HEAPF32.set(dc, inPtr / 4);
      mod._fe_process(state, inPtr, outPtr);
      const out = mod.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);

      let maxAbs = 0;
      for (let i = 0; i < out.length; i++) {
        expect(Number.isFinite(out[i])).toBe(true);
        const abs = Math.abs(out[i]);
        if (abs > maxAbs) maxAbs = abs;
      }
      expect(maxAbs).toBeLessThan(10.0);
    }

    cleanup(mod, state, weightsPtr);
  });
});
