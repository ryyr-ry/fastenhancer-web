/**
 * wasm-differential.test.ts — 4系統差分テスト (4-E)
 *
 * C native / WASM scalar / WASM SIMD の出力を同一入力で比較し、
 * 全系統が一致することを検証する。
 *
 * C native出力はgolden vectorファイルが担う（PyTorch→C native→golden vector）。
 * WASM scalar ↔ WASM SIMD の差分は直接比較する。
 */
import { describe, it, expect } from 'vitest';
import {
  loadWasmModule,
  loadWeights,
  loadGoldenVectors,
  processAllFrames,
  computeMSE,
  computeMaxAbsDiff,
  type ModelSize,
} from './wasm-helpers';

const HOP_SIZE = 512;
const MODEL_IDS: Record<ModelSize, number> = { tiny: 0, base: 1, small: 2 };

describe('4系統差分テスト', () => {
  for (const model of ['tiny', 'base', 'small'] as ModelSize[]) {
    describe(model, () => {
      it('C native (golden) ↔ WASM scalar: MSE < 1e-10', async () => {
        const module = await loadWasmModule(model, 'scalar');
        const { ptr, len } = loadWeights(module, model);
        const state = module._fe_init(MODEL_IDS[model], ptr, len);

        const { input, output: nativeOutput } = loadGoldenVectors(model);
        const wasmOutput = processAllFrames(module, state, input, HOP_SIZE);

        const mse = computeMSE(wasmOutput, nativeOutput);
        const maxDiff = computeMaxAbsDiff(wasmOutput, nativeOutput);

        expect(mse).toBeLessThan(1e-10);
        expect(maxDiff).toBeLessThan(1e-5);

        module._fe_destroy(state);
        module._free(ptr);
      });

      it('C native (golden) ↔ WASM SIMD: MSE < 1e-10', async () => {
        const module = await loadWasmModule(model, 'simd');
        const { ptr, len } = loadWeights(module, model);
        const state = module._fe_init(MODEL_IDS[model], ptr, len);

        const { input, output: nativeOutput } = loadGoldenVectors(model);
        const wasmOutput = processAllFrames(module, state, input, HOP_SIZE);

        const mse = computeMSE(wasmOutput, nativeOutput);
        const maxDiff = computeMaxAbsDiff(wasmOutput, nativeOutput);

        expect(mse).toBeLessThan(1e-10);
        expect(maxDiff).toBeLessThan(1e-5);

        module._fe_destroy(state);
        module._free(ptr);
      });

      it('WASM scalar ↔ WASM SIMD: MSE < 1e-10', async () => {
        const scalarMod = await loadWasmModule(model, 'scalar');
        const simdMod = await loadWasmModule(model, 'simd');

        const scalarW = loadWeights(scalarMod, model);
        const simdW = loadWeights(simdMod, model);

        const scalarState = scalarMod._fe_init(MODEL_IDS[model], scalarW.ptr, scalarW.len);
        const simdState = simdMod._fe_init(MODEL_IDS[model], simdW.ptr, simdW.len);

        const { input } = loadGoldenVectors(model);
        const scalarOut = processAllFrames(scalarMod, scalarState, input, HOP_SIZE);
        const simdOut = processAllFrames(simdMod, simdState, input, HOP_SIZE);

        const mse = computeMSE(scalarOut, simdOut);
        const maxDiff = computeMaxAbsDiff(scalarOut, simdOut);

        expect(mse).toBeLessThan(1e-10);
        expect(maxDiff).toBeLessThan(1e-5);

        scalarMod._fe_destroy(scalarState);
        scalarMod._free(scalarW.ptr);
        simdMod._fe_destroy(simdState);
        simdMod._free(simdW.ptr);
      });
    });
  }

  it('全系統の出力サンプルが有限値 (全モデル)', async () => {
    for (const model of ['tiny', 'base', 'small'] as ModelSize[]) {
      const scalarMod = await loadWasmModule(model, 'scalar');
      const simdMod = await loadWasmModule(model, 'simd');

      const sw = loadWeights(scalarMod, model);
      const simW = loadWeights(simdMod, model);

      const ss = scalarMod._fe_init(MODEL_IDS[model], sw.ptr, sw.len);
      const sims = simdMod._fe_init(MODEL_IDS[model], simW.ptr, simW.len);

      const { input, output: nativeOut } = loadGoldenVectors(model);
      const scalarOut = processAllFrames(scalarMod, ss, input, HOP_SIZE);
      const simdOut = processAllFrames(simdMod, sims, input, HOP_SIZE);

      for (let i = 0; i < nativeOut.length; i++) {
        expect(Number.isFinite(nativeOut[i])).toBe(true);
        expect(Number.isFinite(scalarOut[i])).toBe(true);
        expect(Number.isFinite(simdOut[i])).toBe(true);
      }

      scalarMod._fe_destroy(ss);
      scalarMod._free(sw.ptr);
      simdMod._fe_destroy(sims);
      simdMod._free(simW.ptr);
    }
  });
});
