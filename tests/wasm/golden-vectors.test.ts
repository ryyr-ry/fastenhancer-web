/**
 * golden-vectors.test.ts — WASM golden vector 検証テスト (4-C)
 *
 * PyTorch streaming推論出力とWASM出力の数値一致を検証する。
 * scalar / SIMD 両バリアントで全3モデルサイズをテスト。
 *
 * 判定基準: MSE < 1e-5 (plan.md Phase 4-C)
 */
import { describe, it, expect, afterAll } from 'vitest';
import {
  loadWasmModule,
  loadWeights,
  loadGoldenVectors,
  processAllFrames,
  computeMSE,
  computeMaxAbsDiff,
  type EmscriptenModule,
  type ModelSize,
  type Variant,
} from './wasm-helpers';

const MSE_THRESHOLD = 1e-5;
const MAX_DIFF_THRESHOLD = 0.01;
const HOP_SIZE = 512;

interface TestContext {
  module: EmscriptenModule;
  state: number;
  weightPtr: number;
}

async function setupModel(
  model: ModelSize,
  variant: Variant,
): Promise<TestContext> {
  const module = await loadWasmModule(model, variant);
  const { ptr, len } = loadWeights(module, model);
  const modelId = { tiny: 0, base: 1, small: 2 }[model];
  const state = module._fe_init(modelId, ptr, len);
  if (!state) throw new Error(`fe_init failed for ${model}/${variant}`);
  return { module, state, weightPtr: ptr };
}

function teardown(ctx: TestContext): void {
  ctx.module._fe_destroy(ctx.state);
  ctx.module._free(ctx.weightPtr);
}

describe('WASM golden vector 検証', () => {
  describe('Tiny scalar', () => {
    let ctx: TestContext;

    afterAll(() => { if (ctx) teardown(ctx); });

    it('MSE < 1e-5 (PyTorch出力との一致)', async () => {
      ctx = await setupModel('tiny', 'scalar');
      const { input, output: expected } = loadGoldenVectors('tiny');
      const actual = processAllFrames(ctx.module, ctx.state, input, HOP_SIZE);
      const mse = computeMSE(actual, expected);
      const maxDiff = computeMaxAbsDiff(actual, expected);

      expect(mse).toBeLessThan(MSE_THRESHOLD);
      expect(maxDiff).toBeLessThan(MAX_DIFF_THRESHOLD);
    });
  });

  describe('Tiny SIMD', () => {
    let ctx: TestContext;

    afterAll(() => { if (ctx) teardown(ctx); });

    it('MSE < 1e-5 (PyTorch出力との一致)', async () => {
      ctx = await setupModel('tiny', 'simd');
      const { input, output: expected } = loadGoldenVectors('tiny');
      const actual = processAllFrames(ctx.module, ctx.state, input, HOP_SIZE);
      const mse = computeMSE(actual, expected);
      const maxDiff = computeMaxAbsDiff(actual, expected);

      expect(mse).toBeLessThan(MSE_THRESHOLD);
      expect(maxDiff).toBeLessThan(MAX_DIFF_THRESHOLD);
    });
  });

  describe('Base scalar', () => {
    let ctx: TestContext;

    afterAll(() => { if (ctx) teardown(ctx); });

    it('MSE < 1e-5 (PyTorch出力との一致)', async () => {
      ctx = await setupModel('base', 'scalar');
      const { input, output: expected } = loadGoldenVectors('base');
      const actual = processAllFrames(ctx.module, ctx.state, input, HOP_SIZE);
      const mse = computeMSE(actual, expected);
      const maxDiff = computeMaxAbsDiff(actual, expected);

      expect(mse).toBeLessThan(MSE_THRESHOLD);
      expect(maxDiff).toBeLessThan(MAX_DIFF_THRESHOLD);
    });
  });

  describe('Base SIMD', () => {
    let ctx: TestContext;

    afterAll(() => { if (ctx) teardown(ctx); });

    it('MSE < 1e-5 (PyTorch出力との一致)', async () => {
      ctx = await setupModel('base', 'simd');
      const { input, output: expected } = loadGoldenVectors('base');
      const actual = processAllFrames(ctx.module, ctx.state, input, HOP_SIZE);
      const mse = computeMSE(actual, expected);
      const maxDiff = computeMaxAbsDiff(actual, expected);

      expect(mse).toBeLessThan(MSE_THRESHOLD);
      expect(maxDiff).toBeLessThan(MAX_DIFF_THRESHOLD);
    });
  });

  describe('Small scalar', () => {
    let ctx: TestContext;

    afterAll(() => { if (ctx) teardown(ctx); });

    it('MSE < 1e-5 (PyTorch出力との一致)', async () => {
      ctx = await setupModel('small', 'scalar');
      const { input, output: expected } = loadGoldenVectors('small');
      const actual = processAllFrames(ctx.module, ctx.state, input, HOP_SIZE);
      const mse = computeMSE(actual, expected);
      const maxDiff = computeMaxAbsDiff(actual, expected);

      expect(mse).toBeLessThan(MSE_THRESHOLD);
      expect(maxDiff).toBeLessThan(MAX_DIFF_THRESHOLD);
    });
  });

  describe('Small SIMD', () => {
    let ctx: TestContext;

    afterAll(() => { if (ctx) teardown(ctx); });

    it('MSE < 1e-5 (PyTorch出力との一致)', async () => {
      ctx = await setupModel('small', 'simd');
      const { input, output: expected } = loadGoldenVectors('small');
      const actual = processAllFrames(ctx.module, ctx.state, input, HOP_SIZE);
      const mse = computeMSE(actual, expected);
      const maxDiff = computeMaxAbsDiff(actual, expected);

      expect(mse).toBeLessThan(MSE_THRESHOLD);
      expect(maxDiff).toBeLessThan(MAX_DIFF_THRESHOLD);
    });
  });

  describe('Scalar vs SIMD 差分', () => {
    for (const model of ['tiny', 'base', 'small'] as ModelSize[]) {
      it(`${model}: 同一入力で scalar / SIMD の出力が一致する (MSE < 1e-10)`, async () => {
        const scalarCtx = await setupModel(model, 'scalar');
        const simdCtx = await setupModel(model, 'simd');

        const { input } = loadGoldenVectors(model);
        const scalarOut = processAllFrames(scalarCtx.module, scalarCtx.state, input, HOP_SIZE);
        const simdOut = processAllFrames(simdCtx.module, simdCtx.state, input, HOP_SIZE);

        const mse = computeMSE(scalarOut, simdOut);
        const maxDiff = computeMaxAbsDiff(scalarOut, simdOut);

        expect(mse).toBeLessThan(1e-10);
        expect(maxDiff).toBeLessThan(1e-5);

        teardown(scalarCtx);
        teardown(simdCtx);
      });
    }
  });
});
