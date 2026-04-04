/**
 * wasm-benchmark.test.ts — WASMベンチマーク (4-D)
 *
 * scalar / SIMD の処理時間を計測する。
 *
 * 時間予算: hop=512 / 48kHz = 10.667ms
 *
 * 重要: P99 < 10.67ms はブラウザAudioWorklet内の実時間制約。
 * Node.js V8環境ではGCスパイクによりP99が数十〜数百msに跳ねるため、
 * medianで予算内であることを検証する。
 * P99の実環境検証はPhase 8のPlaywright E2Eテストで行う。
 */
import { describe, it, expect } from 'vitest';
import {
  loadWasmModule,
  loadWeights,
  type EmscriptenModule,
  type ModelSize,
  type Variant,
} from './wasm-helpers';

const HOP_SIZE = 512;
const SAMPLE_RATE = 48000;
const TIME_BUDGET_MS = (HOP_SIZE / SAMPLE_RATE) * 1000; // 10.667ms
const WARMUP_FRAMES = 100;
const MEASURE_FRAMES = 1000;
const MODEL_IDS: Record<ModelSize, number> = { tiny: 0, base: 1, small: 2 };

interface BenchResult {
  medianMs: number;
  p99Ms: number;
  meanMs: number;
  minMs: number;
  maxMs: number;
}

async function benchmarkVariant(model: ModelSize, variant: Variant): Promise<BenchResult> {
  const module = await loadWasmModule(model, variant);
  const { ptr, len } = loadWeights(module, model);
  const state = module._fe_init(MODEL_IDS[model], ptr, len);
  if (!state) throw new Error(`fe_init failed for ${model}/${variant}`);

  const inPtr = module._fe_get_input_ptr(state);
  const outPtr = module._fe_get_output_ptr(state);

  const testInput = new Float32Array(HOP_SIZE);
  // 固定シードPRNGで再現性のあるテスト入力を生成
  let prngState = 42;
  for (let i = 0; i < HOP_SIZE; i++) {
    prngState = (prngState * 1103515245 + 12345) & 0x7fffffff;
    testInput[i] = ((prngState >> 16) / 32768.0 - 0.5) * 0.2;
  }

  for (let i = 0; i < WARMUP_FRAMES; i++) {
    module.HEAPF32.set(testInput, inPtr / 4);
    module._fe_process(state, inPtr, outPtr);
  }

  const times: number[] = [];
  for (let i = 0; i < MEASURE_FRAMES; i++) {
    module.HEAPF32.set(testInput, inPtr / 4);
    const start = performance.now();
    module._fe_process(state, inPtr, outPtr);
    const elapsed = performance.now() - start;
    times.push(elapsed);
  }

  module._fe_destroy(state);
  module._free(ptr);

  times.sort((a, b) => a - b);
  const medianMs = times[Math.floor(times.length / 2)];
  const p99Ms = times[Math.floor(times.length * 0.99)];
  const meanMs = times.reduce((s, t) => s + t, 0) / times.length;

  return {
    medianMs,
    p99Ms,
    meanMs,
    minMs: times[0],
    maxMs: times[times.length - 1],
  };
}

describe('WASMベンチマーク', () => {
  for (const model of ['tiny', 'base', 'small'] as ModelSize[]) {
    it(`${model} scalar: medianが時間予算(${TIME_BUDGET_MS.toFixed(2)}ms)内`, async () => {
      const result = await benchmarkVariant(model, 'scalar');
      expect(result.medianMs).toBeLessThan(TIME_BUDGET_MS);
    }, 60000);

    it(`${model} SIMD: medianが時間予算(${TIME_BUDGET_MS.toFixed(2)}ms)内`, async () => {
      const result = await benchmarkVariant(model, 'simd');
      expect(result.medianMs).toBeLessThan(TIME_BUDGET_MS);
    }, 60000);
  }
});
