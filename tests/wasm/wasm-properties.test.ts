/**
 * wasm-properties.test.ts — COLA/compression property verification through WASM
 *
 * tests/unit/engine/cola.test.ts and compression-extended.test.ts verify
 * only the TypeScript reference implementation. This test verifies the
 * equivalent properties through the actual WASM module to guarantee the
 * correctness of the C implementation.
 *
 * Full verification chain:
 *   C native: test_stft.c (COLA), test_compression.c (compression)
 *   WASM: golden-vectors.test.ts (full pipeline matches PyTorch)
 *   WASM: wasm-differential.test.ts (C native ↔ WASM differences)
 *   WASM: this test (standalone COLA/compression properties)
 */
import { describe, it, expect } from 'vitest';
import {
  loadWasmModule,
  loadWeights,
  processAllFrames,
  type EmscriptenModule,
  type ModelSize,
  type Variant,
} from './wasm-helpers.js';

const MODELS: ModelSize[] = ['tiny', 'base', 'small'];
const VARIANTS: Variant[] = ['scalar', 'simd'];
const HOP_SIZE = 512;

async function initModel(model: ModelSize, variant: Variant): Promise<{
  mod: EmscriptenModule;
  state: number;
  hopSize: number;
  weightsPtr: number;
}> {
  const modelId = { tiny: 0, base: 1, small: 2 }[model];
  const mod = await loadWasmModule(model, variant);
  const weights = loadWeights(mod, model);
  const state = mod._fe_init(modelId, weights.ptr, weights.len);
  if (state === 0) throw new Error(`fe_init failed for ${model}/${variant}`);
  const hopSize = mod._fe_get_hop_size(state);
  return { mod, state, hopSize, weightsPtr: weights.ptr };
}

function cleanup(mod: EmscriptenModule, state: number, weightsPtr: number) {
  mod._fe_destroy(state);
  mod._free(weightsPtr);
}

function generateSilence(length: number): Float32Array {
  return new Float32Array(length);
}

function generateSine(length: number, freq: number, sr = 48000, amplitude = 0.5): Float32Array {
  const buf = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    buf[i] = amplitude * Math.sin((2 * Math.PI * freq * i) / sr);
  }
  return buf;
}

function generateImpulse(length: number, position: number): Float32Array {
  const buf = new Float32Array(length);
  if (position >= 0 && position < length) buf[position] = 1.0;
  return buf;
}

function generateClip(length: number): Float32Array {
  const buf = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    buf[i] = i % 2 === 0 ? 1.0 : -1.0;
  }
  return buf;
}

function generateSweep(length: number, startFreq: number, endFreq: number, sr = 48000): Float32Array {
  const buf = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const t = i / sr;
    const freq = startFreq + (endFreq - startFreq) * (i / length);
    buf[i] = 0.5 * Math.sin(2 * Math.PI * freq * t);
  }
  return buf;
}

function generateTransient(length: number, burstStart: number, burstLen: number): Float32Array {
  const buf = new Float32Array(length);
  for (let i = burstStart; i < Math.min(burstStart + burstLen, length); i++) {
    buf[i] = 0.9 * Math.sin(2 * Math.PI * 4000 * (i - burstStart) / 48000);
  }
  return buf;
}

function assertAllFinite(data: Float32Array, label: string) {
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      throw new Error(`${label}: non-finite value at index ${i}: ${data[i]}`);
    }
  }
}

function computeRms(data: Float32Array, start = 0, end = data.length): number {
  let sum = 0;
  for (let i = start; i < end; i++) sum += data[i] * data[i];
  return Math.sqrt(sum / (end - start));
}

for (const model of MODELS) {
  for (const variant of VARIANTS) {
    describe(`WASM COLA properties [${model}/${variant}]`, () => {
      it('has no frame-boundary discontinuity for silent input', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const nFrames = 10;
        const output = processAllFrames(mod, state, generateSilence(nFrames * hopSize), hopSize);

        assertAllFinite(output, 'silence output');

        for (let f = 2; f < nFrames - 1; f++) {
          const boundary = f * hopSize;
          const diff = Math.abs(output[boundary] - output[boundary - 1]);
          expect(diff).toBeLessThan(0.1);
        }

        cleanup(mod, state, weightsPtr);
      });

      it('has no frame-boundary amplitude modulation for sine-wave input', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const nFrames = 20;
        const output = processAllFrames(mod, state, generateSine(nFrames * hopSize, 1000), hopSize);

        const startFrame = 5;
        assertAllFinite(output, 'sine output');

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

    describe(`WASM compression properties [${model}/${variant}]`, () => {
      it('keeps output finite and low-amplitude for silent input', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const inPtr = mod._fe_get_input_ptr(state);
        const outPtr = mod._fe_get_output_ptr(state);
        const silence = generateSilence(hopSize);

        for (let f = 0; f < 5; f++) {
          mod.HEAPF32.set(silence, inPtr / 4);
          mod._fe_process(state, inPtr, outPtr);
          const out = mod.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);
          assertAllFinite(out, `silence frame ${f}`);
          expect(computeRms(out)).toBeLessThan(1.0);
        }

        cleanup(mod, state, weightsPtr);
      });

      it('produces no NaN/Inf for maximum-amplitude input', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const inPtr = mod._fe_get_input_ptr(state);
        const outPtr = mod._fe_get_output_ptr(state);
        const clip = generateClip(hopSize);

        for (let f = 0; f < 5; f++) {
          mod.HEAPF32.set(clip, inPtr / 4);
          mod._fe_process(state, inPtr, outPtr);
          const out = mod.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);
          assertAllFinite(out, `clip frame ${f}`);
        }

        cleanup(mod, state, weightsPtr);
      });

      it('does not diverge for DC-offset input', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const inPtr = mod._fe_get_input_ptr(state);
        const outPtr = mod._fe_get_output_ptr(state);
        const dc = new Float32Array(hopSize);
        dc.fill(0.5);

        for (let f = 0; f < 10; f++) {
          mod.HEAPF32.set(dc, inPtr / 4);
          mod._fe_process(state, inPtr, outPtr);
          const out = mod.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);
          assertAllFinite(out, `DC frame ${f}`);
          let maxAbs = 0;
          for (let i = 0; i < out.length; i++) {
            const abs = Math.abs(out[i]);
            if (abs > maxAbs) maxAbs = abs;
          }
          expect(maxAbs).toBeLessThan(10.0);
        }

        cleanup(mod, state, weightsPtr);
      });
    });

    describe(`WASM robustness [${model}/${variant}]`, () => {
      it('handles impulse input without NaN/Inf', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const nFrames = 6;
        const input = generateImpulse(nFrames * hopSize, hopSize);
        const output = processAllFrames(mod, state, input, hopSize);
        assertAllFinite(output, 'impulse output');
        cleanup(mod, state, weightsPtr);
      });

      it('handles frequency sweep without divergence', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const nFrames = 20;
        const input = generateSweep(nFrames * hopSize, 100, 20000);
        const output = processAllFrames(mod, state, input, hopSize);
        assertAllFinite(output, 'sweep output');
        cleanup(mod, state, weightsPtr);
      });

      it('handles transient burst without NaN/Inf', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const nFrames = 10;
        const total = nFrames * hopSize;
        const input = generateTransient(total, hopSize * 3, hopSize * 2);
        const output = processAllFrames(mod, state, input, hopSize);
        assertAllFinite(output, 'transient output');
        cleanup(mod, state, weightsPtr);
      });

      it('fe_reset produces consistent output on re-run', async () => {
        const { mod, state, hopSize, weightsPtr } = await initModel(model, variant);
        const nFrames = 8;
        const input = generateSine(nFrames * hopSize, 440);

        const out1 = processAllFrames(mod, state, input, hopSize);
        mod._fe_reset(state);
        const out2 = processAllFrames(mod, state, input, hopSize);

        for (let i = 0; i < out1.length; i++) {
          expect(Math.abs(out1[i] - out2[i])).toBeLessThan(1e-6);
        }

        cleanup(mod, state, weightsPtr);
      });
    });
  }
}
