/**
 * wasm-helpers.ts — helpers for WASM tests
 *
 * Utilities that let test code call Emscripten module loading,
 * weight loading, and frame processing concisely.
 */
import fs from 'fs';
import path from 'path';

/** Type for an Emscripten module instance */
export interface EmscriptenModule {
  _malloc(size: number): number;
  _free(ptr: number): void;
  _fe_weight_count(modelSize: number): number;
  _fe_init(modelSize: number, weightPtr: number, weightLen: number): number;
  _fe_process(state: number, inPtr: number, outPtr: number): number;
  _fe_destroy(state: number): void;
  _fe_get_input_ptr(state: number): number;
  _fe_get_output_ptr(state: number): number;
  _fe_get_hop_size(state: number): number;
  _fe_get_n_fft(state: number): number;
  _fe_reset(state: number): void;
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
}

/** Model and variant selection */
export type ModelSize = 'tiny' | 'base' | 'small';
export type Variant = 'scalar' | 'simd';

const ROOT = path.resolve(import.meta.dirname, '..', '..');

/** Loads an Emscripten module */
export async function loadWasmModule(
  model: ModelSize,
  variant: Variant,
): Promise<EmscriptenModule> {
  const modulePath = path.join(ROOT, 'dist', 'wasm', `fastenhancer-${model}-${variant}.js`);
  const factory = (await import(modulePath)).default;
  return factory() as Promise<EmscriptenModule>;
}

/** Loads a weight file into WASM memory and returns its pointer */
export function loadWeights(
  module: EmscriptenModule,
  model: ModelSize,
): { ptr: number; len: number } {
  const weightsPath = path.join(ROOT, 'weights', `fe_${model}_48k.bin`);
  const buf = fs.readFileSync(weightsPath);
  const data = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const ptr = module._malloc(data.length);
  if (ptr === 0) {
    throw new Error(`WASM malloc failed: requested ${data.length} bytes for ${model} weights`);
  }
  module.HEAPU8.set(data, ptr);
  return { ptr, len: data.length };
}

/** Loads a golden vector file */
export function loadGoldenVectors(model: ModelSize): {
  input: Float32Array;
  output: Float32Array;
} {
  const dir = model === 'tiny' ? 'golden' : `golden_${model}`;
  const inputPath = path.join(ROOT, 'tests', dir, 'golden_input.bin');
  const outputPath = path.join(ROOT, 'tests', dir, 'golden_output.bin');
  const inBuf = fs.readFileSync(inputPath);
  const outBuf = fs.readFileSync(outputPath);
  return {
    input: new Float32Array(inBuf.buffer, inBuf.byteOffset, inBuf.byteLength / 4),
    output: new Float32Array(outBuf.buffer, outBuf.byteOffset, outBuf.byteLength / 4),
  };
}

/** Processes all frames and returns the output */
export function processAllFrames(
  module: EmscriptenModule,
  state: number,
  input: Float32Array,
  hopSize: number,
): Float32Array {
  const nFrames = Math.floor(input.length / hopSize);
  const inPtr = module._fe_get_input_ptr(state);
  const outPtr = module._fe_get_output_ptr(state);
  const output = new Float32Array(nFrames * hopSize);

  for (let f = 0; f < nFrames; f++) {
    const frameIn = input.subarray(f * hopSize, (f + 1) * hopSize);
    module.HEAPF32.set(frameIn, inPtr / 4);
    module._fe_process(state, inPtr, outPtr);
    const frameOut = module.HEAPF32.slice(outPtr / 4, outPtr / 4 + hopSize);
    output.set(frameOut, f * hopSize);
  }

  return output;
}

/** Calculates MSE (assert matching lengths beforehand) */
export function computeMSE(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Length mismatch: a.length=${a.length}, b.length=${b.length}`);
  }
  const len = a.length;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum / len;
}

/** Calculates maximum absolute error (assert matching lengths beforehand) */
export function computeMaxAbsDiff(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Length mismatch: a.length=${a.length}, b.length=${b.length}`);
  }
  const len = a.length;
  let maxDiff = 0;
  for (let i = 0; i < len; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxDiff) maxDiff = d;
  }
  return maxDiff;
}
