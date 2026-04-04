/**
 * real-model.ts — test helpers that use real WASM binaries
 *
 * Creates Model objects from the real dist/wasm/ WASM modules and
 * real weights/ files without using any mocks.
 *
 * Does not use the .js Emscripten glue, only manual instantiation of .wasm binaries.
 */
import fs from 'fs';
import path from 'path';
import type { WasmInstance, Model } from '../../src/api/index.js';
import { instantiateWasm } from '../../src/api/wasm-instantiate.js';

export type ModelSize = 'tiny' | 'base' | 'small';
export type Variant = 'scalar' | 'simd';

const ROOT = path.resolve(import.meta.dirname, '..', '..');

export async function loadRealWasm(
  model: ModelSize = 'small',
  variant: Variant = 'simd',
): Promise<WasmInstance> {
  const wasmPath = path.join(ROOT, 'dist', 'wasm', `fastenhancer-${model}-${variant}.wasm`);
  const mapPath = path.join(ROOT, 'dist', 'wasm', `fastenhancer-${model}-${variant}-exports.json`);

  const wasmBytes = fs.readFileSync(wasmPath);
  const wasmBuffer = wasmBytes.buffer.slice(wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength);

  const rawJson = fs.readFileSync(mapPath, 'utf-8');
  const exportMap = JSON.parse(rawJson.replace(/^\uFEFF/, '')) as Record<string, string>;

  return instantiateWasm(wasmBuffer, exportMap);
}

export function loadRealWeightData(model: ModelSize = 'small'): ArrayBuffer {
  const weightsPath = path.join(ROOT, 'weights', `fe_${model}_48k.bin`);
  const buf = fs.readFileSync(weightsPath);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

export function createRealModel(
  wasm: WasmInstance,
  model: ModelSize = 'small',
): Model {
  return {
    size: model,
    sampleRate: 48000,
    nFft: 1024,
    hopSize: 512,
    wasmFactory: () => loadRealWasm(model),
    _wasm: wasm,
  };
}

export function createRealModelWithFactory(model: ModelSize = 'small'): Model {
  return {
    size: model,
    sampleRate: 48000,
    nFft: 1024,
    hopSize: 512,
    wasmFactory: () => loadRealWasm(model),
  };
}
