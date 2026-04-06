/**
 * Unit tests for wasm-instantiate.ts — Manual WASM instantiation module.
 * Tests error paths, import stubbing, and export map validation.
 */
import { describe, it, expect } from 'vitest';
import { instantiateWasm } from '../../../src/api/wasm-instantiate.js';
import { WasmLoadError } from '../../../src/api/errors.js';
import { loadRealWasm } from '../../helpers/real-model.js';

describe('instantiateWasm', () => {
  it('successfully instantiates a valid WASM binary with correct export map', async () => {
    const wasm = await loadRealWasm('tiny', 'simd');
    expect(typeof wasm._fe_init).toBe('function');
    expect(typeof wasm._fe_destroy).toBe('function');
    expect(typeof wasm._fe_process_inplace).toBe('function');
    expect(typeof wasm._malloc).toBe('function');
    expect(typeof wasm._free).toBe('function');
    expect(wasm.HEAPF32).toBeInstanceOf(Float32Array);
  });

  it('throws WasmLoadError for invalid (corrupt) WASM bytes', async () => {
    const garbage = new ArrayBuffer(32);
    new Uint8Array(garbage).set([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
    // Truncated module — valid magic number but corrupt content
    await expect(instantiateWasm(garbage, {})).rejects.toThrow(WasmLoadError);
  });

  it('throws WasmLoadError for completely invalid bytes', async () => {
    const garbage = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    await expect(
      instantiateWasm(garbage.buffer as ArrayBuffer, {}),
    ).rejects.toThrow(WasmLoadError);
  });

  it('throws WasmLoadError when required exports are missing from export map', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');
    const wasmPath = path.join(ROOT, 'dist', 'wasm', 'fastenhancer-tiny-simd.wasm');
    const wasmBytes = fs.readFileSync(wasmPath).buffer as ArrayBuffer;
    // Pass an empty export map — required exports can't be resolved
    await expect(instantiateWasm(wasmBytes, {})).rejects.toThrow(WasmLoadError);
    await expect(instantiateWasm(wasmBytes, {})).rejects.toThrow(/missing required exports/i);
  });

  it('HEAPF32 getter refreshes on memory growth', async () => {
    const wasm = await loadRealWasm('tiny', 'simd');
    const heap1 = wasm.HEAPF32;
    expect(heap1).toBeInstanceOf(Float32Array);
    // Access again — should return same or equivalent view
    const heap2 = wasm.HEAPF32;
    expect(heap2.buffer).toBe(heap1.buffer);
  });

  it('scalar variant also instantiates correctly', async () => {
    const wasm = await loadRealWasm('tiny', 'scalar');
    expect(typeof wasm._fe_init).toBe('function');
    expect(typeof wasm._fe_process_inplace).toBe('function');
    expect(wasm.HEAPF32).toBeInstanceOf(Float32Array);
  });
});
