/**
 * wasm-instantiate.ts — Manual WASM binary instantiation
 *
 * Uses the same approach as inside AudioWorklet (processor.js)
 * to create a WasmInstance-compatible object directly from the .wasm binary
 * without using Emscripten JS glue (.js).
 *
 * Responsibility: creating WebAssembly.Module/Instance and export mapping only.
 */

import type { WasmInstance } from './index.js';
import { WasmLoadError } from './errors.js';

/**
 * Creates a WasmInstance from a .wasm binary and export map.
 *
 * Uses the same logic as processor.js, but provides an async version for the main thread.
 *
 * @param wasmBytes - Compiled .wasm binary
 * @param exportMap - Export name mapping (readable → minified)
 * @returns A WasmInstance-compatible object
 */
export async function instantiateWasm(
  wasmBytes: ArrayBuffer,
  exportMap: Record<string, string>,
): Promise<WasmInstance> {
  let module: WebAssembly.Module;
  try {
    module = await WebAssembly.compile(wasmBytes);
  } catch (err) {
    throw new WasmLoadError(
      `Failed to compile WASM module: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  const needed = WebAssembly.Module.imports(module);
  const importObject: WebAssembly.Imports = {};
  for (const imp of needed) {
    if (!importObject[imp.module]) {
      importObject[imp.module] = {};
    }
    const mod = importObject[imp.module] as Record<string, WebAssembly.ImportValue>;
    if (imp.kind === 'function') {
      mod[imp.name] =
        imp.name === 'c'
          ? () => { throw new Error('WASM abort'); }
          : () => 0;
    }
  }

  let instance: WebAssembly.Instance;
  try {
    instance = await WebAssembly.instantiate(module, importObject);
  } catch (err) {
    throw new WasmLoadError(
      `Failed to instantiate WASM instance: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  let memory: WebAssembly.Memory | null = null;
  for (const val of Object.values(instance.exports)) {
    if (val instanceof WebAssembly.Memory) {
      memory = val;
      break;
    }
  }
  if (!memory) {
    throw new WasmLoadError('Memory export was not found in the WASM module');
  }

  const exports: Record<string, unknown> = {};
  for (const [readable, minified] of Object.entries(exportMap)) {
    const exp = instance.exports[minified];
    if (typeof exp === 'function') {
      exports[readable] = exp;
    }
  }

  const memoryRef = memory;
  let cachedBuffer = memoryRef.buffer;
  let cachedHeapF32 = new Float32Array(cachedBuffer);

  const wasmInstance = {
    ...exports,
    get HEAPF32(): Float32Array {
      if (cachedBuffer !== memoryRef.buffer) {
        cachedBuffer = memoryRef.buffer;
        cachedHeapF32 = new Float32Array(cachedBuffer);
      }
      return cachedHeapF32;
    },
  } as unknown as WasmInstance;

  return wasmInstance;
}
