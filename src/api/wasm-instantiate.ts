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

const ABORT_IMPORTS = new Set(['abort', 'c', '__assert_fail', '__cxa_throw']);
const KNOWN_SAFE_STUBS = new Set([
  'emscripten_notify_memory_growth',
  'emscripten_resize_heap',
  '__cxa_atexit',
  '__syscall_openat',
  '__syscall_fcntl64',
  '__syscall_ioctl',
  'fd_write',
  'fd_read',
  'fd_close',
  'fd_seek',
  'environ_sizes_get',
  'environ_get',
  'proc_exit',
  'clock_time_get',
  'args_sizes_get',
  'args_get',
]);
const REQUIRED_EXPORTS = [
  '_malloc', '_free', '_fe_init', '_fe_destroy',
  '_fe_process_inplace', '_fe_get_input_ptr', '_fe_get_output_ptr',
  '_fe_get_hop_size', '_fe_get_n_fft',
  '_fe_set_agc', '_fe_set_hpf', '_fe_reset',
];
const OPTIONAL_EXPORTS = ['_fe_process'];

// Emscripten -O3 minifies module and function names to short lowercase identifiers.
// e.g., module "a" with functions "a", "b", "c", "d", "e".
const MINIFIED_NAME_RE = /^[a-z]{1,2}$/;
function isEmscriptenMinified(moduleName: string, funcName: string): boolean {
  return MINIFIED_NAME_RE.test(moduleName) && MINIFIED_NAME_RE.test(funcName);
}

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
  const unknownImports: string[] = [];
  for (const imp of needed) {
    if (!importObject[imp.module]) {
      importObject[imp.module] = {};
    }
    const mod = importObject[imp.module] as Record<string, WebAssembly.ImportValue>;
    if (imp.kind === 'function') {
      if (ABORT_IMPORTS.has(imp.name)) {
        mod[imp.name] = () => {
          throw new Error(`WASM abort import called: ${imp.module}.${imp.name}`);
        };
      } else if (KNOWN_SAFE_STUBS.has(imp.name)) {
        mod[imp.name] = () => 0;
      } else if (isEmscriptenMinified(imp.module, imp.name)) {
        // Emscripten -O3 minifies import names to single letters.
        // These are safe runtime stubs (memory growth, fd ops, etc.).
        mod[imp.name] = () => 0;
      } else {
        unknownImports.push(`${imp.module}.${imp.name}`);
      }
    }
  }
  if (unknownImports.length > 0) {
    throw new WasmLoadError(
      `Unknown WASM imports detected: ${unknownImports.join(', ')}. ` +
      `If these are safe to stub, add them to KNOWN_SAFE_STUBS.`,
    );
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
  const exportedNames = new Set([...Object.keys(exportMap), ...OPTIONAL_EXPORTS]);
  for (const readable of exportedNames) {
    const minified = exportMap[readable];
    if (!minified) {
      continue;
    }
    const exp = instance.exports[minified];
    if (typeof exp === 'function') {
      exports[readable] = exp;
    }
  }
  const missingExports: string[] = [];
  for (const name of REQUIRED_EXPORTS) {
    if (typeof exports[name] !== 'function') {
      missingExports.push(name);
    }
  }
  if (missingExports.length > 0) {
    throw new WasmLoadError(
      `WASM module is missing required exports: ${missingExports.join(', ')}. ` +
      `Check that the export map matches the compiled WASM binary.`,
    );
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
