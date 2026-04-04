/**
 * wasm-instantiate.ts — WASMバイナリの手動インスタンス化
 *
 * AudioWorklet内(processor.js)と同じ手法で、
 * Emscripten JS glue(.js)を使わずに .wasm バイナリから
 * WasmInstance互換オブジェクトを生成する。
 *
 * 責務: WebAssembly.Module/Instance の生成 + エクスポートマッピング のみ。
 */

import type { WasmInstance } from './index.js';
import { WasmLoadError } from './errors.js';

/**
 * .wasm バイナリとエクスポートマップから WasmInstance を生成する。
 *
 * processor.js と同じロジックだが、メインスレッド用に async 版を提供。
 *
 * @param wasmBytes - コンパイル済み .wasm バイナリ
 * @param exportMap - エクスポート名マッピング（readable → minified）
 * @returns WasmInstance 互換オブジェクト
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
      `WASMモジュールのコンパイルに失敗しました: ${err instanceof Error ? err.message : String(err)}`,
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
      `WASMインスタンスの生成に失敗しました: ${err instanceof Error ? err.message : String(err)}`,
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
    throw new WasmLoadError('WASMモジュールからMemoryエクスポートが見つかりません');
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
