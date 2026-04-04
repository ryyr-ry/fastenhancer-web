/**
 * wasm-binary.ts — WASM binary path generation
 *
 * AudioWorklet cannot execute Emscripten glue code,
 * so the raw .wasm binary must be loaded directly with WebAssembly.instantiate().
 * This module generates the WASM binary URL from the model size and variant.
 *
 * Responsibility: path/URL generation only. Actual fetching is handled by Layer 2.
 */

import type { WasmVariant } from './wasm-loader';

/** Supported model sizes */
export type ModelSize = 'tiny' | 'base' | 'small';

/**
 * Generates the path for a WASM binary file.
 *
 * @param modelSize - Model size ('tiny' | 'base' | 'small')
 * @param variant - WASM variant ('scalar' | 'simd')
 * @param baseUrl - Base URL. If omitted, only the file name is returned.
 * @returns The WASM binary file path or URL
 */
export function getWasmBinaryPath(
  modelSize: ModelSize,
  variant: WasmVariant,
  baseUrl?: string,
): string {
  const filename = `fastenhancer-${modelSize}-${variant}.wasm`;

  if (!baseUrl) {
    return filename;
  }

  const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  return `${normalizedBase}${filename}`;
}
