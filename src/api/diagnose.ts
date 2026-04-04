/**
 * Browser compatibility diagnostics.
 * Detects support for WebAssembly, WASM SIMD, AudioContext, and AudioWorklet.
 */

import { detectSimdSupport } from './simd-detect.js';

export interface DiagnoseResult {
  wasm: boolean;
  simd: boolean;
  audioContext: boolean;
  audioWorklet: boolean;
  /** Whether the minimum required features are available (wasm && audioContext && audioWorklet) */
  overall: boolean;
  /** Whether the recommended environment is satisfied (overall && simd) */
  recommended: boolean;
  issues: string[];
}

async function detectSimd(): Promise<boolean> {
  try {
    return detectSimdSupport();
  } catch {
    return false;
  }
}

export async function diagnose(): Promise<DiagnoseResult> {
  const issues: string[] = [];

  const wasm = typeof WebAssembly !== 'undefined';
  if (!wasm) {
    issues.push('WebAssembly is not available. A modern browser is required.');
  }

  const simd = await detectSimd();
  if (wasm && !simd) {
    issues.push('WASM SIMD is not available. Chrome 91+, Firefox 89+, or Safari 16.4+ is required. The scalar fallback will still work, but performance will be reduced.');
  }

  const audioContext = typeof AudioContext !== 'undefined';
  if (!audioContext) {
    issues.push('AudioContext is not available. A browser with Web Audio API support is required.');
  }

  const audioWorklet = typeof AudioWorkletNode !== 'undefined';
  if (!audioWorklet) {
    issues.push('AudioWorklet is not available. Chrome 66+, Firefox 76+, or Safari 14.1+ is required.');
  }

  const overall = wasm && audioContext && audioWorklet;
  const recommended = overall && simd;

  return { wasm, simd, audioContext, audioWorklet, overall, recommended, issues };
}
