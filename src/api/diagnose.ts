/**
 * ブラウザ互換性診断
 * WebAssembly, WASM SIMD, AudioContext, AudioWorkletの対応状況を検出する。
 */

import { detectSimdSupport } from './simd-detect.js';

export interface DiagnoseResult {
  wasm: boolean;
  simd: boolean;
  audioContext: boolean;
  audioWorklet: boolean;
  /** 最低限動作するか (wasm && audioContext && audioWorklet) */
  overall: boolean;
  /** 推奨環境を満たすか (overall && simd) */
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
    issues.push('WebAssemblyが利用できません。モダンブラウザが必要です。');
  }

  const simd = await detectSimd();
  if (wasm && !simd) {
    issues.push('WASM SIMDが利用できません。Chrome 91+/Firefox 89+/Safari 16.4+が必要です。scalarフォールバックで動作しますが、パフォーマンスが低下します。');
  }

  const audioContext = typeof AudioContext !== 'undefined';
  if (!audioContext) {
    issues.push('AudioContextが利用できません。Web Audio APIに対応したブラウザが必要です。');
  }

  const audioWorklet = typeof AudioWorkletNode !== 'undefined';
  if (!audioWorklet) {
    issues.push('AudioWorkletが利用できません。Chrome 66+/Firefox 76+/Safari 14.1+が必要です。');
  }

  const overall = wasm && audioContext && audioWorklet;
  const recommended = overall && simd;

  return { wasm, simd, audioContext, audioWorklet, overall, recommended, issues };
}
