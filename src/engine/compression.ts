/**
 * パワー圧縮/復号 + 複素マスク適用モジュール (TypeScript参照実装)
 * Cエンジンのcompression.c と同等の純TypeScript実装。
 */

export interface CompressResult {
  mag: number;
  phase: number;
}

export interface ComplexResult {
  real: number;
  imag: number;
}

const COMPRESS_EXP = 0.3;

/**
 * パワー圧縮: 複素数 (re, im) → 圧縮後の (mag^0.3, phase)
 */
export function powerCompress(re: number, im: number): CompressResult {
  const mag = Math.sqrt(re * re + im * im);
  const phase = Math.atan2(im, re);
  return {
    mag: mag === 0 ? 0 : Math.pow(mag, COMPRESS_EXP),
    phase,
  };
}

/**
 * パワー復号: (compressedMag, phase) → 元の複素数 (re, im)
 * compressedMag = originalMag^0.3 なので originalMag = compressedMag^(1/0.3)
 */
export function powerDecompress(compressedMag: number, phase: number): ComplexResult {
  if (compressedMag === 0) return { real: 0, imag: 0 };
  const originalMag = Math.pow(compressedMag, 1.0 / COMPRESS_EXP);
  return {
    real: originalMag * Math.cos(phase),
    imag: originalMag * Math.sin(phase),
  };
}

/**
 * 複素マスク適用 (複素数乗算)
 * (inRe + inIm*i) * (maskRe + maskIm*i)
 */
export function applyComplexMask(
  inRe: number,
  inIm: number,
  maskRe: number,
  maskIm: number,
): ComplexResult {
  return {
    real: inRe * maskRe - inIm * maskIm,
    imag: inRe * maskIm + inIm * maskRe,
  };
}
