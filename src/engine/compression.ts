/**
 * Power compression/decompression + complex mask application module
 * (TypeScript reference implementation)
 * Pure TypeScript implementation equivalent to the C engine's compression.c.
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
 * Power compression: complex number (re, im) → compressed (mag^0.3, phase)
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
 * Power decompression: (compressedMag, phase) → original complex number (re, im)
 * compressedMag = originalMag^0.3, so originalMag = compressedMag^(1/0.3)
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
 * Apply complex mask (complex multiplication)
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
