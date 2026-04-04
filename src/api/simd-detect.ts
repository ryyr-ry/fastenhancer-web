/**
 * simd-detect.ts — Detect WASM SIMD runtime support
 *
 * Passes a minimal WASM binary containing a v128 type to WebAssembly.validate()
 * to determine whether the browser/runtime supports WASM SIMD.
 *
 * Responsibility: SIMD detection only. Load selection is handled by wasm-loader.ts.
 */

/**
 * Byte sequence for a minimal WASM module that uses the v128 type.
 *
 * (module
 *   (func (param v128) (result v128)
 *     local.get 0))
 *
 * Section layout:
 * - magic + version (8 bytes)
 * - type section: (v128) -> v128
 * - function section
 * - code section: local.get 0 + end
 */
const SIMD_TEST_MODULE: readonly number[] = [
  0x00, 0x61, 0x73, 0x6d, // magic: \0asm
  0x01, 0x00, 0x00, 0x00, // version: 1
  // Type section (id=1)
  0x01, 0x06,             // section id=1, size=6
  0x01,                   // 1 type entry
  0x60,                   // func type
  0x01, 0x7b,             // 1 param: v128 (0x7b)
  0x01, 0x7b,             // 1 result: v128 (0x7b)
  // Function section (id=3)
  0x03, 0x02,             // section id=3, size=2
  0x01,                   // 1 function
  0x00,                   // type index 0
  // Code section (id=10)
  0x0a, 0x06,             // section id=10, size=6
  0x01,                   // 1 code entry
  0x04,                   // body size=4
  0x00,                   // 0 locals
  0x20, 0x00,             // local.get 0
  0x0b,                   // end
];

/**
 * Returns the SIMD detection test binary as a new Uint8Array.
 * Creates an independent instance on each call.
 */
export function getSimdTestBytes(): Uint8Array {
  return new Uint8Array(SIMD_TEST_MODULE);
}

/**
 * Detects whether the current environment supports WASM SIMD.
 *
 * Passes a minimal module containing a v128 type to WebAssembly.validate().
 * If it returns true, SIMD is supported; if it returns false or throws, SIMD is not supported.
 *
 * @returns true if SIMD is supported, otherwise false
 */
export function detectSimdSupport(): boolean {
  try {
    if (typeof WebAssembly === 'undefined' || !WebAssembly.validate) {
      return false;
    }
    return WebAssembly.validate(getSimdTestBytes() as BufferSource);
  } catch {
    // If WebAssembly.validate() throws, treat SIMD as unsupported.
    // This can happen normally in older browsers or environments without v128 support.
    return false;
  }
}
