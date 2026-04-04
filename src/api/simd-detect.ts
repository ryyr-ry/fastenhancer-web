/**
 * simd-detect.ts — WASM SIMDランタイムサポート検出
 *
 * WebAssembly.validate() に v128 型を含む最小WASMバイナリを渡し、
 * ブラウザ/ランタイムがWASM SIMDをサポートしているか判定する。
 *
 * 責務: SIMD検出のみ。ロード判断はwasm-loader.tsが行う。
 */

/**
 * v128型を使用する最小WASMモジュールのバイト列。
 *
 * (module
 *   (func (param v128) (result v128)
 *     local.get 0))
 *
 * セクション構成:
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
 * SIMD検出用テストバイナリを新しいUint8Arrayとして返す。
 * 呼び出しごとに独立したインスタンスを生成する。
 */
export function getSimdTestBytes(): Uint8Array {
  return new Uint8Array(SIMD_TEST_MODULE);
}

/**
 * 現在の環境がWASM SIMDをサポートしているか検出する。
 *
 * WebAssembly.validate()にv128型を含む最小モジュールを渡し、
 * trueが返ればSIMD対応、falseまたは例外ならSIMD非対応と判定。
 *
 * @returns SIMD対応ならtrue、非対応ならfalse
 */
export function detectSimdSupport(): boolean {
  try {
    if (typeof WebAssembly === 'undefined' || !WebAssembly.validate) {
      return false;
    }
    return WebAssembly.validate(getSimdTestBytes() as BufferSource);
  } catch {
    // WebAssembly.validate()が例外を投げた場合はSIMD非対応と判定。
    // 古いブラウザやv128未対応環境で発生しうる正常系。
    return false;
  }
}
