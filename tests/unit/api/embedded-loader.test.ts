/**
 * embedded-loader.test.ts — 埋込ルーターモジュールのテスト
 *
 * src/embedded/*.ts から動的importしたバイナリ資産が
 * 正しいフォーマットで返されることを検証する。
 *
 * モック不使用: 実際にコード生成された埋込モジュールを読み込む。
 */
import { describe, it, expect, afterAll } from 'vitest';

import {
  loadEmbeddedWasm,
  loadEmbeddedWeights,
  loadEmbeddedExportMap,
  initProcessorBlobUrl,
  revokeProcessorBlobUrl,
} from '../../../src/api/embedded-loader.js';

describe('loadEmbeddedWasm', () => {
  it('tiny-simd: returns an ArrayBuffer starting with WASM magic bytes [0x00,0x61,0x73,0x6d]', async () => {
    const buf = await loadEmbeddedWasm('tiny', 'simd');
    expect(buf).toBeInstanceOf(ArrayBuffer);
    expect(buf.byteLength).toBeGreaterThan(0);
    const header = new Uint8Array(buf, 0, 4);
    expect(Array.from(header)).toEqual([0x00, 0x61, 0x73, 0x6d]);
  });

  it('tiny-scalar: returns an ArrayBuffer starting with WASM magic bytes and differing from simd', async () => {
    const scalar = await loadEmbeddedWasm('tiny', 'scalar');
    const simd = await loadEmbeddedWasm('tiny', 'simd');
    expect(scalar).toBeInstanceOf(ArrayBuffer);
    const header = new Uint8Array(scalar, 0, 4);
    expect(Array.from(header)).toEqual([0x00, 0x61, 0x73, 0x6d]);
    // Scalar and simd binaries differ.
    expect(scalar.byteLength !== simd.byteLength || !buffersEqual(scalar, simd)).toBe(true);
  });

  it('base-simd: returns a valid WASM binary', async () => {
    const buf = await loadEmbeddedWasm('base', 'simd');
    const header = new Uint8Array(buf, 0, 4);
    expect(Array.from(header)).toEqual([0x00, 0x61, 0x73, 0x6d]);
  });

  it('small-scalar: returns a valid WASM binary', async () => {
    const buf = await loadEmbeddedWasm('small', 'scalar');
    const header = new Uint8Array(buf, 0, 4);
    expect(Array.from(header)).toEqual([0x00, 0x61, 0x73, 0x6d]);
  });

  it('returns a fresh ArrayBuffer copy on every call for transferable safety', async () => {
    const a = await loadEmbeddedWasm('tiny', 'simd');
    const b = await loadEmbeddedWasm('tiny', 'simd');
    expect(a).not.toBe(b);
    expect(a.byteLength).toBe(b.byteLength);
  });

  it('throws ValidationError for an invalid model name', async () => {
    await expect(
      loadEmbeddedWasm('invalid' as any, 'simd'),
    ).rejects.toThrow('Invalid');
  });

  it('throws ValidationError for an invalid variant', async () => {
    await expect(
      loadEmbeddedWasm('tiny', 'avx' as any),
    ).rejects.toThrow('Invalid');
  });
});

describe('loadEmbeddedWeights', () => {
  it('tiny: returns an ArrayBuffer starting with FEW1 magic [0x46,0x45,0x57,0x31]', async () => {
    const buf = await loadEmbeddedWeights('tiny');
    expect(buf).toBeInstanceOf(ArrayBuffer);
    expect(buf.byteLength).toBeGreaterThan(0);
    const header = new Uint8Array(buf, 0, 4);
    expect(Array.from(header)).toEqual([0x46, 0x45, 0x57, 0x31]);
  });

  it('base: returns an ArrayBuffer with a valid FEW1 header', async () => {
    const buf = await loadEmbeddedWeights('base');
    const header = new Uint8Array(buf, 0, 4);
    expect(Array.from(header)).toEqual([0x46, 0x45, 0x57, 0x31]);
  });

  it('small: returns an ArrayBuffer with a valid FEW1 header', async () => {
    const buf = await loadEmbeddedWeights('small');
    const header = new Uint8Array(buf, 0, 4);
    expect(Array.from(header)).toEqual([0x46, 0x45, 0x57, 0x31]);
  });

  it('tiny is smaller than base', async () => {
    const tiny = await loadEmbeddedWeights('tiny');
    const base = await loadEmbeddedWeights('base');
    expect(tiny.byteLength).toBeLessThan(base.byteLength);
  });

  it('base is smaller than small', async () => {
    const base = await loadEmbeddedWeights('base');
    const small = await loadEmbeddedWeights('small');
    expect(base.byteLength).toBeLessThan(small.byteLength);
  });

  it('throws ValidationError for an invalid model name', async () => {
    await expect(
      loadEmbeddedWeights('invalid' as any),
    ).rejects.toThrow('Invalid');
  });
});

describe('loadEmbeddedExportMap', () => {
  it('tiny-simd: returns a Record containing the _fe_init key', async () => {
    const map = await loadEmbeddedExportMap('tiny', 'simd');
    expect(map).toBeDefined();
    expect(typeof map).toBe('object');
    expect(map).toHaveProperty('_fe_init');
    expect(map).toHaveProperty('_malloc');
    expect(map).toHaveProperty('_fe_process_inplace');
    expect(typeof map['_fe_init']).toBe('string');
  });

  it('base-scalar: returns a Record containing the _fe_init key', async () => {
    const map = await loadEmbeddedExportMap('base', 'scalar');
    expect(map).toHaveProperty('_fe_init');
    expect(map).toHaveProperty('_free');
  });

  it('simd and scalar export maps share required core exports', async () => {
    const simd = await loadEmbeddedExportMap('tiny', 'simd');
    const scalar = await loadEmbeddedExportMap('tiny', 'scalar');

    const REQUIRED_EXPORTS = [
      '_malloc', '_free', '_fe_init', '_fe_destroy',
      '_fe_process_inplace', '_fe_get_input_ptr', '_fe_get_output_ptr',
      '_fe_get_hop_size',
    ];

    for (const key of REQUIRED_EXPORTS) {
      expect(simd).toHaveProperty(key);
      expect(scalar).toHaveProperty(key);
    }
  });

  it('throws ValidationError for invalid arguments', async () => {
    await expect(
      loadEmbeddedExportMap('huge' as any, 'simd'),
    ).rejects.toThrow('Invalid');
  });
});

describe('initProcessorBlobUrl', () => {
  afterAll(() => {
    revokeProcessorBlobUrl();
  });

  it('returns a string starting with blob: or data:', async () => {
    const url = await initProcessorBlobUrl();
    expect(typeof url).toBe('string');
    expect(url.length).toBeGreaterThan(0);
    // Node.js may expose Blob without URL.createObjectURL, so allow the data: fallback.
    expect(url.startsWith('blob:') || url.startsWith('data:')).toBe(true);
  });

  it('returns the same string from cache', async () => {
    const a = await initProcessorBlobUrl();
    const b = await initProcessorBlobUrl();
    expect(a).toBe(b);
  });
});

/** ArrayBuffer同値比較ユーティリティ */
function buffersEqual(a: ArrayBuffer, b: ArrayBuffer): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const va = new Uint8Array(a);
  const vb = new Uint8Array(b);
  for (let i = 0; i < va.length; i++) {
    if (va[i] !== vb[i]) return false;
  }
  return true;
}

describe('defensive copy', () => {
  it('loadEmbeddedWasm returns distinct ArrayBuffers on consecutive calls', async () => {
    const a = await loadEmbeddedWasm('tiny', 'simd');
    const b = await loadEmbeddedWasm('tiny', 'simd');
    expect(a).not.toBe(b);
    expect(buffersEqual(a, b)).toBe(true);
  });

  it('loadEmbeddedWeights returns distinct ArrayBuffers on consecutive calls', async () => {
    const a = await loadEmbeddedWeights('tiny');
    const b = await loadEmbeddedWeights('tiny');
    expect(a).not.toBe(b);
    expect(buffersEqual(a, b)).toBe(true);
  });

  it('mutation of returned wasm buffer does not affect subsequent calls', async () => {
    const a = await loadEmbeddedWasm('tiny', 'simd');
    new Uint8Array(a)[0] = 0xff;
    const b = await loadEmbeddedWasm('tiny', 'simd');
    expect(new Uint8Array(b)[0]).not.toBe(0xff);
  });

  it('mutation of returned weight buffer does not affect subsequent calls', async () => {
    const a = await loadEmbeddedWeights('tiny');
    new Uint8Array(a)[0] = 0xff;
    const b = await loadEmbeddedWeights('tiny');
    expect(new Uint8Array(b)[0]).not.toBe(0xff);
  });
});
