/**
 * loader.test.ts — real fetch tests for loadModel()
 *
 * loadModel(modelSize, { baseUrl?, simd? }) fetches three resources
 * (.wasm/.bin/.json) together, and LoadedModel contains
 * wasmBytes/weightData/exportMap/createDenoiser/createStreamDenoiser.
 * Cache behavior: identical arguments return the same Promise.
 *
 * Does not use vi.mock()/vi.fn() at all.
 * Verifies behavior with a local HTTP server that serves dist/wasm/ and
 * weights/ as a flat directory structure via real fetch.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { loadModel, clearModelCache, clearCachedModel } from '../../../src/api/loader.js';
import { ValidationError, WasmLoadError } from '../../../src/api/errors.js';

const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');

/**
 * HTTP server that serves dist/wasm/ and weights/ together as a flat structure.
 * Simulates CDN delivery when baseUrl is specified.
 */
function createFlatServer(): Promise<{ server: http.Server; port: number }> {
  return new Promise((resolve, reject) => {
    const MIME: Record<string, string> = {
      '.js': 'application/javascript; charset=UTF-8',
      '.json': 'application/json; charset=UTF-8',
      '.bin': 'application/octet-stream',
      '.wasm': 'application/wasm',
    };

    const SEARCH_DIRS = [
      path.join(ROOT, 'dist', 'wasm'),
      path.join(ROOT, 'weights'),
    ];

    const srv = http.createServer((req, res) => {
      const filename = decodeURIComponent(req.url!.split('?')[0]).replace(
        /^\/+/,
        '',
      );

      if (!filename || filename.includes('..')) {
        res.writeHead(400);
        res.end('Bad Request');
        return;
      }

      for (const dir of SEARCH_DIRS) {
        const filePath = path.join(dir, filename);
        const resolved = path.resolve(filePath);
        if (!resolved.startsWith(path.resolve(dir) + path.sep)) continue;

        try {
          const data = fs.readFileSync(filePath);
          const ext = path.extname(filePath);
          const mime = MIME[ext] || 'application/octet-stream';
          res.writeHead(200, { 'Content-Type': mime });
          res.end(data);
          return;
        } catch {
          continue;
        }
      }

      res.writeHead(404);
      res.end('Not found');
    });

    srv.listen(0, () => {
      const addr = srv.address() as { port: number };
      resolve({ server: srv, port: addr.port });
    });

    srv.on('error', reject);
  });
}

describe('loadModel', () => {
  let server: http.Server;
  let port: number;
  let baseUrl: string;

  beforeAll(async () => {
    const result = await createFlatServer();
    server = result.server;
    port = result.port;
    baseUrl = `http://localhost:${port}/`;
  });

  afterAll(
    () => new Promise<void>((resolve) => server.close(() => resolve())),
  );

  /* ================================================================
   * Input validation
   * ================================================================ */

  describe('Input validation', () => {
    it('throws ValidationError for an invalid model size', () => {
      expect(() => loadModel('invalid' as any, { baseUrl })).toThrow(
        ValidationError,
      );
    });

    it('includes the invalid size name in the ValidationError message', async () => {
      try {
        await loadModel('huge' as any, { baseUrl });
        expect.unreachable('ValidationError should be thrown');
      } catch (err) {
        expect(err).toBeInstanceOf(ValidationError);
        expect((err as Error).message).toContain('huge');
      }
    });
  });

  /* ================================================================
   * Successful loading — metadata
   * ================================================================ */

  describe('Successful loading', () => {
    it('returns a LoadedModel with correct metadata for the tiny model', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });

      expect(model.size).toBe('tiny');
      expect(model.variant).toBe('simd');
      expect(model.sampleRate).toBe(48000);
      expect(model.hopSize).toBe(512);
      expect(model.nFft).toBe(1024);
      expect(model.modelSizeId).toBe(0);
    });

    it('returns wasmBytes as an ArrayBuffer matching the .wasm file size', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });
      const expectedSize = fs.statSync(
        path.join(ROOT, 'dist', 'wasm', 'fastenhancer-tiny-simd.wasm'),
      ).size;

      expect(model.wasmBytes).toBeInstanceOf(ArrayBuffer);
      expect(model.wasmBytes.byteLength).toBe(expectedSize);
    });

    it('returns weightData as an ArrayBuffer matching the .bin file size', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });
      const expectedSize = fs.statSync(
        path.join(ROOT, 'weights', 'fe_tiny_48k.bin'),
      ).size;

      expect(model.weightData).toBeInstanceOf(ArrayBuffer);
      expect(model.weightData.byteLength).toBe(expectedSize);
    });

    it('returns a non-empty exportMap Record containing the correct keys', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });

      expect(model.exportMap).toBeDefined();
      expect(typeof model.exportMap).toBe('object');

      const keys = Object.keys(model.exportMap);
      expect(keys.length).toBeGreaterThan(0);

      expect(model.exportMap).toHaveProperty('_fe_init');
      expect(model.exportMap).toHaveProperty('_fe_process');
      expect(model.exportMap).toHaveProperty('_malloc');
      expect(model.exportMap).toHaveProperty('_free');
      expect(model.exportMap).toHaveProperty('_fe_destroy');
      expect(model.exportMap).toHaveProperty('_fe_get_input_ptr');
      expect(model.exportMap).toHaveProperty('_fe_get_output_ptr');

      for (const [key, value] of Object.entries(model.exportMap)) {
        expect(typeof key).toBe('string');
        expect(typeof value).toBe('string');
        expect(key.startsWith('_')).toBe(true);
        expect(value.length).toBeGreaterThan(0);
      }
    });

    it('exposes createDenoiser / createStreamDenoiser / wasmFactory as functions', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });

      expect(typeof model.createDenoiser).toBe('function');
      expect(typeof model.createStreamDenoiser).toBe('function');
      expect(typeof model.wasmFactory).toBe('function');
    });

    it('loads the base model correctly', async () => {
      const model = await loadModel('base', { baseUrl, simd: true });

      expect(model.size).toBe('base');
      expect(model.modelSizeId).toBe(1);
      expect(model.variant).toBe('simd');
      expect(model.wasmBytes).toBeInstanceOf(ArrayBuffer);
      expect(model.wasmBytes.byteLength).toBe(
        fs.statSync(
          path.join(ROOT, 'dist', 'wasm', 'fastenhancer-base-simd.wasm'),
        ).size,
      );
      expect(model.weightData.byteLength).toBe(
        fs.statSync(path.join(ROOT, 'weights', 'fe_base_48k.bin')).size,
      );
    });

    it('loads the small model correctly', async () => {
      const model = await loadModel('small', { baseUrl, simd: true });

      expect(model.size).toBe('small');
      expect(model.modelSizeId).toBe(2);
      expect(model.variant).toBe('simd');
      expect(model.wasmBytes).toBeInstanceOf(ArrayBuffer);
      expect(model.wasmBytes.byteLength).toBe(
        fs.statSync(
          path.join(ROOT, 'dist', 'wasm', 'fastenhancer-small-simd.wasm'),
        ).size,
      );
      expect(model.weightData.byteLength).toBe(
        fs.statSync(path.join(ROOT, 'weights', 'fe_small_48k.bin')).size,
      );
    });

    it('selects the scalar variant when simd: false', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: false });

      expect(model.variant).toBe('scalar');
      expect(model.wasmBytes.byteLength).toBe(
        fs.statSync(
          path.join(ROOT, 'dist', 'wasm', 'fastenhancer-tiny-scalar.wasm'),
        ).size,
      );
    });
  });

  /* ================================================================
   * Cache
   * ================================================================ */

  describe('Cache', () => {
    it('returns the same Promise for identical arguments and different Promises for different arguments', () => {
      const p1 = loadModel('tiny', { baseUrl, simd: true });
      const p2 = loadModel('tiny', { baseUrl, simd: true });
      const p3 = loadModel('base', { baseUrl, simd: true });
      const p4 = loadModel('tiny', { baseUrl, simd: false });

      expect(p1).toBe(p2);
      expect(p1).not.toBe(p3);
      expect(p1).not.toBe(p4);
    });

    it('clearModelCache causes subsequent loadModel to return a new Promise', async () => {
      const p1 = loadModel('tiny', { baseUrl, simd: true });
      await p1;
      clearModelCache();
      const p2 = loadModel('tiny', { baseUrl, simd: true });
      expect(p1).not.toBe(p2);
      await p2;
    });

    it('clearCachedModel evicts only the specified model', async () => {
      clearModelCache();
      const pTiny = loadModel('tiny', { baseUrl, simd: true });
      const pBase = loadModel('base', { baseUrl, simd: true });
      await Promise.all([pTiny, pBase]);

      const removed = clearCachedModel('tiny', { baseUrl, simd: true });
      expect(removed).toBe(true);

      const pTiny2 = loadModel('tiny', { baseUrl, simd: true });
      expect(pTiny2).not.toBe(pTiny);

      const pBase2 = loadModel('base', { baseUrl, simd: true });
      expect(pBase2).toBe(pBase);
    });

    it('clearCachedModel returns false for non-cached model', () => {
      clearModelCache();
      const removed = clearCachedModel('small', { baseUrl, simd: true });
      expect(removed).toBe(false);
    });
  });

  /* ================================================================
   * Error handling
   * ================================================================ */

  describe('Error handling', () => {
    it('throws WasmLoadError for an unreachable baseUrl', async () => {
      await expect(
        loadModel('tiny', {
          baseUrl: 'http://127.0.0.1:1/',
          simd: true,
        }),
      ).rejects.toThrow(WasmLoadError);
    });

    it('throws WasmLoadError for a baseUrl pointing to a missing path', async () => {
      await expect(
        loadModel('tiny', {
          baseUrl: `http://localhost:${port}/nonexistent_dir/`,
          simd: true,
        }),
      ).rejects.toThrow(WasmLoadError);
    });
  });

  /* ================================================================
   * Embedded path (no baseUrl)
   * ================================================================ */

  describe('Embedded path (no baseUrl)', () => {
    it('loadModel without baseUrl uses embedded assets and returns a valid LoadedModel', async () => {
      const model = await loadModel('tiny', { simd: true });

      expect(model.size).toBe('tiny');
      expect(model.wasmBytes.byteLength).toBeGreaterThan(0);
      expect(model.weightData.byteLength).toBeGreaterThan(0);
      expect(model.exportMap).toHaveProperty('_fe_init');

      const wasmHeader = new Uint8Array(model.wasmBytes, 0, 4);
      expect(Array.from(wasmHeader)).toEqual([0x00, 0x61, 0x73, 0x6d]);

      const weightHeader = new Uint8Array(model.weightData, 0, 4);
      expect(Array.from(weightHeader)).toEqual([0x46, 0x45, 0x57, 0x31]);
    });

    it('embedded path returns same-shaped data as fetch path', async () => {
      const embedded = await loadModel('tiny', { simd: true });
      const fetched = await loadModel('tiny', { baseUrl, simd: true });

      expect(embedded.wasmBytes.byteLength).toBe(fetched.wasmBytes.byteLength);
      expect(embedded.weightData.byteLength).toBe(fetched.weightData.byteLength);
      expect(Object.keys(embedded.exportMap).sort()).toEqual(
        Object.keys(fetched.exportMap).sort(),
      );
    });
  });
});
