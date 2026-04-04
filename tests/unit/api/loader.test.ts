/**
 * loader.test.ts — loadModel() 実fetch テスト
 *
 * loadModel(modelSize, { baseUrl?, simd? }) で3リソース(.wasm/.bin/.json)を一括取得し、
 * LoadedModel に wasmBytes/weightData/exportMap/createDenoiser/createStreamDenoiser を持つ。
 * キャッシュ機能: 同一引数で同一Promiseを返す。
 *
 * vi.mock()/vi.fn() は一切使わない。
 * ローカルHTTPサーバーで dist/wasm/ と weights/ をフラットに配信し、
 * 実fetchで検証する。
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { loadModel } from '../../../src/api/loader.js';
import { ValidationError, WasmLoadError } from '../../../src/api/errors.js';

const ROOT = path.resolve(import.meta.dirname, '..', '..', '..');

/**
 * dist/wasm/ と weights/ を統合フラットに配信する HTTP サーバー。
 * baseUrl 指定時のCDN配信シミュレーション。
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
   * 入力検証
   * ================================================================ */

  describe('入力検証', () => {
    it('不正なモデルサイズで ValidationError', () => {
      expect(() => loadModel('invalid' as any, { baseUrl })).toThrow(
        ValidationError,
      );
    });

    it('ValidationError メッセージに不正なサイズ名を含む', async () => {
      try {
        await loadModel('huge' as any, { baseUrl });
        expect.unreachable('ValidationError が throw されるべき');
      } catch (err) {
        expect(err).toBeInstanceOf(ValidationError);
        expect((err as Error).message).toContain('huge');
      }
    });
  });

  /* ================================================================
   * 正常ロード — メタデータ
   * ================================================================ */

  describe('正常ロード', () => {
    it('tiny モデルの正しいメタデータを持つ LoadedModel を返す', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });

      expect(model.size).toBe('tiny');
      expect(model.variant).toBe('simd');
      expect(model.sampleRate).toBe(48000);
      expect(model.hopSize).toBe(512);
      expect(model.nFft).toBe(1024);
      expect(model.modelSizeId).toBe(0);
    });

    it('wasmBytes が .wasm ファイルサイズと一致する ArrayBuffer', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });
      const expectedSize = fs.statSync(
        path.join(ROOT, 'dist', 'wasm', 'fastenhancer-tiny-simd.wasm'),
      ).size;

      expect(model.wasmBytes).toBeInstanceOf(ArrayBuffer);
      expect(model.wasmBytes.byteLength).toBe(expectedSize);
    });

    it('weightData が .bin ファイルサイズと一致する ArrayBuffer', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });
      const expectedSize = fs.statSync(
        path.join(ROOT, 'weights', 'fe_tiny_48k.bin'),
      ).size;

      expect(model.weightData).toBeInstanceOf(ArrayBuffer);
      expect(model.weightData.byteLength).toBe(expectedSize);
    });

    it('exportMap が正しいキーを含む非空 Record', async () => {
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

    it('createDenoiser / createStreamDenoiser / wasmFactory が関数', async () => {
      const model = await loadModel('tiny', { baseUrl, simd: true });

      expect(typeof model.createDenoiser).toBe('function');
      expect(typeof model.createStreamDenoiser).toBe('function');
      expect(typeof model.wasmFactory).toBe('function');
    });

    it('base モデルを正しくロード', async () => {
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

    it('small モデルを正しくロード', async () => {
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

    it('simd: false で scalar バリアント選択', async () => {
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
   * キャッシュ
   * ================================================================ */

  describe('キャッシュ', () => {
    it('同一引数で同一Promise、異なる引数で異なるPromise', () => {
      const p1 = loadModel('tiny', { baseUrl, simd: true });
      const p2 = loadModel('tiny', { baseUrl, simd: true });
      const p3 = loadModel('base', { baseUrl, simd: true });
      const p4 = loadModel('tiny', { baseUrl, simd: false });

      expect(p1).toBe(p2);
      expect(p1).not.toBe(p3);
      expect(p1).not.toBe(p4);
    });
  });

  /* ================================================================
   * エラーハンドリング
   * ================================================================ */

  describe('エラーハンドリング', () => {
    it('到達不能な baseUrl で WasmLoadError', async () => {
      await expect(
        loadModel('tiny', {
          baseUrl: 'http://127.0.0.1:1/',
          simd: true,
        }),
      ).rejects.toThrow(WasmLoadError);
    });

    it('存在しないパスの baseUrl で WasmLoadError', async () => {
      await expect(
        loadModel('tiny', {
          baseUrl: `http://localhost:${port}/nonexistent_dir/`,
          simd: true,
        }),
      ).rejects.toThrow(WasmLoadError);
    });
  });
});
