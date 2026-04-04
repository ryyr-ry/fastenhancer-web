import { describe, it, expect } from 'vitest';
import { createFrameBuffer } from '../../../src/worklet/frame-buffer.js';

const QUANTUM = 128;

describe('FrameBuffer', () => {
  describe('hop=512 (Tiny/Base/Small)', () => {
    const HOP = 512;

    it('4 quantum蓄積で1フレーム', () => {
      const buf = createFrameBuffer(HOP);
      let frameCount = 0;
      for (let i = 0; i < 4; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(1);
    });

    it('フレームの長さ = hopSize', () => {
      const buf = createFrameBuffer(HOP);
      let outputFrame: Float32Array | null = null;
      for (let i = 0; i < 4; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) outputFrame = frame;
      }
      expect(outputFrame).not.toBeNull();
      expect(outputFrame!).toHaveLength(HOP);
    });

    it('3 quantum以下ではフレームが生成されない', () => {
      const buf = createFrameBuffer(HOP);
      for (let i = 0; i < 3; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        expect(frame).toBeNull();
      }
    });

    it('8 quantum で2フレーム', () => {
      const buf = createFrameBuffer(HOP);
      let frameCount = 0;
      for (let i = 0; i < 8; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(2);
    });
  });

  describe('hop=320 (Medium)', () => {
    const HOP = 320;

    it('5 quantum(640サンプル)で2フレーム', () => {
      const buf = createFrameBuffer(HOP);
      let frameCount = 0;
      for (let i = 0; i < 5; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) {
          expect(frame).toHaveLength(HOP);
          frameCount++;
        }
      }
      expect(frameCount).toBe(2);
    });

    it('push毎のフレーム生成パターン: [null, null, frame, null, frame]', () => {
      const buf = createFrameBuffer(HOP);
      const pattern: boolean[] = [];
      for (let i = 0; i < 5; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        pattern.push(frame !== null);
      }
      expect(pattern).toEqual([false, false, true, false, true]);
    });

    it('10 quantum(1280サンプル)で4フレーム', () => {
      const buf = createFrameBuffer(HOP);
      let frameCount = 0;
      for (let i = 0; i < 10; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(4);
    });
  });

  describe('hop=200 (Large)', () => {
    const HOP = 200;

    it('25 quantum(3200サンプル)で16フレーム', () => {
      const buf = createFrameBuffer(HOP);
      let frameCount = 0;
      for (let i = 0; i < 25; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) {
          expect(frame).toHaveLength(HOP);
          frameCount++;
        }
      }
      expect(frameCount).toBe(16);
    });
  });

  it('入力データの順序が保存される', () => {
    const buf = createFrameBuffer(512);
    for (let i = 0; i < 4; i++) {
      const chunk = new Float32Array(QUANTUM);
      chunk.fill(i + 1);
      const frame = buf.push(chunk);
      if (frame !== null) {
        expect(frame[0]).toBe(1);
        expect(frame[128]).toBe(2);
        expect(frame[256]).toBe(3);
        expect(frame[384]).toBe(4);
      }
    }
  });

  it('連続する複数サイクルでデータが一貫', () => {
    const buf = createFrameBuffer(512);
    for (let cycle = 0; cycle < 3; cycle++) {
      for (let i = 0; i < 4; i++) {
        const chunk = new Float32Array(QUANTUM);
        chunk.fill(cycle * 4 + i);
        const frame = buf.push(chunk);
        if (frame !== null) {
          expect(frame).toHaveLength(512);
          expect(frame[0]).toBe(cycle * 4);
        }
      }
    }
  });

  describe('バックプレッシャー (T-19)', () => {
    it('大量push(1000回)でもクラッシュしない', () => {
      const buf = createFrameBuffer(512);
      let frameCount = 0;
      for (let i = 0; i < 1000; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(250);
    });

    it('内部バッファが無制限に増加しない', () => {
      const buf = createFrameBuffer(512);
      const quantaPerFrame = 4;
      const pushCount = 10000;
      let frameCount = 0;
      for (let i = 0; i < pushCount; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(pushCount / quantaPerFrame);
    });

    it('hop=320で大量pushしても正しいフレーム数', () => {
      const buf = createFrameBuffer(320);
      let frameCount = 0;
      const pushCount = 500;
      for (let i = 0; i < pushCount; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      const expected = Math.floor((pushCount * QUANTUM) / 320);
      expect(frameCount).toBe(expected);
    });

    it('pendingSamplesが常にhopSize未満', () => {
      const buf = createFrameBuffer(512);
      for (let i = 0; i < 1000; i++) {
        buf.push(new Float32Array(QUANTUM));
        expect(buf.pendingSamples).toBeLessThan(512);
        expect(buf.pendingSamples).toBeGreaterThanOrEqual(0);
      }
    });

    it('hop=320でpendingSamplesが常にhopSize未満', () => {
      const buf = createFrameBuffer(320);
      for (let i = 0; i < 500; i++) {
        buf.push(new Float32Array(QUANTUM));
        expect(buf.pendingSamples).toBeLessThan(320);
        expect(buf.pendingSamples).toBeGreaterThanOrEqual(0);
      }
    });

    it('大量pushでデータ整合性が保たれる', () => {
      const buf = createFrameBuffer(512);
      let nextExpectedStart = 0;
      for (let cycle = 0; cycle < 100; cycle++) {
        for (let q = 0; q < 4; q++) {
          const chunk = new Float32Array(QUANTUM);
          const base = (cycle * 4 + q) * QUANTUM;
          for (let i = 0; i < QUANTUM; i++) {
            chunk[i] = base + i;
          }
          const frame = buf.push(chunk);
          if (frame !== null) {
            expect(frame[0]).toBe(nextExpectedStart);
            expect(frame[511]).toBe(nextExpectedStart + 511);
            nextExpectedStart += 512;
          }
        }
      }
      expect(nextExpectedStart).toBe(100 * 512);
    });
  });
});
