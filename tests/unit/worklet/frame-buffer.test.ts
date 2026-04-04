import { describe, it, expect } from 'vitest';
import { createFrameBuffer } from '../../../src/worklet/frame-buffer.js';

const QUANTUM = 128;

describe('FrameBuffer', () => {
  describe('hop=512 (Tiny/Base/Small)', () => {
    const HOP = 512;

    it('produces 1 frame after accumulating 4 quanta', () => {
      const buf = createFrameBuffer(HOP);
      let frameCount = 0;
      for (let i = 0; i < 4; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(1);
    });

    it('uses hopSize as the frame length', () => {
      const buf = createFrameBuffer(HOP);
      let outputFrame: Float32Array | null = null;
      for (let i = 0; i < 4; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) outputFrame = frame;
      }
      expect(outputFrame).not.toBeNull();
      expect(outputFrame!).toHaveLength(HOP);
    });

    it('does not produce a frame with 3 or fewer quanta', () => {
      const buf = createFrameBuffer(HOP);
      for (let i = 0; i < 3; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        expect(frame).toBeNull();
      }
    });

    it('produces 2 frames from 8 quanta', () => {
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

    it('produces 2 frames from 5 quanta (640 samples)', () => {
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

    it('uses the per-push frame generation pattern: [null, null, frame, null, frame]', () => {
      const buf = createFrameBuffer(HOP);
      const pattern: boolean[] = [];
      for (let i = 0; i < 5; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        pattern.push(frame !== null);
      }
      expect(pattern).toEqual([false, false, true, false, true]);
    });

    it('produces 4 frames from 10 quanta (1280 samples)', () => {
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

    it('produces 16 frames from 25 quanta (3200 samples)', () => {
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

  it('preserves the input data order', () => {
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

  it('keeps data consistent across multiple consecutive cycles', () => {
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

  describe('Backpressure (T-19)', () => {
    it('does not crash under heavy push load (1000 pushes)', () => {
      const buf = createFrameBuffer(512);
      let frameCount = 0;
      for (let i = 0; i < 1000; i++) {
        const frame = buf.push(new Float32Array(QUANTUM));
        if (frame !== null) frameCount++;
      }
      expect(frameCount).toBe(250);
    });

    it('does not let the internal buffer grow without bound', () => {
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

    it('keeps the correct frame count under heavy push load with hop=320', () => {
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

    it('keeps pendingSamples always below hopSize', () => {
      const buf = createFrameBuffer(512);
      for (let i = 0; i < 1000; i++) {
        buf.push(new Float32Array(QUANTUM));
        expect(buf.pendingSamples).toBeLessThan(512);
        expect(buf.pendingSamples).toBeGreaterThanOrEqual(0);
      }
    });

    it('keeps pendingSamples always below hopSize for hop=320', () => {
      const buf = createFrameBuffer(320);
      for (let i = 0; i < 500; i++) {
        buf.push(new Float32Array(QUANTUM));
        expect(buf.pendingSamples).toBeLessThan(320);
        expect(buf.pendingSamples).toBeGreaterThanOrEqual(0);
      }
    });

    it('preserves data integrity under heavy push load', () => {
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
