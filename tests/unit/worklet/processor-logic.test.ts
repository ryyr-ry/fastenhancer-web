import { describe, it, expect } from 'vitest';
import { createProcessorScheduler } from '../../../src/worklet/processor-logic.js';

describe('ProcessorScheduler', () => {
  describe('処理予算の計算', () => {
    it('hop=512, sr=48000 → 約10.67ms', () => {
      const s = createProcessorScheduler(512, 48000);
      expect(s.budgetMs).toBeCloseTo(512 / 48000 * 1000, 2);
    });

    it('hop=320, sr=48000 → 約6.67ms', () => {
      const s = createProcessorScheduler(320, 48000);
      expect(s.budgetMs).toBeCloseTo(320 / 48000 * 1000, 2);
    });

    it('hop=200, sr=48000 → 約4.17ms', () => {
      const s = createProcessorScheduler(200, 48000);
      expect(s.budgetMs).toBeCloseTo(200 / 48000 * 1000, 2);
    });
  });

  describe('パススルー遷移', () => {
    it('予算内の処理では通常モード維持', () => {
      const s = createProcessorScheduler(512, 48000);
      s.recordProcessingTime(5.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('1回の超過ではパススルーにならない', () => {
      const s = createProcessorScheduler(512, 48000);
      s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('連続4回超過ではまだパススルーにならない', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 4; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('連続5回超過でパススルーに遷移', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('パススルー中に1回成功しても復帰しない', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('パススルー中に2回連続成功しても復帰しない', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('パススルー中に3回連続成功で通常モードに復帰', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('復帰後に再度5連続超過でパススルーに再遷移', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      for (let i = 0; i < 3; i++) s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(false);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('復帰シーケンス中に超過が入るとカウントリセット', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(15.0);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(true);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(false);
    });
  });

  describe('フレームドロップ情報', () => {
    it('予算超過時: dropped=true', () => {
      const s = createProcessorScheduler(512, 48000);
      const info = s.recordProcessingTime(15.0);
      expect(info.dropped).toBe(true);
      expect(info.processingTimeMs).toBeCloseTo(15.0, 2);
      expect(info.budgetMs).toBeCloseTo(512 / 48000 * 1000, 2);
    });

    it('予算内: dropped=false', () => {
      const s = createProcessorScheduler(512, 48000);
      const info = s.recordProcessingTime(3.0);
      expect(info.dropped).toBe(false);
      expect(info.processingTimeMs).toBeCloseTo(3.0, 2);
    });
    it('予算ちょうど境界: dropped=false', () => {
      const s = createProcessorScheduler(512, 48000);
      const budget = 512 / 48000 * 1000;
      const info = s.recordProcessingTime(budget);
      expect(info.dropped).toBe(false);
    });

    it('予算を0.001ms超過: dropped=true', () => {
      const s = createProcessorScheduler(512, 48000);
      const budget = 512 / 48000 * 1000;
      const info = s.recordProcessingTime(budget + 0.001);
      expect(info.dropped).toBe(true);
    });
  });
});
