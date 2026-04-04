import { describe, it, expect } from 'vitest';
import { createProcessorScheduler } from '../../../src/worklet/processor-logic.js';

describe('ProcessorScheduler', () => {
  describe('Processing budget calculation', () => {
    it('hop=512, sr=48000 → about 10.67ms', () => {
      const s = createProcessorScheduler(512, 48000);
      expect(s.budgetMs).toBeCloseTo(512 / 48000 * 1000, 2);
    });

    it('hop=320, sr=48000 → about 6.67ms', () => {
      const s = createProcessorScheduler(320, 48000);
      expect(s.budgetMs).toBeCloseTo(320 / 48000 * 1000, 2);
    });

    it('hop=200, sr=48000 → about 4.17ms', () => {
      const s = createProcessorScheduler(200, 48000);
      expect(s.budgetMs).toBeCloseTo(200 / 48000 * 1000, 2);
    });
  });

  describe('Passthrough transitions', () => {
    it('stays in normal mode when processing stays within budget', () => {
      const s = createProcessorScheduler(512, 48000);
      s.recordProcessingTime(5.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('does not enter passthrough after a single overrun', () => {
      const s = createProcessorScheduler(512, 48000);
      s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('still does not enter passthrough after 4 consecutive overruns', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 4; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('enters passthrough after 5 consecutive overruns', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('does not recover during passthrough after one successful run', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('does not recover during passthrough after two consecutive successful runs', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('returns to normal mode after three consecutive successful runs during passthrough', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(false);
    });

    it('re-enters passthrough after 5 new consecutive overruns following recovery', () => {
      const s = createProcessorScheduler(512, 48000);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
      for (let i = 0; i < 3; i++) s.recordProcessingTime(3.0);
      expect(s.isPassthrough).toBe(false);
      for (let i = 0; i < 5; i++) s.recordProcessingTime(15.0);
      expect(s.isPassthrough).toBe(true);
    });

    it('resets the recovery count when an overrun occurs during the recovery sequence', () => {
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

  describe('Frame drop information', () => {
    it('sets dropped=true when over budget', () => {
      const s = createProcessorScheduler(512, 48000);
      const info = s.recordProcessingTime(15.0);
      expect(info.dropped).toBe(true);
      expect(info.processingTimeMs).toBeCloseTo(15.0, 2);
      expect(info.budgetMs).toBeCloseTo(512 / 48000 * 1000, 2);
    });

    it('sets dropped=false when within budget', () => {
      const s = createProcessorScheduler(512, 48000);
      const info = s.recordProcessingTime(3.0);
      expect(info.dropped).toBe(false);
      expect(info.processingTimeMs).toBeCloseTo(3.0, 2);
    });
    it('keeps dropped=false exactly at the budget boundary', () => {
      const s = createProcessorScheduler(512, 48000);
      const budget = 512 / 48000 * 1000;
      const info = s.recordProcessingTime(budget);
      expect(info.dropped).toBe(false);
    });

    it('sets dropped=true when exceeding the budget by 0.001ms', () => {
      const s = createProcessorScheduler(512, 48000);
      const budget = 512 / 48000 * 1000;
      const info = s.recordProcessingTime(budget + 0.001);
      expect(info.dropped).toBe(true);
    });
  });
});
