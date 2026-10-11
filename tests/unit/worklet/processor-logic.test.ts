import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { processorSource } from '../../../src/embedded/processor-source.js';
import { createWorkletHarness } from '../../helpers/worklet-harness.js';

function run(h: ReturnType<typeof createWorkletHarness>, frames: number) {
  for (let i = 0; i < frames; i++) h.frame();
}
function overload(h: ReturnType<typeof createWorkletHarness>) {
  h.setCost(h.budgetMs * 2.5);
  run(h, 3);
  expect(h.automaticEvents).toEqual([true]);
}

describe('shipped Worklet performance monitoring', () => {
  it('keeps denoising during a long run just above the nominal budget', () => {
    const h = createWorkletHarness();
    h.setCost(h.budgetMs + 0.001);
    run(h, 500);
    expect(h.calls).toBe(500);
    expect(h.automaticEvents).toEqual([]);
    h.send({ type: 'get_stats' });
    expect(h.messages.at(-1)).toMatchObject({ overBudgetFrames: 500, passthroughFrames: 0 });
  });

  it('absorbs an 85 ms moderate load burst, including at startup', () => {
    for (const warmup of [0, 50]) {
      const h = createWorkletHarness();
      h.setCost(h.budgetMs * 0.4); run(h, warmup);
      h.setCost(h.budgetMs * 1.8); run(h, 8);
      h.setCost(h.budgetMs * 0.4); run(h, 100);
      expect(h.automaticEvents).toEqual([]);
    }
  });

  it('ignores isolated large stalls after which the load is already healthy', () => {
    const h = createWorkletHarness();
    for (let i = 0; i < 100; i++) {
      h.setCost(h.budgetMs * (i % 20 === 0 ? 12 : 0.5)); h.frame();
    }
    expect(h.automaticEvents).toEqual([]);
  });

  it('uses sustainable average load for alternating slow and fast frames', () => {
    const h = createWorkletHarness();
    for (let i = 0; i < 300; i++) {
      h.setCost(h.budgetMs * (i % 2 ? 0.2 : 1.6)); h.frame();
    }
    expect(h.automaticEvents).toEqual([]);
  });

  it('eventually protects against unsustainable alternating load despite a low slow-frame fraction', () => {
    const h = createWorkletHarness();
    for (let i = 0; i < 45; i++) {
      h.setCost(h.budgetMs * (i % 2 ? 0.8 : 1.6)); h.frame();
    }
    expect(h.automaticEvents).toEqual([true]);
  });

  it('detects persistent overload even when occasional fast frames break a streak', () => {
    const h = createWorkletHarness();
    for (let i = 0; i < 25; i++) {
      h.setCost(h.budgetMs * (i % 5 === 4 ? 0.5 : 1.5)); h.frame();
    }
    expect(h.automaticEvents).toEqual([true]);
  });

  it('requires sustained evidence and accumulated excess time for a modest overload', () => {
    const h = createWorkletHarness();
    h.setCost(h.budgetMs * 1.12); run(h, 19);
    expect(h.automaticEvents).toEqual([]);
    run(h, 7);
    expect(h.automaticEvents).toEqual([true]);
  });

  it('protects promptly from repeated stalls at twice the full-frame budget', () => {
    const h = createWorkletHarness(); overload(h);
    expect(h.calls).toBe(3);
    expect(h.processor._monitor.mode).toBe('bypass');
  });

  it('recovers in under 100 ms of audio after a transient overload', () => {
    const h = createWorkletHarness(); overload(h);
    const calls = h.calls;
    h.setCost(h.budgetMs * 0.7); run(h, 2);
    expect(h.calls).toBe(calls);
    run(h, 4);
    expect(h.automaticEvents).toEqual([true, false]);
    expect(h.resets).toBe(1);
    expect(h.processor._monitor.mode).toBe('processing');
    expect(6 * h.budgetMs).toBeLessThan(100);
  });

  it('requires consecutive live frames to recover, not isolated cheap probes', () => {
    const h = createWorkletHarness(); overload(h);
    h.setCost(h.budgetMs * 0.6); run(h, 3);
    expect(h.processor._monitor.mode).toBe('recovering');
    h.setCost(h.budgetMs * 4); h.frame();
    expect(h.processor._monitor.mode).toBe('bypass');
    expect(h.automaticEvents).toEqual([true]);
    h.setCost(h.budgetMs * 0.6); run(h, 9);
    expect(h.automaticEvents).toEqual([true, false]);
    expect(h.resets).toBe(2);
  });

  it('allows integer millisecond readings during recovery', () => {
    const h = createWorkletHarness({ timer: 'callback-frozen' }); overload(h);
    h.setCost(11); run(h, 6); // Formerly failed every probe against 10.67 ms.
    expect(h.automaticEvents).toEqual([true, false]);
    h.send({ type: 'get_state' });
    expect(h.messages.at(-1).timingSource).toBe('date');
  });

  it('accepts a moderately slow recovery frame when the trial average is healthy', () => {
    const h = createWorkletHarness(); overload(h);
    h.setCost(h.budgetMs * 1.2); run(h, 3);
    h.setCost(h.budgetMs * 0.7); run(h, 3);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it('does not remain stuck when the first cold probe is slow but consecutive processing fits', () => {
    const h = createWorkletHarness(); overload(h);
    run(h, 3); // One cold probe costs 2.5 budgets.
    h.setCost(h.budgetMs * 0.5); run(h, 3);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it('backs off expensive probes but keeps the next attempt within 128 ms of audio', () => {
    const h = createWorkletHarness(); overload(h);
    const callFrames: number[] = [];
    let previousCalls = h.calls;
    for (let f = 1; f <= 100; f++) {
      h.frame();
      if (h.calls !== previousCalls) { callFrames.push(f); previousCalls = h.calls; }
    }
    const gaps = callFrames.slice(1).map((frame, i) => frame - callFrames[i]);
    expect(callFrames.slice(0, 6)).toEqual([3, 4, 10, 11, 23, 24]);
    expect(Math.max(...gaps) * h.budgetMs).toBeLessThanOrEqual(128 + h.budgetMs);
    expect(h.calls).toBeLessThan(25);
    expect(h.automaticEvents).toEqual([true]);
    h.setCost(h.budgetMs * 0.5); run(h, 16);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it('does not flap on the former 10.67 ms boundary after recovery', () => {
    const h = createWorkletHarness(); overload(h);
    h.setCost(h.budgetMs * 1.03); run(h, 306);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it('uses the combined stereo cost and changes both channels together', () => {
    const h = createWorkletHarness({ channels: 2 });
    h.setCost(h.budgetMs * 0.8); run(h, 50);
    expect(h.automaticEvents).toEqual([]);
    h.setCost(h.budgetMs * 1.5); run(h, 20);
    expect(h.automaticEvents).toEqual([true]);
  });

  it('extends background evidence without carrying old spikes into foreground', () => {
    const h = createWorkletHarness();
    h.send({ type: 'set_background_mode', enabled: true });
    h.setCost(h.budgetMs * 1.5); run(h, 50);
    expect(h.automaticEvents).toEqual([]);
    h.send({ type: 'set_background_mode', enabled: false });
    h.setCost(h.budgetMs * 0.5); run(h, 100);
    expect(h.automaticEvents).toEqual([]);
  });

  it('retries immediately on foreground return even at the longest interval', () => {
    const h = createWorkletHarness(); overload(h); run(h, 22);
    h.send({ type: 'set_background_mode', enabled: true });
    h.send({ type: 'set_background_mode', enabled: false });
    h.setCost(h.budgetMs * 0.5);
    const calls = h.calls; h.frame();
    expect(h.calls).toBe(calls + 1);
    run(h, 3);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it.each(['missing', 'backwards'] as const)('does not infer overload from a %s clock', timer => {
    const h = createWorkletHarness({ timer });
    h.setCost(h.budgetMs * 4); run(h, 50);
    expect(h.automaticEvents).toEqual([]);
    h.send({ type: 'get_stats' });
    expect(h.messages.at(-1)).toMatchObject({ totalFrames: 0, timingSource: 'unavailable' });
  });

  it('resets both states after an engine error and keeps rendering aligned dry audio', () => {
    const h = createWorkletHarness({ channels: 2 });
    h.frame([0.25, -0.4]); h.failNext(1);
    const output = h.frame([0.7, -0.6]);
    expect(output[0].every(v => v === 0.25)).toBe(true);
    expect(output[1].every(v => Math.abs(v + 0.4) < 1e-6)).toBe(true);
    expect(h.automaticEvents).toEqual([true]); run(h, 6);
    expect(h.resets).toBe(2);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it('clears load history on manual bypass and warms the model on return', () => {
    const h = createWorkletHarness(); overload(h);
    h.send({ type: 'set_bypass', enabled: true });
    const calls = h.calls; run(h, 100);
    expect(h.calls).toBe(calls);
    h.setCost(h.budgetMs * 0.5); h.send({ type: 'set_bypass', enabled: false });
    expect(h.frame()[0].every(v => v === 0.25)).toBe(true); run(h, 30);
    expect(h.automaticEvents).toEqual([true, false]);
  });

  it('keeps stats flowing during bypass and counts audio frames separately from probes', () => {
    const h = createWorkletHarness(); h.send({ type: 'start_stats', intervalFrames: 5 });
    overload(h); run(h, 17);
    const stats = h.messages.filter(message => message.type === 'stats');
    expect(stats).toHaveLength(4);
    expect(stats.at(-1).audioFrames).toBe(20);
    expect(stats.at(-1).passthroughFrames).toBeGreaterThan(10);
    expect(stats.at(-1).frameCount).toBeLessThan(20);
  });

  it('runs the same policy in embedded and self-hosted demo assets', () => {
    const demoSource = readFileSync(resolve(import.meta.dirname, '../../../demo/public/assets/worklet/processor.js'), 'utf8');
    for (const source of [processorSource, demoSource]) {
      const h = createWorkletHarness({ source }); overload(h);
      h.setCost(h.budgetMs * 0.7); run(h, 6);
      expect(h.automaticEvents).toEqual([true, false]);
    }
  });
});

describe('aligned audio during protection and recovery', () => {
  it('retains the same delay for manual bypass and denoising', () => {
    for (const manual of [false, true]) {
      const h = createWorkletHarness({ channels: 2 });
      if (manual) h.send({ type: 'set_bypass', enabled: true });
      const rendered = [new Float32Array(1280), new Float32Array(1280)];
      const input = Float32Array.from({ length: 1280 }, (_, i) => i / 1280);
      for (let q = 0; q < 10; q++) {
        const chunk = input.slice(q * 128, (q + 1) * 128);
        const output = h.quantum([chunk, chunk.map(v => -v)]);
        for (let c = 0; c < 2; c++) rendered[c].set(output[c], q * 128);
      }
      // 384 samples of frame assembly + 512 of STFT delay.
      expect(rendered[0].slice(0, 896).every(v => v === 0)).toBe(true);
      for (let i = 896; i < 1280; i++) {
        const expected = input[i - 896] * (manual ? 1 : 0.5);
        expect(rendered[0][i]).toBeCloseTo(expected, 7);
        expect(rendered[1][i]).toBeCloseTo(-expected, 7);
      }
    }
  });

  it('never emits isolated probes or zero startup output while bypassing', () => {
    const h = createWorkletHarness({ channels: 2 }); overload(h);
    for (let f = 0; f < 30; f++) {
      const output = h.frame([0.25, -0.125]);
      // On the first frame right is still the previous default input.
      if (f === 0) continue;
      expect(output[0].every(v => v === 0.25)).toBe(true);
      expect(output[1].every(v => v === -0.125)).toBe(true);
    }
  });

  it('crossfades both channels with the same ramp only after recovery completes', () => {
    const h = createWorkletHarness({ channels: 2 }); overload(h); run(h, 1);
    h.setCost(h.budgetMs * 0.5); h.frame([0.25, -0.5]);
    for (let i = 0; i < 3; i++) {
      const dry = h.frame([0.25, -0.5]);
      expect(dry[0].every(v => v === 0.25)).toBe(true);
      expect(dry[1].every(v => v === -0.5)).toBe(true);
    }
    const mixed = h.frame([0.25, -0.5]);
    expect(h.automaticEvents).toEqual([true, false]);
    expect(mixed[0][0]).toBe(0.25);
    expect(mixed[0][511]).toBe(0.125);
    for (let i = 0; i < 512; i++) expect(mixed[1][i]).toBeCloseTo(-2 * mixed[0][i], 7);
  });
});
