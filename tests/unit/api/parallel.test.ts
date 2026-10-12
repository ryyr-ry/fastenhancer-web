import { describe, expect, it } from 'vitest';
// Exercise emitted module entry points and real Node Worker threads.
import { enhancePcm as enhanceNode } from '../../../dist/api/node.js';
import { enhancePcm as enhanceInline } from '../../../src/api/enhance.js';

const audio = (rate: number, length = 70001) => ({ sampleRate: rate, channels: [
  Float32Array.from({ length }, (_, i) => 0.15 * Math.sin(i * 0.07)),
  Float32Array.from({ length }, (_, i) => 0.05 * Math.sin(i * 0.19)),
] });
describe('exact parallel processing with real Worker threads', () => {
  it.each(['tiny', 'base', 'small'] as const)('preserves %s output over multiple processing tiles', async model => {
    const input = audio(44100), original = input.channels.map(c => c.slice());
    const expected = await enhanceInline(input, { model, simd: true });
    for (const maxWorkers of [1, 3]) {
      const progress: number[] = [];
      const result = await enhanceNode(input, { model, simd: true, parallel: { maxWorkers }, onProgress: p => progress.push(p) });
      expect(result).toEqual(expected); expect(input.channels).toEqual(original);
      expect(progress[0]).toBe(0); expect(progress.at(-1)).toBe(1);
      expect(progress).toEqual([...progress].sort((a, b) => a - b));
    }
  });
  it('cancels a running pool without poisoning the next job', async () => {
    const controller = new AbortController();
    await expect(enhanceNode(audio(48000, 200001), { model: 'small', parallel: { maxWorkers: 3 }, signal: controller.signal,
      onProgress(p) { if (p > 0.1) controller.abort(); },
    })).rejects.toMatchObject({ name: 'AbortError' });
    const result = await enhanceNode(audio(48000, 513), { model: 'tiny', parallel: { maxWorkers: 2 } });
    expect(result.channels[0]).toHaveLength(513);
  });
  it('propagates callback failures and releases all worker slots', async () => {
    await expect(enhanceNode(audio(48000, 513), { model: 'tiny', parallel: {}, onProgress() { throw new Error('callback'); } })).rejects.toThrow('callback');
    const result = await enhanceNode(audio(48000, 1), { model: 'tiny', parallel: {} });
    expect(result.channels[0]).toHaveLength(1);
  });
});
