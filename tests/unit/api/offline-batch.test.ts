import { describe, expect, it } from 'vitest';
import { enhancePcm as enhanceNode } from '../../../dist/api/node.js';
import { enhancePcm as enhanceInline } from '../../../src/api/enhance.js';

const input = (length: number, rate = 48000) => ({ sampleRate: rate, channels: [
  Float32Array.from({ length }, (_, i) => 0.12 * Math.sin(i * 0.07) + 0.02 * Math.sin(i * 0.31)),
  Float32Array.from({ length }, (_, i) => 0.04 * Math.sin(i * 0.19)),
] });
describe('state-preserving internal offline parallelism', () => {
  it.each(['tiny', 'base', 'small'] as const)('preserves %s handling of finite PCM whose unused Nyquist value overflows', async model => {
    const channels = [Float32Array.from({ length: 70001 }, (_, i) => i >= 40000 && i < 60000 ? (i % 2 === 0 ? 1e38 : -1e38) : 0.1 * Math.sin(i * 0.07))];
    const audio = { channels, sampleRate: 48000 }, reference = await enhanceInline(audio, { model, simd: true });
    expect(reference.channels[0].every(Number.isFinite)).toBe(true);
    expect(await enhanceNode(audio, { model, simd: true, parallel: { engine: 'batch', maxWorkers: 3 } })).toEqual(reference);
  });
  it.each(['tiny', 'base', 'small'] as const)('matches the %s SIMD engine at every sample, with one or several workers', async model => {
    const audio = input(70001), reference = await enhanceInline(audio, { model, simd: true });
    for (const maxWorkers of [1, 3]) expect(await enhanceNode(audio, { model, simd: true, parallel: { engine: 'batch', maxWorkers } })).toEqual(reference);
  });
  it.each(['tiny', 'base', 'small'] as const)('keeps legacy scalar rounding for %s even when batch is requested', async model => {
    const audio = input(1703), reference = await enhanceInline(audio, { model, simd: false });
    expect(await enhanceNode(audio, { model, simd: false, parallel: { engine: 'batch', maxWorkers: 3 } })).toEqual(reference);
  });
  it.each([0, 1, 511, 512, 513, 65537])('aligns the header and tail for %i samples', async length => {
    const audio = input(length, 44100), reference = await enhanceInline(audio, { model: 'tiny', simd: true });
    expect(await enhanceNode(audio, { model: 'tiny', simd: true, parallel: { engine: 'batch', maxWorkers: 3 } })).toEqual(reference);
  });
  it('preserves a silent channel and repeated input throughout several batches', async () => {
    const source = input(70001).channels[0];
    const result = await enhanceNode({ channels: [source, source], sampleRate: 48000 }, { model: 'base', simd: true, parallel: { engine: 'batch', maxWorkers: 3 } });
    expect(result.channels[0]).toEqual(result.channels[1]);
    const silent = await enhanceNode({ channels: [source, new Float32Array(source.length)], sampleRate: 48000 }, { model: 'base', simd: true, parallel: { engine: 'batch', maxWorkers: 3 } });
    expect(silent.channels[1].every(v => v === 0)).toBe(true);
  });
  it('terminates a batch after cancellation and can initialize fresh recurrent state', async () => {
    const controller = new AbortController();
    await expect(enhanceNode(input(200001), { model: 'small', simd: true, parallel: { engine: 'batch', maxWorkers: 3 }, signal: controller.signal,
      onProgress(p) { if (p > 0.1) controller.abort(); },
    })).rejects.toMatchObject({ name: 'AbortError' });
    const audio = input(513), reference = await enhanceInline(audio, { model: 'small', simd: true });
    expect(await enhanceNode(audio, { model: 'small', simd: true, parallel: { engine: 'batch', maxWorkers: 3 } })).toEqual(reference);
  });
});
