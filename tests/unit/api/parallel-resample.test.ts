import { describe, expect, it } from 'vitest';
import { resamplePcm } from '../../../src/api/offline-pcm.js';
import { resampleInputRange, resampleTile } from '../../../src/api/offline-resample.js';

const cases = [8000, 16000, 22050, 44100, 48000, 96000, 192000].flatMap(rate =>
  [false, true].flatMap(reverse => [1, 513, 1703].flatMap(length => [1, 2].map(count => ({
    sourceRate: reverse ? 48000 : rate, targetRate: reverse ? rate : 48000, length, count,
  })))));

describe('parallel resampling preserves the existing filter exactly', () => {
  it.each(cases)('$sourceRate → $targetRate, length $length, $count channels', async ({ sourceRate, targetRate, length, count }) => {
    const channels = Array.from({ length: count }, (_, c) => Float32Array.from({ length }, (_, i) => 0.3 * Math.sin(i * (c + 1) * 0.17) + (i === length - 1 ? 0.1 : 0)));
    const reference = await resamplePcm({ channels, sampleRate: sourceRate }, targetRate);
    for (const tileSize of [1, 127, 513]) {
      const output = channels.map(() => new Float32Array(reference.channels[0].length));
      for (let start = 0; start < output[0].length; start += tileSize) {
        const end = Math.min(output[0].length, start + tileSize);
        const [inputBase, inputEnd] = resampleInputRange(sourceRate, targetRate, length, start, end);
        const tile = resampleTile({ channels: channels.map(c => c.slice(inputBase, inputEnd)), sourceRate, targetRate, sourceLength: length, inputBase, outputStart: start, outputEnd: end });
        tile.forEach((c, i) => output[i].set(c, start));
      }
      expect(output).toEqual(reference.channels);
    }
  });
});
