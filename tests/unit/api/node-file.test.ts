import { describe, it, expect } from 'vitest';
import { mkdtemp, readFile, writeFile, rm, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { enhanceFile } from '../../../src/api/node.js';
import { decodeWav, encodeWav } from '../../../src/api/wav.js';

describe('one-call WAV file processing', () => {
  it('reads and saves a stereo file at its original sample rate', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'fastenhancer-file-'));
    try {
      const left = Float32Array.from({ length: 1703 }, (_, i) => 0.2 * Math.sin(2 * Math.PI * 400 * i / 16000));
      await writeFile(join(dir, 'input.wav'), new Uint8Array(encodeWav({ channels: [left, new Float32Array(left.length)], sampleRate: 16000 })));
      await enhanceFile(join(dir, 'input.wav'), join(dir, 'output.wav'), { model: 'tiny' });
      const bytes = await readFile(join(dir, 'output.wav'));
      const result = decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      expect(result.sampleRate).toBe(16000);
      expect(result.channels.map(c => c.length)).toEqual([1703, 1703]);
      expect(result.channels[1].every(v => v === 0)).toBe(true);
      expect(await readdir(dir)).toEqual(['input.wav', 'output.wav']);
    } finally { await rm(dir, { recursive: true, force: true }); }
  });

  it('keeps an existing output intact when processing is cancelled', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'fastenhancer-cancel-'));
    try {
      await writeFile(join(dir, 'input.wav'), new Uint8Array(encodeWav({ channels: [new Float32Array(48000)], sampleRate: 48000 })));
      await writeFile(join(dir, 'output.wav'), 'existing output');
      const controller = new AbortController();
      await expect(enhanceFile(join(dir, 'input.wav'), join(dir, 'output.wav'), {
        model: 'tiny', signal: controller.signal,
        onProgress(p) { if (p >= 0.1) controller.abort(); },
      })).rejects.toMatchObject({ name: 'AbortError' });
      expect(await readFile(join(dir, 'output.wav'), 'utf8')).toBe('existing output');
      expect(await readdir(dir)).toEqual(['input.wav', 'output.wav']);
    } finally { await rm(dir, { recursive: true, force: true }); }
  });
});
