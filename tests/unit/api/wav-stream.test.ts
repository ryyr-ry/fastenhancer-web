import { describe, expect, it } from 'vitest';
import { mkdtemp, readFile, writeFile, rm, readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { enhanceAudio, enhanceWavStream } from '../../../src/api/enhance.js';
import { encodeWav } from '../../../src/api/wav.js';
import { enhanceFile } from '../../../dist/api/node.js';
import { encodeWavHeader } from '../../../src/api/wav-codec.js';

const wav = (rate: number, length = 70001) => encodeWav({ sampleRate: rate, channels: [
  Float32Array.from({ length }, (_, i) => 0.1 * Math.sin(i * 0.07) + (i === length - 1 ? 0.2 : 0)),
  new Float32Array(length),
] });
async function streamed(file: Blob, options = {}) {
  const parts: Uint8Array[] = [];
  for await (const block of enhanceWavStream(file, { model: 'tiny', worker: false, ...options })) parts.push(block);
  return new Uint8Array(await new Blob(parts as BlobPart[]).arrayBuffer());
}

function encodedFixture(kind: number | 'float' | 'extensible', length = 513): ArrayBuffer {
  if (kind === 'extensible') {
    const original = wav(48000, length), result = new ArrayBuffer(original.byteLength + 24), view = new DataView(result);
    new Uint8Array(result).set(new Uint8Array(original, 0, 36));
    new Uint8Array(result).set(new Uint8Array(original, 36), 60);
    view.setUint32(4, result.byteLength - 8, true); view.setUint32(16, 40, true); view.setUint16(20, 0xfffe, true);
    view.setUint16(36, 22, true); view.setUint16(38, 16, true); view.setUint32(40, 3, true); view.setUint32(44, 1, true);
    new Uint8Array(result).set([0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113], 48);
    return result;
  }
  const bits = kind === 'float' ? 32 : kind, size = bits / 8, result = new ArrayBuffer(44 + length * 2 * size), view = new DataView(result);
  new Uint8Array(result).set(encodeWavHeader(2, 48000, 0));
  view.setUint32(4, result.byteLength - 8, true); view.setUint16(20, kind === 'float' ? 3 : 1, true);
  view.setUint32(28, 48000 * size * 2, true); view.setUint16(32, size * 2, true); view.setUint16(34, bits, true); view.setUint32(40, length * size * 2, true);
  for (let i = 0; i < length; i++) for (let c = 0; c < 2; c++) {
    const value = c === 0 ? 0.2 * Math.sin(i * 0.07) : 0, offset = 44 + (i * 2 + c) * size;
    if (kind === 'float') view.setFloat32(offset, value, true);
    else if (bits === 8) view.setUint8(offset, Math.round(value * 127) + 128);
    else if (bits === 24) { const raw = Math.round(value * 8388607); for (let b = 0; b < 3; b++) view.setUint8(offset + b, (raw >> (b * 8)) & 255); }
    else view.setInt32(offset, Math.round(value * 2147483647), true);
  }
  return result;
}
describe('bounded WAV input/output', () => {
  it.each([8, 24, 32, 'float', 'extensible'] as const)('preserves the existing %s WAV decoding', async kind => {
    const file = new Blob([encodedFixture(kind)]);
    const expected = new Uint8Array(await (await enhanceAudio(file, { model: 'tiny', worker: false })).arrayBuffer());
    expect(await streamed(file)).toEqual(expected);
  });
  it.each([8000, 16000, 44100, 48000, 96000, 192000])('preserves every output byte at %i Hz across tile boundaries', async rate => {
    const file = new Blob([wav(rate)]);
    const expected = new Uint8Array(await (await enhanceAudio(file, { model: 'tiny', worker: false })).arrayBuffer());
    expect(await streamed(file)).toEqual(expected);
  });
  it.each([0, 1, 511, 512, 513])('flushes a %i-sample file exactly', async length => {
    const file = new Blob([wav(44100, length)]);
    const expected = new Uint8Array(await (await enhanceAudio(file, { model: 'tiny', worker: false })).arrayBuffer());
    expect(await streamed(file)).toEqual(expected);
  });
  it('reads bounded ranges without requesting the whole input', async () => {
    let entire = 0, maximum = 0;
    class RangedBlob extends Blob {
      override arrayBuffer() { entire++; return super.arrayBuffer(); }
      override slice(start = 0, end = this.size, type?: string) { maximum = Math.max(maximum, end - start); return super.slice(start, end, type); }
    }
    const file = new RangedBlob([wav(192000, 270001)]);
    await streamed(file);
    expect(entire).toBe(0); expect(maximum).toBeLessThan(file.size);
    expect(maximum).toBeLessThan(1100000);
  });
  it('releases an inline session when a consumer stops after the header', async () => {
    for await (const block of enhanceWavStream(new Blob([wav(48000)]), { model: 'tiny', worker: false })) { expect(block).toHaveLength(44); break; }
    expect((await streamed(new Blob([wav(48000, 1)]))).length).toBe(48);
  });
  it('rejects malformed format and truncated data consistently', async () => {
    const bytes = wav(48000, 3);
    await expect(streamed(new Blob([bytes.slice(0, -1)]))).rejects.toThrow('Truncated');
    const invalid = bytes.slice(0); new DataView(invalid).setUint16(20, 7, true);
    await expect(streamed(new Blob([invalid]))).rejects.toThrow('WAV encodings');
  });
  it('saves the same bytes through Node threads and supports replacing its input', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'fastenhancer-stream-'));
    try {
      const bytes = wav(44100), file = join(dir, 'audio.wav');
      const expected = Buffer.from(await streamed(new Blob([bytes])));
      await writeFile(file, new Uint8Array(bytes));
      await enhanceFile(file, file, { model: 'tiny', parallel: { maxWorkers: 3 } });
      expect(await readFile(file)).toEqual(expected);
      expect(await readdir(dir)).toEqual(['audio.wav']);
    } finally { await rm(dir, { force: true, recursive: true }); }
  });
  it.each(['abort', 'callback'])('keeps the old file and removes temporary data on late %s', async mode => {
    const dir = await mkdtemp(join(tmpdir(), 'fastenhancer-stream-cancel-'));
    try {
      const source = join(dir, 'input.wav'), output = join(dir, 'output.wav'), controller = new AbortController();
      await writeFile(source, new Uint8Array(wav(48000))); await writeFile(output, 'original');
      const promise = enhanceFile(source, output, { model: 'tiny', parallel: { maxWorkers: 2 }, signal: controller.signal,
        onProgress(p) { if (p === 1) { if (mode === 'abort') controller.abort(); else throw new Error('callback'); } },
      });
      if (mode === 'abort') await expect(promise).rejects.toMatchObject({ name: 'AbortError' });
      else await expect(promise).rejects.toThrow('callback');
      expect(await readFile(output, 'utf8')).toBe('original');
      expect((await readdir(dir)).sort()).toEqual(['input.wav', 'output.wav']);
    } finally { await rm(dir, { force: true, recursive: true }); }
  });
  it('discards partial output when a late float32 sample is invalid', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'fastenhancer-stream-invalid-'));
    try {
      const bytes = encodedFixture('float', 70001); new DataView(bytes).setFloat32(bytes.byteLength - 8, NaN, true);
      const source = join(dir, 'input.wav'), output = join(dir, 'output.wav');
      await writeFile(source, new Uint8Array(bytes)); await writeFile(output, 'original');
      await expect(enhanceFile(source, output, { model: 'tiny', parallel: { maxWorkers: 3 } })).rejects.toThrow('non-finite');
      expect(await readFile(output, 'utf8')).toBe('original');
      expect((await readdir(dir)).sort()).toEqual(['input.wav', 'output.wav']);
    } finally { await rm(dir, { force: true, recursive: true }); }
  });
});
