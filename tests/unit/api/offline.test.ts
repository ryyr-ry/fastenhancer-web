import { describe, expect, it } from 'vitest';
import { enhanceAudio, enhancePcm, loadModel, ValidationError } from '../../../src/api/index.js';
import { resamplePcm } from '../../../src/api/offline-pcm.js';
import { decodeWav, encodeWav } from '../../../src/api/wav.js';

const tone = (length: number, frequency = 400, rate = 48000) =>
  Float32Array.from({ length }, (_, i) => 0.2 * Math.sin(2 * Math.PI * frequency * i / rate));

describe('offline enhancement (real WASM)', () => {
  it.each([1, 511, 512, 513, 1703])('flushes and aligns a %i-sample stereo clip', async length => {
    const channels = [tone(length), tone(length, 1700)];
    const original = channels.map(c => c.slice());
    const result = await enhancePcm({ channels, sampleRate: 48000 }, { model: 'tiny' });
    expect(result.sampleRate).toBe(48000);
    expect(result.channels.map(c => c.length)).toEqual([length, length]);
    expect(channels).toEqual(original);

    // Independent frame-level reference: one extra zero frame drains the STFT.
    const model = await loadModel('tiny');
    for (let c = 0; c < 2; c++) {
      const mono = await model.createDenoiser();
      try {
        const count = Math.ceil(length / 512) + 1;
        const raw = new Float32Array(count * 512);
        for (let frame = 0; frame < count; frame++) {
          const input = new Float32Array(512);
          if (frame * 512 < length) input.set(channels[c].subarray(frame * 512, (frame + 1) * 512));
          raw.set(mono.processFrame(input), frame * 512);
        }
        expect(result.channels[c]).toEqual(raw.slice(512, 512 + length));
      } finally { mono.destroy(); }
    }
  });

  it('retains a signal at the end of a clip and a silent opposite channel', async () => {
    const left = new Float32Array(1733);
    left.set(tone(333), 1400);
    const result = await enhancePcm({ channels: [left, new Float32Array(1733)], sampleRate: 48000 }, { model: 'tiny' });
    expect(result.channels[0].slice(1400).some(v => Math.abs(v) > 1e-6)).toBe(true);
    expect(result.channels[1].every(v => v === 0)).toBe(true);
  });

  it.each([16000, 44100, 96000])('preserves %i Hz and exact sample count', async rate => {
    const result = await enhancePcm({ channels: [tone(1301, 400, rate)], sampleRate: rate }, { model: 'tiny' });
    expect(result.sampleRate).toBe(rate);
    expect(result.channels[0]).toHaveLength(1301);
    expect(result.channels[0].every(Number.isFinite)).toBe(true);
  });

  it('reports monotonic progress and supports cancellation during processing', async () => {
    const controller = new AbortController();
    const progress: number[] = [];
    await expect(enhancePcm({ channels: [tone(48000)], sampleRate: 48000 }, {
      model: 'tiny', signal: controller.signal,
      onProgress(p) { progress.push(p); if (p > 0.1) controller.abort(); },
    })).rejects.toMatchObject({ name: 'AbortError' });
    expect(progress).toEqual([...progress].sort((a, b) => a - b));
    // A cancelled job must not poison the model cache or leak its state.
    const result = await enhancePcm({ channels: [tone(513)], sampleRate: 48000 }, { model: 'tiny' });
    expect(result.channels[0]).toHaveLength(513);
  });

  it('rejects mismatched channels, missing rates, NaN and unsupported channel counts', async () => {
    for (const audio of [
      { channels: [tone(1), tone(2)], sampleRate: 48000 },
      { channels: [tone(1)], sampleRate: NaN },
      { channels: [new Float32Array([NaN])], sampleRate: 48000 },
      { channels: [tone(1), tone(1), tone(1)], sampleRate: 48000 },
    ]) await expect(enhancePcm(audio)).rejects.toThrow(ValidationError);
  });

  it('returns a ready-to-save stereo WAV Blob in one call', async () => {
    const source = { channels: [tone(1703), new Float32Array(1703)], sampleRate: 48000 };
    const blob = await enhanceAudio(new Blob([encodeWav(source)]), { model: 'tiny' });
    expect(blob.type).toBe('audio/wav');
    const clean = decodeWav(await blob.arrayBuffer());
    expect(clean.channels).toHaveLength(2);
    expect(clean.channels[0]).toHaveLength(1703);
    expect(clean.channels[1].every(v => v === 0)).toBe(true);
    expect(clean.channels[0].some(v => Math.abs(v) > 1e-5)).toBe(true);
  });
});

describe('anti-aliasing resampling', () => {
  it('rejects a 12 kHz tone when downsampling from 48 to 16 kHz', async () => {
    const result = await resamplePcm({ channels: [tone(4800, 12000)], sampleRate: 48000 }, 16000);
    const rms = Math.sqrt(result.channels[0].slice(100, -100).reduce((s, v) => s + v * v, 0) / 1400);
    expect(rms).toBeLessThan(0.001);
  });
  it('retains a 1 kHz tone and matching stereo channels at 44.1 kHz', async () => {
    const source = tone(4800, 1000);
    const result = await resamplePcm({ channels: [source, source], sampleRate: 48000 }, 44100);
    expect(result.channels[0]).toEqual(result.channels[1]);
    expect(result.channels[0]).toHaveLength(4410);
    const reference = tone(4410, 1000, 44100);
    const mse = result.channels[0].slice(100, -100).reduce((s, v, i) => s + (v - reference[i + 100]) ** 2, 0) / 4210;
    expect(mse).toBeLessThan(1e-7);
  });
});

describe('WAV container validation', () => {
  it('decodes signed 24-bit stereo samples with their correct channel order', () => {
    const hex = '524946463000000057415645666d7420100000000100020080bb00000065040006001800646174610c0000000000800000400000200000c0';
    const bytes = Uint8Array.from(hex.match(/../g)!, b => parseInt(b, 16));
    const audio = decodeWav(bytes.buffer);
    expect(audio.sampleRate).toBe(48000);
    expect(audio.channels[0]).toEqual(new Float32Array([-1, 0.25]));
    expect(audio.channels[1]).toEqual(new Float32Array([0.5, -0.5]));
  });
  it('interleaves and decodes opposite-polarity stereo without mixing', () => {
    const left = tone(511);
    const right = left.map(v => -v);
    const result = decodeWav(encodeWav({ channels: [left, right], sampleRate: 44100 }));
    expect(result.sampleRate).toBe(44100);
    for (let i = 0; i < left.length; i++) {
      expect(result.channels[0][i]).toBeCloseTo(left[i], 4);
      expect(result.channels[1][i]).toBeCloseTo(right[i], 4);
    }
  });
  it('rejects truncated chunks rather than silently discarding audio', () => {
    const valid = encodeWav({ channels: [tone(3)], sampleRate: 48000 });
    expect(() => decodeWav(valid.slice(0, -1))).toThrow(ValidationError);
    new DataView(valid).setUint32(40, 0xffffffff, true);
    expect(() => decodeWav(valid)).toThrow(ValidationError);
  });
  it('accepts extensible stereo PCM and rejects a different speaker layout', () => {
    const original = encodeWav({ channels: [tone(3), tone(3, 700)], sampleRate: 48000 });
    const buffer = new ArrayBuffer(original.byteLength + 24);
    new Uint8Array(buffer).set(new Uint8Array(original, 0, 36));
    new Uint8Array(buffer).set(new Uint8Array(original, 36), 60);
    const view = new DataView(buffer);
    view.setUint32(4, buffer.byteLength - 8, true);
    view.setUint32(16, 40, true); view.setUint16(20, 0xfffe, true);
    view.setUint16(36, 22, true); view.setUint16(38, 16, true);
    view.setUint32(40, 3, true); view.setUint32(44, 1, true);
    new Uint8Array(buffer).set([0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113], 48);
    expect(decodeWav(buffer)).toEqual(decodeWav(original));
    view.setUint32(40, 12, true);
    expect(() => decodeWav(buffer)).toThrow(ValidationError);
  });
});
