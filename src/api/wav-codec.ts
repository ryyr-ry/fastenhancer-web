import { ValidationError } from './errors.js';
import { decodeWav, encodeWav } from './wav.js';
import type { PcmAudio } from './offline-pcm.js';

export interface WavFormat { format: number; channelCount: 1 | 2; sampleRate: number; bits: number; blockAlign: number }
export interface WavInfo extends WavFormat { dataOffset: number; dataLength: number; length: number }

export function wavTag(view: DataView, offset: number): string {
  return String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset + offset, 4));
}
function putTag(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
}

/** Validate with the existing codec, including extensible GUID/layout checks. */
export function parseWavFormat(view: DataView, start: number, size: number): WavFormat {
  const count = Math.min(size, 40), padded = count + (count & 1);
  const buffer = new ArrayBuffer(28 + padded), wrapper = new DataView(buffer);
  putTag(wrapper, 0, 'RIFF'); wrapper.setUint32(4, buffer.byteLength - 8, true); putTag(wrapper, 8, 'WAVE');
  putTag(wrapper, 12, 'fmt '); wrapper.setUint32(16, count, true);
  new Uint8Array(buffer, 20, count).set(new Uint8Array(view.buffer, view.byteOffset + start, count));
  putTag(wrapper, 20 + padded, 'data'); wrapper.setUint32(24 + padded, 0, true);
  const audio = decodeWav(buffer);
  const format = view.getUint16(start, true);
  return { format: format === 0xfffe ? view.getUint32(start + 24, true) : format,
    channelCount: audio.channels.length as 1 | 2, sampleRate: audio.sampleRate,
    bits: view.getUint16(start + 14, true), blockAlign: view.getUint16(start + 12, true) };
}

export function wavInfo(format: WavFormat | undefined, dataOffset: number, dataLength: number): WavInfo {
  if (!format || dataOffset < 0 || dataLength % format.blockAlign !== 0) throw new ValidationError('Invalid WAV data or block alignment');
  return { ...format, dataOffset, dataLength, length: dataLength / format.blockAlign };
}

/** Decode bounded blocks with the same codec rather than a second PCM implementation. */
export function decodeWavSamples(view: DataView, info: WavFormat): PcmAudio {
  if (view.byteLength % info.blockAlign !== 0) throw new ValidationError('Invalid WAV data or block alignment');
  const buffer = new ArrayBuffer(44 + view.byteLength), header = new DataView(buffer);
  new Uint8Array(buffer, 0, 44).set(encodeWavHeader(info.channelCount, info.sampleRate, 0));
  header.setUint32(4, buffer.byteLength - 8, true); header.setUint16(20, info.format, true);
  header.setUint32(28, info.sampleRate * info.blockAlign, true); header.setUint16(32, info.blockAlign, true);
  header.setUint16(34, info.bits, true); header.setUint32(40, view.byteLength, true);
  new Uint8Array(buffer, 44).set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return decodeWav(buffer);
}

export function encodeWavHeader(channelCount: 1 | 2, sampleRate: number, length: number): Uint8Array {
  const dataLength = length * channelCount * 2;
  if (!Number.isSafeInteger(length) || length < 0 || dataLength > 0xffffffff - 36) throw new ValidationError('Audio is too large for RIFF/WAVE');
  const buffer = encodeWav({ channels: Array.from({ length: channelCount }, () => new Float32Array(0)), sampleRate });
  const view = new DataView(buffer); view.setUint32(4, 36 + dataLength, true); view.setUint32(40, dataLength, true);
  return new Uint8Array(buffer);
}
export function encodePcm16(audio: PcmAudio): Uint8Array { return new Uint8Array(encodeWav(audio), 44); }
