import { ValidationError } from './errors.js';
import { validatePcm, type PcmAudio } from './offline-pcm.js';

function tag(view: DataView, offset: number): string {
  return String.fromCharCode(...new Uint8Array(view.buffer, view.byteOffset + offset, 4));
}

export function isWav(buffer: ArrayBuffer): boolean {
  if (buffer.byteLength < 12) return false;
  const view = new DataView(buffer);
  return tag(view, 0) === 'RIFF' && tag(view, 8) === 'WAVE';
}

/** RIFF/WAVE PCM 8/16/24/32-bit and IEEE float32, mono or stereo. */
export function decodeWav(buffer: ArrayBuffer): PcmAudio {
  if (!isWav(buffer)) throw new ValidationError('Expected a RIFF/WAVE audio file');
  const view = new DataView(buffer);
  const riffEnd = view.getUint32(4, true) + 8;
  if (riffEnd > buffer.byteLength || riffEnd < 12) throw new ValidationError('Truncated WAV file');
  let format = 0, channelCount = 0, sampleRate = 0, bits = 0, blockAlign = 0;
  let dataOffset = -1, dataLength = 0;
  for (let offset = 12; offset + 8 <= riffEnd;) {
    const id = tag(view, offset);
    const size = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (start + size > riffEnd) throw new ValidationError('Truncated WAV chunk');
    if (id === 'fmt ') {
      if (size < 16) throw new ValidationError('Invalid WAV format chunk');
      format = view.getUint16(start, true);
      channelCount = view.getUint16(start + 2, true);
      sampleRate = view.getUint32(start + 4, true);
      blockAlign = view.getUint16(start + 12, true);
      bits = view.getUint16(start + 14, true);
      if (format === 0xfffe) {
        // WAVE_FORMAT_EXTENSIBLE carries the PCM/float encoding in a GUID.
        const guidTail = [0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113];
        if (size < 40 || view.getUint16(start + 16, true) < 22
          || !guidTail.every((byte, i) => view.getUint8(start + 28 + i) === byte)) {
          throw new ValidationError('Unsupported extensible WAV format');
        }
        const validBits = view.getUint16(start + 18, true);
        const channelMask = view.getUint32(start + 20, true);
        if (validBits > bits || (channelCount === 2 && channelMask !== 0 && channelMask !== 3)) {
          throw new ValidationError('Unsupported WAV valid bits or stereo channel layout');
        }
        format = view.getUint32(start + 24, true);
      }
    } else if (id === 'data' && dataOffset === -1) {
      dataOffset = start;
      dataLength = size;
    }
    offset = start + size + (size & 1);
  }
  if ((format !== 1 && format !== 3) || (format === 3 && bits !== 32)
    || ![8, 16, 24, 32].includes(bits)) {
    throw new ValidationError('Supported WAV encodings: PCM 8/16/24/32-bit or IEEE float32');
  }
  if (channelCount !== 1 && channelCount !== 2) throw new ValidationError('Only mono/stereo WAV files are supported');
  const bytes = bits / 8;
  if (blockAlign !== channelCount * bytes || dataOffset < 0 || dataLength % blockAlign !== 0) {
    throw new ValidationError('Invalid WAV data or block alignment');
  }
  const length = dataLength / blockAlign;
  const channels = Array.from({ length: channelCount }, () => new Float32Array(length));
  for (let i = 0; i < length; i++) {
    for (let c = 0; c < channelCount; c++) {
      const offset = dataOffset + i * blockAlign + c * bytes;
      let value: number;
      if (format === 3) value = view.getFloat32(offset, true);
      else if (bits === 8) value = (view.getUint8(offset) - 128) / 128;
      else if (bits === 16) value = view.getInt16(offset, true) / 32768;
      else if (bits === 24) {
        const raw = view.getUint8(offset) | view.getUint8(offset + 1) << 8 | view.getUint8(offset + 2) << 16;
        value = (raw & 0x800000 ? raw - 0x1000000 : raw) / 8388608;
      } else value = view.getInt32(offset, true) / 2147483648;
      if (!Number.isFinite(value)) throw new ValidationError('WAV contains non-finite samples');
      channels[c][i] = value;
    }
  }
  const audio = { channels, sampleRate };
  validatePcm(audio);
  return audio;
}

/** Interleaved PCM16 WAV. Samples outside [-1, 1] are clipped on export. */
export function encodeWav(audio: PcmAudio): ArrayBuffer {
  validatePcm(audio);
  const channelCount = audio.channels.length;
  const length = audio.channels[0].length;
  const dataLength = length * channelCount * 2;
  if (dataLength > 0xffffffff - 36) throw new ValidationError('Audio is too large for RIFF/WAVE');
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  const writeTag = (offset: number, value: string): void => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  writeTag(0, 'RIFF'); view.setUint32(4, 36 + dataLength, true); writeTag(8, 'WAVE');
  writeTag(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true); view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * channelCount * 2, true);
  view.setUint16(32, channelCount * 2, true); view.setUint16(34, 16, true);
  writeTag(36, 'data'); view.setUint32(40, dataLength, true);
  for (let i = 0; i < length; i++) {
    for (let c = 0; c < channelCount; c++) {
      const sample = audio.channels[c][i];
      if (!Number.isFinite(sample)) throw new ValidationError('PCM contains non-finite samples');
      const clipped = Math.max(-1, Math.min(1, sample));
      view.setInt16(44 + (i * channelCount + c) * 2, Math.round(clipped * (clipped < 0 ? 32768 : 32767)), true);
    }
  }
  return buffer;
}
