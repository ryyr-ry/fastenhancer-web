import { ValidationError } from './errors.js';
import { checkAbort, type EnhanceOptions } from './offline-pcm.js';
import { OfflinePool } from './offline-pool.js';
import { OFFLINE_TILE_SAMPLES } from './offline-parallel.js';
import { createInferenceSession } from './offline-batch.js';
import { resampleInputRange } from './offline-resample.js';
import { decodeWavSamples, encodePcm16, encodeWavHeader, parseWavFormat, wavInfo, wavTag, type WavFormat, type WavInfo } from './wav-codec.js';

export interface AudioReader { readonly size: number; read(offset: number, length: number): Promise<ArrayBuffer> }
async function readExact(reader: AudioReader, offset: number, length: number, signal?: AbortSignal): Promise<ArrayBuffer> {
  checkAbort(signal);
  const bytes = await reader.read(offset, length);
  checkAbort(signal);
  if (bytes.byteLength !== length) throw new ValidationError('Truncated WAV file');
  return bytes;
}
export async function readWavInfo(reader: AudioReader, signal?: AbortSignal): Promise<WavInfo> {
  if (reader.size < 12) throw new ValidationError('Expected a RIFF/WAVE audio file');
  const header = new DataView(await readExact(reader, 0, 12, signal));
  if (wavTag(header, 0) !== 'RIFF' || wavTag(header, 8) !== 'WAVE') throw new ValidationError('Expected a RIFF/WAVE audio file');
  const riffEnd = header.getUint32(4, true) + 8;
  if (riffEnd > reader.size || riffEnd < 12) throw new ValidationError('Truncated WAV file');
  let format: WavFormat | undefined, dataOffset = -1, dataLength = 0;
  for (let offset = 12; offset + 8 <= riffEnd;) {
    const chunk = new DataView(await readExact(reader, offset, 8, signal)), size = chunk.getUint32(4, true), start = offset + 8;
    if (start + size > riffEnd) throw new ValidationError('Truncated WAV chunk');
    const id = wavTag(chunk, 0);
    if (id === 'fmt ') format = parseWavFormat(new DataView(await readExact(reader, start, Math.min(size, 40), signal)), 0, size);
    else if (id === 'data' && dataOffset === -1) { dataOffset = start; dataLength = size; }
    offset = start + size + (size & 1);
  }
  return wavInfo(format, dataOffset, dataLength);
}

class PcmWindow {
  start = 0;
  channels: Float32Array[];
  constructor(count: number) { this.channels = Array.from({ length: count }, () => new Float32Array(0)); }
  get end(): number { return this.start + this.channels[0].length; }
  append(channels: Float32Array[]): void {
    this.channels = this.channels.map((c, i) => { const next = new Float32Array(c.length + channels[i].length); next.set(c); next.set(channels[i], c.length); return next; });
  }
  read(start: number, end: number): Float32Array[] {
    if (start < this.start || end > this.end) throw new ValidationError('Missing resampling context');
    return this.channels.map(c => c.slice(start - this.start, end - this.start));
  }
  discardBefore(start: number): void {
    const next = Math.min(this.end, Math.max(this.start, start));
    this.channels = this.channels.map(c => c.slice(next - this.start));
    this.start = next;
  }
}
async function convertRange(read: (start: number, end: number) => Promise<Float32Array[]>, sourceRate: number, targetRate: number, sourceLength: number, start: number, end: number, pool: OfflinePool): Promise<Float32Array[]> {
  if (sourceRate === targetRate) return read(start, Math.min(end, sourceLength));
  const tiles = Math.min(pool.size, Math.max(1, Math.ceil((end - start) / 8192)));
  let output: Float32Array[] | undefined;
  await pool.map(tiles, async (tile, worker) => {
    const a = start + Math.floor((end - start) * tile / tiles), b = start + Math.floor((end - start) * (tile + 1) / tiles);
    const [inputBase, inputEnd] = resampleInputRange(sourceRate, targetRate, sourceLength, a, b);
    const channels = await read(inputBase, inputEnd);
    const result = await pool.run<Float32Array[]>(worker, { type: 'resample', tile: { channels, sourceRate, targetRate, sourceLength, inputBase, outputStart: a, outputEnd: b } }, channels.map(c => c.buffer as ArrayBuffer));
    output ??= result.map(() => new Float32Array(end - start));
    result.forEach((c, i) => output![i].set(c, a - start));
  });
  return output!;
}

/** Bounded WAV processing. Encoded blocks are yielded under consumer backpressure. */
export async function* enhanceWavChunks(reader: AudioReader, options: EnhanceOptions, pool: OfflinePool): AsyncGenerator<Uint8Array> {
  checkAbort(options.signal);
  options.onProgress?.(0);
  const info = await readWavInfo(reader, options.signal);
  yield encodeWavHeader(info.channelCount, info.sampleRate, info.length);
  checkAbort(options.signal);
  if (info.length === 0) { options.onProgress?.(1); checkAbort(options.signal); return; }
  const session = await createInferenceSession(pool, info.channelCount, options, info.length / info.sampleRate);
  const model = session.info;
  const inputLength = Math.ceil(info.length * model.sampleRate / info.sampleRate);
  const frames = Math.ceil((inputLength + model.latency) / model.hopSize), tileFrames = session.framesPerTile;
  const window = new PcmWindow(info.channelCount);
  const readSource = async (start: number, end: number): Promise<Float32Array[]> => {
    const bytes = await readExact(reader, info.dataOffset + start * info.blockAlign, (end - start) * info.blockAlign, options.signal);
    return decodeWavSamples(new DataView(bytes), info).channels as Float32Array[];
  };
  const outputTile = Math.max(1, Math.floor(OFFLINE_TILE_SAMPLES * info.sampleRate / model.sampleRate));
  let written = 0;
  for (let f = 0; f < frames; f += tileFrames) {
    checkAbort(options.signal);
    const start = f * model.hopSize, count = Math.min(tileFrames, frames - f) * model.hopSize;
    const available = Math.min(inputLength, start + count);
    const samples = start < inputLength ? await convertRange(readSource, info.sampleRate, model.sampleRate, info.length, start, available, pool) : Array.from({ length: info.channelCount }, () => new Float32Array(0));
    const outputStart = start - model.latency, from = Math.max(0, -outputStart), to = Math.min(count, inputLength - outputStart);
    const produced = Array.from({ length: info.channelCount }, () => new Float32Array(Math.max(0, to - from)));
    const padded = samples.map(c => { const value = new Float32Array(count); value.set(c); return value; });
    const result = await session.process(padded);
    if (to > from) result.forEach((c, key) => produced[key].set(c.subarray(from, to)));
    window.append(produced);
    while (written < info.length) {
      const end = Math.min(info.length, written + outputTile);
      const [, neededEnd] = resampleInputRange(model.sampleRate, info.sampleRate, inputLength, written, end);
      if (neededEnd > window.end) break;
      const channels = await convertRange(async (a, b) => window.read(a, b), model.sampleRate, info.sampleRate, inputLength, written, end, pool);
      yield encodePcm16({ channels, sampleRate: info.sampleRate });
      checkAbort(options.signal);
      written = end;
      const [retain] = resampleInputRange(model.sampleRate, info.sampleRate, inputLength, written, written + 1);
      window.discardBefore(retain);
    }
    options.onProgress?.(0.99 * Math.min(frames, f + tileFrames) / frames);
    checkAbort(options.signal);
  }
  if (written !== info.length) throw new ValidationError('Incomplete WAV output');
  options.onProgress?.(1);
  checkAbort(options.signal);
}
