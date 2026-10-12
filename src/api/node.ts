/// <reference types="node" />
import { open, readFile, writeFile, rename, rm } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { enhancePcm as enhancePortablePcm } from './enhance.js';
import { checkAbort, validatePcm, type EnhanceOptions, type PcmAudio } from './offline-pcm.js';
import { createNodePool, enhanceNodeParallel } from './offline-node.js';
import { enhanceWavChunks } from './offline-wav.js';
import { decodeWav, encodeWav } from './wav.js';

export type { EnhanceOptions, PcmAudio } from './offline-pcm.js';

/** Node/Bun PCM adapter, including opt-in Worker threads. */
export async function enhancePcm(audio: PcmAudio, options: EnhanceOptions = {}): Promise<PcmAudio> {
  checkAbort(options.signal);
  validatePcm(audio);
  if (options.parallel !== undefined && options.parallel !== false) return enhanceNodeParallel(audio, options);
  return enhancePortablePcm(audio, options);
}

/** Node/Bun: read a mono/stereo WAV, enhance it, and save a PCM16 WAV. */
export async function enhanceFile(
  inputPath: string | URL,
  outputPath: string | URL,
  options: EnhanceOptions = {},
): Promise<void> {
  checkAbort(options.signal);
  if (options.parallel !== undefined && options.parallel !== false) {
    await enhanceFileStreaming(inputPath, outputPath, options);
    return;
  }
  const bytes = await readFile(inputPath, { signal: options.signal });
  const audio = decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const clean = await enhancePcm(audio, options);
  checkAbort(options.signal);
  const destination = typeof outputPath === 'string' ? outputPath : fileURLToPath(outputPath);
  const temporary = join(dirname(destination), `.${basename(destination)}.${randomUUID()}.tmp`);
  try {
    await writeFile(temporary, new Uint8Array(encodeWav(clean)), { signal: options.signal, flag: 'wx' });
    checkAbort(options.signal);
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { force: true });
  }
}

async function enhanceFileStreaming(inputPath: string | URL, outputPath: string | URL, options: EnhanceOptions): Promise<void> {
  const destination = typeof outputPath === 'string' ? outputPath : fileURLToPath(outputPath);
  const temporary = join(dirname(destination), `.${basename(destination)}.${randomUUID()}.tmp`);
  const pool = await createNodePool(options);
  let input: Awaited<ReturnType<typeof open>> | undefined, output: Awaited<ReturnType<typeof open>> | undefined, created = false;
  try {
    input = await open(inputPath, 'r');
    const size = (await input.stat()).size;
    output = await open(temporary, 'wx'); created = true;
    const reader = { size, async read(offset: number, length: number): Promise<ArrayBuffer> {
      const bytes = new Uint8Array(length);
      let position = 0;
      while (position < length) {
        checkAbort(options.signal);
        const { bytesRead } = await input!.read(bytes, position, length - position, offset + position);
        if (!bytesRead) break;
        position += bytesRead;
      }
      return bytes.buffer.slice(0, position);
    } };
    for await (const chunk of enhanceWavChunks(reader, options, pool)) {
      let position = 0;
      while (position < chunk.byteLength) {
        checkAbort(options.signal);
        const { bytesWritten } = await output.write(chunk, position, chunk.byteLength - position);
        if (!bytesWritten) throw new Error('Could not write enhanced WAV');
        position += bytesWritten;
      }
    }
    checkAbort(options.signal);
    await input.close(); input = undefined;
    await output.close(); output = undefined;
    checkAbort(options.signal);
    await rename(temporary, destination);
  } finally {
    try {
      await Promise.allSettled([input?.close(), output?.close()]);
      if (created) await rm(temporary, { force: true });
    } finally { await pool.close(); }
  }
}
