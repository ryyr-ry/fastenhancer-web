/// <reference types="node" />
import { readFile, writeFile, rename, rm } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { enhancePcm } from './enhance.js';
import { checkAbort, type EnhanceOptions } from './offline-pcm.js';
import { decodeWav, encodeWav } from './wav.js';

/** Node/Bun: read a mono/stereo WAV, enhance it, and save a PCM16 WAV. */
export async function enhanceFile(
  inputPath: string | URL,
  outputPath: string | URL,
  options: EnhanceOptions = {},
): Promise<void> {
  checkAbort(options.signal);
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
