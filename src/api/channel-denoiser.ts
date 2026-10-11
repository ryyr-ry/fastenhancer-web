import { DestroyedError, ValidationError } from './errors.js';
import type { Denoiser } from './index.js';
import type { LoadedModel } from './loader.js';

export type ChannelCount = 1 | 2;

/** Planar mono/stereo frames. Each channel has its own GRU and STFT state. */
export interface ChannelDenoiser {
  readonly channelCount: ChannelCount;
  readonly hopSize: number;
  readonly sampleRate: number;
  readonly latencySamples: number;
  readonly state: 'ready' | 'destroyed';
  processFrame(channels: readonly Float32Array[]): Float32Array[];
  bypass: boolean;
  agcEnabled: boolean;
  hpfEnabled: boolean;
  reset(): void;
  destroy(): void;
}

export async function createChannelDenoiser(
  model: LoadedModel,
  channelCount: ChannelCount = 2,
): Promise<ChannelDenoiser> {
  if (channelCount !== 1 && channelCount !== 2) {
    throw new ValidationError('channelCount must be 1 (mono) or 2 (stereo)');
  }
  const denoisers: Denoiser[] = [];
  try {
    for (let c = 0; c < channelCount; c++) denoisers.push(await model.createDenoiser({ isolated: true }));
  } catch (error) {
    denoisers.forEach(d => d.destroy());
    throw error;
  }
  let destroyed = false;
  const ensureReady = (): void => {
    if (destroyed) throw new DestroyedError('Channel denoiser is destroyed');
  };
  return {
    channelCount,
    hopSize: model.hopSize,
    sampleRate: model.sampleRate,
    // Streaming STFT shifts by nFft - hopSize samples (currently one hop).
    latencySamples: model.nFft - model.hopSize,
    get state() { return destroyed ? 'destroyed' : 'ready'; },
    processFrame(channels) {
      ensureReady();
      if (channels.length !== channelCount) {
        throw new ValidationError(`Expected ${channelCount} channels, got ${channels.length}`);
      }
      // Validate every channel before advancing any temporal state.
      for (const channel of channels) {
        if (!(channel instanceof Float32Array) || channel.length !== model.hopSize) {
          throw new ValidationError(`Each channel must contain ${model.hopSize} Float32 samples`);
        }
        if (!channel.every(Number.isFinite)) {
          throw new ValidationError('Input contains non-finite samples');
        }
      }
      return channels.map((channel, c) => denoisers[c].processFrame(channel));
    },
    get bypass() { return denoisers[0].bypass; },
    set bypass(value) { ensureReady(); denoisers.forEach(d => { d.bypass = value; }); },
    get agcEnabled() { return denoisers[0].agcEnabled; },
    set agcEnabled(value) { ensureReady(); denoisers.forEach(d => { d.agcEnabled = value; }); },
    get hpfEnabled() { return denoisers[0].hpfEnabled; },
    set hpfEnabled(value) { ensureReady(); denoisers.forEach(d => { d.hpfEnabled = value; }); },
    reset() { ensureReady(); denoisers.forEach(d => d.reset()); },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      denoisers.forEach(d => d.destroy());
    },
  };
}
