import { describe, expect, it } from 'vitest';
import { loadModel, ValidationError, DestroyedError } from '../../../src/api/index.js';

function tone(frame: number, frequency: number): Float32Array {
  return Float32Array.from({ length: 512 }, (_, i) => 0.2 * Math.sin(2 * Math.PI * frequency * (frame * 512 + i) / 48000));
}

describe('independent stereo states (real WASM)', () => {
  for (const size of ['tiny', 'base', 'small'] as const) {
    for (const simd of [false, true]) {
      it(`${size}/${simd ? 'simd' : 'scalar'} matches two independent mono engines`, async () => {
        const model = await loadModel(size, { simd });
        const stereo = await model.createChannelDenoiser(2);
        const left = await model.createDenoiser({ isolated: true });
        const right = await model.createDenoiser({ isolated: true });
        try {
          for (let frame = 0; frame < 4; frame++) {
            const inputs = [tone(frame, 400), tone(frame, 1700)];
            const output = stereo.processFrame(inputs);
            expect(output[0]).toEqual(left.processFrame(inputs[0]));
            expect(output[1]).toEqual(right.processFrame(inputs[1]));
          }
        } finally { stereo.destroy(); left.destroy(); right.destroy(); }
      });
    }
  }

  it('does not leak a left signal into a silent right channel', async () => {
    const stereo = await (await loadModel('tiny')).createChannelDenoiser();
    try {
      let leftEnergy = 0;
      for (let frame = 0; frame < 12; frame++) {
        const output = stereo.processFrame([tone(frame, 400), new Float32Array(512)]);
        leftEnergy += output[0].reduce((sum, v) => sum + v * v, 0);
        expect(output[1].every(v => v === 0)).toBe(true);
      }
      expect(leftEnergy).toBeGreaterThan(0);
    } finally { stereo.destroy(); }
  });

  it('rejects invalid right input before advancing the left state', async () => {
    const model = await loadModel('tiny');
    const stereo = await model.createChannelDenoiser();
    const reference = await model.createChannelDenoiser();
    try {
      const right = tone(0, 1700); right[0] = NaN;
      expect(() => stereo.processFrame([tone(0, 400), right])).toThrow(ValidationError);
      expect(() => stereo.processFrame([tone(0, 400)])).toThrow(ValidationError);
      const inputs = [tone(0, 400), tone(0, 1700)];
      expect(stereo.processFrame(inputs)).toEqual(reference.processFrame(inputs));
      stereo.reset();
      reference.reset();
      expect(stereo.processFrame(inputs)).toEqual(reference.processFrame(inputs));
    } finally { stereo.destroy(); reference.destroy(); }
    expect(() => stereo.reset()).toThrow(DestroyedError);
  });

  it('bypasses both channels without changing caller buffers', async () => {
    const stereo = await (await loadModel('tiny')).createChannelDenoiser();
    try {
      stereo.bypass = true;
      const inputs = [tone(0, 300), tone(0, 900)];
      const output = stereo.processFrame(inputs);
      expect(output).toEqual(inputs);
      expect(output[0]).not.toBe(inputs[0]);
      stereo.destroy();
      stereo.destroy();
      expect(() => stereo.processFrame(inputs)).toThrow(DestroyedError);
    } finally { stereo.destroy(); }
  });

  it('cleans up the first engine if the second cannot initialize', async () => {
    const model = await loadModel('tiny');
    const first = await model.createDenoiser({ isolated: true });
    let calls = 0;
    const failingModel = { ...model, createDenoiser: async () => {
      if (calls++ === 0) return first;
      throw new Error('second engine unavailable');
    } };
    await expect((await import('../../../src/api/channel-denoiser.js')).createChannelDenoiser(failingModel)).rejects.toThrow('second engine unavailable');
    expect(first.state).toBe('destroyed');
  });
});
