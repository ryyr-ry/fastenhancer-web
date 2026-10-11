import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/tests/browser/offline-test.html');
});

for (const size of ['tiny', 'base', 'small'] as const) {
  for (const simd of [false, true]) {
    test(`stereo AudioWorklet ${size}/${simd ? 'simd' : 'scalar'} matches independent engines`, async ({ page }) => {
      const result = await page.evaluate(async ({ size, simd }) => {
        const modulePath = '/dist/api/index.js';
        const { loadModel } = await import(modulePath);
        const model = await loadModel(size, { simd });
        // Keep DSP correctness independent of CPU timing: four frames cannot
        // trigger the existing five-overrun real-time fallback policy.
        const length = 2048;
        const context = new OfflineAudioContext(2, length, 48000);
        const buffer = context.createBuffer(2, length, 48000);
        for (let i = 0; i < length; i++) {
          buffer.getChannelData(0)[i] = 0.2 * Math.sin(2 * Math.PI * 400 * i / 48000);
          buffer.getChannelData(1)[i] = 0.1 * Math.sin(2 * Math.PI * 1700 * i / 48000);
        }
        await context.audioWorklet.addModule('/dist/worklet/processor.js');
        const node = new AudioWorkletNode(context, 'fastenhancer-processor', {
          numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
          channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete',
          processorOptions: { channelCount: 2 },
        });
        const bypassEvents: boolean[] = [];
        const ready = new Promise<void>((resolve, reject) => {
          node.port.onmessage = event => {
            if (event.data.type === 'auto_bypass') bypassEvents.push(event.data.enabled);
            if (event.data.type === 'ready') resolve();
            if (event.data.type === 'error') reject(new Error(event.data.message));
          };
        });
        node.port.postMessage({ type: 'init', wasmBytes: model.wasmBytes,
          weightBytes: model.weightData, exportMap: model.exportMap, modelSize: model.modelSizeId, channelCount: 2 });
        await ready;
        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(node).connect(context.destination);
        source.start();
        const rendered = await context.startRendering();
        const errors: number[] = [];
        for (let c = 0; c < 2; c++) {
          const reference = await model.createDenoiser({ isolated: true });
          try {
            const raw = new Float32Array(length);
            for (let f = 0; f < length / 512; f++) {
              raw.set(reference.processFrame(buffer.getChannelData(c).slice(f * 512, (f + 1) * 512)), f * 512);
            }
            // Worklet buffers three 128-sample quanta before the first 512-sample frame.
            let maxError = 0;
            for (let i = 384; i < length; i++) maxError = Math.max(maxError, Math.abs(rendered.getChannelData(c)[i] - raw[i - 384]));
            errors.push(maxError);
          } finally { reference.destroy(); }
        }
        node.port.postMessage({ type: 'destroy' });
        node.disconnect();
        return { channels: rendered.numberOfChannels, errors, bypassEvents };
      }, { size, simd });
      expect(result.channels).toBe(2);
      expect(result.bypassEvents).toEqual([]);
      expect(result.errors[0]).toBeLessThan(1e-6);
      expect(result.errors[1]).toBeLessThan(1e-6);
    });
  }
}

test('module Worker returns a complete stereo WAV without detaching input', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const apiPath = '/dist/api/index.js', wavPath = '/dist/api/wav.js';
    const { enhanceAudio } = await import(apiPath);
    const { encodeWav, decodeWav } = await import(wavPath);
    const length = 4411;
    const left = Float32Array.from({ length }, (_, i) => 0.2 * Math.sin(2 * Math.PI * 400 * i / 44100));
    const input = new Blob([encodeWav({ channels: [left, new Float32Array(length)], sampleRate: 44100 })]);
    const progress: number[] = [];
    const output = await enhanceAudio(input, { model: 'tiny', worker: true, onProgress: (p: number) => progress.push(p) });
    const decoded = decodeWav(await output.arrayBuffer());
    return {
      mime: output.type, rate: decoded.sampleRate, lengths: decoded.channels.map((c: Float32Array) => c.length),
      silentRight: decoded.channels[1].every((v: number) => v === 0),
      hasLeft: decoded.channels[0].some((v: number) => Math.abs(v) > 1e-5),
      originalLength: left.length, progress,
    };
  });
  expect(result).toMatchObject({ mime: 'audio/wav', rate: 44100, lengths: [4411, 4411], silentRight: true, hasLeft: true, originalLength: 4411 });
  expect(result.progress[0]).toBe(0);
  expect(result.progress.at(-1)).toBe(1);
  expect(result.progress).toEqual([...result.progress].sort((a, b) => a - b));
});

test('stereo bypass retains distinct left/right signals', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const apiPath = '/dist/api/index.js';
    const { loadModel } = await import(apiPath);
    const model = await loadModel('tiny');
    const context = new OfflineAudioContext(2, 2048, 48000);
    const buffer = context.createBuffer(2, 2048, 48000);
    buffer.getChannelData(0).fill(0.2);
    buffer.getChannelData(1).fill(-0.1);
    await context.audioWorklet.addModule('/dist/worklet/processor.js');
    const node = new AudioWorkletNode(context, 'fastenhancer-processor', {
      outputChannelCount: [2], channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete',
      processorOptions: { channelCount: 2 },
    });
    const ready = new Promise<void>((resolve, reject) => {
      node.port.onmessage = event => {
        if (event.data.type === 'ready') resolve();
        if (event.data.type === 'error') reject(new Error(event.data.message));
      };
    });
    node.port.postMessage({ type: 'init', wasmBytes: model.wasmBytes, weightBytes: model.weightData,
      exportMap: model.exportMap, modelSize: 0, channelCount: 2 });
    await ready;
    // Controls are asynchronous: wait for the port's state response before rendering.
    const bypassReady = new Promise<void>((resolve, reject) => {
      node.port.onmessage = event => {
        if (event.data.type === 'state' && event.data.requestId === 'bypass-ready') {
          if (event.data.bypass) resolve();
          else reject(new Error('Bypass was not enabled'));
        }
      };
    });
    node.port.postMessage({ type: 'set_bypass', enabled: true });
    node.port.postMessage({ type: 'get_state', requestId: 'bypass-ready' });
    await bypassReady;
    const source = context.createBufferSource(); source.buffer = buffer;
    source.connect(node).connect(context.destination); source.start();
    const output = await context.startRendering();
    node.port.postMessage({ type: 'destroy' }); node.disconnect();
    return [output.getChannelData(0)[1024], output.getChannelData(1)[1024]];
  });
  expect(result[0]).toBeCloseTo(0.2, 6);
  expect(result[1]).toBeCloseTo(-0.1, 6);
});

test('stream adapter infers stereo and retains the injected AudioContext', async ({ page }) => {
  await page.locator('body').click();
  const result = await page.evaluate(async () => {
    const apiPath = '/dist/api/index.js';
    const { loadModel } = await import(apiPath);
    const context = new AudioContext({ sampleRate: 48000 });
    const input = context.createMediaStreamDestination();
    input.channelCount = 2;
    const model = await loadModel('tiny');
    const denoiser = await model.createStreamDenoiser(input.stream, { audioContext: context });
    try {
      const state = await denoiser.getWorkletState();
      await denoiser.destroyAsync();
      return { channels: denoiser.channelCount, initialized: state.initialized, contextAlive: context.state !== 'closed' };
    } finally { await context.close(); input.stream.getTracks().forEach(track => track.stop()); }
  });
  expect(result).toEqual({ channels: 2, initialized: true, contextAlive: true });
});

test('enhanceBuffer preserves stereo metadata and input samples', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const apiPath = '/dist/api/index.js';
    const { enhanceBuffer } = await import(apiPath);
    const input = new AudioBuffer({ length: 1703, sampleRate: 48000, numberOfChannels: 2 });
    input.getChannelData(1).set(Float32Array.from({ length: 1703 }, (_, i) => 0.2 * Math.sin(2 * Math.PI * 400 * i / 48000)));
    const original = input.getChannelData(1).slice();
    const output = await enhanceBuffer(input, { model: 'tiny' });
    return {
      channels: output.numberOfChannels, length: output.length, rate: output.sampleRate,
      silentLeft: output.getChannelData(0).every(v => v === 0),
      hasRight: output.getChannelData(1).some(v => Math.abs(v) > 1e-6),
      untouched: input.getChannelData(1).every((v, i) => v === original[i]),
    };
  });
  expect(result).toEqual({ channels: 2, length: 1703, rate: 48000, silentLeft: true, hasRight: true, untouched: true });
});

test('terminates a cancelled Worker and can process another job', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const apiPath = '/dist/api/index.js';
    const { enhancePcm } = await import(apiPath);
    const controller = new AbortController();
    let errorName = '';
    try {
      await enhancePcm({ channels: [new Float32Array(480000)], sampleRate: 48000 }, {
        signal: controller.signal, worker: true, onProgress: () => controller.abort(),
      });
    } catch (error) { errorName = (error as Error).name; }
    const output = await enhancePcm({ channels: [new Float32Array(513)], sampleRate: 48000 }, { model: 'tiny', worker: true });
    return { errorName, length: output.channels[0].length };
  });
  expect(result).toEqual({ errorName: 'AbortError', length: 513 });
});

test('enhances a browser-decodable compressed recording', async ({ page }) => {
  const supported = await page.evaluate(() => typeof MediaRecorder !== 'undefined'
    && MediaRecorder.isTypeSupported('audio/webm;codecs=opus'));
  test.skip(!supported, 'This browser cannot generate the WebM/Opus fixture');
  await page.locator('body').click();
  const result = await page.evaluate(async () => {
    const apiPath = '/dist/api/index.js', wavPath = '/dist/api/wav.js';
    const { enhanceAudio } = await import(apiPath);
    const { decodeWav } = await import(wavPath);
    const context = new AudioContext({ sampleRate: 48000 });
    const destination = context.createMediaStreamDestination();
    const oscillator = context.createOscillator();
    oscillator.frequency.value = 400;
    oscillator.connect(destination); oscillator.start();
    let recording: Blob;
    try {
      await context.resume();
      recording = await new Promise<Blob>((resolve, reject) => {
        const chunks: Blob[] = [];
        const recorder = new MediaRecorder(destination.stream, { mimeType: 'audio/webm;codecs=opus' });
        recorder.ondataavailable = event => chunks.push(event.data);
        recorder.onerror = () => reject(new Error('Could not create compressed fixture'));
        recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType }));
        recorder.start(); setTimeout(() => recorder.stop(), 250);
      });
    } finally { oscillator.stop(); destination.stream.getTracks().forEach(track => track.stop()); await context.close(); }
    const decoded = await new OfflineAudioContext(2, 1, 48000).decodeAudioData(await recording.arrayBuffer());
    const output = decodeWav(await (await enhanceAudio(recording, { model: 'tiny' })).arrayBuffer());
    return { expectedLength: decoded.length, expectedChannels: decoded.numberOfChannels,
      length: output.channels[0].length, channels: output.channels.length, rate: output.sampleRate };
  });
  expect(result.length).toBeGreaterThan(0);
  expect(result.length).toBe(result.expectedLength);
  expect(result.channels).toBe(result.expectedChannels);
  expect(result.rate).toBe(48000);
});
