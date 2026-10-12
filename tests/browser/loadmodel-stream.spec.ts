import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

const PLATFORM_LIMIT_RE =
  /Can't find variable: AudioContext|AudioContext is not defined|relaxed simd instructions not supported/;

test('loadModel → createStreamDenoiser full-path E2E', async ({ page, browserName }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/loadmodel-stream-test.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('result');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 30000 },
  );

  const resultContent = await page.textContent('#result');

  if (resultContent && PLATFORM_LIMIT_RE.test(resultContent)) {
    expect(resultContent).toContain('[FAIL]');
    return;
  }

  // Direct property verification via evaluate()
  const data = await page.evaluate(() => (window as any).__testData);
  expect(data).toBeDefined();
  expect(data.completed).toBe(true);

  // Model properties
  expect(data.size).toBe('tiny');
  expect(data.sampleRate).toBe(48000);
  expect(data.hopSize).toBe(512);
  expect(data.nFft).toBe(1024);
  expect(data.wasmBytesLength).toBeGreaterThan(0);
  expect(data.weightDataLength).toBeGreaterThan(0);
  expect(data.exportMapKeys.length).toBeGreaterThan(0);
  expect(data.hasWasmFactory).toBe(true);
  expect(data.hasCreateDenoiser).toBe(true);
  expect(data.hasCreateStreamDenoiser).toBe(true);

  // Stream properties
  expect(data.streamOutputType).toBe('MediaStream');
  expect(data.streamTrackCount).toBeGreaterThan(0);
  expect(data.streamState).toBe('running');
  expect(data.stateAfterDestroy).toBe('destroyed');

  expect(errors).toHaveLength(0);
});

for (const simd of [false, true]) {
  for (const selfHosted of [false, true]) {
    test(`Small/${simd ? 'simd' : 'scalar'}: public stream with ${selfHosted ? 'static' : 'embedded'} Worklet keeps denoising`, async ({ page }) => {
      await page.goto('/tests/browser/offline-test.html');
      const result = await page.evaluate(async ({ simd, selfHosted }) => {
        const apiPath = '/dist/api/index.js';
        const { loadModel } = await import(apiPath);
        const model = await loadModel('small', { simd });
        const context = new AudioContext({ sampleRate: 48000 });
        await context.resume();
        const oscillator = context.createOscillator();
        const input = context.createMediaStreamDestination();
        input.channelCount = 1;
        oscillator.frequency.value = 400;
        oscillator.connect(input); oscillator.start();
        const events: boolean[] = [];
        let denoiser: any;
        try {
          denoiser = await model.createStreamDenoiser(input.stream, {
            audioContext: context, channelCount: 1,
            workletUrl: selfHosted ? '/dist/worklet/processor.js' : undefined,
            onAutoBypass: (enabled: boolean) => events.push(enabled),
          });
          let state = await denoiser.getWorkletState();
          for (let attempt = 0; attempt < 100 && (state.audioFrames ?? 0) < 80; attempt++) {
            await new Promise(resolve => setTimeout(resolve, 25));
            state = await denoiser.getWorkletState();
          }
          return { events, state, autoPassthrough: denoiser.autoPassthrough, channelCount: denoiser.channelCount,
            outputTracks: denoiser.outputStream.getAudioTracks().length };
        } finally {
          await denoiser?.destroyAsync();
          oscillator.stop(); oscillator.disconnect();
          input.stream.getTracks().forEach(track => track.stop());
          await context.close();
        }
      }, { simd, selfHosted });
      expect(result.events).toEqual([]);
      expect(result.autoPassthrough).toBe(false);
      expect(result.outputTracks).toBe(1);
      expect(result.channelCount).toBe(1);
      expect(result.state.audioFrames).toBeGreaterThanOrEqual(80);
      expect(result.state).toMatchObject({ performanceMode: 'processing', passthroughFrames: 0 });
      expect(result.state.averageProcessingMs).toBeGreaterThanOrEqual(0);
    });
  }
}
