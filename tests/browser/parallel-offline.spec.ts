import { test, expect } from '@playwright/test';

for (const model of ['tiny', 'base', 'small']) for (const count of [1, 2]) {
  test(`internal batch ${model}/${count} channels retains every sample`, async ({ page }) => {
    await page.goto('/tests/browser/offline-test.html');
    const result = await page.evaluate(async ({ model, count }) => {
      const { enhancePcm } = await import('/dist/api/enhance.js');
      const audio = { channels: Array.from({ length: count }, (_, c) => Float32Array.from({ length: 70001 }, (_, i) => 0.1 * Math.sin(i * (c + 1) * 0.07))), sampleRate: 48000 };
      const expected = await enhancePcm(audio, { model, simd: true, worker: false });
      const actual = await enhancePcm(audio, { model, simd: true, parallel: { engine: 'batch', maxWorkers: 3 } });
      let difference = 0;
      for (let c = 0; c < count; c++) for (let i = 0; i < 70001; i++) difference = Math.max(difference, Math.abs(actual.channels[c][i] - expected.channels[c][i]));
      return difference;
    }, { model, count });
    expect(result).toBe(0);
  });
}

for (const model of ['tiny', 'base', 'small']) for (const simd of [false, true]) {
  test(`parallel ${model}/${simd ? 'simd' : 'scalar'} retains exact stereo output and progress`, async ({ page }) => {
    await page.goto('/tests/browser/offline-test.html');
    const result = await page.evaluate(async ({ model, simd }) => {
      const { enhancePcm } = await import('/dist/api/enhance.js');
      const length = 70001, channels = [Float32Array.from({ length }, (_, i) => 0.1 * Math.sin(i * 0.07)), new Float32Array(length)];
      const expected = await enhancePcm({ channels, sampleRate: 44100 }, { model, simd, worker: false });
      const progress: number[] = [];
      const actual = await enhancePcm({ channels, sampleRate: 44100 }, { model, simd, parallel: { maxWorkers: 3 }, onProgress: (p: number) => progress.push(p) });
      let error = 0;
      for (let c = 0; c < 2; c++) for (let i = 0; i < length; i++) error = Math.max(error, Math.abs(expected.channels[c][i] - actual.channels[c][i]));
      return { error, progress, inputLengths: channels.map(c => c.length), outputLengths: actual.channels.map((c: Float32Array) => c.length) };
    }, { model, simd });
    expect(result.error).toBe(0); expect(result.inputLengths).toEqual([70001, 70001]);
    expect(result.outputLengths).toEqual([70001, 70001]);
    expect(result.progress[0]).toBe(0); expect(result.progress.at(-1)).toBe(1);
    expect(result.progress).toEqual([...result.progress].sort((a, b) => a - b));
  });
}
test('parallel WAV processing uses ranges and matches the legacy WAV bytes', async ({ page }) => {
  await page.goto('/tests/browser/offline-test.html');
  const result = await page.evaluate(async () => {
    const { enhanceAudio } = await import('/dist/api/enhance.js');
    const { encodeWav } = await import('/dist/api/wav.js');
    const bytes = encodeWav({ channels: [Float32Array.from({ length: 70001 }, (_, i) => 0.1 * Math.sin(i * 0.07)), new Float32Array(70001)], sampleRate: 44100 });
    const expected = new Uint8Array(await (await enhanceAudio(new Blob([bytes]), { model: 'tiny', worker: false })).arrayBuffer());
    let wholeReads = 0;
    class Input extends Blob { override arrayBuffer() { wholeReads++; return super.arrayBuffer(); } }
    const actual = new Uint8Array(await (await enhanceAudio(new Input([bytes]), { model: 'tiny', parallel: { engine: 'batch', maxWorkers: 3 } })).arrayBuffer());
    return { wholeReads, length: actual.length, differences: actual.reduce((n, value, i) => n + Number(value !== expected[i]), 0) };
  });
  expect(result.wholeReads).toBe(0); expect(result.length).toBe(280048); expect(result.differences).toBe(0);
});
test('cancellation and early stream return terminate all workers', async ({ page }) => {
  await page.goto('/tests/browser/offline-test.html');
  const result = await page.evaluate(async () => {
    const { enhancePcm, enhanceWavStream } = await import('/dist/api/enhance.js');
    const { encodeWav } = await import('/dist/api/wav.js');
    const Original = window.Worker; let active = 0, peak = 0;
    window.Worker = class extends Original {
      constructor(url: string | URL, options?: WorkerOptions) { super(url, options); active++; peak = Math.max(peak, active); }
      override terminate() { active--; super.terminate(); }
    };
    try {
      const controller = new AbortController(); let name = '';
      try { await enhancePcm({ channels: [new Float32Array(200001)], sampleRate: 48000 }, {
        model: 'small', parallel: { maxWorkers: 3 }, signal: controller.signal,
        onProgress: (p: number) => { if (p > 0.1) controller.abort(); },
      }); } catch (error) { name = (error as Error).name; }
      const afterCancel = active;
      const file = new Blob([encodeWav({ channels: [new Float32Array(70001)], sampleRate: 48000 })]);
      for await (const block of enhanceWavStream(file, { parallel: { maxWorkers: 3 } })) { if (block.length !== 44) throw new Error('header'); break; }
      const afterReturn = active;
      const next = await enhancePcm({ channels: [new Float32Array(513)], sampleRate: 48000 }, { model: 'tiny', parallel: { maxWorkers: 2 } });
      return { name, afterCancel, afterReturn, active, peak, nextLength: next.channels[0].length };
    } finally { window.Worker = Original; }
  });
  expect(result.name).toBe('AbortError'); expect(result.afterCancel).toBe(0); expect(result.afterReturn).toBe(0);
  expect(result.active).toBe(0); expect(result.peak).toBeLessThanOrEqual(3); expect(result.nextLength).toBe(513);
});
test('a CSP failure rejects parallel processing and inline processing remains usable', async ({ page }) => {
  await page.route('**/tests/browser/offline-test.html', async route => {
    const response = await route.fetch(); await route.fulfill({ response, headers: { ...response.headers(), 'content-security-policy': "worker-src 'none'" } });
  });
  await page.goto('/tests/browser/offline-test.html');
  const result = await page.evaluate(async () => {
    const { enhancePcm } = await import('/dist/api/enhance.js');
    const audio = { channels: [new Float32Array(513)], sampleRate: 48000 }; let message = '';
    try { await enhancePcm(audio, { parallel: {} }); } catch (error) { message = (error as Error).message; }
    const result = await enhancePcm(audio, { worker: false, model: 'tiny' });
    return { message, length: result.channels[0].length };
  });
  expect(result.message).toContain('Worker'); expect(result.length).toBe(513);
});
