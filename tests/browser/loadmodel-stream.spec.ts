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
