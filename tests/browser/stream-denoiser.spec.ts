import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

const PLATFORM_LIMIT_RE =
  /Can't find variable: AudioContext|AudioContext is not defined|relaxed simd instructions not supported/;

test('createStreamDenoiser() creates a real-time noise-removal stream', async ({ page, browserName }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/stream-denoiser-test.html`);
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
  expect(data.outputStreamType).toBe('MediaStream');
  expect(data.audioTrackCount).toBeGreaterThan(0);
  expect(data.stateAfterCreate).toBe('running');
  expect(data.stateAfterDestroy).toBe('destroyed');
  expect(data.workletBypass).toBe(true);
  expect(data.workletAgc).toBe(true);
  expect(data.workletHpf).toBe(true);
  expect(data.destroyedErrorName).toBe('DestroyedError');
  expect(data.bypassAfterDestroyThrew).toBe(true);
  expect(data.agcAfterDestroyThrew).toBe(true);
  expect(data.hpfAfterDestroyThrew).toBe(true);

  expect(errors).toHaveLength(0);
});
