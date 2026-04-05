import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

const PLATFORM_LIMIT_RE =
  /Can't find variable: AudioContext|AudioContext is not defined|relaxed simd instructions not supported/;

test('E2E denoising verification: noise energy reduction', async ({ page, browserName }) => {
  test.setTimeout(30000);

  await page.goto(`${BASE_URL}/tests/browser/denoising-verify-test.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('log');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 25000 },
  );

  const logContent = await page.textContent('#log');

  if (logContent && PLATFORM_LIMIT_RE.test(logContent)) {
    expect(logContent).toContain('[FAIL]');
    return;
  }

  const data = await page.evaluate(() => (window as any).__verifyData);
  expect(data).toBeDefined();
  expect(data.error).toBeUndefined();

  // Core denoising assertions
  expect(data.allPassed).toBe(true);
  expect(data.checks.energyReduced).toBe(true);
  expect(data.checks.notIdentity).toBe(true);
  expect(data.checks.allFinite).toBe(true);
  expect(data.checks.hasSignal).toBe(true);
  expect(data.checks.enoughSamples).toBe(true);

  // Quantitative bounds
  expect(data.energyReductionRatio).toBeLessThan(0.85);
  expect(data.correlation).toBeLessThan(0.95);
  expect(data.outputRMS).toBeGreaterThan(0);
  expect(data.capturedSamples).toBeGreaterThan(24000);
});
