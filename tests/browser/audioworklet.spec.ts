import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

test('AudioWorklet WASM initialization and audio processing', async ({ page }) => {
  const logs: string[] = [];
  const errors: string[] = [];

  page.on('console', (msg) => logs.push(msg.text()));
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/test-page.html`);

  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('log');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 30000 }
  );

  const logContent = await page.textContent('#log');

  expect(logContent).toContain('[PASS]');
  expect(logContent).toContain('initSuccess');
  expect(logContent).toContain('hopSizeCorrect');
  expect(logContent).toContain('hasProcessedFrames');
  expect(logContent).toContain('noNaN');
  expect(logContent).toContain('noInf');
});

test('AudioWorklet destroy safety', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/test-page.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('log');
      return el && el.textContent!.includes('Cleanup complete');
    },
    { timeout: 30000 }
  );

  const logContent = await page.textContent('#log');
  expect(logContent).toContain('Cleanup complete');
  expect(errors).toHaveLength(0);
});

test('AudioWorklet performance measurement', async ({ page }) => {
  const logs: string[] = [];
  page.on('console', (msg) => logs.push(msg.text()));

  await page.goto(`${BASE_URL}/tests/browser/perf-page.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('log');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 30000 }
  );

  const logContent = await page.textContent('#log');

  expect(logContent).toContain('[PASS]');
  expect(logContent).toContain('enoughFrames');
  expect(logContent).toContain('medianUnderBudget');
});
