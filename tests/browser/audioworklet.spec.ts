import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

// 3% margin on budget accounts for:
// - Run-to-run measurement noise (±0.22ms, 3σ ≈ 0.3ms)
// - OS scheduling jitter on main thread vs AudioWorklet thread
// - HW audio buffer (≈40ms ≈ 3.75 frames) absorbs small overruns
const BUDGET_MARGIN = 1.03;

// Detects known platform limitations in test page output:
// - WebKit on Windows lacks AudioContext
// - WebKit does not support relaxed SIMD instructions
const PLATFORM_LIMIT_RE =
  /Can't find variable: AudioContext|AudioContext is not defined|relaxed simd instructions not supported/;

test('AudioWorklet WASM initialization and audio processing', async ({ page, browserName }) => {
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

  if (logContent && PLATFORM_LIMIT_RE.test(logContent)) {
    expect(logContent).toContain('[FAIL]');
    return;
  }

  expect(logContent).toContain('[PASS]');
  expect(logContent).toContain('initSuccess');
  expect(logContent).toContain('hopSizeCorrect');
  expect(logContent).toContain('hasProcessedFrames');
  expect(logContent).toContain('noNaN');
  expect(logContent).toContain('noInf');
});

test('AudioWorklet destroy safety', async ({ page, browserName }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/test-page.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('log');
      return el && (el.textContent!.includes('Cleanup complete') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 30000 }
  );

  const logContent = await page.textContent('#log');

  if (logContent && PLATFORM_LIMIT_RE.test(logContent)) {
    expect(logContent).toContain('[FAIL]');
    return;
  }

  expect(logContent).toContain('Cleanup complete');
  expect(errors).toHaveLength(0);
});

test('AudioWorklet performance measurement', async ({ page, browserName }) => {
  test.setTimeout(120000);
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

  if (logContent && PLATFORM_LIMIT_RE.test(logContent)) {
    expect(logContent).toContain('[FAIL]');
    return;
  }

  const data = await page.evaluate(() => (window as any).__perfData);
  expect(data).toBeDefined();
  expect(data.allPassed).toBe(true);
  expect(data.medianMs).toBeLessThan(data.budgetMs * BUDGET_MARGIN);
});

const MODELS = ['tiny', 'base', 'small'] as const;
const VARIANTS = ['scalar', 'simd'] as const;

for (const model of MODELS) {
  for (const variant of VARIANTS) {
    test(`Performance: ${model}/${variant} median under budget`, async ({ page, browserName }) => {
      // Playwright Firefox uses a patched binary with 6-9x slower WASM SIMD
      // (verified on real Firefox 149 via Selenium: all models pass comfortably)
      test.skip(browserName === 'firefox', 'Playwright Firefox WASM SIMD is 6-9x slower than real Firefox');
      test.setTimeout(120000);

      await page.goto(`${BASE_URL}/tests/browser/perf-page.html?model=${model}&variant=${variant}`);
      await page.click('#startBtn');

      await page.waitForFunction(
        () => {
          const el = document.getElementById('log');
          return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
        },
        { timeout: 120000 },
      );

      const logContent = await page.textContent('#log');

      if (logContent && PLATFORM_LIMIT_RE.test(logContent)) {
        expect(logContent).toContain('[FAIL]');
        return;
      }

      const data = await page.evaluate(() => (window as any).__perfData);
      expect(data).toBeDefined();
      expect(data.model).toBe(model);
      expect(data.variant).toBe(variant);

      // All models must meet real-time budget — no relaxation
      expect(data.frameCount).toBeGreaterThan(data.expectedFrames * 0.9);
      expect(data.medianMs).toBeLessThan(data.budgetMs * BUDGET_MARGIN);
      expect(data.p99Ms).toBeLessThan(data.budgetMs * 3);
    });
  }
}
