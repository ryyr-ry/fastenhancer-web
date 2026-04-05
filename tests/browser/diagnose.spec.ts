import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

test('diagnose() detects full feature support in Chromium', async ({ page, browserName }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/diagnose-test.html`);

  await page.waitForFunction(
    () => {
      const el = document.getElementById('result');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 10000 }
  );

  const content = await page.textContent('#result');

  const hasAudioContext = await page.evaluate(
    () => typeof AudioContext !== 'undefined'
  );

  if (!hasAudioContext) {
    // Platform lacks AudioContext — verify diagnose correctly reports limitations
    expect(content).toContain('OK: wasm');
    expect(content).toContain('FAIL: audioContext');
    expect(content).toContain('[FAIL]');
    return;
  }

  expect(errors).toHaveLength(0);
  expect(content).toContain('[PASS]');
  expect(content).toContain('OK: wasm');
  expect(content).toContain('OK: simd');
  expect(content).toContain('OK: audioContext');
  expect(content).toContain('OK: audioWorklet');
  expect(content).toContain('OK: overall');
  expect(content).toContain('OK: issuesEmpty');
});

test('diagnose() returns issues (simulated environment without AudioWorklet support)', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(globalThis, 'AudioWorkletNode', {
      value: undefined,
      configurable: true,
      writable: true,
    });
  });

  await page.goto(`${BASE_URL}/tests/browser/diagnose-test.html`);

  await page.waitForFunction(
    () => {
      const el = document.getElementById('result');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 10000 }
  );

  const content = await page.textContent('#result');
  expect(content).toContain('FAIL: audioWorklet');
  expect(content).toContain('FAIL: overall');
  expect(content).toContain('[FAIL]');
});
