import { test, expect } from '@playwright/test';

test('WASM SINGLE_FILE loads and executes inside AudioWorklet', async ({ page }) => {
  const logs: string[] = [];

  page.on('console', (msg) => {
    logs.push(msg.text());
  });

  page.on('pageerror', (err) => {
    logs.push(`PAGE_ERROR: ${err.message}`);
  });

  await page.goto('http://localhost:3456/tests/feasibility/index.html');

  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const logEl = document.getElementById('log');
      return logEl && (logEl.textContent!.includes('[PASS]') || logEl.textContent!.includes('[FAIL]'));
    },
    { timeout: 10000 }
  );

  const logContent = await page.textContent('#log');

  expect(logContent).toContain('[PASS]');
  expect(logContent).toContain('add_two(1, 2) = 3');
  expect(logContent).toContain('multiply_f32(2, 3) = 6');
});
