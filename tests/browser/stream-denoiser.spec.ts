import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

test('createStreamDenoiser() creates a real-time noise-removal stream', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/stream-denoiser-test.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('result');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 30000 }
  );

  const content = await page.textContent('#result');

  expect(errors).toHaveLength(0);
  expect(content).toContain('[PASS]');
  expect(content).toContain('OK: outputStreamExists');
  expect(content).toContain('OK: outputStreamHasTracks');
  expect(content).toContain('OK: stateRunning');
  expect(content).toContain('OK: bypassOn');
  expect(content).toContain('OK: bypassOff');
  expect(content).toContain('OK: stateDestroyed');
  expect(content).toContain('OK: destroyIdempotent');
  expect(content).toContain('OK: bypassAfterDestroyThrows');
  expect(content).toContain('OK: destroyedErrorName');
  expect(content).toContain('OK: agcAfterDestroyThrows');
  expect(content).toContain('OK: hpfAfterDestroyThrows');
});
