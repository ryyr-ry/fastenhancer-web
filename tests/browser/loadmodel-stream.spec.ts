import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

test('loadModel→createStreamDenoiser フルパスE2E', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/loadmodel-stream-test.html`);
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
  expect(content).toContain('OK: modelSizeTiny');
  expect(content).toContain('OK: sampleRate48k');
  expect(content).toContain('OK: wasmBytesExist');
  expect(content).toContain('OK: weightDataExist');
  expect(content).toContain('OK: exportMapExist');
  expect(content).toContain('OK: createStreamDenoiserFn');
  expect(content).toContain('OK: outputStreamExists');
  expect(content).toContain('OK: outputStreamHasTracks');
  expect(content).toContain('OK: stateRunning');
  expect(content).toContain('OK: bypassOn');
  expect(content).toContain('OK: bypassOff');
  expect(content).toContain('OK: stateDestroyed');
});
