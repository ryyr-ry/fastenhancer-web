import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

/**
 * Helper that waits until the data-state attribute reaches the specified value
 */
async function waitForState(page: import('@playwright/test').Page, expectedState: string, timeout = 15000) {
  await page.waitForFunction(
    (s) => document.getElementById('hook-state')?.getAttribute('data-state') === s,
    expectedState,
    { timeout }
  );
}

test.describe('useDenoiser React Hook E2E', () => {
  test('starts in idle and transitions start → processing → stop → idle', async ({ page, browserName }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`${BASE_URL}/tests/browser/react-test.html`);

    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'ready',
      { timeout: 10000 }
    );

    const hookEl = page.locator('#hook-state');

    await expect(hookEl).toHaveAttribute('data-state', 'idle');
    await expect(hookEl).toHaveAttribute('data-has-stream', 'false');
    await expect(hookEl).toHaveAttribute('data-bypass', 'false');

    const hasAudioContext = await page.evaluate(() => typeof AudioContext !== 'undefined');

    if (!hasAudioContext) {
      // Platform lacks AudioContext — verify start attempt doesn't crash
      await page.click('#btn-start');
      await page.waitForTimeout(2000);
      await expect(hookEl).toHaveAttribute('data-state', 'idle');
      expect(errors).toHaveLength(0);
      return;
    }

    await page.click('#btn-start');
    await waitForState(page, 'processing');

    await expect(hookEl).toHaveAttribute('data-has-stream', 'true');

    await page.click('#btn-stop');
    await waitForState(page, 'idle');

    await expect(hookEl).toHaveAttribute('data-has-stream', 'false');

    expect(errors).toHaveLength(0);
  });

  test('reflects bypass toggling in state', async ({ page, browserName }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`${BASE_URL}/tests/browser/react-test.html`);
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'ready',
      { timeout: 10000 }
    );

    const hookEl = page.locator('#hook-state');
    const hasAudioContext = await page.evaluate(() => typeof AudioContext !== 'undefined');

    if (!hasAudioContext) {
      // Without AudioContext, start() cannot proceed — verify no crash
      await page.click('#btn-start');
      await page.waitForTimeout(2000);
      await expect(hookEl).toHaveAttribute('data-state', 'idle');
      expect(errors).toHaveLength(0);
      return;
    }

    await page.click('#btn-start');
    await waitForState(page, 'processing');

    await page.click('#btn-bypass-on');
    await expect(hookEl).toHaveAttribute('data-bypass', 'true');

    await page.click('#btn-bypass-off');
    await expect(hookEl).toHaveAttribute('data-bypass', 'false');

    await page.click('#btn-stop');
    await waitForState(page, 'idle');

    expect(errors).toHaveLength(0);
  });

  test('transitions to destroyed state on destroy', async ({ page, browserName }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`${BASE_URL}/tests/browser/react-test.html`);
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'ready',
      { timeout: 10000 }
    );

    const hasAudioContext = await page.evaluate(() => typeof AudioContext !== 'undefined');

    if (!hasAudioContext) {
      // Verify destroy works from idle when AudioContext is unavailable
      await page.click('#btn-destroy');
      await waitForState(page, 'destroyed');
      expect(errors).toHaveLength(0);
      return;
    }

    await page.click('#btn-start');
    await waitForState(page, 'processing');

    await page.click('#btn-destroy');
    await waitForState(page, 'destroyed');

    expect(errors).toHaveLength(0);
  });

  test('keeps destroyed state terminal after destroy', async ({ page, browserName }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`${BASE_URL}/tests/browser/react-test.html`);
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'ready',
      { timeout: 10000 }
    );

    await page.click('#btn-destroy');
    await waitForState(page, 'destroyed');

    const hookEl = page.locator('#hook-state');

    await page.click('#btn-start');
    await expect(hookEl).toHaveAttribute('data-state', 'destroyed');
    await expect(hookEl).toHaveAttribute('data-error', '');

    await page.click('#btn-stop');
    await expect(hookEl).toHaveAttribute('data-state', 'destroyed');

    await page.click('#btn-bypass-on');
    await expect(hookEl).toHaveAttribute('data-bypass', 'false');

    expect(errors).toHaveLength(0);
  });
});
