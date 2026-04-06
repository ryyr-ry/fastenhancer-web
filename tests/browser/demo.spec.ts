import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

/**
 * Demo page smoke tests:
 * Verify that the demo page loads successfully, renders expected UI elements,
 * and can initialize the denoiser engine through user interaction.
 */
test.describe('Demo page smoke tests', () => {
  test('demo page loads and renders all UI elements', async ({ page }) => {
    await page.goto(`${BASE_URL}/tests/browser/demo.html`);

    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('#modelSelect')).toBeVisible();
    await expect(page.locator('#sourceSelect')).toBeVisible();
    await expect(page.locator('#btnInit')).toBeVisible();
    await expect(page.locator('#btnPlay')).toBeVisible();
    await expect(page.locator('#btnStop')).toBeVisible();
    await expect(page.locator('#statusBadge')).toContainText('Not Initialized');
    await expect(page.locator('#chkBypass')).toBeVisible();
    await expect(page.locator('#chkAGC')).toBeVisible();
    await expect(page.locator('#chkHPF')).toBeVisible();
    await expect(page.locator('#log')).toBeVisible();
  });

  test('demo page initializes the denoiser on button click', async ({ page, browserName }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`${BASE_URL}/tests/browser/demo.html`);

    // Select the tiny model for fastest init
    await page.selectOption('#modelSelect', 'tiny');

    await page.click('#btnInit');

    // Wait for initialization to complete (look for status badge change)
    await page.waitForFunction(
      () => {
        const badge = document.getElementById('statusBadge');
        return badge && (badge.textContent!.includes('Ready') || badge.textContent!.includes('Error'));
      },
      { timeout: 30000 }
    );

    const logContent = await page.textContent('#log');
    expect(logContent).toBeTruthy();

    const status = await page.textContent('#statusBadge');

    // WebKit (Playwright) lacks AudioContext — initialization failure is expected
    if (browserName === 'webkit' && status!.includes('Error')) {
      expect(logContent).toContain('AudioContext');
      return;
    }

    expect(status).toContain('Ready');
    expect(errors).toHaveLength(0);
  });

  test('demo page model selector has all three options', async ({ page }) => {
    await page.goto(`${BASE_URL}/tests/browser/demo.html`);

    const options = await page.locator('#modelSelect option').allTextContents();
    const values = await page.locator('#modelSelect option').evaluateAll(
      (opts) => (opts as HTMLOptionElement[]).map(o => o.value)
    );

    expect(values).toContain('tiny');
    expect(values).toContain('base');
    expect(values).toContain('small');
    expect(options.length).toBe(3);
  });
});
