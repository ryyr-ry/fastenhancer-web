import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

/**
 * CSP validation: verify that the library works without unsafe-eval
 *
 * Policy:
 *   script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'
 *   worker-src 'self'
 *   connect-src 'self'
 *
 * unsafe-inline is for inline scripts on the test page.
 * wasm-unsafe-eval is for WebAssembly.instantiate().
 * unsafe-eval is intentionally omitted to guarantee that eval()/new Function() are not used.
 */
const CSP_POLICY = [
  "default-src 'none'",
  "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'",
  "worker-src 'self'",
  "connect-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self'",
].join('; ');

test.describe('CSP compatibility validation', () => {
  test('succeeds in AudioWorklet + WASM initialization without unsafe-eval', async ({ page }) => {
    const cspViolations: string[] = [];
    const pageErrors: string[] = [];

    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.route('**/*', async (route) => {
      const response = await route.fetch();
      const headers = { ...response.headers() };
      headers['content-security-policy'] = CSP_POLICY;
      await route.fulfill({
        response,
        headers,
      });
    });

    page.on('console', (msg) => {
      if (msg.text().includes('Content-Security-Policy') ||
          msg.text().includes('EvalError') ||
          msg.text().includes('Refused to evaluate')) {
        cspViolations.push(msg.text());
      }
    });

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

    expect(cspViolations).toHaveLength(0);
    expect(pageErrors).toHaveLength(0);
    expect(logContent).toContain('[PASS]');
    expect(logContent).toContain('initSuccess');
    expect(logContent).toContain('hasProcessedFrames');
  });

  test('succeeds in StreamDenoiser creation without unsafe-eval', async ({ page }) => {
    const cspViolations: string[] = [];
    const pageErrors: string[] = [];

    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.route('**/*', async (route) => {
      const response = await route.fetch();
      const headers = { ...response.headers() };
      headers['content-security-policy'] = CSP_POLICY;
      await route.fulfill({
        response,
        headers,
      });
    });

    page.on('console', (msg) => {
      if (msg.text().includes('Content-Security-Policy') ||
          msg.text().includes('EvalError') ||
          msg.text().includes('Refused to evaluate')) {
        cspViolations.push(msg.text());
      }
    });

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

    expect(cspViolations).toHaveLength(0);
    expect(pageErrors).toHaveLength(0);
    expect(content).toContain('[PASS]');
  });

  test('blocks WASM instantiation without wasm-unsafe-eval', async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'Enforcement of CSP wasm-unsafe-eval is Chromium-specific behavior');
    const STRICT_CSP = [
      "default-src 'none'",
      "script-src 'self' 'unsafe-inline'",
      "worker-src 'self'",
      "connect-src 'self'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self'",
    ].join('; ');

    await page.route('**/*', async (route) => {
      const response = await route.fetch();
      const headers = { ...response.headers() };
      headers['content-security-policy'] = STRICT_CSP;
      await route.fulfill({
        response,
        headers,
      });
    });

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
    expect(logContent).toContain('[FAIL]');
  });
});
