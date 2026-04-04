import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';

/**
 * CSP検証: unsafe-eval なしでライブラリが動作することを確認
 *
 * ポリシー:
 *   script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'
 *   worker-src 'self'
 *   connect-src 'self'
 *
 * unsafe-inline はテストページのインラインスクリプト用。
 * wasm-unsafe-eval はWebAssembly.instantiate()用。
 * unsafe-eval は含めない → eval()/new Function() を使っていないことを保証。
 */
const CSP_POLICY = [
  "default-src 'none'",
  "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'",
  "worker-src 'self'",
  "connect-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self'",
].join('; ');

test.describe('CSP互換性検証', () => {
  test('unsafe-eval なしでAudioWorklet + WASM初期化が成功する', async ({ page }) => {
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

  test('unsafe-eval なしでStreamDenoiser生成が成功する', async ({ page }) => {
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

  test('wasm-unsafe-eval なしではWASMインスタンス化がブロックされる', async ({ page, browserName }) => {
    test.skip(browserName !== 'chromium', 'CSP wasm-unsafe-evalの強制はChromium固有の挙動');
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
