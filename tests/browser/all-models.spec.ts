import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:3457';
const MODELS = ['tiny', 'base', 'small'] as const;

const PLATFORM_LIMIT_RE =
  /Can't find variable: AudioContext|AudioContext is not defined|relaxed simd instructions not supported/;

test('loadModel → createStreamDenoiser E2E for all models (tiny/base/small)', async ({ page, browserName }) => {
  const errors: string[] = [];
  page.on('pageerror', (err) => errors.push(err.message));

  await page.goto(`${BASE_URL}/tests/browser/all-models-test.html`);
  await page.click('#startBtn');

  await page.waitForFunction(
    () => {
      const el = document.getElementById('result');
      return el && (el.textContent!.includes('[PASS]') || el.textContent!.includes('[FAIL]'));
    },
    { timeout: 60000 },
  );

  const content = await page.textContent('#result');

  if (content && PLATFORM_LIMIT_RE.test(content)) {
    expect(content).toContain('[FAIL]');
    return;
  }

  expect(errors).toHaveLength(0);
  expect(content).toContain('[PASS]');

  for (const model of MODELS) {
    expect(content).toContain(`OK: ${model}_size`);
    expect(content).toContain(`OK: ${model}_sampleRate`);
    expect(content).toContain(`OK: ${model}_wasmBytes`);
    expect(content).toContain(`OK: ${model}_weightData`);
    expect(content).toContain(`OK: ${model}_exportMap`);
    expect(content).toContain(`OK: ${model}_outputStream`);
    expect(content).toContain(`OK: ${model}_tracks`);
    expect(content).toContain(`OK: ${model}_running`);
    expect(content).toContain(`OK: ${model}_bypassOn`);
    expect(content).toContain(`OK: ${model}_bypassOff`);
    expect(content).toContain(`OK: ${model}_destroyed`);
  }
});
