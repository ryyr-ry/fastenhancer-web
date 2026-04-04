import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  testMatch: '*.spec.ts',
  timeout: 30000,
  projects: [
    {
      name: 'chromium',
      use: {
        browserName: 'chromium',
        headless: true,
        baseURL: 'http://localhost:3457',
      },
    },
    {
      name: 'firefox',
      use: {
        browserName: 'firefox',
        headless: true,
        baseURL: 'http://localhost:3457',
      },
    },
  ],
  webServer: {
    command: 'node tests/browser/server.cjs',
    url: 'http://localhost:3457/package.json',
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
});
