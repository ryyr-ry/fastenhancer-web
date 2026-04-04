import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'tests/**/*.test.ts',
      'tests/**/*.test.tsx',
    ],
    exclude: [],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
    },
    testTimeout: 30000,
    hookTimeout: 30000,
  },
});
