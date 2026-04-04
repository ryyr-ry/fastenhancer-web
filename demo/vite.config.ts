import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react({ include: ['src/**/*.{ts,tsx,js,jsx}', '../src/**/*.{ts,tsx,js,jsx}'] })],
  base: './',
  resolve: {
    alias: [
      {
        find: /^react$/,
        replacement: path.resolve(__dirname, 'node_modules/react'),
      },
      {
        find: /^react\/jsx-runtime$/,
        replacement: path.resolve(__dirname, 'node_modules/react/jsx-runtime'),
      },
      {
        find: /^react-dom$/,
        replacement: path.resolve(__dirname, 'node_modules/react-dom'),
      },
      {
        find: /^fastenhancer-web\/react$/,
        replacement: path.resolve(__dirname, '../src/react/index.ts'),
      },
      {
        find: /^fastenhancer-web\/loader$/,
        replacement: path.resolve(__dirname, '../src/api/loader.ts'),
      },
      {
        find: /^fastenhancer-web\/errors$/,
        replacement: path.resolve(__dirname, '../src/api/errors.ts'),
      },
      {
        find: /^fastenhancer-web\/stream$/,
        replacement: path.resolve(__dirname, '../src/api/stream-denoiser.ts'),
      },
      {
        find: /^fastenhancer-web$/,
        replacement: path.resolve(__dirname, '../src/api/index.ts'),
      },
    ],
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
})
