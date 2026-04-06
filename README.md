<div align="center">

# 🎙️ fastenhancer-web

### Real-time voice noise removal for the browser

[![npm version](https://img.shields.io/npm/v/fastenhancer-web.svg?style=flat-square)](https://www.npmjs.com/package/fastenhancer-web)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](./LICENSE)
[![CI](https://github.com/ryyr-ry/fastenhancer-web/actions/workflows/deploy-demo.yml/badge.svg?style=flat-square)](https://github.com/ryyr-ry/fastenhancer-web/actions/workflows/deploy-demo.yml)
[![Tests: 581](https://img.shields.io/badge/tests-581%20passed-brightgreen.svg?style=flat-square)](#test-architecture)

Powered by [FastEnhancer](https://github.com/aask1357/fastenhancer) (ICASSP 2026)<br>
Written in C → WebAssembly SIMD → TypeScript API

<br>

[**🎧 Live Demo**](https://ryyr-ry.github.io/fastenhancer-web/) &nbsp;·&nbsp; [**📖 API Reference**](#api-reference) &nbsp;·&nbsp; [**🇯🇵 日本語**](./README.ja.md)

<br>

| Model | Bundle (gzip) | Latency | Budget |
|:-----:|:-------------:|:-------:|:------:|
| **Tiny** | **124 KB** | 0.45 ms | 4.2% |
| **Base** | **391 KB** | 1.63 ms | 15% |
| **Small** | **780 KB** | 3.88 ms | 36% |

<sub>ONNX Runtime Web WASM alone is 11.79 MB. fastenhancer-web Tiny is <b>95× smaller</b>.</sub>

</div>

<br>

## ✨ Features

<table>
<tr>
<td width="50%">

🎯 **Zero-config**<br>
WASM + weights embedded as base64 in JS.<br>
No `.wasm` to serve. No CDN. No CORS.

</td>
<td width="50%">

⚡ **Tiny footprint**<br>
124 KB gzip (Tiny model). Neural net inference<br>
in C — no generic runtime overhead.

</td>
</tr>
<tr>
<td>

🔊 **48 kHz native**<br>
No resampling. Processes at native sample rate<br>
with 10.67 ms frame budget.

</td>
<td>

🔒 **No special headers**<br>
No SharedArrayBuffer, no COOP/COEP.<br>
CSP: only `wasm-unsafe-eval`.

</td>
</tr>
<tr>
<td>

🧩 **3-layer API**<br>
React hook (1 line) · Stream (3 lines)<br>
Frame-level control.

</td>
<td>

📦 **Every bundler**<br>
Vite · webpack · esbuild · Rollup · Bun<br>
ESM-only. Tree-shakable.

</td>
</tr>
</table>

<br>

## 📦 Installation

```bash
# npm
npm install fastenhancer-web

# bun
bun add fastenhancer-web

# pnpm
pnpm add fastenhancer-web

# yarn
yarn add fastenhancer-web
```

> **ESM-only.** Requires a bundler (Vite, esbuild, webpack 5+) or a runtime with ES module support (Node.js 18+ with `"type": "module"`). CommonJS `require()` is not supported.
>
> **TypeScript users:** This package uses `"moduleResolution": "bundler"`. If your project uses `"node16"` or `"nodenext"`, you may need to set `"moduleResolution": "bundler"` in your `tsconfig.json`.

---

## 🚀 Quick Start

### Layer 3 — React Hook

```tsx
import { useDenoiser } from 'fastenhancer-web/react';

function CallScreen() {
  const { outputStream, start, stop, state } = useDenoiser('small');

  return (
    <div>
      <button onClick={start} disabled={state === 'loading'}>
        {state === 'processing' ? 'Denoising...' : 'Start Noise Removal'}
      </button>
      <button onClick={stop} disabled={state !== 'processing'}>
        Stop
      </button>
      {outputStream && <audio autoPlay ref={el => {
        if (el) el.srcObject = outputStream;
      }} />}
    </div>
  );
}
```

### Layer 2 — Stream API

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createStreamDenoiser(micStream);
const cleanStream = denoiser.outputStream;
// ...
denoiser.destroy();
```

### Layer 1 — Frame-level

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createDenoiser();
const output = denoiser.processFrame(inputFloat32Array); // 512 samples @ 48 kHz
denoiser.destroy();
```

### 💡 Zero-config vs Self-hosted

By default, all WASM binaries and model weights are **embedded as base64 inside JavaScript modules**. This means zero external dependencies — no `.wasm` files to serve, no separate weight files to host, no CDN setup:

```typescript
// Zero-config: everything is embedded (default)
const model = await loadModel('small');

// Self-hosted: fetch from your own CDN
const model = await loadModel('small', { baseUrl: 'https://cdn.example.com/models/' });
```

Bundlers with tree-shaking will only include the model you actually import.

---

## 📖 API Reference

### `useDenoiser(modelSize, options?)` — React Hook

```typescript
import { useDenoiser } from 'fastenhancer-web/react';

const {
  state,          // 'idle' | 'loading' | 'processing' | 'error' | 'destroyed'
  error,          // Error | null
  inputStream,    // MediaStream | null
  outputStream,   // MediaStream | null
  bypass,         // boolean
  start,          // (inputStream?: MediaStream) => Promise<void>
  stop,           // () => void
  setBypass,      // (enabled: boolean) => void
  destroy,        // () => void
} = useDenoiser('small');
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `modelSize` | `'tiny' \| 'base' \| 'small'` | Model size to use (required) |
| `options.baseUrl` | `string` | Base URL for WASM/weight files |
| `options.simd` | `boolean` | Force SIMD on/off (auto-detected by default) |
| `options.workletUrl` | `string` | Custom AudioWorklet processor URL |
| `options.audioConstraints` | `MediaTrackConstraints` | Custom constraints for auto-getUserMedia |
| `options.onWarning` | `(msg: string) => void` | Warning callback |
| `options.onError` | `(err: Error) => void` | Error callback |

**Features:**
- `start()` with no arguments automatically acquires microphone via getUserMedia
- `start(stream)` uses an existing MediaStream (WebRTC, custom constraints, etc.)
- Automatic mic release on stop/destroy/unmount (only for hook-acquired streams)
- Automatic cleanup on unmount
- React 18+ Strict Mode safe (double mount/unmount resilient)
- Race condition safe (stale start() results are automatically discarded)

### `loadModel(modelSize, options?)` — Model & Resource Loader

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
// model.createDenoiser()         — Layer 1: frame-level processing
// model.createStreamDenoiser()   — Layer 2: real-time AudioWorklet stream
// model.wasmBytes                — raw WASM binary (advanced use)
// model.weightData               — weight binary (advanced use)
// model.exportMap                — WASM export name mapping (advanced use)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `modelSize` | `'tiny' \| 'base' \| 'small'` | Model size to use (required) |
| `options.baseUrl` | `string` | Base URL for resource files (omit for zero-config embedded loading) |
| `options.simd` | `boolean` | Force SIMD on/off (auto-detected by default) |

Results are cached — calling `loadModel('small')` twice returns the same Promise.

### `createDenoiser(options)` — Frame-level Processing API (Low-level)

```typescript
import { createDenoiser } from 'fastenhancer-web';
```

> **Note:** For most use cases, prefer `loadModel('small').then(m => m.createDenoiser())` which handles resource loading automatically. The direct `createDenoiser()` requires manually providing WASM factory and weight data.

### `createStreamDenoiser(options)` — AudioWorklet Integration (Low-level)

```typescript
import { createStreamDenoiser } from 'fastenhancer-web';
```

> **Note:** For most use cases, prefer `model.createStreamDenoiser(micStream)` via `loadModel()` which handles WASM, weights, and export map automatically. The direct `createStreamDenoiser()` requires manually providing all binary resources.

### `diagnose()` — Browser Compatibility Check

```typescript
import { diagnose } from 'fastenhancer-web';

const result = await diagnose();
// { wasm: true, simd: true, audioContext: true, audioWorklet: true, overall: true, recommended: true, issues: [] }
```

### `getModels()` / `recommendModel(options?)` — Model Selection

```typescript
import { getModels, recommendModel } from 'fastenhancer-web';

const models = getModels();
// [{ id: 'tiny', ... }, { id: 'base', ... }, { id: 'small', ... }]

const recommended = recommendModel({ priority: 'quality' });
// { id: 'small', reason: 'Highest-quality noise removal. Recommended for most environments.' }
```

### Error Classes

All errors extend `FastEnhancerError` with a machine-readable `code` property:

| Class | Code | When |
|-------|------|------|
| `WasmLoadError` | `WASM_LOAD_FAILED` | WASM module failed to load |
| `ModelInitError` | `MODEL_INIT_FAILED` | Model initialization failed (corrupt weights or incompatible format) |
| `AudioContextError` | `AUDIO_CONTEXT_ERROR` | AudioContext creation/state error |
| `WorkletError` | `WORKLET_ERROR` | AudioWorklet registration/communication failure |
| `ValidationError` | `VALIDATION_ERROR` | Invalid arguments |
| `DestroyedError` | `DESTROYED_ERROR` | Operation on a destroyed instance |

```typescript
import { WasmLoadError, DestroyedError } from 'fastenhancer-web/errors';
```

---

## 📊 Model Comparison

All models process 48 kHz audio natively with a 512-sample hop size (10.67 ms frame budget).

| | Tiny | Base | Small |
|---|------|------|-------|
| **Parameters** | 28K | 101K | 207K |
| **WASM SIMD size** | 46 KB | 45 KB | 46 KB |
| **Weights size** | 111 KB | 397 KB | 814 KB |
| **Total (gzip)** | **124 KB** | **391 KB** | **780 KB** |
| **Processing time (SIMD median)** | 0.45 ms | 1.63 ms | 3.88 ms |
| **Processing time (SIMD P99)** | 0.67 ms | 1.89 ms | 4.77 ms |
| **Budget utilization** | 4.2% | 15% | 36% |
| **RNNFormer blocks** | 2 | 3 | 3 |
| **Encoder blocks** | 2 | 2 | 3 |
| **Channels** | 24 | 48 | 64 |

**Recommendation:**
- **Small** — Recommended for most applications. Best noise removal quality.
- **Base** — Good balance of quality and speed. Suitable for mobile devices.
- **Tiny** — Minimal resource usage. For low-power or latency-critical scenarios.

---

## 🏗️ Architecture

```
Microphone Input (48 kHz)
    │
    ▼
┌─────────────────────────────┐
│  AudioWorkletProcessor      │  ← runs on audio thread
│  ┌───────────────────────┐  │
│  │  Frame Buffer          │  │  4 × 128 quantum → 512 samples
│  │  WASM SIMD Engine      │  │  FFT → Encoder → RNNFormer → Decoder → iFFT
│  │  Frame Drop Scheduler  │  │  5 consecutive drops → auto bypass
│  └───────────────────────┘  │
└─────────────────────────────┘
    │
    ▼
Clean Audio Output (48 kHz)
```

**Key design decisions:**

| Decision | Detail |
|----------|--------|
| Loading strategy | Embedded JS modules by default; `fetch()` when `baseUrl` provided |
| Worklet loading | Inline Blob URL (no external file); `workletUrl` for strict CSP |
| WASM instantiation | Raw `WebAssembly.instantiate()` — no Emscripten glue in worklet |
| Audio channels | Mono only — stereo inputs downmixed to first channel |
| Memory | All buffers pre-allocated at init (zero runtime malloc) |
| Package size | All 3 models + both variants embedded (~1.85 MB tarball); tree-shakable |

---

## 🌐 Browser Support

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome 91+ | ✅ Fully supported | WASM SIMD + AudioWorklet |
| Edge 91+ | ✅ Fully supported | Chromium-based |
| Firefox 89+ | ✅ E2E tested | WASM SIMD supported since 89 |
| Safari 16.4+ | ⚠️ Untested | WASM SIMD supported since 16.4 |

**Requirements:**
- WASM SIMD support (falls back to scalar if unavailable)
- AudioWorklet support
- No `SharedArrayBuffer` / `Cross-Origin-Isolation` headers needed

<details>
<summary><b>CSP directives</b></summary>

```
script-src 'self' 'wasm-unsafe-eval' blob:;
worker-src blob:;
```

- `wasm-unsafe-eval` — required for `WebAssembly.instantiate()`
- `blob:` — for default worklet loading (not needed if using `workletUrl` option)

</details>

---

## ⚖️ Comparison

| | fastenhancer-web | rnnoise-wasm | ONNX Runtime Web |
|---|---|---|---|
| **Smallest bundle** | **124 KB** (Tiny, gzip) | ~95 KB | 11.79 MB (WASM only) |
| **Zero-config** | ✅ Embedded in JS | ❌ Requires `.wasm` | ❌ Requires `.wasm` + `.onnx` |
| **Special headers** | ❌ Not needed | ❌ Not needed | ✅ SharedArrayBuffer (multi-thread) |
| **Sample rate** | 48 kHz native | 48 kHz | Model-dependent |
| **Model quality** | FastEnhancer (ICASSP 2026) | RNNoise (2018) | Varies |
| **Runtime malloc** | Zero | Zero | Yes |
| **Tree-shakable** | ✅ Per-model | N/A | N/A |
| **TypeScript types** | ✅ Built-in | ❌ | ✅ Built-in |
| **React hook** | ✅ Built-in | ❌ | ❌ |

---

## 🛠️ Development

```bash
# Install dependencies
bun install

# Run all vitest tests (305 tests across 22 files)
bun run test

# Build TypeScript
bun run build:ts

# Build all WASM variants (requires Emscripten SDK)
bun run build:wasm:all

# Build everything (WASM + TS)
bun run build:all

# Run E2E browser tests (requires Playwright browsers)
bunx playwright test
```

### Test Architecture

| Suite | Environment | Framework | Tests | Purpose |
|-------|-------------|-----------|-------|---------|
| C native | gcc (MinGW) | Unity | ~201 | C module correctness |
| WASM | Emscripten (scalar + SIMD) | vitest | 30 | Emscripten conversion + SIMD correctness |
| TypeScript unit | Node.js | vitest | 275 | API / worklet / lifecycle correctness |
| Browser E2E | Chrome + Firefox + WebKit | Playwright | 75 | AudioWorklet integration |
| **Total** | | | **~581** | |

---

## 🔬 Numerical Accuracy

PyTorch reference ↔ WASM SIMD output comparison (40 frames):

| Model | MSE | Max Absolute Difference |
|-------|-----|------------------------|
| Tiny | 2.69 × 10⁻¹⁴ | 3.52 × 10⁻⁶ |
| Base | 1.37 × 10⁻¹⁴ | 4.23 × 10⁻⁷ |
| Small | 1.69 × 10⁻¹⁴ | 5.85 × 10⁻⁷ |

---

## License

MIT

## Credits

- [FastEnhancer](https://github.com/aask1357/fastenhancer) — Original neural network architecture (ICASSP 2026)
- [fastenhancer.c.wasm](https://github.com/kdrkdrkdr/fastenhancer.c.wasm) — Reference C/WASM implementation
