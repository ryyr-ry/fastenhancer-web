<div align="center">

# 🎙️ fastenhancer-web

### Voice noise removal for live and recorded audio

[![npm version](https://img.shields.io/npm/v/fastenhancer-web.svg?style=flat-square)](https://www.npmjs.com/package/fastenhancer-web)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](./LICENSE)
[![CI](https://github.com/ryyr-ry/fastenhancer-web/actions/workflows/deploy-demo.yml/badge.svg?style=flat-square)](https://github.com/ryyr-ry/fastenhancer-web/actions/workflows/deploy-demo.yml)

Powered by [FastEnhancer](https://github.com/aask1357/fastenhancer) (ICASSP 2026)<br>
Written in C → WebAssembly SIMD → TypeScript API

<br>

[**🎧 Live Demo**](https://ryyr-ry.github.io/fastenhancer-web/) &nbsp;·&nbsp; [**📁 Audio File Demo**](https://ryyr-ry.github.io/fastenhancer-web/#/file) &nbsp;·&nbsp; [**📖 API Reference**](#api-reference) &nbsp;·&nbsp; [**🇯🇵 日本語**](./README.ja.md)

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
Core inference uses a 10.67 ms frame budget.<br>
Offline APIs convert and retain input sample rates.

</td>
<td>

🔒 **No special headers**<br>
No SharedArrayBuffer, no COOP/COEP.<br>
CSP supports `wasm-unsafe-eval`<br>
and explicit worklet/worker sources.

</td>
</tr>
<tr>
<td>

🧩 **Files and live audio**<br>
One-call File, AudioBuffer and PCM processing.<br>
React hook, stream and mono/stereo frame APIs.

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

> Stereo and offline APIs are available on `main` and in the demo. The current npm release (0.1.1) predates these additions.

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

### Recorded audio — one call

```typescript
import { enhanceAudio, enhanceBuffer, enhancePcm } from 'fastenhancer-web';

const cleanWav = await enhanceAudio(file);       // File/Blob → ready-to-save WAV Blob
const fastWav = await enhanceAudio(file, { parallel: { maxWorkers: 4 } });
const cleanBuffer = await enhanceBuffer(buffer); // AudioBuffer → AudioBuffer
const cleanPcm = await enhancePcm({ channels: [left, right], sampleRate: 44100 });
```

Mono/stereo channels and duration are retained. WAV and AudioBuffer inputs retain their sample rate; other browser-decodable files are decoded at 48 kHz. Model loading, resampling, frame splitting, delayed tail output and WAV encoding are handled automatically. The default is Small; browsers run inference in a module Worker. Use `{ model: 'tiny', signal, onProgress }` for a smaller model, cancellation and progress. Output WAV is PCM16. Node/Bun WAV processing is also one call:

```typescript
import { enhanceFile } from 'fastenhancer-web/node';
await enhanceFile('input.wav', 'output.wav');
```

### Stereo frames

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const stereo = await model.createChannelDenoiser(2);
const [cleanLeft, cleanRight] = stereo.processFrame([leftFrame, rightFrame]);
stereo.destroy();
```

Frames contain 512 samples per channel at 48 kHz. Left/right use independent states and WASM heaps. This is speech enhancement applied per channel; the original stereo image is not guaranteed. Stream APIs infer channel count from track settings; pass `{ channelCount: 2 }` when needed. See [design, deployment and limitations](docs/stereo-offline.md).

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

<a id="api-reference"></a>

## 📖 API Reference

### Recorded audio APIs

| Function | Input → Output | Runtime |
|----------|----------------|---------|
| `enhanceAudio(file, options?)` | `File`/`Blob` → PCM16 WAV `Blob` | Browser; WAV also works in Node/Bun |
| `enhanceBuffer(buffer, options?)` | `AudioBuffer` → `AudioBuffer` | Browser |
| `enhancePcm(audio, options?)` | `{ channels: Float32Array[], sampleRate }` → same PCM shape | Browser / Node / Bun |
| `enhanceFile(inputPath, outputPath, options?)` | WAV path → saved WAV | Node/Bun, import from `fastenhancer-web/node` |
| `enhanceWavStream(file, options?)` | WAV `Blob` → async iterable of encoded WAV byte chunks | Browser; cooperative inline fallback |

All accept mono/stereo input. PCM and WAV support integer sample rates from 8–192 kHz. Files use browser-supported codecs; non-WAV files produce 48 kHz output. Processing retains channel order and decoded sample count. Parallel WAV file processing reads and encodes bounded blocks; PCM/AudioBuffer and compressed decoding still require full-size arrays. Returning a Blob retains the final encoded output. Use `enhanceWavStream` to consume bytes incrementally.

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `'small'` | `'tiny'`, `'base'` or `'small'` |
| `simd` | Auto-detected | Force SIMD on/off |
| `baseUrl` | Embedded assets | Self-hosted model resources |
| `signal` | — | `AbortSignal` for cancellation |
| `onProgress` | — | Callback with processing progress from 0 to 1 |
| `worker` | Automatic | Browser module Worker; `true` requires one, `false` uses cooperative inline processing |
| `parallel` | `false` | `{ maxWorkers?: 1–32, engine?: 'auto' \| 'channels' \| 'batch' }`; exact offline parallel processing |

`parallel` is opt-in and incompatible with `worker: false`. Its worker limit is shared across a job and capped by available CPU capacity; jobs in one coordinator share a budget. `auto` uses the additional SIMD batch engine for recordings of at least four seconds when useful, preserving recurrent history while distributing frames and frequency positions. Scalar, short clips, one-worker automatic processing and existing self-hosted resources use the legacy per-channel engine. All modes preserve output samples; time segments are never independently reset.

Node/Bun callers should import `enhanceFile` and the parallel PCM adapter `enhancePcm` from `fastenhancer-web/node`, and omit `worker`. Parallel `enhanceFile` reads/writes WAV incrementally and replaces the output only after success. The default file API retains its existing inline behavior. WAV output is PCM16 and does not retain source container metadata. Cancellation prevents returning or saving a cancelled result; browser codec decoding cannot be interrupted mid-call. An async byte-stream consumer must discard partial output if iteration fails. See [parallel processing, performance and deployment](docs/offline-parallel-processing.md).

### `model.createChannelDenoiser(channelCount?)` — Mono/stereo frames

The default channel count is 2; pass 1 for mono. `processFrame([left, right])` accepts exactly 512 finite samples per channel at 48 kHz and returns separate output arrays. `reset()` clears temporal history. `destroy()` releases both engines. `bypass`, `agcEnabled` and `hpfEnabled` apply to every channel together. Stereo inference and memory costs are approximately twice those of mono; the timing tables below measure mono.

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
| `options.channelCount` | `1 \| 2` | Output channels (inferred from input track settings; defaults to mono if absent) |
| `options.onWarning` | `(msg: string) => void` | Warning callback |
| `options.onAutoBypass` | `(enabled: boolean) => void` | Automatic performance protection/recovery callback, separate from manual bypass |
| `options.onError` | `(err: Error) => void` | Error callback |
| `options.keepAliveInBackground` | `boolean` | Keep audio alive when app is backgrounded (mobile) |

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
// model.createChannelDenoiser(2) — independent left/right frame processing
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
│  │  Load / Recovery      │  │  sustained overload → dry audio → consecutive recovery
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
| Audio channels | Mono/stereo — independent processing per channel |
| Memory | All buffers pre-allocated at init (zero runtime malloc) |
| Package size | All 3 models + both variants embedded (~1.85 MB tarball); tree-shakable |

Real-time protection evaluates about 200 ms of average load plus accumulated excess time, rather than bypassing after five slightly slow frames. Repeated severe stalls still trigger promptly. Recovery retries after 32 ms, backs off up to 128 ms, and validates four consecutive frames before crossfading back to denoising. Both dry and processed audio retain the same STFT delay, including manual bypass. The React and stream demos show when automatic protection is active. See [performance monitoring design and verification](docs/performance-monitoring.md).

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

AudioWorklet is required for stream processing. Offline APIs use module Workers when available and browser codecs for non-WAV files. The new stereo/offline paths have been tested in Chromium; Firefox and Safari validation remains pending.

<details>
<summary><b>CSP directives</b></summary>

```
script-src 'self' 'wasm-unsafe-eval' blob:;
worker-src 'self' blob:;
```

- `wasm-unsafe-eval` — required for `WebAssembly.instantiate()`
- `blob:` — for default worklet loading (not needed if using `workletUrl` option)
- `'self'` in `worker-src` — for the offline module Worker and its bundled assets

</details>

---

## 📱 Background Audio (Mobile)

On mobile, switching to another app may suspend the browser's audio processing. Enable `keepAliveInBackground` to maintain audio continuity:

```typescript
// Layer 3 — React Hook
const denoiser = useDenoiser('small', { keepAliveInBackground: true });

// Layer 2 — Stream API
const sd = await model.createStreamDenoiser(mic, { keepAliveInBackground: true });
```

**What it does:**
- Creates a silent oscillator to keep AudioContext active
- Registers [Media Session API](https://developer.mozilla.org/docs/Web/API/Media_Session_API) as live audio — prevents OS from killing the audio pipeline
- Blocks pause/stop actions so users can't accidentally interrupt processing
- Extends the overload evidence window and severe-stall count by 4× while in background
- Automatically resumes AudioContext on foreground return

> **Note:** Effectiveness varies by browser and OS. Chrome/Brave on Android benefit the most. iOS Safari may still suspend audio despite these hints.

---

## ⚖️ Comparison

| | fastenhancer-web | rnnoise-wasm | ONNX Runtime Web |
|---|---|---|---|
| **Smallest bundle** | **124 KB** (Tiny, gzip) | ~95 KB | 11.79 MB (WASM only) |
| **Zero-config** | ✅ Embedded in JS | ❌ Requires `.wasm` | ❌ Requires `.wasm` + `.onnx` |
| **Special headers** | ✅ Not needed | ✅ Not needed | ❌ SharedArrayBuffer (multi-thread) |
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

# Run all vitest tests
bun run test

# Run stereo/offline tests using committed embedded assets (no WASM rebuild)
bunx vitest run tests/unit/api/channel-denoiser.test.ts tests/unit/api/offline.test.ts tests/unit/api/node-file.test.ts

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

| Suite | Environment | Framework | Purpose |
|-------|-------------|-----------|---------|
| C native | gcc (MinGW) | Unity | C module correctness |
| WASM | Emscripten (scalar + SIMD) | vitest | Emscripten conversion + SIMD correctness |
| TypeScript unit / adversarial | Node.js | vitest | API, worklet, lifecycle, stereo isolation and file processing |
| Browser E2E | Chromium / Firefox / WebKit | Playwright | AudioWorklet and offline module Worker integration |

Stereo/offline, recovery and parallel processing validation: **420 unit/adversarial tests** and **41 selected Chromium tests** passed, including Base/Small scalar/SIMD performance checks, controlled stereo recovery and exact internal parallel processing. Bun also matched the legacy PCM and WAV results. The production demo passed mono/stereo and standard/parallel WAV round trips under its deployment subpath. These results do not imply a full run of every native/WASM or browser suite. See [parallel processing results and limitations](docs/offline-parallel-processing.md).

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
