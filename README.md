# fastenhancer-web

Real-time voice noise removal for the browser, powered by [FastEnhancer](https://github.com/aask1357/fastenhancer) (ICASSP 2026). Written in C, compiled to WebAssembly SIMD, and exposed through a clean TypeScript API.

> **日本語のドキュメントは [README.ja.md](./README.ja.md) をご覧ください。**

---

## Why fastenhancer-web?

Most browser-based noise removal solutions depend on generic inference runtimes. For example, the standard ONNX Runtime Web v1.24.3 WASM file (`ort-wasm-simd-threaded.wasm`) is **11.79 MB** alone (verified on jsdelivr CDN). Add JS glue and model weights, and users face multi-megabyte downloads. **fastenhancer-web implements neural network inference directly in C, eliminating runtime dependencies and achieving minimal bundle size:**

| Model | Parameters | WASM + Weights (gzip) | Processing Time (SIMD) | Budget Utilization |
|-------|-----------|----------------------|----------------------|-------------------|
| **Tiny** | 28K | **124 KB** | 0.45 ms | 4.2% |
| **Base** | 101K | **391 KB** | 1.63 ms | 15% |
| **Small** | 207K | **780 KB** | 3.88 ms | 36% |

- 48 kHz native — no resampling artifacts
- WASM SIMD acceleration with relaxed-simd FMA
- Zero runtime `malloc` — all memory pre-allocated at initialization
- No SharedArrayBuffer / COOP / COEP headers required
- CSP-compatible — requires only `wasm-unsafe-eval` (no `unsafe-eval`)

---

## Installation

```bash
npm install fastenhancer-web
```

---

## Quick Start

### Layer 3: React Hook

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

### Layer 2: Vanilla JavaScript (3 lines)

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createStreamDenoiser(micStream);
const cleanStream = denoiser.outputStream;
// ...
denoiser.destroy();
```

### Layer 1: Frame-level Processing

```typescript
import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createDenoiser();
// Process raw Float32Array frames (512 samples at 48 kHz)
const output = denoiser.processFrame(inputFloat32Array);
denoiser.destroy();
```

---

## API Reference

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

## Model Comparison

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

## Architecture

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
- `loadModel()` loads WASM binary, weights, and export map automatically — by default via embedded JS modules (`import()`) for zero-config bundler compatibility; when `baseUrl` is provided, falls back to `fetch()`
- Default worklet loading uses inline Blob URL (no external file dependency); `workletUrl` option available for strict CSP environments that disallow `blob:`
- AudioWorklet uses raw `WebAssembly.instantiate()` — no Emscripten glue inside the worklet
- Audio processing is mono-only — stereo inputs are downmixed to the first channel
- WASM binary is transferred as ArrayBuffer from main thread to worklet via `postMessage`
- All neural network buffers are pre-allocated at init time (zero runtime malloc)
- All 3 models (tiny/base/small) and both variants (scalar/SIMD) are embedded in the package (~1.85 MB tarball). This is intentional for zero-config usage — bundlers with tree-shaking will only include the model you actually `import()`. If you need to minimize initial download size, use the `baseUrl` option with `loadModel()` to fetch models on demand from a CDN or your own server

---

## Browser Support

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome 91+ | ✅ Fully supported | WASM SIMD + AudioWorklet |
| Edge 91+ | ✅ Fully supported | Chromium-based |
| Firefox 89+ | ✅ E2E tested | WASM SIMD supported since 89 |
| Safari 16.4+ | ⚠️ Untested | WASM SIMD supported since 16.4 |

**Requirements:**
- WASM SIMD support (falls back to scalar if unavailable)
- AudioWorklet support
- CSP must allow `wasm-unsafe-eval` (for `WebAssembly.instantiate()`)
- Default worklet loading uses `blob:` URL — if your CSP blocks `blob:`, provide a `workletUrl` option
- No `SharedArrayBuffer` / `Cross-Origin-Isolation` headers needed

**Required CSP directives:**
- `script-src`: `'self'` (or wherever your scripts are served from)
- `script-src` or `worker-src`: `blob:` (for default worklet loading; not needed if using `workletUrl` option)
- `script-src`: `'wasm-unsafe-eval'` (required for `WebAssembly.instantiate()`)

WASM SIMD sizes above are approximate and depend on whether you count raw `.wasm` bytes or embedded JS wrapper output.

---

## Exports Map

```json
{
  ".":            "Main API (createDenoiser, createStreamDenoiser, loadModel, diagnose, getModels, ...)",
  "./react":      "React Hook (useDenoiser)",
  "./stream":     "AudioWorklet integration (createStreamDenoiser)",
  "./loader":     "WASM loader utilities (loadModel)",
  "./errors":     "Error classes"
}
```

---

## Development

```bash
# Install dependencies
bun install

# Run all vitest tests (187 tests: unit + WASM + adversarial)
bun run test

# Build TypeScript
bun run build:ts

# Build all WASM variants (requires Emscripten SDK)
bun run build:wasm:all

# Build everything
bun run build:all

# Run C engine native tests (201 tests, requires gcc/MinGW)
# Individual test executables are built and run via build scripts
```

### Test Architecture

| Suite | Environment | Framework | Tests | Purpose |
|-------|-------------|-----------|-------|---------|
| C native | gcc (MinGW) | Unity | 201 | C module correctness |
| WASM | Emscripten (scalar + SIMD) | vitest | 30 | Emscripten conversion + SIMD correctness |
| TypeScript unit | Node.js | vitest | 157 | API / worklet / lifecycle correctness |
| Browser E2E | Chrome + Firefox | Playwright | 30 | AudioWorklet integration |

**Differential debugging:** C↔WASM scalar = Emscripten issue, scalar↔SIMD = SIMD issue, SIMD↔Browser = integration issue.

---

## Numerical Accuracy

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
