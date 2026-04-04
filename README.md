# fastenhancer-web

Real-time voice noise removal for the browser, powered by [FastEnhancer](https://github.com/aask1357/fastenhancer) (ICASSP 2026). Written in C, compiled to WebAssembly SIMD, and exposed through a clean TypeScript API.

> **日本語のドキュメントは [README.ja.md](./README.ja.md) をご覧ください。**

---

## Why fastenhancer-web?

Most browser-based noise removal solutions depend on generic inference runtimes. For example, the standard ONNX Runtime Web v1.24.3 WASM file (`ort-wasm-simd-threaded.wasm`) is **11.79 MB** alone (verified on jsdelivr CDN). Add JS glue and model weights, and users face multi-megabyte downloads. **fastenhancer-web implements neural network inference directly in C, eliminating runtime dependencies and achieving minimal bundle size:**

| Model | Parameters | WASM + Weights (gzip) | Processing Time (SIMD) | Budget Utilization |
|-------|-----------|----------------------|----------------------|-------------------|
| **Tiny** | 28K | **128 KB** | 0.51 ms | 4.8% |
| **Base** | 101K | **395 KB** | 1.71 ms | 16% |
| **Small** | 207K | **783 KB** | 3.84 ms | 36% |

- 48 kHz native — no resampling artifacts
- WASM SIMD acceleration with relaxed-simd FMA
- Zero runtime `malloc` — all memory pre-allocated at initialization
- No SharedArrayBuffer / COOP / COEP headers required
- CSP-compatible (no `unsafe-eval`)

---

## Installation

```bash
npm install fastenhancer-web
```

---

## Quick Start

### Layer 3: React Hook (1 line)

```tsx
import { useDenoiser } from 'fastenhancer-web/react';

function CallScreen() {
  const { outputStream, start, stop, state } = useDenoiser('small');

  const handleStart = async () => {
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    await start(mic);
  };

  return (
    <div>
      <button onClick={handleStart} disabled={state === 'processing'}>
        Start Noise Removal
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
  outputStream,   // MediaStream | null
  bypass,         // boolean
  start,          // (inputStream: MediaStream) => Promise<void>
  stop,           // () => void
  setBypass,      // (enabled: boolean) => void
  destroy,        // () => void
} = useDenoiser('small');
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `modelSize` | `'tiny' \| 'base' \| 'small'` | Model size to use (default: `'small'`) |
| `options.baseUrl` | `string` | Base URL for WASM/weight files |
| `options.simd` | `boolean` | Force SIMD on/off (auto-detected by default) |
| `options.workletUrl` | `string` | Custom AudioWorklet processor URL |
| `options.onWarning` | `(msg: string) => void` | Warning callback |
| `options.onError` | `(err: Error) => void` | Error callback |

**Features:**
- Automatic cleanup on unmount
- React 18+ Strict Mode safe (double mount/unmount resilient)
- Race condition safe (stale start() results are automatically discarded)

### `createDenoiser(options)` — Frame-level Processing API

```typescript
import { createDenoiser } from 'fastenhancer-web';
```

### `createStreamDenoiser(options)` — AudioWorklet Integration

```typescript
import { createStreamDenoiser } from 'fastenhancer-web';
```

### `diagnose()` — Browser Compatibility Check

```typescript
import { diagnose } from 'fastenhancer-web';

const result = await diagnose();
// { wasm: true, simd: true, audioContext: true, audioWorklet: true, overall: true, issues: [] }
```

### `getModels()` / `recommendModel(priority)` — Model Selection

```typescript
import { getModels, recommendModel } from 'fastenhancer-web';

const models = getModels();
// [{ id: 'tiny', ... }, { id: 'base', ... }, { id: 'small', ... }]

const recommended = recommendModel();
// { id: 'small', reason: '最高品質のノイズ除去。多くの環境で推奨。' }
```

### Error Classes

All errors extend `FastEnhancerError` with a machine-readable `code` property:

| Class | Code | When |
|-------|------|------|
| `WasmLoadError` | `WASM_LOAD_FAILED` | WASM module failed to load |
| `ModelInitError` | `MODEL_INIT_FAILED` | Model initialization failed (corrupt weights, CRC mismatch) |
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
| **WASM SIMD size** | 60 KB | 58 KB | 60 KB |
| **Weights size** | 111 KB | 397 KB | 814 KB |
| **Total (gzip)** | **128 KB** | **395 KB** | **783 KB** |
| **Processing time (SIMD median)** | 0.51 ms | 1.71 ms | 3.84 ms |
| **Processing time (SIMD P99)** | 0.76 ms | 2.09 ms | 4.42 ms |
| **Budget utilization** | 4.8% | 16% | 36% |
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
- WASM binary is embedded in JS via Emscripten SINGLE_FILE mode (no separate .wasm fetch)
- AudioWorklet uses raw `WebAssembly.instantiate()` — no Emscripten glue inside the worklet
- WASM binary is transferred as ArrayBuffer from main thread to worklet via `postMessage`
- All neural network buffers are pre-allocated at init time (zero runtime malloc)

---

## Browser Support

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome 91+ | ✅ Fully supported | WASM SIMD + AudioWorklet |
| Edge 91+ | ✅ Fully supported | Chromium-based |
| Firefox 89+ | ⚠️ Untested | WASM SIMD supported since 89 |
| Safari 16.4+ | ⚠️ Untested | WASM SIMD supported since 16.4 |

**Requirements:**
- WASM SIMD support (falls back to scalar if unavailable)
- AudioWorklet support
- CSP must allow `blob:` URLs (for SINGLE_FILE WASM)
- No `SharedArrayBuffer` / `Cross-Origin-Isolation` headers needed

---

## Exports Map

```json
{
  ".":            "Main API (createDenoiser, createStreamDenoiser, diagnose, getModels, ...)",
  "./react":      "React Hook (useDenoiser)",
  "./stream":     "AudioWorklet integration (createStreamDenoiser)",
  "./loader":     "WASM loader utilities",
  "./errors":     "Error classes",
  "./wasm/*":     "WASM SINGLE_FILE modules (tiny/base/small × scalar/simd)"
}
```

---

## Development

```bash
# Install dependencies
bun install

# Run TypeScript unit tests (162 tests)
bun run test

# Build TypeScript
bun run build:ts

# Build all WASM variants (requires Emscripten SDK)
bun run build:wasm:all

# Build everything
bun run build:all

# Run C engine native tests (154 tests, requires gcc/MinGW)
# Individual test executables are built and run via build scripts
```

### Test Architecture

| Suite | Environment | Framework | Tests | Purpose |
|-------|-------------|-----------|-------|---------|
| C native | gcc (MinGW) | Unity | 154 | C module correctness |
| WASM scalar | Emscripten (no SIMD) | vitest | 31 | Emscripten conversion |
| WASM SIMD | Emscripten (-msimd128) | vitest | 31 | SIMD-specific bugs |
| TypeScript unit | Node.js | vitest | 162 | API layer correctness |
| Browser E2E | Chrome | Playwright | 9 | AudioWorklet integration |

**Differential debugging:** C↔WASM scalar = Emscripten issue, scalar↔SIMD = SIMD issue, SIMD↔Browser = integration issue.

---

## Numerical Accuracy

PyTorch reference ↔ WASM SIMD output comparison (1000 frames):

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
