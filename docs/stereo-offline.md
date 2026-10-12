# Stereo and one-call offline processing

## Implementation order

1. Add an independent mono engine for each channel, with separate temporal state and WASM memory.
2. Preserve channel count in the existing AudioWorklet and stream adapters.
3. Build duration-preserving offline PCM processing on the channel engine.
4. Add AudioBuffer, File/Blob and Node WAV adapters, with a file-processing demo.
5. Validate real WASM, emitted browser modules, bundled Workers and file round trips.

These steps are implemented. Joint stereo enhancement, video remuxing and a batch-job API are separate follow-up work.

## Stereo contract

The shipped neural models accept mono input. Stereo support uses two independent engines, one for left and one for right. This preserves channel order and synchronized frame positions; it does not guarantee unchanged spatial imaging. The models target speech rather than music or speaker separation. Identical channels produce identical results; a silent channel cannot acquire audio from its neighbor.

Each channel needs an isolated WASM instance because the published binaries have fixed heaps that cannot hold two Base/Small states. Model resources remain cached by `loadModel`; the AudioWorklet shares compiled code between instances. Destroying a channel processor frees every native state. A failure during construction releases already-created states.

`model.createChannelDenoiser(2)` processes planar `[left, right]` frames at 48 kHz, 512 samples per channel. Every input channel is validated before any state advances. `reset()` clears GRU/STFT history. Existing `createDenoiser()` and `processFrame(Float32Array)` remain mono APIs; `createDenoiser({ isolated: true })` is available when independent heaps are required.

The stream adapter infers mono/stereo from the input track's settings, falling back to mono when unspecified. `channelCount: 2` explicitly selects stereo when settings are absent. Channel count is fixed for a processor's lifetime. Channel interpretation is discrete: explicit mono output selects the first input channel, and a missing right input is silence. HPF, AGC, bypass and fallback decisions apply to both channels together. Stereo roughly doubles inference and memory costs; existing timing tables measure mono.

## Offline contract

- `enhancePcm({ channels, sampleRate })` returns planar PCM with the same channel count, sample rate and sample count. Supported rates are integer 8–192 kHz, with mono or stereo Float32 data. Input buffers are not modified or detached.
- `enhanceBuffer(audioBuffer)` returns an AudioBuffer with matching metadata.
- `enhanceAudio(fileOrBlob)` returns a ready-to-save `audio/wav` Blob.
- `enhanceFile(inputPath, outputPath)` from `fastenhancer-web/node` reads and writes WAV files without browser APIs.

All default to the Small model. `model`, `simd` and `baseUrl` are optional. The processing layer converts to 48 kHz using a windowed-sinc anti-aliasing filter, processes all frames, pads the last partial frame, drains the one-hop STFT delay and removes that delay. Resampling back uses the original sample count, avoiding duration rounding errors. Offline processing never invokes real-time auto-bypass.

Browser calls use a module Worker automatically when available, with progress from 0 to 1 and cancellation via `AbortSignal`. `worker: true` requires a browser Worker; `worker: false` uses the same engine with cooperative yielding. If CSP or deployment blocks the Worker, the call reports an error instead of silently moving heavy inference to the main thread. Deploy `dist/api` and the embedded modules together; bundlers must support module workers and model dynamic imports. Vite's `worker.format: 'es'` keeps model chunks lazy-loaded. WASM still needs `wasm-unsafe-eval` under strict CSP.

Default file calls retain their existing in-memory behavior. Opt-in `parallel: { maxWorkers: 4 }` adds exact per-channel/resampling parallelism and a SIMD batch engine for suitable longer recordings. Parallel WAV file processing reads and encodes bounded blocks; `enhanceWavStream` yields encoded chunks under consumer backpressure. Parallel Node `enhanceFile` also streams its input/output and keeps its temporary-file/rename contract. PCM/AudioBuffer APIs and compressed browser decoding still need full-size arrays, and a returned Blob retains final output bytes. See [parallel processing](offline-parallel-processing.md).

The built-in WAV adapter supports RIFF PCM 8/16/24/32-bit and IEEE float32, including compatible extensible headers. WAV input retains its original sample rate. Other files use the browser's supported decoders at 48 kHz, so their output is 48 kHz. Node/Bun currently accepts WAV only. Output is interleaved PCM16 WAV: values outside [-1, 1] are clipped, and source container metadata is not retained. Node writes a temporary sibling file and renames it after successful processing, preserving an existing destination on cancellation or processing failure.

## Validation

Unit tests compare all three models and both WASM variants against independent mono engines. They cover channel isolation, invalid-input atomicity, reset, frame boundaries, delayed tail output, exact durations at multiple rates, anti-aliasing, WAV parsing, cancellation and Node file output.

Browser tests compare stereo AudioWorklet output with frame-level references, exercise actual module Workers, AudioBuffer round trips and cancellation. Short DSP comparisons avoid timing-dependent protection. Longer Base/Small stereo tests inject measured durations into the shipped processor while running real WASM, verifying continuous dry output and freshly reset engine references after recovery. See [performance monitoring](performance-monitoring.md). Browser compatibility beyond the locally tested engine must still be verified in CI.
