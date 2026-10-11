/**
 * FastEnhancer AudioWorkletProcessor
 *
 * Directly instantiates the WASM engine inside the AudioWorklet
 * and performs real-time noise removal.
 *
 * Initialization flow:
 *   Main thread → postMessage({ type:'init', wasmBytes, weightBytes, exportMap, modelSize })
 *   Worklet     → synchronously create WebAssembly.Module/Instance → fe_init() → postMessage({ type:'ready' })
 *
 * process() flow:
 *   1. Accumulate 128 input samples into the frame buffer
 *   2. Once hopSize samples have accumulated, run WASM processing → update outputFrame
 *   3. Output 128 samples from the current position of outputFrame
 */

/* global registerProcessor, AudioWorkletProcessor, sampleRate, currentTime */

const DEFAULT_HOP_SIZE = 512;
const QUANTUM = 128;
const MAX_STORED_TIMES = 2000;
const DEFAULT_STATS_INTERVAL = 100;
const OVERRUN_THRESHOLD = 5;
const RECOVERY_THRESHOLD = 3;
const PROBE_INTERVAL = 16;
const NONZERO_THRESHOLD = 1e-8;

const _perf = typeof performance !== 'undefined' ? performance : null;
const _dateNow = typeof Date !== 'undefined' ? Date.now : null;

function _now() {
  if (_perf) {
    const t = _perf.now();
    if (t > 0) return t;
  }
  return _dateNow ? _dateNow() : currentTime * 1000;
}

let _timerChecked = false;
let _perfNowFrozen = false;
let _timerCheckCount = 0;
let _timerCheckPrev = 0;
const TIMER_CHECK_CALLBACKS = 3;

function _checkTimerAcrossCallbacks() {
  if (_timerChecked) return;
  if (!_perf) { _timerChecked = true; _perfNowFrozen = true; return; }

  const now = _perf.now();
  if (_timerCheckCount === 0) {
    _timerCheckPrev = now;
    _timerCheckCount = 1;
    return;
  }

  if (now === _timerCheckPrev) {
    _timerCheckCount++;
    if (_timerCheckCount >= TIMER_CHECK_CALLBACKS) {
      _perfNowFrozen = true;
      _timerChecked = true;
    }
    return;
  }

  // Timer is progressing normally
  _timerChecked = true;
  _perfNowFrozen = false;
}

function _nowReliable() {
  if (!_perfNowFrozen && _perf) return _perf.now();
  return _dateNow ? _dateNow() : currentTime * 1000;
}

class FastEnhancerProcessor extends AudioWorkletProcessor {
  constructor(options = {}) {
    super();
    this._channelCount = options.processorOptions?.channelCount ?? 1;
    this._channels = [];
    this._initialized = false;
    this._destroyed = false;
    this._bypass = false;
    this._agcEnabled = false;
    this._hpfEnabled = false;

    this._hopSize = DEFAULT_HOP_SIZE;
    this._budgetMs = (DEFAULT_HOP_SIZE / 48000) * 1000;
    this._consecutiveOverruns = 0;
    this._consecutiveSuccesses = 0;
    this._autoPassthrough = false;
    this._probeCounter = 0;
    this._backgroundMode = false;
    this._overrunThreshold = OVERRUN_THRESHOLD;
    this._inputWritePos = 0;

    this._outputPos = 0;
    this._lastFramePassthrough = false;


    this._frameCount = 0;
    this._processingTimes = new Float64Array(MAX_STORED_TIMES);
    this._maxStoredTimes = MAX_STORED_TIMES;
    this._timesIndex = 0;
    this._timesCount = 0;
    this._statsInterval = 0;
    this._statsTimer = 0;
    this._outputRmsSum = 0;
    this._outputNonZeroCount = 0;
    this._outputTotalSamples = 0;
    this._hasNaN = false;
    this._hasInf = false;

    this.port.onmessage = (e) => this._handleMessage(e);
  }

  _handleMessage(e) {
    const msg = e.data;
    switch (msg.type) {
      case 'init':
        this._initWasm(msg);
        break;
      case 'destroy':
        this._destroy();
        break;
      case 'set_hpf':
        this._hpfEnabled = !!msg.enabled;
        this._forEachState('_fe_set_hpf', msg.enabled ? 1 : 0);
        break;
      case 'set_agc':
        this._agcEnabled = !!msg.enabled;
        this._forEachState('_fe_set_agc', msg.enabled ? 1 : 0);
        break;
      case 'reset':
        this._forEachState('_fe_reset');
        this._inputWritePos = 0;
        this._outputPos = this._hopSize;
        break;
      case 'get_stats':
        this._sendStats();
        break;
      case 'start_stats':
        this._statsInterval = msg.intervalFrames || DEFAULT_STATS_INTERVAL;
        this._statsTimer = 0;
        break;
      case 'stop_stats':
        this._statsInterval = 0;
        break;
      case 'get_output_info':
        this._sendOutputInfo();
        break;
      case 'set_bypass':
        this._bypass = !!msg.enabled;
        break;
      case 'set_background_mode':
        this._handleBackgroundMode(!!msg.enabled);
        break;
      case 'get_state':
        this.port.postMessage({
          type: 'state',
          requestId: msg.requestId,
          bypass: this._bypass,
          agcEnabled: this._agcEnabled,
          hpfEnabled: this._hpfEnabled,
          initialized: this._initialized,
          destroyed: this._destroyed,
          autoPassthrough: this._autoPassthrough,
        });
        break;
    }
  }

  _handleBackgroundMode(enabled) {
    this._backgroundMode = enabled;
    if (enabled) {
      this._overrunThreshold = OVERRUN_THRESHOLD * 4;
    } else {
      this._overrunThreshold = OVERRUN_THRESHOLD;
      if (this._autoPassthrough) {
        this._probeCounter = PROBE_INTERVAL - 1;
      }
    }
  }

  _forEachState(method, ...args) {
    for (const channel of this._channels) channel.wasm[method](channel.statePtr, ...args);
  }

  _releaseStates() {
    this._forEachState('_fe_destroy');
    this._channels = [];
    this._initialized = false;
  }

  _initWasm({ wasmBytes, weightBytes, exportMap, modelSize, channelCount = this._channelCount }) {
    this._releaseStates();
    try {
      if (channelCount !== 1 && channelCount !== 2) throw new Error('channelCount must be 1 or 2');
      this._channelCount = channelCount;
      // NOTE: Synchronous compilation is required inside AudioWorklet (no async in process()).
      // This blocks the audio thread for the duration of compilation (~50-200ms).
      // WebAssembly.compileStreaming is not available in AudioWorkletGlobalScope.
      const module = new WebAssembly.Module(wasmBytes);

      const needed = WebAssembly.Module.imports(module);
      const importObject = {};
      const ABORT_IMPORT_NAMES = new Set(['abort', 'c', '__assert_fail', '__cxa_throw']);
      const KNOWN_SAFE_STUBS = new Set([
        'emscripten_notify_memory_growth',
        'emscripten_resize_heap',
        '__cxa_atexit',
        '__syscall_openat',
        '__syscall_fcntl64',
        '__syscall_ioctl',
        'fd_write',
        'fd_read',
        'fd_close',
        'fd_seek',
        'environ_sizes_get',
        'environ_get',
        'proc_exit',
        'clock_time_get',
        'args_sizes_get',
        'args_get',
      ]);
      const MINIFIED_NAME_RE = /^[a-z]{1,2}$/;
      const unknownImports = [];
      for (const imp of needed) {
        if (!importObject[imp.module]) importObject[imp.module] = {};
        if (imp.kind === 'function') {
          if (ABORT_IMPORT_NAMES.has(imp.name)) {
            importObject[imp.module][imp.name] = (...args) => {
              throw new Error(`WASM fatal: ${imp.name}(${args.join(', ')})`);
            };
          } else if (KNOWN_SAFE_STUBS.has(imp.name)) {
            importObject[imp.module][imp.name] = () => 0;
          } else if (MINIFIED_NAME_RE.test(imp.module) && MINIFIED_NAME_RE.test(imp.name)) {
            // Emscripten -O3 minified import — safe to stub
            importObject[imp.module][imp.name] = () => 0;
          } else {
            unknownImports.push(`${imp.module}.${imp.name}`);
          }
        }
      }
      if (unknownImports.length > 0) {
        throw new Error(
          `Unknown WASM imports detected: ${unknownImports.join(', ')}. ` +
          'If these are safe to stub, add them to KNOWN_SAFE_STUBS.'
        );
      }

      // Share compiled code, but give each channel a separate fixed-size heap.
      const createEngine = () => {
        const instance = new WebAssembly.Instance(module, importObject);

        let memory = null;
        for (const val of Object.values(instance.exports)) {
          if (val instanceof WebAssembly.Memory) { memory = val; break; }
        }
        if (!memory) throw new Error('WASM memory export not found');

        const wasm = {};
        for (const [readable, minified] of Object.entries(exportMap)) {
          const exp = instance.exports[minified];
          if (typeof exp === 'function') wasm[readable] = exp;
        }

        const WORKLET_REQUIRED_EXPORTS = [
          '_malloc', '_free', '_fe_init', '_fe_destroy',
          '_fe_process_inplace', '_fe_get_input_ptr', '_fe_get_output_ptr',
          '_fe_get_hop_size', '_fe_get_n_fft',
          '_fe_set_agc', '_fe_set_hpf', '_fe_reset',
        ];
        const missingExports = [];
        for (const name of WORKLET_REQUIRED_EXPORTS) {
          if (typeof wasm[name] !== 'function') missingExports.push(name);
        }
        if (missingExports.length > 0) {
          throw new Error(
            `WASM module is missing required exports: ${missingExports.join(', ')}. ` +
            'Check that the export map matches the compiled WASM binary.'
          );
        }
        return { wasm, memory };

      };

      const weightLen = weightBytes.byteLength;
      for (let c = 0; c < channelCount; c++) {
        const { wasm, memory } = createEngine();
        const weightPtr = wasm._malloc(weightLen);
        if (!weightPtr) throw new Error('malloc failed for weight allocation');
        try {
          new Uint8Array(memory.buffer).set(new Uint8Array(weightBytes), weightPtr);
          const statePtr = wasm._fe_init(modelSize, weightPtr, weightLen);
          if (!statePtr) throw new Error('fe_init returned null (weight validation failed?)');
          const hopSize = wasm._fe_get_hop_size(statePtr);
          this._channels.push({
            wasm, memory, statePtr,
            inputPtr: wasm._fe_get_input_ptr(statePtr),
            outputPtr: wasm._fe_get_output_ptr(statePtr),
            inputBuffer: new Float32Array(hopSize + QUANTUM),
            passthroughBuffer: new Float32Array(hopSize),
            cachedHeapF32: null,
            cachedBuffer: null,
          });
          this._hopSize = hopSize;
          wasm._fe_set_hpf(statePtr, this._hpfEnabled ? 1 : 0);
          wasm._fe_set_agc(statePtr, this._agcEnabled ? 1 : 0);
        } finally {
          wasm._free(weightPtr);
        }
      }
      const hopSize = this._hopSize;
      this._inputWritePos = 0;
      this._outputPos = hopSize;
      this._lastFramePassthrough = false;
      this._budgetMs = (hopSize / sampleRate) * 1000;
      this._consecutiveOverruns = 0;
      this._consecutiveSuccesses = 0;
      this._autoPassthrough = false;
      this._probeCounter = 0;

      // Run timer diagnostics during init (avoids busy-waiting inside process())
      _checkTimerAcrossCallbacks();
      if (_perfNowFrozen) {
        this.port.postMessage({
          type: 'timer_info',
          perfNowFrozen: true,
          fallback: _dateNow ? 'Date.now' : 'currentTime',
        });
      }

      this._initialized = true;
      this.port.postMessage({ type: 'ready', hopSize, channelCount });
    } catch (err) {
      this._releaseStates();
      this.port.postMessage({ type: 'error', message: err.message });
    }
  }

  _destroy() {
    this._releaseStates();
    this._destroyed = true;
    this.port.postMessage({ type: 'destroyed' });
  }

  _sendStats() {
    const count = this._timesCount;
    if (count === 0) {
      this.port.postMessage({ type: 'stats', frameCount: 0, totalFrames: 0 });
      return;
    }
    const sorted = new Float64Array(count);
    for (let i = 0; i < count; i++) sorted[i] = this._processingTimes[i];
    sorted.sort();
    let sum = 0;
    for (let i = 0; i < count; i++) sum += sorted[i];
    const budgetMs = (this._hopSize / sampleRate) * 1000;
    let dropped = 0;
    for (let i = 0; i < count; i++) {
      if (sorted[i] > budgetMs) dropped++;
    }
    this.port.postMessage({
      type: 'stats',
      frameCount: this._frameCount,
      totalFrames: count,
      avgMs: sum / count,
      medianMs: sorted[Math.floor(count / 2)],
      p99Ms: sorted[Math.floor(count * 0.99)],
      minMs: sorted[0],
      maxMs: sorted[count - 1],
      budgetMs,
      droppedFrames: dropped,
      dropRate: dropped / count,
    });
  }

  _sendOutputInfo() {
    const total = this._outputTotalSamples;
    const rms = total > 0 ? Math.sqrt(this._outputRmsSum / total) : 0;
    const nonZeroRatio = total > 0 ? this._outputNonZeroCount / total : 0;
    this.port.postMessage({
      type: 'output_info',
      frameCount: this._frameCount,
      rms,
      nonZeroRatio,
      totalSamples: total,
      hasNaN: this._hasNaN,
      hasInf: this._hasInf,
    });
  }

  _copyInputFrameToOutput(hopSize) {
    for (const channel of this._channels) {
      channel.passthroughBuffer.set(channel.inputBuffer.subarray(0, hopSize));
    }
    this._lastFramePassthrough = true;
  }

  _getHeapF32(channel) {
    if (!channel.cachedHeapF32 || channel.cachedBuffer !== channel.memory.buffer) {
      channel.cachedBuffer = channel.memory.buffer;
      channel.cachedHeapF32 = new Float32Array(channel.cachedBuffer);
    }
    return channel.cachedHeapF32;
  }

  _updateOutputDiagnostics(hopSize) {
    if (this._statsInterval <= 0) return;

    for (const channel of this._channels) {
      const heapF32 = this._getHeapF32(channel);
      const outputOffset = channel.outputPtr >> 2;
      for (let i = 0; i < hopSize; i++) {
        const v = heapF32[outputOffset + i];
        if (v !== v) {
          this._hasNaN = true;
          continue;
        }
        if (v === Infinity || v === -Infinity) {
          this._hasInf = true;
          continue;
        }
        this._outputRmsSum += v * v;
        if (v > NONZERO_THRESHOLD || v < -NONZERO_THRESHOLD) this._outputNonZeroCount++;
        this._outputTotalSamples++;
      }
    }
  }

  _runWasmFrame(hopSize) {
    this._lastFramePassthrough = false;
    const t0 = _nowReliable();

    for (const channel of this._channels) {
      const heapF32 = this._getHeapF32(channel);
      heapF32.set(channel.inputBuffer.subarray(0, hopSize), channel.inputPtr >> 2);
      channel.wasm._fe_process_inplace(channel.statePtr);
    }

    const elapsed = _nowReliable() - t0;
    this._frameCount++;
    this._processingTimes[this._timesIndex] = elapsed;
    this._timesIndex = (this._timesIndex + 1) % this._maxStoredTimes;
    this._timesCount = Math.min(this._timesCount + 1, this._maxStoredTimes);

    this._updateOutputDiagnostics(hopSize);

    if (this._statsInterval > 0) {
      this._statsTimer++;
      if (this._statsTimer >= this._statsInterval) {
        this._statsTimer = 0;
        this._sendStats();
      }
    }

    return elapsed;
  }

  _handleNormalProcessingResult(elapsed) {
    if (elapsed > this._budgetMs) {
      this._consecutiveOverruns++;
      this._consecutiveSuccesses = 0;
      if (this._consecutiveOverruns >= this._overrunThreshold) {
        this._autoPassthrough = true;
        this._consecutiveOverruns = 0;
        this._probeCounter = 0;
        this.port.postMessage({ type: 'auto_bypass', enabled: true });
      }
      return;
    }

    this._consecutiveOverruns = 0;
  }

  _handleProbeResult(elapsed) {
    if (elapsed > this._budgetMs) {
      this._consecutiveSuccesses = 0;
      return;
    }

    this._consecutiveSuccesses++;
    if (this._consecutiveSuccesses >= RECOVERY_THRESHOLD) {
      this._autoPassthrough = false;
      this._consecutiveSuccesses = 0;
      this._consecutiveOverruns = 0;
      this._probeCounter = 0;
      this.port.postMessage({ type: 'auto_bypass', enabled: false });
    }
  }

  _processFrame(hopSize) {
    if (this._bypass) {
      this._copyInputFrameToOutput(hopSize);
      return;
    }

    if (this._autoPassthrough) {
      this._probeCounter++;
      if (this._probeCounter < PROBE_INTERVAL) {
        this._copyInputFrameToOutput(hopSize);
        return;
      }

      this._probeCounter = 0;
      this._handleProbeResult(this._runWasmFrame(hopSize));
      return;
    }

    this._handleNormalProcessingResult(this._runWasmFrame(hopSize));
  }

  process(inputs, outputs) {
    if (this._destroyed) return false;
    if (!_timerChecked) _checkTimerAcrossCallbacks();

    const outputChannels = outputs[0];
    if (!outputChannels || !outputChannels[0]) return true;
    const inputChannels = inputs[0] || [];
    const len = Math.min(QUANTUM, outputChannels[0].length);

    if (!this._initialized) {
      for (let c = 0; c < outputChannels.length; c++) {
        outputChannels[c].fill(0);
        if (inputChannels[c]) outputChannels[c].set(inputChannels[c].subarray(0, len));
      }
      return true;
    }

    // Both channels share frame positions and bypass decisions, but never model state.
    const hopSize = this._hopSize;
    for (let c = 0; c < this._channels.length; c++) {
      const buffer = this._channels[c].inputBuffer;
      buffer.fill(0, this._inputWritePos, this._inputWritePos + len);
      if (inputChannels[c]) buffer.set(inputChannels[c].subarray(0, len), this._inputWritePos);
    }
    this._inputWritePos += len;

    if (this._inputWritePos >= hopSize) {
      try {
        this._processFrame(hopSize);
      } catch (err) {
        this._forEachState('_fe_reset');
        this._copyInputFrameToOutput(hopSize);
        this._lastError = err.message || String(err);
        this.port.postMessage({ type: 'process_error', message: this._lastError });
      }
      const leftover = this._inputWritePos - hopSize;
      if (leftover > 0) {
        for (const channel of this._channels) channel.inputBuffer.copyWithin(0, hopSize, this._inputWritePos);
      }
      this._inputWritePos = leftover;
      this._outputPos = 0;
    }

    const available = this._outputPos < hopSize ? Math.min(len, hopSize - this._outputPos) : 0;
    for (let c = 0; c < outputChannels.length; c++) {
      const output = outputChannels[c];
      output.fill(0);
      const channel = this._channels[c];
      if (!channel || available === 0) continue;
      const heapF32 = this._getHeapF32(channel);
      const frame = this._lastFramePassthrough ? channel.passthroughBuffer
        : heapF32.subarray(channel.outputPtr >> 2, (channel.outputPtr >> 2) + hopSize);
      output.set(frame.subarray(this._outputPos, this._outputPos + available));
    }
    this._outputPos += len;
    return true;
  }

}

registerProcessor('fastenhancer-processor', FastEnhancerProcessor);
