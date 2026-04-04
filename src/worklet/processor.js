/**
 * FastEnhancer AudioWorkletProcessor
 *
 * WASMエンジンをAudioWorklet内で直接インスタンス化し、
 * リアルタイムノイズ除去を行う。
 *
 * 初期化フロー:
 *   Main thread → postMessage({ type:'init', wasmBytes, weightBytes, exportMap, modelSize })
 *   Worklet     → WebAssembly.Module/Instance を同期生成 → fe_init() → postMessage({ type:'ready' })
 *
 * process()フロー:
 *   1. outputFrame の現在位置から128サンプルを出力
 *   2. input 128サンプルをフレームバッファに蓄積
 *   3. hopSize分蓄積したらWASM処理 → outputFrame更新
 */

/* global registerProcessor, AudioWorkletProcessor, sampleRate, currentTime */

const DEFAULT_HOP_SIZE = 512;
const QUANTUM = 128;
const MAX_STORED_TIMES = 2000;
const DEFAULT_STATS_INTERVAL = 100;
const OVERRUN_THRESHOLD = 5;
const RECOVERY_THRESHOLD = 3;
const NONZERO_THRESHOLD = 1e-8;
const TIMER_CHECK_SPINS = 100000;

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

function _checkTimer() {
  if (_timerChecked) return;
  _timerChecked = true;
  if (!_perf) { _perfNowFrozen = true; return; }
  const a = _perf.now();
  let spins = 0;
  while (spins < TIMER_CHECK_SPINS) { spins++; }
  const b = _perf.now();
  if (b === a) _perfNowFrozen = true;
}

function _nowReliable() {
  if (!_perfNowFrozen && _perf) return _perf.now();
  return _dateNow ? _dateNow() : currentTime * 1000;
}

class FastEnhancerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
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
    this._inputBuffer = new Float32Array(this._hopSize + QUANTUM);
    this._inputWritePos = 0;

    this._outputFrame = new Float32Array(this._hopSize);
    this._outputPos = 0;

    this._wasm = null;
    this._statePtr = 0;
    this._inputPtr = 0;
    this._outputPtr = 0;
    this._memory = null;
    this._cachedHeapF32 = null;
    this._cachedBuffer = null;

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
        if (this._wasm && this._statePtr)
          this._wasm._fe_set_hpf(this._statePtr, msg.enabled ? 1 : 0);
        break;
      case 'set_agc':
        this._agcEnabled = !!msg.enabled;
        if (this._wasm && this._statePtr)
          this._wasm._fe_set_agc(this._statePtr, msg.enabled ? 1 : 0);
        break;
      case 'reset':
        if (this._wasm && this._statePtr)
          this._wasm._fe_reset(this._statePtr);
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

  _initWasm({ wasmBytes, weightBytes, exportMap, modelSize }) {
    try {
      const module = new WebAssembly.Module(wasmBytes);

      const needed = WebAssembly.Module.imports(module);
      const importObject = {};
      for (const imp of needed) {
        if (!importObject[imp.module]) importObject[imp.module] = {};
        if (imp.kind === 'function') {
          importObject[imp.module][imp.name] = () => 0;
        }
      }

      const instance = new WebAssembly.Instance(module, importObject);

      let memory = null;
      for (const val of Object.values(instance.exports)) {
        if (val instanceof WebAssembly.Memory) { memory = val; break; }
      }
      if (!memory) throw new Error('WASM memory export not found');
      this._memory = memory;

      const wasm = {};
      for (const [readable, minified] of Object.entries(exportMap)) {
        const exp = instance.exports[minified];
        if (typeof exp === 'function') wasm[readable] = exp;
      }
      this._wasm = wasm;

      const weightLen = weightBytes.byteLength;
      const weightPtr = wasm._malloc(weightLen);
      if (!weightPtr) throw new Error('malloc failed for weight allocation');

      new Uint8Array(memory.buffer).set(new Uint8Array(weightBytes), weightPtr);

      let statePtr;
      try {
        statePtr = wasm._fe_init(modelSize, weightPtr, weightLen);
      } finally {
        wasm._free(weightPtr);
      }
      if (!statePtr) throw new Error('fe_init returned null (weight validation failed?)');
      this._statePtr = statePtr;

      const hopSize = wasm._fe_get_hop_size(statePtr);
      this._hopSize = hopSize;
      this._inputPtr = wasm._fe_get_input_ptr(statePtr);
      this._outputPtr = wasm._fe_get_output_ptr(statePtr);

      this._inputBuffer = new Float32Array(hopSize + QUANTUM);
      this._inputWritePos = 0;
      this._outputFrame = new Float32Array(hopSize);
      this._outputPos = 0;
      this._budgetMs = (hopSize / sampleRate) * 1000;

      // タイマー診断をinit時に実行（process()内のbusy-waitを回避）
      _checkTimer();
      if (_perfNowFrozen) {
        this.port.postMessage({
          type: 'timer_info',
          perfNowFrozen: true,
          fallback: _dateNow ? 'Date.now' : 'currentTime',
        });
      }

      this._initialized = true;
      this.port.postMessage({ type: 'ready', hopSize });
    } catch (err) {
      this.port.postMessage({ type: 'error', message: err.message });
    }
  }

  _destroy() {
    if (this._wasm && this._statePtr) {
      this._wasm._fe_destroy(this._statePtr);
      this._statePtr = 0;
    }
    this._initialized = false;
    this._destroyed = true;
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

  process(inputs, outputs) {
    if (this._destroyed) return false;

    const output = outputs[0] && outputs[0][0];
    if (!output) return true;

    const input = inputs[0] && inputs[0][0];

    if (!this._initialized || !input) {
      if (input) output.set(input);
      else output.fill(0);
      return true;
    }

    const hopSize = this._hopSize;
    const len = Math.min(QUANTUM, output.length);

    const remaining = hopSize - this._outputPos;
    if (remaining >= len) {
      output.set(this._outputFrame.subarray(this._outputPos, this._outputPos + len));
    } else {
      output.set(this._outputFrame.subarray(this._outputPos, this._outputPos + remaining));
      output.fill(0, remaining);
    }
    this._outputPos += len;

    this._inputBuffer.set(input.subarray(0, len), this._inputWritePos);
    this._inputWritePos += len;

    if (this._inputWritePos >= hopSize) {
      if (this._bypass || this._autoPassthrough) {
        this._outputFrame.set(this._inputBuffer.subarray(0, hopSize));
      } else {
        try {
          const t0 = _nowReliable();

          if (!this._cachedHeapF32 || this._cachedBuffer !== this._memory.buffer) {
            this._cachedBuffer = this._memory.buffer;
            this._cachedHeapF32 = new Float32Array(this._cachedBuffer);
          }
          const heapF32 = this._cachedHeapF32;
          const inOff = this._inputPtr >> 2;
          heapF32.set(this._inputBuffer.subarray(0, hopSize), inOff);

          this._wasm._fe_process_inplace(this._statePtr);

          if (this._cachedBuffer !== this._memory.buffer) {
            this._cachedBuffer = this._memory.buffer;
            this._cachedHeapF32 = new Float32Array(this._cachedBuffer);
          }
          const outHeap = this._cachedHeapF32;
          const outOff = this._outputPtr >> 2;
          this._outputFrame.set(outHeap.subarray(outOff, outOff + hopSize));

          const elapsed = _nowReliable() - t0;
          this._frameCount++;
          this._processingTimes[this._timesIndex] = elapsed;
          this._timesIndex = (this._timesIndex + 1) % this._maxStoredTimes;
          this._timesCount = Math.min(this._timesCount + 1, this._maxStoredTimes);

          if (elapsed > this._budgetMs) {
            this._consecutiveOverruns++;
            this._consecutiveSuccesses = 0;
            if (!this._autoPassthrough && this._consecutiveOverruns >= OVERRUN_THRESHOLD) {
              this._autoPassthrough = true;
              this.port.postMessage({ type: 'auto_bypass', enabled: true });
            }
          } else {
            this._consecutiveOverruns = 0;
            if (this._autoPassthrough) {
              this._consecutiveSuccesses++;
              if (this._consecutiveSuccesses >= RECOVERY_THRESHOLD) {
                this._autoPassthrough = false;
                this._consecutiveSuccesses = 0;
                this.port.postMessage({ type: 'auto_bypass', enabled: false });
              }
            }
          }

          for (let si = 0; si < hopSize; si++) {
            const v = this._outputFrame[si];
            if (v !== v) { this._hasNaN = true; continue; }
            if (v === Infinity || v === -Infinity) { this._hasInf = true; continue; }
            this._outputRmsSum += v * v;
            if (Math.abs(v) > NONZERO_THRESHOLD) this._outputNonZeroCount++;
            this._outputTotalSamples++;
          }

          if (this._statsInterval > 0) {
            this._statsTimer++;
            if (this._statsTimer >= this._statsInterval) {
              this._statsTimer = 0;
              this._sendStats();
            }
          }
        } catch (err) {
          this._lastError = err.message || String(err);
          this.port.postMessage({ type: 'process_error', message: this._lastError });
        }
      }

      const leftover = this._inputWritePos - hopSize;
      if (leftover > 0) {
        this._inputBuffer.copyWithin(0, hopSize, this._inputWritePos);
      }
      this._inputWritePos = leftover;
      this._outputPos = 0;
    }

    return true;
  }
}

registerProcessor('fastenhancer-processor', FastEnhancerProcessor);
