import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { runInNewContext } from 'node:vm';

/** Run the shipped Worklet with controlled timing and a delayed test engine.
 * Buffering, monitoring, recovery, mixing and messages use production code.
 */
export function createWorkletHarness(options: {
  channels?: number;
  timer?: 'performance' | 'callback-frozen' | 'missing' | 'backwards';
  source?: string;
} = {}) {
  const channelCount = options.channels ?? 1;
  const clock = { now: 1000, callbackTime: 1000 };
  let costMs = 1, failChannel = -1, calls = 0, resets = 0;
  const messages: any[] = [];
  let Processor: any;
  const context = {
    performance: options.timer === 'missing' ? undefined : {
      now: () => options.timer === 'callback-frozen' ? clock.callbackTime : clock.now,
    },
    Date: options.timer === 'missing' ? undefined : { now: () => clock.now },
    currentTime: 0, sampleRate: 48000,
    Float32Array, Float64Array, Uint8Array, ArrayBuffer, WebAssembly,
    AudioWorkletProcessor: class {
      port = { onmessage: null, postMessage: (message: any) => messages.push(message) };
    },
    registerProcessor: (_name: string, value: any) => { Processor = value; },
  };
  const source = options.source ?? readFileSync(resolve(import.meta.dirname, '../../src/worklet/processor.js'), 'utf8');
  runInNewContext(source, context);
  const processor = new Processor({ processorOptions: { channelCount } });
  processor._initialized = true;
  processor._outputPos = 512;
  processor._channels = Array.from({ length: channelCount }, (_, channel) => {
    const memory = { buffer: new ArrayBuffer(4096) };
    const heap = new Float32Array(memory.buffer);
    const previous = new Float32Array(512);
    return {
      memory, statePtr: channel + 1, inputPtr: 0, outputPtr: 2048,
      inputBuffer: new Float32Array(640), passthroughBuffer: new Float32Array(512),
      dryDelayBuffer: new Float32Array(512), outputBuffer: new Float32Array(512),
      cachedHeapF32: null, cachedBuffer: null,
      wasm: {
        _fe_process_inplace() {
          calls++;
          clock.now += (options.timer === 'backwards' ? -1 : 1) * costMs / channelCount;
          if (failChannel === channel) { failChannel = -1; throw new Error('injected engine failure'); }
          for (let i = 0; i < 512; i++) heap[512 + i] = previous[i] * 0.5;
          previous.set(heap.subarray(0, 512));
        },
        _fe_reset() { previous.fill(0); resets++; },
        _fe_destroy() {}, _fe_set_agc() {}, _fe_set_hpf() {},
      },
    };
  });
  function quantum(input: Float32Array[]) {
    clock.now += 128 / 48000 * 1000;
    clock.callbackTime = clock.now;
    context.currentTime += 128 / 48000;
    const output = input.map(() => new Float32Array(128));
    processor.process([input], [output]);
    return output;
  }
  function frame(values: number[] = Array(channelCount).fill(0.25)) {
    for (let q = 0; q < 4; q++) quantum(values.map(value => new Float32Array(128).fill(value)));
    return processor._channels.map((channel: any) => channel.outputBuffer.slice()) as Float32Array[];
  }
  return {
    processor, messages, quantum, frame,
    get budgetMs() { return processor._budgetMs as number; },
    get calls() { return calls; },
    get resets() { return resets; },
    setCost(ms: number) { costMs = ms; },
    failNext(channel = 0) { failChannel = channel; },
    send(message: any) { processor.port.onmessage({ data: message }); },
    get automaticEvents() { return messages.filter(message => message.type === 'auto_bypass').map(message => message.enabled); },
  };
}
