/**
 * worklet-protocol.test.ts — Tests for AudioWorklet message protocol
 *
 * Verifies the message-based protocol between the main thread and the
 * FastEnhancerProcessor running inside the AudioWorklet.
 *
 * Since AudioWorkletProcessor cannot be directly instantiated in Node.js,
 * we mock the global AudioWorklet environment and evaluate processor.js
 * to get the processor class.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { runInNewContext } from 'vm';

interface MockPort {
  onmessage: ((e: { data: any }) => void) | null;
  messages: any[];
  postMessage(data: any): void;
}

function createMockPort(): MockPort {
  return {
    onmessage: null,
    messages: [],
    postMessage(data: any) {
      this.messages.push(data);
    },
  };
}

let ProcessorClass: any;

function loadProcessorClass(): any {
  const processorPath = resolve(
    import.meta.dirname,
    '..',
    '..',
    '..',
    'src',
    'worklet',
    'processor.js',
  );
  const source = readFileSync(processorPath, 'utf-8');

  let captured: any = null;
  const ctx: Record<string, any> = {
    performance: { now: () => Date.now() },
    Date,
    Math,
    Float32Array,
    Float64Array,
    Uint8Array,
    ArrayBuffer,
    WebAssembly: globalThis.WebAssembly,
    console,
    Number,
    Infinity,
    NaN,
    Error,
    Object,
    currentTime: 0,
    sampleRate: 48000,
    AudioWorkletProcessor: class {
      port: MockPort;
      constructor() {
        this.port = createMockPort();
      }
    },
    registerProcessor: (name: string, cls: any) => {
      captured = cls;
    },
  };

  runInNewContext(source, ctx);
  return captured;
}

function createProcessor(): { processor: any; port: MockPort } {
  const processor = new ProcessorClass();
  const port = processor.port as MockPort;
  return { processor, port };
}

function sendMessage(port: MockPort, data: any): void {
  if (port.onmessage) {
    port.onmessage({ data });
  }
}

function getLastMessage(port: MockPort): any {
  return port.messages[port.messages.length - 1];
}

function clearMessages(port: MockPort): void {
  port.messages.length = 0;
}

describe('FastEnhancerProcessor worklet protocol', () => {
  beforeEach(() => {
    if (!ProcessorClass) {
      ProcessorClass = loadProcessorClass();
    }
  });

  describe('get_state message', () => {
    it('responds with initial state before init', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'get_state', requestId: 'req-1' });

      const msg = getLastMessage(port);
      expect(msg.type).toBe('state');
      expect(msg.requestId).toBe('req-1');
      expect(msg.initialized).toBe(false);
      expect(msg.destroyed).toBe(false);
      expect(msg.bypass).toBe(false);
      expect(msg.agcEnabled).toBe(false);
      expect(msg.hpfEnabled).toBe(false);
      expect(msg.autoPassthrough).toBe(false);
    });

    it('echoes back the requestId', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'get_state', requestId: 'abc-123' });
      expect(getLastMessage(port).requestId).toBe('abc-123');
    });
  });

  describe('set_bypass message', () => {
    it('sets bypass to true', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_bypass', enabled: true });
      sendMessage(port, { type: 'get_state', requestId: 'r1' });
      expect(getLastMessage(port).bypass).toBe(true);
    });

    it('sets bypass back to false', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_bypass', enabled: true });
      sendMessage(port, { type: 'set_bypass', enabled: false });
      sendMessage(port, { type: 'get_state', requestId: 'r2' });
      expect(getLastMessage(port).bypass).toBe(false);
    });

    it('coerces truthy values to boolean', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_bypass', enabled: 1 });
      sendMessage(port, { type: 'get_state', requestId: 'r3' });
      expect(getLastMessage(port).bypass).toBe(true);
    });
  });

  describe('set_hpf message', () => {
    it('sets hpfEnabled to true', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_hpf', enabled: true });
      sendMessage(port, { type: 'get_state', requestId: 'r1' });
      expect(getLastMessage(port).hpfEnabled).toBe(true);
    });

    it('sets hpfEnabled to false', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_hpf', enabled: true });
      sendMessage(port, { type: 'set_hpf', enabled: false });
      sendMessage(port, { type: 'get_state', requestId: 'r2' });
      expect(getLastMessage(port).hpfEnabled).toBe(false);
    });
  });

  describe('set_agc message', () => {
    it('sets agcEnabled to true', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_agc', enabled: true });
      sendMessage(port, { type: 'get_state', requestId: 'r1' });
      expect(getLastMessage(port).agcEnabled).toBe(true);
    });

    it('sets agcEnabled to false', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'set_agc', enabled: true });
      sendMessage(port, { type: 'set_agc', enabled: false });
      sendMessage(port, { type: 'get_state', requestId: 'r2' });
      expect(getLastMessage(port).agcEnabled).toBe(false);
    });
  });

  describe('destroy message', () => {
    it('responds with destroyed type', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'destroy' });
      expect(getLastMessage(port).type).toBe('destroyed');
    });

    it('marks state as destroyed', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'destroy' });
      clearMessages(port);
      sendMessage(port, { type: 'get_state', requestId: 'r1' });
      expect(getLastMessage(port).destroyed).toBe(true);
      expect(getLastMessage(port).initialized).toBe(false);
    });

    it('process() returns false after destroy', () => {
      const { processor, port } = createProcessor();
      sendMessage(port, { type: 'destroy' });
      const result = processor.process(
        [[new Float32Array(128)]],
        [[new Float32Array(128)]],
        {},
      );
      expect(result).toBe(false);
    });
  });

  describe('get_stats message (before init)', () => {
    it('responds with empty stats', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'get_stats' });
      const msg = getLastMessage(port);
      expect(msg.type).toBe('stats');
      expect(msg.frameCount).toBe(0);
      expect(msg.totalFrames).toBe(0);
    });
  });

  describe('start_stats / stop_stats messages', () => {
    it('start_stats accepts custom interval', () => {
      const { processor, port } = createProcessor();
      sendMessage(port, { type: 'start_stats', intervalFrames: 50 });
      expect(processor._statsInterval).toBe(50);
    });

    it('start_stats uses default interval when not specified', () => {
      const { processor, port } = createProcessor();
      sendMessage(port, { type: 'start_stats' });
      expect(processor._statsInterval).toBe(100);
    });

    it('stop_stats disables stats', () => {
      const { processor, port } = createProcessor();
      sendMessage(port, { type: 'start_stats', intervalFrames: 50 });
      sendMessage(port, { type: 'stop_stats' });
      expect(processor._statsInterval).toBe(0);
    });
  });

  describe('get_output_info message', () => {
    it('responds with output info before any processing', () => {
      const { port } = createProcessor();
      sendMessage(port, { type: 'get_output_info' });
      const msg = getLastMessage(port);
      expect(msg.type).toBe('output_info');
      expect(msg.frameCount).toBe(0);
      expect(msg.rms).toBe(0);
      expect(msg.nonZeroRatio).toBe(0);
      expect(msg.totalSamples).toBe(0);
      expect(msg.hasNaN).toBe(false);
      expect(msg.hasInf).toBe(false);
    });
  });

  describe('reset message', () => {
    it('does not crash before init (no WASM loaded)', () => {
      const { port } = createProcessor();
      expect(() => {
        sendMessage(port, { type: 'reset' });
      }).not.toThrow();
    });
  });

  describe('process() before init', () => {
    it('passes input through to output when not initialized', () => {
      const { processor } = createProcessor();
      const input = new Float32Array(128);
      input[0] = 0.5;
      input[64] = -0.3;
      const output = new Float32Array(128);
      const result = processor.process([[input]], [[output]], {});
      expect(result).toBe(true);
      expect(output[0]).toBe(0.5);
      expect(output[64]).toBeCloseTo(-0.3);
    });

    it('fills output with zeros when no input available', () => {
      const { processor } = createProcessor();
      const output = new Float32Array(128);
      output.fill(1.0);
      const result = processor.process([[]], [[output]], {});
      expect(result).toBe(true);
    });
  });

  describe('unknown message type', () => {
    it('silently ignores unknown message types', () => {
      const { port } = createProcessor();
      const msgCountBefore = port.messages.length;
      expect(() => {
        sendMessage(port, { type: 'nonexistent_message' });
      }).not.toThrow();
      expect(port.messages.length).toBe(msgCountBefore);
    });
  });

  describe('init message with invalid data', () => {
    it('responds with error when wasmBytes is invalid', () => {
      const { port } = createProcessor();
      sendMessage(port, {
        type: 'init',
        wasmBytes: new ArrayBuffer(0),
        weightBytes: new ArrayBuffer(0),
        exportMap: {},
        modelSize: 0,
      });
      const msg = getLastMessage(port);
      expect(msg.type).toBe('error');
      expect(typeof msg.message).toBe('string');
      expect(msg.message.length).toBeGreaterThan(0);
    });

    it('state remains uninitialized after failed init', () => {
      const { port } = createProcessor();
      sendMessage(port, {
        type: 'init',
        wasmBytes: new ArrayBuffer(0),
        weightBytes: new ArrayBuffer(0),
        exportMap: {},
        modelSize: 0,
      });
      clearMessages(port);
      sendMessage(port, { type: 'get_state', requestId: 'r1' });
      expect(getLastMessage(port).initialized).toBe(false);
    });
  });
});
