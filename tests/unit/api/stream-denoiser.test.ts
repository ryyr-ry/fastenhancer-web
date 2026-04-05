/**
 * stream-denoiser.test.ts — StreamDenoiser error path unit tests
 *
 * Tests failure modes in createStreamDenoiser:
 * - AudioContext creation failure
 * - AudioContext.resume() failure
 * - audioWorklet.addModule() failure
 * - Worklet init timeout
 * - Worklet init error message
 * - Destroyed state rejection
 * - destroyAsync idempotency
 * - Sample rate warning
 * - bypass/agc/hpf setters
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { AudioContextError, WorkletError, DestroyedError } from '../../../src/api/errors.js';
import { createStreamDenoiser, type StreamDenoiserOptions } from '../../../src/api/stream-denoiser.js';

// Mock embedded-loader
vi.mock('../../../src/api/embedded-loader.js', () => ({
  initProcessorBlobUrl: vi.fn().mockResolvedValue('blob:mock-processor'),
}));

// ─── Mock MessagePort ─────────────────────────────────────────────────────────

class MockPort {
  private _listeners = new Map<string, Set<(evt: any) => void>>();

  addEventListener(type: string, handler: (evt: any) => void): void {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type)!.add(handler);
  }

  removeEventListener(type: string, handler: (evt: any) => void): void {
    this._listeners.get(type)?.delete(handler);
  }

  postMessage = vi.fn();
  start = vi.fn();

  fire(type: string, data: any): void {
    const handlers = this._listeners.get(type);
    if (handlers) {
      for (const h of handlers) {
        h({ data } as MessageEvent);
      }
    }
  }
}

let activeMockPort: MockPort;

beforeEach(() => {
  activeMockPort = new MockPort();

  // AudioWorkletNode must be a real class (not vi.fn) so `new` works
  (globalThis as any).AudioWorkletNode = class {
    port = activeMockPort;
    connect = vi.fn();
    disconnect = vi.fn();
  };
});

afterEach(() => {
  vi.restoreAllMocks();
  delete (globalThis as any).AudioWorkletNode;
});

// ─── Helpers ──────────────────────────────────────────────────────────────────

function mockCtx(overrides?: Record<string, any>): any {
  return {
    sampleRate: 48000,
    state: 'running',
    resume: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    audioWorklet: { addModule: vi.fn().mockResolvedValue(undefined) },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    createMediaStreamSource: vi.fn(() => ({ connect: vi.fn(), disconnect: vi.fn() })),
    createMediaStreamDestination: vi.fn(() => ({
      stream: { getTracks: () => [], id: 'out', active: true },
    })),
    ...overrides,
  };
}

function fakeStream(): MediaStream {
  return {
    getTracks: () => [], getAudioTracks: () => [], getVideoTracks: () => [],
    addTrack: vi.fn(), removeTrack: vi.fn(), clone: vi.fn(),
    id: 'in', active: true, onaddtrack: null, onremovetrack: null,
    addEventListener: vi.fn(), removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(() => true),
  } as unknown as MediaStream;
}

function opts(audioContext: any, extra?: Partial<StreamDenoiserOptions>): StreamDenoiserOptions {
  return {
    inputStream: fakeStream(),
    wasmBytes: new ArrayBuffer(16),
    weightBytes: new ArrayBuffer(16),
    exportMap: { _fe_init: 'a' },
    modelSize: 0,
    audioContext,
    ...extra,
  };
}

async function createWithReady(o: StreamDenoiserOptions) {
  const p = createStreamDenoiser(o);
  await new Promise((r) => setTimeout(r, 0));
  activeMockPort.fire('message', { type: 'ready' });
  return p;
}

// ═══════════════════════════════════════════════════════════════════════════════

describe('createStreamDenoiser', () => {
  describe('AudioContext creation failures', () => {
    it('throws AudioContextError when constructor fails', async () => {
      const orig = (globalThis as any).AudioContext;
      try {
        (globalThis as any).AudioContext = class { constructor() { throw new Error('hw unavail'); } };
        const o: StreamDenoiserOptions = {
          inputStream: fakeStream(), wasmBytes: new ArrayBuffer(8),
          weightBytes: new ArrayBuffer(8), exportMap: {}, modelSize: 0,
        };
        await expect(createStreamDenoiser(o)).rejects.toThrow(AudioContextError);
        await expect(createStreamDenoiser(o)).rejects.toThrow(/Failed to create AudioContext/);
      } finally {
        if (orig) (globalThis as any).AudioContext = orig;
        else delete (globalThis as any).AudioContext;
      }
    });
  });

  describe('AudioContext.resume() failures', () => {
    it('throws AudioContextError when resume fails on suspended context', async () => {
      const orig = (globalThis as any).AudioContext;
      try {
        const _resume = vi.fn().mockRejectedValue(new Error('autoplay blocked'));
        const _close = vi.fn().mockResolvedValue(undefined);
        (globalThis as any).AudioContext = class {
          sampleRate = 48000; state = 'suspended';
          resume = _resume; close = _close;
          audioWorklet = { addModule: vi.fn() };
          addEventListener = vi.fn(); removeEventListener = vi.fn();
          createMediaStreamSource = vi.fn(); createMediaStreamDestination = vi.fn();
        };
        const o: StreamDenoiserOptions = {
          inputStream: fakeStream(), wasmBytes: new ArrayBuffer(8),
          weightBytes: new ArrayBuffer(8), exportMap: {}, modelSize: 0,
        };
        await expect(createStreamDenoiser(o)).rejects.toThrow(AudioContextError);
        await expect(createStreamDenoiser(o)).rejects.toThrow(/autoplay/);
      } finally {
        if (orig) (globalThis as any).AudioContext = orig;
        else delete (globalThis as any).AudioContext;
      }
    });
  });

  describe('audioWorklet.addModule() failures', () => {
    it('throws WorkletError when addModule fails', async () => {
      const ctx = mockCtx();
      ctx.audioWorklet.addModule.mockRejectedValue(new Error('CSP blocked'));
      await expect(createStreamDenoiser(opts(ctx))).rejects.toThrow(WorkletError);
      await expect(createStreamDenoiser(opts(ctx))).rejects.toThrow(/Failed to register AudioWorklet/);
    });

    it('includes CSP hint when blob URL is used', async () => {
      const ctx = mockCtx();
      ctx.audioWorklet.addModule.mockRejectedValue(new Error('blocked'));
      try {
        await createStreamDenoiser(opts(ctx));
      } catch (err) {
        expect(err).toBeInstanceOf(WorkletError);
        expect((err as Error).message).toMatch(/CSP/i);
      }
    });
  });

  describe('worklet initialization timeout', () => {
    it('throws WorkletError on init timeout', async () => {
      vi.useFakeTimers();
      try {
        const ctx = mockCtx();
        const p = createStreamDenoiser(opts(ctx));
        // Attach a no-op catch to prevent unhandled rejection warning
        // while we advance fake timers
        let caught: unknown;
        const guarded = p.catch((e) => { caught = e; });
        await vi.advanceTimersByTimeAsync(10_001);
        await guarded;
        expect(caught).toBeInstanceOf(WorkletError);
        expect((caught as Error).message).toMatch(/timed out/);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('worklet initialization error message', () => {
    it('throws WorkletError when worklet reports error', async () => {
      const ctx = mockCtx();
      const p = createStreamDenoiser(opts(ctx));
      await new Promise((r) => setTimeout(r, 0));
      activeMockPort.fire('message', { type: 'error', message: 'WASM instantiation failed' });
      await expect(p).rejects.toThrow(WorkletError);
      await expect(p).rejects.toThrow(/WASM instantiation failed/);
    });
  });

  describe('destroyed state rejection', () => {
    it('throws DestroyedError when setting bypass after destroy', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.destroy();
      await new Promise((r) => setTimeout(r, 0));
      expect(() => { sd.bypass = true; }).toThrow(DestroyedError);
    });

    it('throws DestroyedError when setting agcEnabled after destroy', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.destroy();
      await new Promise((r) => setTimeout(r, 0));
      expect(() => { sd.agcEnabled = true; }).toThrow(DestroyedError);
    });

    it('throws DestroyedError when setting hpfEnabled after destroy', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.destroy();
      await new Promise((r) => setTimeout(r, 0));
      expect(() => { sd.hpfEnabled = true; }).toThrow(DestroyedError);
    });

    it('rejects getWorkletState after destroy', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.destroy();
      await new Promise((r) => setTimeout(r, 0));
      await expect(sd.getWorkletState()).rejects.toThrow(DestroyedError);
    });
  });

  describe('destroyAsync', () => {
    it('is idempotent — multiple calls all resolve without error', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      const results = await Promise.all([sd.destroyAsync(), sd.destroyAsync(), sd.destroyAsync()]);
      expect(results).toHaveLength(3);
      expect(sd.state).toBe('destroyed');
    });

    it('sets state to destroyed', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      expect(sd.state).toBe('running');
      await sd.destroyAsync();
      expect(sd.state).toBe('destroyed');
    });
  });

  describe('sample rate warning', () => {
    it('emits warning when sample rate is not 48000', async () => {
      const onWarning = vi.fn();
      const sd = await createWithReady(opts(mockCtx({ sampleRate: 44100 }), { onWarning }));
      expect(onWarning).toHaveBeenCalledWith(expect.stringContaining('44100'));
      sd.destroy();
    });
  });

  describe('bypass / agc / hpf setters', () => {
    it('posts set_bypass message', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.bypass = true;
      expect(activeMockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'set_bypass', enabled: true }),
      );
      sd.destroy();
    });

    it('posts set_agc message', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.agcEnabled = true;
      expect(activeMockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'set_agc', enabled: true }),
      );
      sd.destroy();
    });

    it('posts set_hpf message', async () => {
      const sd = await createWithReady(opts(mockCtx()));
      sd.hpfEnabled = true;
      expect(activeMockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'set_hpf', enabled: true }),
      );
      sd.destroy();
    });
  });
});
