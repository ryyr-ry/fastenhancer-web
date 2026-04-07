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

  describe('keepAliveInBackground', () => {
    let mockOscillator: any;
    let mockGainNode: any;
    let mockDestination: any;
    let documentListeners: Map<string, Set<(...args: any[]) => void>>;
    let savedDocument: any;
    let savedMediaMetadata: any;
    let navigatorDescriptor: PropertyDescriptor | undefined;
    let mockMediaSession: any;

    beforeEach(() => {
      savedDocument = (globalThis as any).document;
      savedMediaMetadata = (globalThis as any).MediaMetadata;
      navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'navigator');

      mockOscillator = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        start: vi.fn(),
        stop: vi.fn(),
      };
      mockGainNode = {
        connect: vi.fn(),
        disconnect: vi.fn(),
        gain: { value: 1 },
      };
      mockDestination = {};
      documentListeners = new Map();

      (globalThis as any).document = {
        visibilityState: 'visible',
        addEventListener: vi.fn((type: string, handler: (...a: any[]) => void) => {
          if (!documentListeners.has(type)) documentListeners.set(type, new Set());
          documentListeners.get(type)!.add(handler);
        }),
        removeEventListener: vi.fn((type: string, handler: (...a: any[]) => void) => {
          documentListeners.get(type)?.delete(handler);
        }),
      };

      mockMediaSession = {
        metadata: null,
        playbackState: 'none',
      };

      Object.defineProperty(globalThis, 'navigator', {
        value: { mediaSession: mockMediaSession },
        writable: true,
        configurable: true,
      });

      (globalThis as any).MediaMetadata = class {
        title: string;
        artist: string;
        constructor(init: { title: string; artist: string }) {
          this.title = init.title;
          this.artist = init.artist;
        }
      };
    });

    afterEach(() => {
      if (savedDocument !== undefined) {
        (globalThis as any).document = savedDocument;
      } else {
        delete (globalThis as any).document;
      }
      if (navigatorDescriptor) {
        Object.defineProperty(globalThis, 'navigator', navigatorDescriptor);
      } else {
        delete (globalThis as any).navigator;
      }
      if (savedMediaMetadata !== undefined) {
        (globalThis as any).MediaMetadata = savedMediaMetadata;
      } else {
        delete (globalThis as any).MediaMetadata;
      }
    });

    function keepAliveCtx(overrides?: Record<string, any>): any {
      return mockCtx({
        createOscillator: vi.fn(() => mockOscillator),
        createGain: vi.fn(() => mockGainNode),
        destination: mockDestination,
        ...overrides,
      });
    }

    it('creates silent oscillator connected through gain to destination', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      expect(ctx.createOscillator).toHaveBeenCalled();
      expect(ctx.createGain).toHaveBeenCalled();
      expect(mockGainNode.gain.value).toBe(0);
      expect(mockOscillator.connect).toHaveBeenCalledWith(mockGainNode);
      expect(mockGainNode.connect).toHaveBeenCalledWith(mockDestination);
      expect(mockOscillator.start).toHaveBeenCalled();
      sd.destroy();
    });

    it('registers visibilitychange listener on document', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      expect((globalThis as any).document.addEventListener).toHaveBeenCalledWith(
        'visibilitychange',
        expect.any(Function),
      );
      sd.destroy();
    });

    it('posts set_background_mode when document becomes hidden', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      activeMockPort.postMessage.mockClear();

      (globalThis as any).document.visibilityState = 'hidden';
      const handlers = documentListeners.get('visibilitychange');
      handlers?.forEach((h) => h());

      expect(activeMockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'set_background_mode', enabled: true }),
      );
      sd.destroy();
    });

    it('posts set_background_mode disabled and resumes AudioContext when returning to foreground', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      activeMockPort.postMessage.mockClear();
      ctx.resume.mockClear();

      (globalThis as any).document.visibilityState = 'visible';
      const handlers = documentListeners.get('visibilitychange');
      handlers?.forEach((h) => h());

      expect(activeMockPort.postMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'set_background_mode', enabled: false }),
      );
      expect(ctx.resume).toHaveBeenCalled();
      sd.destroy();
    });

    it('sets mediaSession metadata and playbackState when available', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      expect(mockMediaSession.metadata).not.toBeNull();
      expect(mockMediaSession.playbackState).toBe('playing');
      sd.destroy();
    });

    it('cleans up oscillator, gain, visibilitychange listener, and mediaSession on destroy', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      await sd.destroyAsync();

      expect(mockOscillator.stop).toHaveBeenCalled();
      expect(mockOscillator.disconnect).toHaveBeenCalled();
      expect(mockGainNode.disconnect).toHaveBeenCalled();
      expect((globalThis as any).document.removeEventListener).toHaveBeenCalledWith(
        'visibilitychange',
        expect.any(Function),
      );
      expect(mockMediaSession.playbackState).toBe('none');
    });

    it('does not create keepalive resources when option is false', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: false }));
      expect(ctx.createOscillator).not.toHaveBeenCalled();
      expect(ctx.createGain).not.toHaveBeenCalled();
      sd.destroy();
    });

    it('does not create keepalive resources when option is omitted', async () => {
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx));
      expect(ctx.createOscillator).not.toHaveBeenCalled();
      expect(ctx.createGain).not.toHaveBeenCalled();
      sd.destroy();
    });

    it('does not throw in SSR environment without document', async () => {
      delete (globalThis as any).document;
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      expect(sd.state).toBe('running');
      // Oscillator is still created (AudioContext is available, document is not)
      expect(ctx.createOscillator).toHaveBeenCalled();
      sd.destroy();
    });

    it('does not throw when navigator.mediaSession is unavailable', async () => {
      Object.defineProperty(globalThis, 'navigator', {
        value: {},
        writable: true,
        configurable: true,
      });
      const ctx = keepAliveCtx();
      const sd = await createWithReady(opts(ctx, { keepAliveInBackground: true }));
      expect(sd.state).toBe('running');
      sd.destroy();
    });
  });
});
