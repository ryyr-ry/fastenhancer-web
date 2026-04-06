/**
 * @vitest-environment jsdom
 */

/**
 * useDenoiser.test.ts — React hook unit tests
 *
 * Tests the React lifecycle management, state machine transitions,
 * race conditions, and cleanup logic of useDenoiser.
 *
 * Uses mocked loadModel/createStreamDenoiser to isolate the hook logic
 * from actual WASM/AudioWorklet processing.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDenoiser } from '../../../src/react/useDenoiser.js';

// Type for the mock StreamDenoiser
interface MockStreamDenoiser {
  outputStream: MediaStream;
  state: string;
  bypass: boolean;
  destroy: ReturnType<typeof vi.fn>;
}

// Type for LoadedModel mock
interface MockLoadedModel {
  size: string;
  wasmBytes: ArrayBuffer;
  weightData: ArrayBuffer;
  exportMap: Record<string, string>;
  createDenoiser: ReturnType<typeof vi.fn>;
  createStreamDenoiser: ReturnType<typeof vi.fn>;
}

let mockLoadModel: ReturnType<typeof vi.fn>;

vi.mock('../../../src/api/loader.js', () => ({
  loadModel: (...args: any[]) => mockLoadModel(...args),
}));

function createMockMediaStream(): MediaStream {
  return {
    getTracks: () => [],
    getAudioTracks: () => [],
    getVideoTracks: () => [],
    addTrack: vi.fn(),
    removeTrack: vi.fn(),
    clone: vi.fn(),
    id: 'mock-stream',
    active: true,
    onaddtrack: null,
    onremovetrack: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(() => true),
  } as unknown as MediaStream;
}

function createMockStreamDenoiser(): MockStreamDenoiser {
  return {
    outputStream: createMockMediaStream(),
    state: 'running',
    bypass: false,
    destroy: vi.fn(),
  };
}

function createMockLoadedModel(
  streamDenoiser?: MockStreamDenoiser,
): MockLoadedModel {
  const sd = streamDenoiser ?? createMockStreamDenoiser();
  return {
    size: 'tiny',
    wasmBytes: new ArrayBuffer(8),
    weightData: new ArrayBuffer(8),
    exportMap: { _fe_init: 'a' },
    createDenoiser: vi.fn(),
    createStreamDenoiser: vi.fn().mockResolvedValue(sd),
  };
}

describe('useDenoiser', () => {
  beforeEach(() => {
    mockLoadModel = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('initial state', () => {
    it('starts in idle state with no stream and no error', () => {
      mockLoadModel.mockResolvedValue(createMockLoadedModel());
      const { result } = renderHook(() => useDenoiser('tiny'));

      expect(result.current.state).toBe('idle');
      expect(result.current.error).toBeNull();
      expect(result.current.outputStream).toBeNull();
      expect(result.current.bypass).toBe(false);
    });
  });

  describe('start → processing flow', () => {
    it('transitions idle → loading → processing on successful start', async () => {
      const sd = createMockStreamDenoiser();
      const model = createMockLoadedModel(sd);
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      expect(result.current.state).toBe('processing');
      expect(result.current.outputStream).toBe(sd.outputStream);
      expect(result.current.error).toBeNull();
    });

    it('calls loadModel with the correct model size and options', async () => {
      const model = createMockLoadedModel();
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() =>
        useDenoiser('base', { simd: true, baseUrl: '/assets/' }),
      );

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      expect(mockLoadModel).toHaveBeenCalledWith('base', expect.objectContaining({
        baseUrl: '/assets/',
        simd: true,
      }));
    });
  });

  describe('stop flow', () => {
    it('transitions processing → idle and clears output stream', async () => {
      const sd = createMockStreamDenoiser();
      const model = createMockLoadedModel(sd);
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });
      expect(result.current.state).toBe('processing');

      act(() => {
        result.current.stop();
      });

      expect(result.current.state).toBe('idle');
      expect(result.current.outputStream).toBeNull();
      expect(sd.destroy).toHaveBeenCalled();
    });
  });

  describe('destroy flow', () => {
    it('transitions to destroyed state', () => {
      mockLoadModel.mockResolvedValue(createMockLoadedModel());
      const { result } = renderHook(() => useDenoiser('tiny'));

      act(() => {
        result.current.destroy();
      });

      expect(result.current.state).toBe('destroyed');
    });

    it('destroyed is terminal — start after destroy silently no-ops', async () => {
      mockLoadModel.mockResolvedValue(createMockLoadedModel());
      const { result } = renderHook(() => useDenoiser('tiny'));

      act(() => {
        result.current.destroy();
      });
      expect(result.current.state).toBe('destroyed');

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      expect(result.current.state).toBe('destroyed');
      expect(mockLoadModel).not.toHaveBeenCalled();
    });

    it('stop after destroy silently no-ops', () => {
      mockLoadModel.mockResolvedValue(createMockLoadedModel());
      const { result } = renderHook(() => useDenoiser('tiny'));

      act(() => {
        result.current.destroy();
      });

      act(() => {
        result.current.stop();
      });

      expect(result.current.state).toBe('destroyed');
    });

    it('setBypass after destroy silently no-ops', () => {
      mockLoadModel.mockResolvedValue(createMockLoadedModel());
      const { result } = renderHook(() => useDenoiser('tiny'));

      act(() => {
        result.current.destroy();
      });

      act(() => {
        result.current.setBypass(true);
      });

      expect(result.current.bypass).toBe(false);
    });

    it('destroy is idempotent', async () => {
      const sd = createMockStreamDenoiser();
      const model = createMockLoadedModel(sd);
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      act(() => {
        result.current.destroy();
      });
      expect(sd.destroy).toHaveBeenCalledTimes(1);

      act(() => {
        result.current.destroy();
      });
      expect(sd.destroy).toHaveBeenCalledTimes(1);
      expect(result.current.state).toBe('destroyed');
    });
  });

  describe('error handling', () => {
    it('transitions to error state on loadModel failure', async () => {
      const loadError = new Error('WASM load failed');
      mockLoadModel.mockRejectedValue(loadError);
      const onError = vi.fn();

      const { result } = renderHook(() =>
        useDenoiser('tiny', { onError }),
      );

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      expect(result.current.state).toBe('error');
      expect(result.current.error).toBe(loadError);
      expect(onError).toHaveBeenCalledWith(loadError);
    });

    it('transitions to error on createStreamDenoiser failure', async () => {
      const sdError = new Error('AudioWorklet init failed');
      const model = createMockLoadedModel();
      model.createStreamDenoiser.mockRejectedValue(sdError);
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      expect(result.current.state).toBe('error');
      expect(result.current.error).toBe(sdError);
    });
  });

  describe('bypass', () => {
    it('updates bypass state', async () => {
      const sd = createMockStreamDenoiser();
      const model = createMockLoadedModel(sd);
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      act(() => {
        result.current.setBypass(true);
      });

      expect(result.current.bypass).toBe(true);
      expect(sd.bypass).toBe(true);
    });

    it('toggles bypass off after on', async () => {
      const sd = createMockStreamDenoiser();
      const model = createMockLoadedModel(sd);
      mockLoadModel.mockResolvedValue(model);

      const { result } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      act(() => {
        result.current.setBypass(true);
      });
      act(() => {
        result.current.setBypass(false);
      });

      expect(result.current.bypass).toBe(false);
      expect(sd.bypass).toBe(false);
    });
  });

  describe('race conditions', () => {
    it('cancels stale start when a new start is called', async () => {
      let resolveFirst: (value: MockLoadedModel) => void;
      const firstPromise = new Promise<MockLoadedModel>((r) => { resolveFirst = r; });
      const secondModel = createMockLoadedModel();

      mockLoadModel
        .mockReturnValueOnce(firstPromise)
        .mockResolvedValueOnce(secondModel);

      const { result } = renderHook(() => useDenoiser('tiny'));

      const stream1 = createMockMediaStream();
      const stream2 = createMockMediaStream();

      let firstStartDone = false;
      act(() => {
        result.current.start(stream1).then(() => { firstStartDone = true; });
      });

      await act(async () => {
        await result.current.start(stream2);
      });

      resolveFirst!(createMockLoadedModel());
      await act(async () => {
        await firstPromise.catch(() => {});
      });

      expect(result.current.state).toBe('processing');
    });

    it('stop during loading cancels the pending start', async () => {
      let resolveLoad: (value: MockLoadedModel) => void;
      const loadPromise = new Promise<MockLoadedModel>((r) => { resolveLoad = r; });
      mockLoadModel.mockReturnValue(loadPromise);

      const { result } = renderHook(() => useDenoiser('tiny'));

      act(() => {
        result.current.start(createMockMediaStream());
      });

      act(() => {
        result.current.stop();
      });
      expect(result.current.state).toBe('idle');

      resolveLoad!(createMockLoadedModel());
      await act(async () => {
        await loadPromise;
      });

      expect(result.current.state).toBe('idle');
    });
  });

  describe('unmount cleanup', () => {
    it('destroys the stream denoiser on unmount', async () => {
      const sd = createMockStreamDenoiser();
      const model = createMockLoadedModel(sd);
      mockLoadModel.mockResolvedValue(model);

      const { result, unmount } = renderHook(() => useDenoiser('tiny'));

      await act(async () => {
        await result.current.start(createMockMediaStream());
      });

      unmount();
      expect(sd.destroy).toHaveBeenCalled();
    });
  });
});
