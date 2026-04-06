/**
 * useDenoiser — React Hook for real-time noise removal (Layer 3)
 *
 * Responsibility: manages the React lifecycle and the denoising stream connection.
 * Delegates resource loading to loadModel and AudioWorklet processing to createStreamDenoiser.
 *
 * start() with no arguments automatically acquires the microphone via getUserMedia.
 * start(existingStream) uses the provided MediaStream instead.
 * The hook owns and releases internally-acquired mic streams on stop/destroy/unmount.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import type { StreamDenoiser } from '../api/stream-denoiser.js';
import { loadModel, type LoadedModel } from '../api/loader.js';

export type ModelSize = 'tiny' | 'base' | 'small';

export type UseDenoiserState =
  | 'idle'
  | 'loading'
  | 'processing'
  | 'error'
  | 'destroyed';

export interface UseDenoiserOptions {
  /** Base URL for all resource files (omitted: auto-detected from import.meta.url) */
  baseUrl?: string;
  /** Explicitly specify SIMD usage (omitted: auto-detected) */
  simd?: boolean;
  /** AudioWorklet JS file URL */
  workletUrl?: string;
  /** Audio constraints passed to getUserMedia when start() is called without arguments */
  audioConstraints?: MediaTrackConstraints;
  /** Warning callback */
  onWarning?: (message: string) => void;
  /** Error callback */
  onError?: (error: Error) => void;
}

export interface UseDenoiserReturn {
  /** Current state */
  state: UseDenoiserState;
  /** Error information (non-null only in the error state) */
  error: Error | null;
  /** Input stream being denoised (non-null only while processing) */
  inputStream: MediaStream | null;
  /** Denoised output stream (non-null only while processing) */
  outputStream: MediaStream | null;
  /** Bypass mode */
  bypass: boolean;
  /** Start denoising — call with no args to auto-acquire mic, or pass your own MediaStream */
  start: (inputStream?: MediaStream) => Promise<void>;
  /** Stop denoising and return to the idle state */
  stop: () => void;
  /** Set bypass mode */
  setBypass: (value: boolean) => void;
  /** Release all resources (cannot be reused) */
  destroy: () => void;
}

/**
 * React Hook: performs real-time noise removal in the browser.
 *
 * @example
 * ```tsx
 * const { outputStream, start, stop, state } = useDenoiser('small');
 *
 * <button onClick={start}>Start</button>
 * <button onClick={stop}>Stop</button>
 * {outputStream && <audio autoPlay ref={el => el && (el.srcObject = outputStream)} />}
 * ```
 */
export function useDenoiser(
  modelSize: ModelSize,
  options?: UseDenoiserOptions,
): UseDenoiserReturn {
  const [state, setState] = useState<UseDenoiserState>('idle');
  const [error, setError] = useState<Error | null>(null);
  const [inputStream, setInputStream] = useState<MediaStream | null>(null);
  const [outputStream, setOutputStream] = useState<MediaStream | null>(null);
  const [bypass, setBypassState] = useState(false);

  const streamDenoiserRef = useRef<StreamDenoiser | null>(null);
  const ownedMicStreamRef = useRef<MediaStream | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);
  const destroyedRef = useRef(false);
  const requestIdRef = useRef(0);
  const modelSizeRef = useRef(modelSize);
  const optionsRef = useRef(options);
  const bypassRef = useRef(bypass);

  modelSizeRef.current = modelSize;
  optionsRef.current = options;
  bypassRef.current = bypass;

  const warn = useCallback((message: string) => {
    const onWarning = optionsRef.current?.onWarning;
    if (onWarning) {
      onWarning(message);
      return;
    }
    console.warn(message);
  }, []);

  const releaseOwnedMicStream = useCallback(() => {
    const mic = ownedMicStreamRef.current;
    if (mic) {
      for (const track of mic.getTracks()) {
        track.stop();
      }
      ownedMicStreamRef.current = null;
    }
  }, []);

  const cleanupStreamDenoiser = useCallback(() => {
    requestIdRef.current++;
    const ac = abortControllerRef.current;
    if (ac) {
      ac.abort();
      abortControllerRef.current = null;
    }
    const sd = streamDenoiserRef.current;
    if (sd) {
      try { sd.destroy(); } catch (e) {
        warn(`useDenoiser destroy failed during cleanup: ${e instanceof Error ? e.message : String(e)}`);
      }
      streamDenoiserRef.current = null;
    }
    releaseOwnedMicStream();
    if (mountedRef.current) {
      setInputStream(null);
      setOutputStream(null);
    }
  }, [warn, releaseOwnedMicStream]);

  const start = useCallback(async (inputStream?: MediaStream) => {
    if (destroyedRef.current || !mountedRef.current) {
      return;
    }

    cleanupStreamDenoiser();
    const thisRequestId = requestIdRef.current;
    const ac = new AbortController();
    abortControllerRef.current = ac;
    setState('loading');
    setError(null);

    try {
      let stream: MediaStream;
      if (inputStream) {
        stream = inputStream;
      } else {
        const constraints = optionsRef.current?.audioConstraints;
        stream = await navigator.mediaDevices.getUserMedia({
          audio: constraints ?? true,
        });
        ownedMicStreamRef.current = stream;
      }

      if (destroyedRef.current || !mountedRef.current || requestIdRef.current !== thisRequestId) {
        if (!inputStream) {
          for (const track of stream.getTracks()) track.stop();
          ownedMicStreamRef.current = null;
        }
        return;
      }

      const opts = optionsRef.current;

      const model: LoadedModel = await loadModel(modelSizeRef.current, {
        baseUrl: opts?.baseUrl,
        simd: opts?.simd,
        signal: ac.signal,
      });

      if (destroyedRef.current || !mountedRef.current || requestIdRef.current !== thisRequestId) {
        if (!inputStream) {
          for (const track of stream.getTracks()) track.stop();
          ownedMicStreamRef.current = null;
        }
        return;
      }

      const sd = await model.createStreamDenoiser(stream, {
        workletUrl: opts?.workletUrl,
        onWarning: (msg: string) => optionsRef.current?.onWarning?.(msg),
        onDestroy: () => {
          if (mountedRef.current && !destroyedRef.current) {
            streamDenoiserRef.current = null;
            releaseOwnedMicStream();
            setInputStream(null);
            setOutputStream(null);
            setState('destroyed');
          }
        },
      });

      if (destroyedRef.current || !mountedRef.current || requestIdRef.current !== thisRequestId) {
        try { sd.destroy(); } catch (e) {
          warn(`useDenoiser destroy failed during race-condition cleanup: ${e instanceof Error ? e.message : String(e)}`);
        }
        if (!inputStream) {
          for (const track of stream.getTracks()) track.stop();
          ownedMicStreamRef.current = null;
        }
        return;
      }

      streamDenoiserRef.current = sd;
      sd.bypass = bypassRef.current;
      setInputStream(stream);
      setOutputStream(sd.outputStream);
      setState('processing');
    } catch (err) {
      if (requestIdRef.current !== thisRequestId) return;
      if (ac.signal.aborted) return;
      releaseOwnedMicStream();
      const e = err instanceof Error ? err : new Error(String(err));
      setError(e);
      setState('error');
      optionsRef.current?.onError?.(e);
    }
  }, [cleanupStreamDenoiser, warn, releaseOwnedMicStream]);

  const stop = useCallback(() => {
    if (destroyedRef.current || !mountedRef.current) return;
    cleanupStreamDenoiser();
    setState('idle');
    setError(null);
  }, [cleanupStreamDenoiser]);

  const setBypass = useCallback((value: boolean) => {
    if (destroyedRef.current || !mountedRef.current) return;
    setBypassState(value);
    const sd = streamDenoiserRef.current;
    if (sd && sd.state !== 'destroyed') {
      sd.bypass = value;
    }
  }, []);

  const destroy = useCallback(() => {
    if (destroyedRef.current || !mountedRef.current) return;
    destroyedRef.current = true;
    cleanupStreamDenoiser();
    setError(null);
    setState('destroyed');
  }, [cleanupStreamDenoiser]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cleanupStreamDenoiser();
    };
  }, [cleanupStreamDenoiser]);

  return {
    state,
    error,
    inputStream,
    outputStream,
    bypass,
    start,
    stop,
    setBypass,
    destroy,
  };
}
