/**
 * useDenoiser — React Hook for real-time noise removal (Layer 3)
 *
 * Responsibility: only manages the React lifecycle and the denoising stream connection.
 * Delegate resource loading to loadModel and AudioWorklet processing to createStreamDenoiser.
 *
 * v2: wasmBytes/weightBytes/exportMap are unnecessary.
 * loadModel automatically fetches the resources and passes them to createStreamDenoiser.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import type { StreamDenoiser } from '../api/stream-denoiser.js';
import { loadModel, type LoadedModel } from '../api/loader.js';
import { DestroyedError } from '../api/errors.js';

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
  /** Denoised output stream (non-null only while processing) */
  outputStream: MediaStream | null;
  /** Bypass mode */
  bypass: boolean;
  /** Start denoising */
  start: (inputStream: MediaStream) => Promise<void>;
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
 * const { outputStream, start, stop, state } = useDenoiser('tiny');
 *
 * const handleStart = async () => {
 *   const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
 *   await start(mic);
 * };
 * ```
 */
export function useDenoiser(
  modelSize: ModelSize,
  options?: UseDenoiserOptions,
): UseDenoiserReturn {
  const [state, setState] = useState<UseDenoiserState>('idle');
  const [error, setError] = useState<Error | null>(null);
  const [outputStream, setOutputStream] = useState<MediaStream | null>(null);
  const [bypass, setBypassState] = useState(false);

  const streamDenoiserRef = useRef<StreamDenoiser | null>(null);
  const destroyedRef = useRef(false);
  const requestIdRef = useRef(0);
  const modelSizeRef = useRef(modelSize);
  const optionsRef = useRef(options);
  const bypassRef = useRef(bypass);

  modelSizeRef.current = modelSize;
  optionsRef.current = options;
  bypassRef.current = bypass;

  const cleanupStreamDenoiser = useCallback(() => {
    requestIdRef.current++;
    const sd = streamDenoiserRef.current;
    if (sd) {
      try { sd.destroy(); } catch (e) {
        console.warn('useDenoiser destroy failed during cleanup:', e instanceof Error ? e.message : String(e));
      }
      streamDenoiserRef.current = null;
    }
    setOutputStream(null);
  }, []);

  const start = useCallback(async (inputStream: MediaStream) => {
    if (destroyedRef.current) {
      setError(new DestroyedError('This useDenoiser instance has already been destroyed'));
      setState('error');
      return;
    }

    cleanupStreamDenoiser();
    const thisRequestId = requestIdRef.current;
    setState('loading');
    setError(null);

    try {
      const opts = optionsRef.current;

      const model: LoadedModel = await loadModel(modelSizeRef.current, {
        baseUrl: opts?.baseUrl,
        simd: opts?.simd,
      });

      if (destroyedRef.current || requestIdRef.current !== thisRequestId) {
        return;
      }

      const sd = await model.createStreamDenoiser(inputStream, {
        workletUrl: opts?.workletUrl,
        onWarning: opts?.onWarning,
      });

      if (destroyedRef.current || requestIdRef.current !== thisRequestId) {
        try { sd.destroy(); } catch (e) {
          console.warn('useDenoiser destroy failed during race-condition cleanup:', e instanceof Error ? e.message : String(e));
        }
        return;
      }

      streamDenoiserRef.current = sd;
      sd.bypass = bypassRef.current;
      setOutputStream(sd.outputStream);
      setState('processing');
    } catch (err) {
      if (requestIdRef.current !== thisRequestId) return;
      const e = err instanceof Error ? err : new Error(String(err));
      setError(e);
      setState('error');
      optionsRef.current?.onError?.(e);
    }
  }, [cleanupStreamDenoiser]);

  const stop = useCallback(() => {
    cleanupStreamDenoiser();
    if (!destroyedRef.current) {
      setState('idle');
      setError(null);
    }
  }, [cleanupStreamDenoiser]);

  const setBypass = useCallback((value: boolean) => {
    setBypassState(value);
    const sd = streamDenoiserRef.current;
    if (sd && sd.state !== 'destroyed') {
      sd.bypass = value;
    }
  }, []);

  const destroy = useCallback(() => {
    if (destroyedRef.current) return;
    destroyedRef.current = true;
    cleanupStreamDenoiser();
    setState('destroyed');
  }, [cleanupStreamDenoiser]);

  useEffect(() => {
    return () => {
      cleanupStreamDenoiser();
    };
  }, [cleanupStreamDenoiser]);

  return {
    state,
    error,
    outputStream,
    bypass,
    start,
    stop,
    setBypass,
    destroy,
  };
}
