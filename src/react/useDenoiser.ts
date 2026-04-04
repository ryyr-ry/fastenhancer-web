/**
 * useDenoiser — React Hook for real-time noise removal (Layer 3)
 *
 * 責務: Reactライフサイクルとノイズ除去ストリームの接続管理のみ。
 * リソースロードはloadModelに、AudioWorklet処理はcreateStreamDenoiserに委譲する。
 *
 * v2: wasmBytes/weightBytes/exportMap は不要。
 * loadModel が自動的にリソースを取得し、createStreamDenoiser に渡す。
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
  /** 全リソースファイルのベースURL（省略時: import.meta.url自動検出） */
  baseUrl?: string;
  /** SIMD使用を明示指定（省略時: 自動検出） */
  simd?: boolean;
  /** AudioWorklet JSファイルURL */
  workletUrl?: string;
  /** 警告コールバック */
  onWarning?: (message: string) => void;
  /** エラーコールバック */
  onError?: (error: Error) => void;
}

export interface UseDenoiserReturn {
  /** 現在の状態 */
  state: UseDenoiserState;
  /** エラー情報（error状態時のみ非null） */
  error: Error | null;
  /** ノイズ除去済み出力ストリーム（processing時のみ非null） */
  outputStream: MediaStream | null;
  /** バイパスモード */
  bypass: boolean;
  /** ノイズ除去を開始する */
  start: (inputStream: MediaStream) => Promise<void>;
  /** ノイズ除去を停止してidle状態に戻る */
  stop: () => void;
  /** バイパスモードを設定する */
  setBypass: (value: boolean) => void;
  /** 全リソースを解放する（再利用不可） */
  destroy: () => void;
}

/**
 * React Hook: ブラウザでリアルタイムノイズ除去を行う。
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
        console.warn('useDenoiser cleanup時のdestroy失敗:', e instanceof Error ? e.message : String(e));
      }
      streamDenoiserRef.current = null;
    }
    setOutputStream(null);
  }, []);

  const start = useCallback(async (inputStream: MediaStream) => {
    if (destroyedRef.current) {
      setError(new DestroyedError('このuseDenoiserは既に破棄されています'));
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
          console.warn('useDenoiser レース条件cleanup時のdestroy失敗:', e instanceof Error ? e.message : String(e));
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
