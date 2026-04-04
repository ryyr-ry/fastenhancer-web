/**
 * stream-denoiser.ts — AudioWorklet 統合 Layer 2 API
 *
 * MediaStream 入力を受け取り、AudioWorklet 内で WASM ノイズ除去を行い、
 * クリーンな MediaStream を出力する。
 *
 * 責務: AudioContext/AudioWorklet/MediaStream の接続管理のみ。
 * WASM推論の詳細は processor.js (Worklet側) が担う。
 */

import { DestroyedError, AudioContextError, WorkletError } from './errors.js';

/** モデルが期待するサンプルレート (Hz) */
const TARGET_SAMPLE_RATE = 48000;

/** Worklet初期化完了を待つ最大時間 (ms) */
const WORKLET_INIT_TIMEOUT_MS = 10_000;

/** Worklet状態取得の最大待機時間 (ms) */
const WORKLET_STATE_TIMEOUT_MS = 3_000;

type StreamDenoiserState = 'initializing' | 'running' | 'destroyed';

/** createStreamDenoiser のオプション */
export interface StreamDenoiserOptions {
  /** 入力マイクストリーム */
  inputStream: MediaStream;
  /** WASM SINGLE_FILE から抽出したバイナリ（ArrayBuffer） */
  wasmBytes: ArrayBuffer;
  /** 重みバイナリ（ArrayBuffer） */
  weightBytes: ArrayBuffer;
  /** WASMエクスポートマップ */
  exportMap: Record<string, string>;
  /** モデルサイズID (0=tiny, 1=base, 2=small) */
  modelSize: number;
  /** AudioWorklet JSファイルのURL（省略時はデフォルト） */
  workletUrl?: string;
  /** 警告コールバック */
  onWarning?: (message: string) => void;
}

/** createStreamDenoiser の返却型 */
export interface StreamDenoiser {
  /** クリーンな出力ストリーム */
  readonly outputStream: MediaStream;
  /** 現在の状態 */
  readonly state: StreamDenoiserState;
  /** バイパスモード */
  bypass: boolean;
  /** AGC有効/無効 */
  agcEnabled: boolean;
  /** HPF有効/無効 */
  hpfEnabled: boolean;
  /** 全リソース解放 */
  destroy(): void;
  /** Worklet側の内部状態を問い合わせる（デバッグ・テスト用） */
  getWorkletState(): Promise<WorkletStateResponse>;
}

/** Worklet側から返される内部状態 */
export interface WorkletStateResponse {
  bypass: boolean;
  agcEnabled: boolean;
  hpfEnabled: boolean;
  initialized: boolean;
  destroyed: boolean;
  autoPassthrough: boolean;
}

/**
 * AudioWorklet経由でリアルタイムノイズ除去を行うストリームデノイザーを生成する。
 */
export async function createStreamDenoiser(
  options: StreamDenoiserOptions,
): Promise<StreamDenoiser> {
  const {
    inputStream,
    wasmBytes,
    weightBytes,
    exportMap,
    modelSize,
    workletUrl,
    onWarning,
  } = options;

  let audioContext: AudioContext;
  try {
    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  } catch (err) {
    throw new AudioContextError(
      `AudioContextの生成に失敗しました: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (audioContext.state === 'suspended') {
    try {
      await audioContext.resume();
    } catch (resumeErr) {
      if (onWarning) {
        onWarning(
          `AudioContext.resume()に失敗しました（自動再生制限の可能性）: ${resumeErr instanceof Error ? resumeErr.message : String(resumeErr)}`,
        );
      }
    }
  }

  if (audioContext.sampleRate !== TARGET_SAMPLE_RATE && onWarning) {
    onWarning(
      `AudioContextのサンプルレートが${audioContext.sampleRate}Hzです（期待: ${TARGET_SAMPLE_RATE}Hz）。` +
      `品質が低下する可能性があります。`,
    );
  }

  const processorUrl = workletUrl ?? new URL('../worklet/processor.js', import.meta.url).href;

  try {
    await audioContext.audioWorklet.addModule(processorUrl);
  } catch (err) {
    await audioContext.close();
    throw new WorkletError(
      `AudioWorkletの登録に失敗しました: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  let workletNode: AudioWorkletNode;
  let source: MediaStreamAudioSourceNode;
  let destination: MediaStreamAudioDestinationNode;
  try {
    workletNode = new AudioWorkletNode(audioContext, 'fastenhancer-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });

    source = audioContext.createMediaStreamSource(inputStream);
    destination = audioContext.createMediaStreamDestination();

    source.connect(workletNode);
    workletNode.connect(destination);
  } catch (err) {
    audioContext.close().catch((closeErr: unknown) => {
      if (onWarning) onWarning(`AudioContext.close()失敗: ${closeErr instanceof Error ? closeErr.message : String(closeErr)}`);
    });
    throw new WorkletError(
      `AudioWorkletノードの構築に失敗しました: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  const initPromise = new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      workletNode.port.removeEventListener('message', handler);
      reject(new WorkletError(`Worklet初期化タイムアウト（${WORKLET_INIT_TIMEOUT_MS / 1000}秒）`));
    }, WORKLET_INIT_TIMEOUT_MS);

    const handler = (event: MessageEvent) => {
      if (event.data?.type === 'ready') {
        clearTimeout(timeout);
        workletNode.port.removeEventListener('message', handler);
        resolve();
      } else if (event.data?.type === 'error') {
        clearTimeout(timeout);
        workletNode.port.removeEventListener('message', handler);
        reject(new WorkletError(
          `Worklet初期化に失敗しました: ${event.data.message ?? '不明なエラー'}`,
        ));
      }
    };
    workletNode.port.addEventListener('message', handler);
    workletNode.port.start();
  });

  const wasmBytesTransfer = wasmBytes.slice(0);
  const weightBytesTransfer = weightBytes.slice(0);
  workletNode.port.postMessage({
    type: 'init',
    wasmBytes: wasmBytesTransfer,
    weightBytes: weightBytesTransfer,
    exportMap,
    modelSize,
  }, [wasmBytesTransfer, weightBytesTransfer]);

  try {
    await initPromise;
  } catch (err) {
    try { source.disconnect(); } catch (e: unknown) {
      if (onWarning) onWarning(`source.disconnect()失敗: ${e instanceof Error ? e.message : String(e)}`);
    }
    try { workletNode.disconnect(); } catch (e: unknown) {
      if (onWarning) onWarning(`workletNode.disconnect()失敗: ${e instanceof Error ? e.message : String(e)}`);
    }
    audioContext.close().catch((closeErr: unknown) => {
      if (onWarning) onWarning(`AudioContext.close()失敗: ${closeErr instanceof Error ? closeErr.message : String(closeErr)}`);
    });
    throw err;
  }

  let currentState: StreamDenoiserState = 'running';
  let bypassMode = false;
  let agcMode = false;
  let hpfMode = false;

  workletNode.port.addEventListener('message', (event: MessageEvent) => {
    if (event.data?.type === 'process_error' && currentState !== 'destroyed') {
      if (options?.onWarning) {
        options.onWarning(`Worklet処理エラー: ${event.data.message ?? '不明'}`);
      }
    }
  });

  const denoiser: StreamDenoiser = {
    get outputStream(): MediaStream {
      return destination.stream;
    },

    get state(): StreamDenoiserState {
      return currentState;
    },

    get bypass(): boolean {
      return bypassMode;
    },
    set bypass(value: boolean) {
      if (currentState === 'destroyed') {
        throw new DestroyedError('このStreamDenoiserは既に破棄されています');
      }
      bypassMode = value;
      workletNode.port.postMessage({ type: 'set_bypass', enabled: value });
    },

    get agcEnabled(): boolean {
      return agcMode;
    },
    set agcEnabled(value: boolean) {
      if (currentState === 'destroyed') {
        throw new DestroyedError('このStreamDenoiserは既に破棄されています');
      }
      agcMode = value;
      workletNode.port.postMessage({ type: 'set_agc', enabled: value });
    },

    get hpfEnabled(): boolean {
      return hpfMode;
    },
    set hpfEnabled(value: boolean) {
      if (currentState === 'destroyed') {
        throw new DestroyedError('このStreamDenoiserは既に破棄されています');
      }
      hpfMode = value;
      workletNode.port.postMessage({ type: 'set_hpf', enabled: value });
    },

    destroy(): void {
      if (currentState === 'destroyed') return;

      currentState = 'destroyed';

      try { source.disconnect(); } catch (e) {
        if (onWarning) onWarning(`destroy時source.disconnect()失敗: ${e instanceof Error ? e.message : String(e)}`);
      }
      try { workletNode.disconnect(); } catch (e) {
        if (onWarning) onWarning(`destroy時workletNode.disconnect()失敗: ${e instanceof Error ? e.message : String(e)}`);
      }

      const destroyAckTimeout = 2000;
      const ackPromise = new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, destroyAckTimeout);
        const handler = (event: MessageEvent) => {
          if (event.data?.type === 'destroyed' || event.data?.type === 'error') {
            clearTimeout(timer);
            workletNode.port.removeEventListener('message', handler);
            resolve();
          }
        };
        workletNode.port.addEventListener('message', handler);
      });

      workletNode.port.postMessage({ type: 'destroy' });

      destination.stream.getTracks().forEach((track) => track.stop());

      ackPromise.then(() => {
        audioContext.close().catch((e) => {
          if (onWarning) onWarning(`destroy時AudioContext.close()失敗: ${e instanceof Error ? e.message : String(e)}`);
        });
      });
    },

    getWorkletState(): Promise<WorkletStateResponse> {
      if (currentState === 'destroyed') {
        return Promise.reject(
          new DestroyedError('このStreamDenoiserは既に破棄されています'),
        );
      }
      const requestId = `state_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      return new Promise<WorkletStateResponse>((resolve, reject) => {
        const timeout = setTimeout(() => {
          workletNode.port.removeEventListener('message', handler);
          reject(new WorkletError(`Worklet状態取得タイムアウト（${WORKLET_STATE_TIMEOUT_MS / 1000}秒）`));
        }, WORKLET_STATE_TIMEOUT_MS);

        const handler = (event: MessageEvent) => {
          if (event.data?.type === 'state' && event.data?.requestId === requestId) {
            clearTimeout(timeout);
            workletNode.port.removeEventListener('message', handler);
            resolve(event.data as WorkletStateResponse);
          }
        };
        workletNode.port.addEventListener('message', handler);
        workletNode.port.start();
        workletNode.port.postMessage({ type: 'get_state', requestId });
      });
    },
  };

  return denoiser;
}
