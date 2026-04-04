/**
 * stream-denoiser.ts — AudioWorklet-integrated Layer 2 API
 *
 * Accepts a MediaStream input, performs WASM noise removal inside AudioWorklet,
 * and outputs a cleaned MediaStream.
 *
 * Responsibility: manages AudioContext/AudioWorklet/MediaStream connections only.
 * The details of WASM inference are handled by processor.js on the worklet side.
 */

import { DestroyedError, AudioContextError, WorkletError } from './errors.js';

/** Sample rate expected by the model (Hz) */
const TARGET_SAMPLE_RATE = 48000;

/** Maximum time to wait for Worklet initialization to complete (ms) */
const WORKLET_INIT_TIMEOUT_MS = 10_000;

/** Maximum time to wait for Worklet state retrieval (ms) */
const WORKLET_STATE_TIMEOUT_MS = 3_000;

type StreamDenoiserState = 'initializing' | 'running' | 'destroyed';

/** Options for createStreamDenoiser */
export interface StreamDenoiserOptions {
  /** Input microphone stream */
  inputStream: MediaStream;
  /** Binary extracted from WASM SINGLE_FILE (ArrayBuffer) */
  wasmBytes: ArrayBuffer;
  /** Weight binary (ArrayBuffer) */
  weightBytes: ArrayBuffer;
  /** WASM export map */
  exportMap: Record<string, string>;
  /** Model size ID (0=tiny, 1=base, 2=small) */
  modelSize: number;
  /** URL of the AudioWorklet JS file (default is used if omitted) */
  workletUrl?: string;
  /** Warning callback */
  onWarning?: (message: string) => void;
}

/** Return type of createStreamDenoiser */
export interface StreamDenoiser {
  /** Clean output stream */
  readonly outputStream: MediaStream;
  /** Current state */
  readonly state: StreamDenoiserState;
  /** Bypass mode */
  bypass: boolean;
  /** AGC enabled/disabled */
  agcEnabled: boolean;
  /** HPF enabled/disabled */
  hpfEnabled: boolean;
  /** Releases all resources */
  destroy(): void;
  /** Queries the internal state on the Worklet side (for debugging and testing) */
  getWorkletState(): Promise<WorkletStateResponse>;
}

/** Internal state returned from the Worklet side */
export interface WorkletStateResponse {
  bypass: boolean;
  agcEnabled: boolean;
  hpfEnabled: boolean;
  initialized: boolean;
  destroyed: boolean;
  autoPassthrough: boolean;
}

/**
 * Creates a stream denoiser that performs real-time noise removal through AudioWorklet.
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
      `Failed to create AudioContext: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (audioContext.state === 'suspended') {
    try {
      await audioContext.resume();
    } catch (resumeErr) {
      if (onWarning) {
        onWarning(
          `AudioContext.resume() failed (possibly due to autoplay restrictions): ${resumeErr instanceof Error ? resumeErr.message : String(resumeErr)}`,
        );
      }
    }
  }

  if (audioContext.sampleRate !== TARGET_SAMPLE_RATE && onWarning) {
    onWarning(
      `AudioContext sample rate is ${audioContext.sampleRate}Hz (expected: ${TARGET_SAMPLE_RATE}Hz).` +
      ` Audio quality may be reduced.`,
    );
  }

  const processorUrl = workletUrl ?? new URL('../worklet/processor.js', import.meta.url).href;

  try {
    await audioContext.audioWorklet.addModule(processorUrl);
  } catch (err) {
    await audioContext.close();
    throw new WorkletError(
      `Failed to register AudioWorklet: ${err instanceof Error ? err.message : String(err)}`,
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
      if (onWarning) onWarning(`AudioContext.close() failed: ${closeErr instanceof Error ? closeErr.message : String(closeErr)}`);
    });
    throw new WorkletError(
      `Failed to construct AudioWorklet node: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  const initPromise = new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      workletNode.port.removeEventListener('message', handler);
      reject(new WorkletError(`Worklet initialization timed out (${WORKLET_INIT_TIMEOUT_MS / 1000}s)`));
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
          `Worklet initialization failed: ${event.data.message ?? 'Unknown error'}`,
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
      if (onWarning) onWarning(`source.disconnect() failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    try { workletNode.disconnect(); } catch (e: unknown) {
      if (onWarning) onWarning(`workletNode.disconnect() failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    audioContext.close().catch((closeErr: unknown) => {
      if (onWarning) onWarning(`AudioContext.close() failed: ${closeErr instanceof Error ? closeErr.message : String(closeErr)}`);
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
        options.onWarning(`Worklet processing error: ${event.data.message ?? 'Unknown'}`);
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
        throw new DestroyedError('This StreamDenoiser has already been destroyed');
      }
      bypassMode = value;
      workletNode.port.postMessage({ type: 'set_bypass', enabled: value });
    },

    get agcEnabled(): boolean {
      return agcMode;
    },
    set agcEnabled(value: boolean) {
      if (currentState === 'destroyed') {
        throw new DestroyedError('This StreamDenoiser has already been destroyed');
      }
      agcMode = value;
      workletNode.port.postMessage({ type: 'set_agc', enabled: value });
    },

    get hpfEnabled(): boolean {
      return hpfMode;
    },
    set hpfEnabled(value: boolean) {
      if (currentState === 'destroyed') {
        throw new DestroyedError('This StreamDenoiser has already been destroyed');
      }
      hpfMode = value;
      workletNode.port.postMessage({ type: 'set_hpf', enabled: value });
    },

    destroy(): void {
      if (currentState === 'destroyed') return;

      currentState = 'destroyed';

      try { source.disconnect(); } catch (e) {
        if (onWarning) onWarning(`source.disconnect() failed during destroy: ${e instanceof Error ? e.message : String(e)}`);
      }
      try { workletNode.disconnect(); } catch (e) {
        if (onWarning) onWarning(`workletNode.disconnect() failed during destroy: ${e instanceof Error ? e.message : String(e)}`);
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
          if (onWarning) onWarning(`AudioContext.close() failed during destroy: ${e instanceof Error ? e.message : String(e)}`);
        });
      });
    },

    getWorkletState(): Promise<WorkletStateResponse> {
      if (currentState === 'destroyed') {
        return Promise.reject(
          new DestroyedError('This StreamDenoiser has already been destroyed'),
        );
      }
      const requestId = `state_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      return new Promise<WorkletStateResponse>((resolve, reject) => {
        const timeout = setTimeout(() => {
          workletNode.port.removeEventListener('message', handler);
          reject(new WorkletError(`Worklet state retrieval timed out (${WORKLET_STATE_TIMEOUT_MS / 1000}s)`));
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
