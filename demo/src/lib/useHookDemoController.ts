import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useDenoiser } from 'fastenhancer-web/react'
import { useT } from '../i18n'
import {
  createAudioPlayback,
  createSampleInputSession,
  formatDemoError,
  stopInputSession,
  type AudioPlayback,
  type DemoSourceMode,
  type InputSession,
  type ModelSize,
  type SampleId,
} from './demo-media'

export interface HookDemoController {
  sourceMode: DemoSourceMode
  sampleId: SampleId
  inputStream: MediaStream | null
  outputStream: MediaStream | null
  state: 'idle' | 'loading' | 'processing' | 'error' | 'destroyed'
  bypass: boolean
  modelSize: ModelSize
  volume: number
  warning: string | null
  errorMessage: string | null
  setSourceMode: (mode: DemoSourceMode) => Promise<void>
  setSampleId: (sampleId: SampleId) => Promise<void>
  setModelSize: (modelSize: ModelSize) => Promise<void>
  setVolume: (volume: number) => void
  setBypass: (value: boolean) => void
  startProcessing: () => Promise<void>
  stopProcessing: () => Promise<void>
}

export function useHookDemoController(): HookDemoController {
  const t = useT()
  const [sourceMode, setSourceModeState] = useState<DemoSourceMode>('sample')
  const [sampleId, setSampleIdState] = useState<SampleId>('speech_48k_noisy_10dB.wav')
  const [modelSize, setModelSizeState] = useState<ModelSize>('small')
  const [volume, setVolume] = useState(0.8)
  const [warning, setWarning] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)

  const sampleSessionRef = useRef<InputSession | null>(null)
  const playbackRef = useRef<AudioPlayback | null>(null)

  const options = useMemo(
    () => ({
      baseUrl: './assets/wasm/',
      workletUrl: './assets/worklet/processor.js',
      audioConstraints: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        sampleRate: 48_000,
      } as MediaTrackConstraints,
      onWarning: (message: string) => setWarning(message),
      onError: (error: Error) => setLocalError(formatDemoError(error, t)),
    }),
    [t],
  )

  const { state, error, inputStream, outputStream, bypass, start, stop, setBypass } = useDenoiser(
    modelSize,
    options,
  )

  useEffect(() => {
    if (outputStream) {
      if (playbackRef.current) void playbackRef.current.cleanup()
      playbackRef.current = createAudioPlayback(outputStream, volume)
    } else if (playbackRef.current) {
      void playbackRef.current.cleanup()
      playbackRef.current = null
    }
  }, [outputStream])

  useEffect(() => {
    playbackRef.current?.setVolume(volume)
  }, [volume])

  const latestStopRef = useRef(stop)
  latestStopRef.current = stop

  useEffect(() => {
    return () => {
      latestStopRef.current()
      if (playbackRef.current) {
        void playbackRef.current.cleanup()
        playbackRef.current = null
      }
      if (sampleSessionRef.current) {
        void stopInputSession(sampleSessionRef.current)
        sampleSessionRef.current = null
      }
    }
  }, [])

  const cleanupSampleSession = useCallback(async () => {
    if (sampleSessionRef.current) {
      await stopInputSession(sampleSessionRef.current)
      sampleSessionRef.current = null
    }
  }, [])

  const stopProcessing = useCallback(async () => {
    stop()
    setWarning(null)
    setLocalError(null)
    if (playbackRef.current) {
      void playbackRef.current.cleanup()
      playbackRef.current = null
    }
    await cleanupSampleSession()
  }, [stop, cleanupSampleSession])

  const startProcessing = useCallback(async () => {
    setWarning(null)
    setLocalError(null)

    try {
      if (sourceMode === 'microphone') {
        await start()
      } else {
        await cleanupSampleSession()
        const session = await createSampleInputSession(sampleId)
        sampleSessionRef.current = session
        await start(session.stream)
      }
    } catch (errorValue) {
      setLocalError(formatDemoError(errorValue, t))
      await cleanupSampleSession()
    }
  }, [sourceMode, sampleId, start, cleanupSampleSession, t])

  const setSourceMode = useCallback(
    async (mode: DemoSourceMode) => {
      if (state === 'processing' || state === 'loading') {
        await stopProcessing()
      }
      setSourceModeState(mode)
      setWarning(null)
      setLocalError(null)
    },
    [state, stopProcessing],
  )

  const setSampleId = useCallback(
    async (nextSampleId: SampleId) => {
      if (state === 'processing' || state === 'loading') {
        await stopProcessing()
      }
      setSampleIdState(nextSampleId)
    },
    [state, stopProcessing],
  )

  const handleSetModelSize = useCallback(
    async (nextModelSize: ModelSize) => {
      if (state === 'processing' || state === 'loading') {
        await stopProcessing()
      }
      setModelSizeState(nextModelSize)
      setWarning(null)
      setLocalError(null)
    },
    [state, stopProcessing],
  )

  const errorMessage = localError ?? (error ? formatDemoError(error, t) : null)

  return {
    sourceMode,
    sampleId,
    inputStream,
    outputStream,
    state,
    bypass,
    modelSize,
    volume,
    warning,
    errorMessage,
    setSourceMode,
    setSampleId,
    setModelSize: handleSetModelSize,
    setVolume,
    setBypass,
    startProcessing,
    stopProcessing,
  }
}
