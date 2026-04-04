import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useDenoiser } from 'fastenhancer-web/react'
import { useT } from '../i18n'
import {
  createAudioPlayback,
  formatDemoError,
  type AudioPlayback,
  type DemoSourceMode,
  type ModelSize,
  type SampleId,
} from './demo-media'
import { useDemoInputSource } from './useDemoInputSource'

export interface HookDemoController {
  sourceMode: ReturnType<typeof useDemoInputSource>['sourceMode']
  sampleId: ReturnType<typeof useDemoInputSource>['sampleId']
  inputStream: MediaStream | null
  outputStream: MediaStream | null
  state: 'idle' | 'loading' | 'processing' | 'error' | 'destroyed'
  bypass: boolean
  modelSize: ModelSize
  volume: number
  warning: string | null
  errorMessage: string | null
  microphoneReady: boolean
  setSourceMode: (mode: ReturnType<typeof useDemoInputSource>['sourceMode']) => Promise<void>
  setSampleId: (sampleId: ReturnType<typeof useDemoInputSource>['sampleId']) => Promise<void>
  setModelSize: (modelSize: ModelSize) => Promise<void>
  setVolume: (volume: number) => void
  setBypass: (value: boolean) => void
  requestMicrophone: () => Promise<void>
  startProcessing: () => Promise<void>
  stopProcessing: () => Promise<void>
}

export function useHookDemoController(): HookDemoController {
  const t = useT()
  const inputController = useDemoInputSource()
  const [modelSize, setModelSizeState] = useState<ModelSize>('small')
  const [volume, setVolume] = useState(0.9)
  const [warning, setWarning] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
  const playbackRef = useRef<AudioPlayback | null>(null)

  const options = useMemo(
    () => ({
      baseUrl: './assets/wasm/',
      workletUrl: './assets/worklet/processor.js',
      onWarning: (message: string) => setWarning(message),
      onError: (error: Error) => setLocalError(formatDemoError(error, t)),
    }),
    [t],
  )

  const { state, error, outputStream, bypass, start, stop, setBypass } = useDenoiser(
    modelSize,
    options,
  )

  useEffect(() => {
    if (outputStream) {
      if (playbackRef.current) {
        void playbackRef.current.cleanup()
      }
      playbackRef.current = createAudioPlayback(outputStream, volume)
    } else if (playbackRef.current) {
      void playbackRef.current.cleanup()
      playbackRef.current = null
    }
  }, [outputStream, volume])

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
    await inputController.resetAfterStop()
  }, [inputController, stop])

  const startProcessing = useCallback(async () => {
    setWarning(null)
    setLocalError(null)

    try {
      const session = await inputController.prepareInputSession()
      await start(session.stream)
    } catch (errorValue) {
      setLocalError(formatDemoError(errorValue, t))
      await inputController.resetAfterStop()
    }
  }, [inputController, start, t])

  const requestMicrophone = useCallback(async () => {
    try {
      await inputController.requestMicrophone()
      setLocalError(null)
    } catch (errorValue) {
      setLocalError(formatDemoError(errorValue, t))
    }
  }, [inputController, t])

  const setSourceMode = useCallback(
    async (mode: DemoSourceMode) => {
      if (state === 'processing' || state === 'loading') {
        await stopProcessing()
      }

      inputController.setSourceMode(mode)
      setWarning(null)
      setLocalError(null)
    },
    [inputController, state, stopProcessing],
  )

  const setSampleId = useCallback(
    async (nextSampleId: SampleId) => {
      if (state === 'processing' || state === 'loading') {
        await stopProcessing()
      }

      inputController.setSampleId(nextSampleId)
    },
    [inputController, state, stopProcessing],
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
    sourceMode: inputController.sourceMode,
    sampleId: inputController.sampleId,
    inputStream: inputController.inputStream,
    outputStream,
    state,
    bypass,
    modelSize,
    volume,
    warning,
    errorMessage,
    microphoneReady: inputController.microphoneReady,
    setSourceMode,
    setSampleId,
    setModelSize: handleSetModelSize,
    setVolume,
    setBypass,
    requestMicrophone,
    startProcessing,
    stopProcessing,
  }
}
