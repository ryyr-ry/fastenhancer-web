import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createSampleInputSession,
  requestMicrophoneSession,
  stopInputSession,
  type DemoSourceMode,
  type InputSession,
  type SampleId,
} from './demo-media'

export interface DemoInputSourceController {
  sourceMode: DemoSourceMode
  sampleId: SampleId
  inputStream: MediaStream | null
  microphoneReady: boolean
  setSourceMode: (mode: DemoSourceMode) => void
  setSampleId: (sampleId: SampleId) => void
  requestMicrophone: () => Promise<MediaStream>
  prepareInputSession: () => Promise<InputSession>
  resetAfterStop: () => Promise<void>
  cleanupAll: () => Promise<void>
}

export function useDemoInputSource(): DemoInputSourceController {
  const [sourceMode, setSourceModeState] = useState<DemoSourceMode>('sample')
  const [sampleId, setSampleIdState] =
    useState<SampleId>('speech_48k_noisy_10dB.wav')
  const [inputStream, setInputStream] = useState<MediaStream | null>(null)
  const [microphoneReady, setMicrophoneReady] = useState(false)

  const microphoneSessionRef = useRef<InputSession | null>(null)
  const activeSampleSessionRef = useRef<InputSession | null>(null)

  const resetAfterStop = useCallback(async () => {
    if (activeSampleSessionRef.current) {
      await stopInputSession(activeSampleSessionRef.current)
      activeSampleSessionRef.current = null
    }

    setInputStream(
      sourceMode === 'microphone' ? microphoneSessionRef.current?.stream ?? null : null,
    )
  }, [sourceMode])

  const requestMicrophone = useCallback(async () => {
    if (!microphoneSessionRef.current) {
      microphoneSessionRef.current = await requestMicrophoneSession()
      setMicrophoneReady(true)
    }

    if (sourceMode === 'microphone') {
      setInputStream(microphoneSessionRef.current.stream)
    }

    return microphoneSessionRef.current.stream
  }, [sourceMode])

  const prepareInputSession = useCallback(async () => {
    if (sourceMode === 'sample') {
      if (activeSampleSessionRef.current) {
        await stopInputSession(activeSampleSessionRef.current)
      }

      const session = await createSampleInputSession(sampleId)
      activeSampleSessionRef.current = session
      setInputStream(session.stream)
      return session
    }

    const stream = await requestMicrophone()
    return {
      kind: 'microphone' as const,
      label: 'Live microphone',
      stream,
      stop: async () => undefined,
    }
  }, [requestMicrophone, sampleId, sourceMode])

  const cleanupAll = useCallback(async () => {
    if (activeSampleSessionRef.current) {
      await stopInputSession(activeSampleSessionRef.current)
      activeSampleSessionRef.current = null
    }

    if (microphoneSessionRef.current) {
      await stopInputSession(microphoneSessionRef.current)
      microphoneSessionRef.current = null
    }

    setMicrophoneReady(false)
    setInputStream(null)
  }, [])

  const setSourceMode = useCallback((mode: DemoSourceMode) => {
    setSourceModeState(mode)
    if (mode === 'microphone' && microphoneSessionRef.current) {
      setInputStream(microphoneSessionRef.current.stream)
      return
    }

    if (mode === 'sample') {
      setInputStream(activeSampleSessionRef.current?.stream ?? null)
      return
    }

    setInputStream(null)
  }, [])

  const setSampleId = useCallback((nextSampleId: SampleId) => {
    setSampleIdState(nextSampleId)
  }, [])

  useEffect(() => {
    return () => {
      void cleanupAll()
    }
  }, [cleanupAll])

  return {
    sourceMode,
    sampleId,
    inputStream,
    microphoneReady,
    setSourceMode,
    setSampleId,
    requestMicrophone,
    prepareInputSession,
    resetAfterStop,
    cleanupAll,
  }
}
