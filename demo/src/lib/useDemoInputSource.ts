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
  setSourceMode: (mode: DemoSourceMode) => void
  setSampleId: (sampleId: SampleId) => void
  prepareInputSession: () => Promise<InputSession>
  resetAfterStop: () => Promise<void>
  cleanupAll: () => Promise<void>
}

export function useDemoInputSource(): DemoInputSourceController {
  const [sourceMode, setSourceModeState] = useState<DemoSourceMode>('sample')
  const [sampleId, setSampleIdState] =
    useState<SampleId>('speech_48k_noisy_10dB.wav')
  const [inputStream, setInputStream] = useState<MediaStream | null>(null)

  const activeSessionRef = useRef<InputSession | null>(null)

  const resetAfterStop = useCallback(async () => {
    if (activeSessionRef.current) {
      await stopInputSession(activeSessionRef.current)
      activeSessionRef.current = null
    }
    setInputStream(null)
  }, [])

  const prepareInputSession = useCallback(async () => {
    if (activeSessionRef.current) {
      await stopInputSession(activeSessionRef.current)
      activeSessionRef.current = null
    }

    const session =
      sourceMode === 'sample'
        ? await createSampleInputSession(sampleId)
        : await requestMicrophoneSession()

    activeSessionRef.current = session
    setInputStream(session.stream)
    return session
  }, [sampleId, sourceMode])

  const cleanupAll = useCallback(async () => {
    if (activeSessionRef.current) {
      await stopInputSession(activeSessionRef.current)
      activeSessionRef.current = null
    }
    setInputStream(null)
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
    setSourceMode: setSourceModeState,
    setSampleId: setSampleIdState,
    prepareInputSession,
    resetAfterStop,
    cleanupAll,
  }
}
