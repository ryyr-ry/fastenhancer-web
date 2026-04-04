import type { TFunction } from '../i18n'

export type ModelSize = 'tiny' | 'base' | 'small'
export type DemoSourceMode = 'sample' | 'microphone'
export type SampleId = 'speech_48k_noisy_10dB.wav' | 'speech_48k_noisy_0dB.wav'

export interface Option<TValue extends string> {
  value: TValue
  label: string
  description?: string
}

export interface InputSession {
  kind: DemoSourceMode
  label: string
  stream: MediaStream
  stop: () => Promise<void>
}

export function getModelOptions(t: TFunction): Option<ModelSize>[] {
  return [
    { value: 'tiny', label: t('model.tiny'), description: t('model.tinyDesc') },
    { value: 'base', label: t('model.base'), description: t('model.baseDesc') },
    { value: 'small', label: t('model.small'), description: t('model.smallDesc') },
  ]
}

export function getSampleOptions(t: TFunction): Option<SampleId>[] {
  return [
    {
      value: 'speech_48k_noisy_10dB.wav',
      label: t('sample.noisy10'),
      description: t('sample.noisy10Desc'),
    },
    {
      value: 'speech_48k_noisy_0dB.wav',
      label: t('sample.noisy0'),
      description: t('sample.noisy0Desc'),
    },
  ]
}

const AUDIO_BASE_URL = './assets/audio/'
const MICROPHONE_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    channelCount: 1,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    sampleRate: 48_000,
  },
  video: false,
}

export async function createSampleInputSession(sampleId: SampleId): Promise<InputSession> {
  const audioContext = new AudioContext({ sampleRate: 48_000 })
  if (audioContext.state === 'suspended') {
    await audioContext.resume()
  }

  const response = await fetch(`${AUDIO_BASE_URL}${sampleId}`)
  if (!response.ok) {
    await audioContext.close().catch(() => undefined)
    throw new Error(`Failed to load sample audio: ${response.status} ${response.statusText}`)
  }

  const audioBuffer = await audioContext.decodeAudioData(await response.arrayBuffer())
  const source = audioContext.createBufferSource()
  const gainNode = audioContext.createGain()
  const destination = audioContext.createMediaStreamDestination()

  source.buffer = audioBuffer
  source.loop = true
  gainNode.gain.value = 1

  source.connect(gainNode)
  gainNode.connect(destination)
  source.start()

  let closed = false

  return {
    kind: 'sample',
    label: sampleId,
    stream: destination.stream,
    stop: async () => {
      if (closed) {
        return
      }

      closed = true

      try {
        source.stop()
      } catch {
        return
      } finally {
        source.disconnect()
        gainNode.disconnect()
        destination.disconnect()
        stopMediaStream(destination.stream)
        await audioContext.close().catch(() => undefined)
      }
    },
  }
}

export async function requestMicrophoneSession(): Promise<InputSession> {
  const stream = await navigator.mediaDevices.getUserMedia(MICROPHONE_CONSTRAINTS)

  return {
    kind: 'microphone',
    label: 'Live microphone',
    stream,
    stop: async () => {
      stopMediaStream(stream)
    },
  }
}

export function stopMediaStream(stream: MediaStream | null | undefined): void {
  stream?.getTracks().forEach((track) => track.stop())
}

export async function stopInputSession(session: InputSession | null | undefined): Promise<void> {
  if (!session) {
    return
  }

  await session.stop().catch(() => undefined)
}

export async function attachStreamToAudioElement(
  element: HTMLAudioElement | null,
  stream: MediaStream | null,
  volume: number,
): Promise<void> {
  if (!element) {
    return
  }

  element.pause()
  element.srcObject = stream
  element.volume = volume
  element.setAttribute('playsinline', '')
  element.autoplay = true
  element.muted = false

  if (stream) {
    await element.play().catch(() => undefined)
  }
}

export function detachAudioElement(element: HTMLAudioElement | null): void {
  if (!element) {
    return
  }

  element.pause()
  element.srcObject = null
}

export interface AudioPlayback {
  setVolume(volume: number): void
  cleanup(): Promise<void>
}

export function createAudioPlayback(
  stream: MediaStream,
  volume: number,
): AudioPlayback {
  const ctx = new AudioContext({ sampleRate: 48_000 })
  if (ctx.state === 'suspended') {
    ctx.resume().catch(() => undefined)
  }
  const source = ctx.createMediaStreamSource(stream)
  const gain = ctx.createGain()
  gain.gain.value = volume
  source.connect(gain)
  gain.connect(ctx.destination)

  return {
    setVolume(v: number) {
      gain.gain.value = v
    },
    async cleanup() {
      try { source.disconnect() } catch { /* already disconnected */ }
      try { gain.disconnect() } catch { /* already disconnected */ }
      await ctx.close().catch(() => undefined)
    },
  }
}

export function formatDemoError(error: unknown, t: TFunction): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError') {
      return t('error.micDenied')
    }

    if (error.name === 'NotFoundError') {
      return t('error.noMic')
    }
  }

  if (error instanceof Error) {
    return error.message
  }

  return t('error.unexpected')
}
