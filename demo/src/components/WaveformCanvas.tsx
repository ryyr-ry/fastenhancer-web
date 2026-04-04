import { useEffect, useRef } from 'react'
import { useT } from '../i18n'

interface WaveformCanvasProps {
  audioSource: MediaStream | AudioNode | null
  label: string
  caption?: string
}

export function WaveformCanvas({ audioSource, label, caption }: WaveformCanvasProps) {
  const t = useT()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const isMediaStream =
      typeof MediaStream !== 'undefined' && audioSource instanceof MediaStream

    let createdContext: AudioContext | null = null
    let analyser: AnalyserNode | null = null
    let sourceNode: AudioNode | null = null
    let frameId = 0

    const draw = () => {
      const context = canvas.getContext('2d')
      if (!context) {
        return
      }

      const ratio = window.devicePixelRatio || 1
      const width = canvas.clientWidth * ratio
      const height = canvas.clientHeight * ratio

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
      }

      context.clearRect(0, 0, width, height)
      context.fillStyle = '#070b12'
      context.fillRect(0, 0, width, height)

      context.strokeStyle = 'rgba(79, 195, 247, 0.12)'
      context.lineWidth = 1
      context.beginPath()
      context.moveTo(0, height / 2)
      context.lineTo(width, height / 2)
      context.stroke()

      if (!analyser) {
        context.fillStyle = '#7f8b97'
        context.font = `${14 * ratio}px system-ui`
        context.fillText(t('common.waveformWaiting'), 18 * ratio, 28 * ratio)
        frameId = window.requestAnimationFrame(draw)
        return
      }

      const data = new Uint8Array(analyser.fftSize)
      analyser.getByteTimeDomainData(data)

      context.lineWidth = 2 * ratio
      context.strokeStyle = '#4ade80'
      context.beginPath()

      for (let index = 0; index < data.length; index += 1) {
        const x = (index / (data.length - 1)) * width
        const y = (data[index] / 255) * height

        if (index === 0) {
          context.moveTo(x, y)
        } else {
          context.lineTo(x, y)
        }
      }

      context.stroke()
      frameId = window.requestAnimationFrame(draw)
    }

    if (audioSource) {
      if (isMediaStream) {
        createdContext = new AudioContext()
        analyser = createdContext.createAnalyser()
        analyser.fftSize = 2048
        sourceNode = createdContext.createMediaStreamSource(audioSource)
        sourceNode.connect(analyser)
      } else {
        const node = audioSource as AudioNode
        analyser = node.context.createAnalyser()
        analyser.fftSize = 2048
        node.connect(analyser)
      }
    }

    frameId = window.requestAnimationFrame(draw)

    return () => {
      window.cancelAnimationFrame(frameId)

      if (sourceNode && analyser) {
        try {
          sourceNode.disconnect(analyser)
        } catch {
          undefined
        }
      }

      if (!isMediaStream && audioSource && analyser) {
        try {
          (audioSource as AudioNode).disconnect(analyser)
        } catch {
          undefined
        }
      }

      analyser?.disconnect()
      void createdContext?.close().catch(() => undefined)
    }
  }, [audioSource])

  return (
    <div className="waveform-shell">
      <div className="waveform-label">
        <div>
          <h3 className="waveform-label__title">{label}</h3>
          <p className="waveform-label__caption">{caption ?? t('common.waveformCaption')}</p>
        </div>
      </div>
      <canvas ref={canvasRef} className="waveform-canvas" />
    </div>
  )
}
