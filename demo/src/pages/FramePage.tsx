import { useCallback, useEffect, useRef, useState } from 'react'
import { loadModel, type Denoiser } from 'fastenhancer-web'
import { CodeBlock } from '../components/CodeBlock'
import { StatusBadge } from '../components/StatusBadge'
import { WaveformCanvas } from '../components/WaveformCanvas'
import { useT, renderCode } from '../i18n'
import { formatDemoError, getModelOptions, getSampleOptions, type ModelSize, type SampleId } from '../lib/demo-media'
import { useDemoInputSource } from '../lib/useDemoInputSource'

interface FrameStats {
  lastMs: number
  avgMs: number
  p99Ms: number
  droppedFrames: number
  totalFrames: number
  inputPreview: number[]
  outputPreview: number[]
}

interface FrameRuntime {
  audioContext: AudioContext
  sourceNode: MediaStreamAudioSourceNode
  processorNode: ScriptProcessorNode
  gainNode: GainNode
  denoiser: Denoiser
}

const codeExample = `import { loadModel } from 'fastenhancer-web';

const model = await loadModel('small');
const denoiser = await model.createDenoiser();
const output = denoiser.processFrame(inputFloat32);`

const EMPTY_STATS: FrameStats = {
  lastMs: 0, avgMs: 0, p99Ms: 0, droppedFrames: 0, totalFrames: 0,
  inputPreview: [], outputPreview: [],
}

export function FramePage() {
  const t = useT()
  const input = useDemoInputSource()
  const [modelSize, setModelSize] = useState<ModelSize>('small')
  const [volume, setVolume] = useState(0.65)
  const [bypass, setBypass] = useState(false)
  const [status, setStatus] = useState<'idle' | 'loading' | 'processing' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [warning, setWarning] = useState<string | null>(null)
  const [inputNode, setInputNode] = useState<AudioNode | null>(null)
  const [outputNode, setOutputNode] = useState<AudioNode | null>(null)
  const [stats, setStats] = useState<FrameStats>(EMPTY_STATS)

  const runtimeRef = useRef<FrameRuntime | null>(null)
  const bypassRef = useRef(bypass)
  const tickRef = useRef(0)

  useEffect(() => {
    bypassRef.current = bypass
    if (runtimeRef.current) runtimeRef.current.denoiser.bypass = bypass
  }, [bypass])

  useEffect(() => {
    runtimeRef.current?.gainNode.gain.setValueAtTime(
      volume, runtimeRef.current.audioContext.currentTime,
    )
  }, [volume])

  const cleanupRuntime = useCallback(async () => {
    if (!runtimeRef.current) {
      setInputNode(null)
      setOutputNode(null)
      return
    }
    const rt = runtimeRef.current
    runtimeRef.current = null
    rt.processorNode.onaudioprocess = null
    rt.sourceNode.disconnect()
    rt.processorNode.disconnect()
    rt.gainNode.disconnect()
    rt.denoiser.destroy()
    setInputNode(null)
    setOutputNode(null)
    await rt.audioContext.close().catch(() => undefined)
    await input.resetAfterStop()
  }, [input])

  const stopProcessing = useCallback(async () => {
    await cleanupRuntime()
    setStatus('idle')
    setWarning(null)
    setErrorMessage(null)
  }, [cleanupRuntime])

  const startProcessing = useCallback(async () => {
    setWarning(null)
    setErrorMessage(null)
    try {
      await cleanupRuntime()
      setStatus('loading')

      const session = await input.prepareInputSession()

      const model = await loadModel(modelSize, { baseUrl: './assets/wasm/' })
      const denoiser = await model.createDenoiser()
      denoiser.bypass = bypassRef.current

      const audioContext = new AudioContext({ sampleRate: model.sampleRate })
      if (audioContext.state === 'suspended') await audioContext.resume().catch(() => undefined)

      const sourceNode = audioContext.createMediaStreamSource(session.stream)
      const processorNode = audioContext.createScriptProcessor(model.hopSize, 1, 1)
      const gainNode = audioContext.createGain()
      gainNode.gain.value = volume

      processorNode.onaudioprocess = (event) => {
        const inputBuf = new Float32Array(event.inputBuffer.getChannelData(0))
        const outputChannel = event.outputBuffer.getChannelData(0)
        try {
          const t0 = performance.now()
          denoiser.bypass = bypassRef.current
          const outputBuf = denoiser.processFrame(inputBuf)
          outputChannel.set(outputBuf)
          const elapsed = performance.now() - t0
          const now = performance.now()
          if (now - tickRef.current > 80) {
            tickRef.current = now
            const perf = denoiser.performance
            setStats({
              lastMs: elapsed,
              avgMs: perf.avgMs,
              p99Ms: perf.p99Ms,
              droppedFrames: perf.droppedFrames,
              totalFrames: perf.totalFrames,
              inputPreview: Array.from(inputBuf.subarray(0, 12)).map((v) => Number(v.toFixed(4))),
              outputPreview: Array.from(outputBuf.subarray(0, 12)).map((v) => Number((v as number).toFixed(4))),
            })
          }
        } catch (err) {
          outputChannel.fill(0)
          setWarning(formatDemoError(err, t))
        }
      }

      sourceNode.connect(processorNode)
      processorNode.connect(gainNode)
      gainNode.connect(audioContext.destination)

      runtimeRef.current = { audioContext, sourceNode, processorNode, gainNode, denoiser }
      setInputNode(sourceNode)
      setOutputNode(processorNode)
      setStatus('processing')
    } catch (err) {
      setStatus('error')
      setErrorMessage(formatDemoError(err, t))
      await cleanupRuntime()
    }
  }, [cleanupRuntime, input, modelSize, volume, t])

  const handleSourceChange = useCallback(
    async (mode: 'sample' | 'microphone') => {
      if (status === 'processing' || status === 'loading') await stopProcessing()
      input.setSourceMode(mode)
    },
    [input, status, stopProcessing],
  )

  const handleSampleChange = useCallback(
    async (id: SampleId) => {
      if (status === 'processing' || status === 'loading') await stopProcessing()
      input.setSampleId(id)
    },
    [input, status, stopProcessing],
  )

  const handleModelChange = useCallback(
    async (size: ModelSize) => {
      if (status === 'processing' || status === 'loading') await stopProcessing()
      setModelSize(size)
    },
    [status, stopProcessing],
  )

  const requestMic = useCallback(async () => {
    try {
      await input.requestMicrophone()
      setErrorMessage(null)
    } catch (err) {
      setErrorMessage(formatDemoError(err, t))
    }
  }, [input, t])

  const latestCleanupRef = useRef(cleanupRuntime)
  latestCleanupRef.current = cleanupRuntime
  const latestInputRef = useRef(input)
  latestInputRef.current = input

  useEffect(() => {
    return () => {
      void latestCleanupRef.current()
      void latestInputRef.current.cleanupAll()
    }
  }, [])

  const busy = status === 'loading'
  const active = status === 'processing'
  const modelOptions = getModelOptions(t)
  const sampleOptions = getSampleOptions(t)

  return (
    <div className="demo">
      <header className="demo__header">
        <div>
          <span className="demo__tag">{t('frame.tag')}</span>
          <h1 className="demo__title">{t('frame.title')}</h1>
          <p className="demo__subtitle">{renderCode(t('frame.subtitle'))}</p>
        </div>
        <StatusBadge status={status} />
      </header>

      {errorMessage && <div className="demo__alert demo__alert--error">{errorMessage}</div>}
      {warning && <div className="demo__alert demo__alert--warn">{warning}</div>}

      <div className="demo__controls">
        <div className="demo__row">
          <div className="demo__field">
            <span className="demo__label">{t('common.source')}</span>
            <div className="seg" role="tablist">
              <button
                type="button"
                className={`seg__btn${input.sourceMode === 'sample' ? ' seg__btn--on' : ''}`}
                disabled={busy}
                onClick={() => void handleSourceChange('sample')}
              >
                {t('common.sample')}
              </button>
              <button
                type="button"
                className={`seg__btn${input.sourceMode === 'microphone' ? ' seg__btn--on' : ''}`}
                disabled={busy}
                onClick={() => void handleSourceChange('microphone')}
              >
                {t('common.mic')}
              </button>
            </div>
          </div>

          {input.sourceMode === 'sample' ? (
            <div className="demo__field">
              <span className="demo__label">{t('common.clip')}</span>
              <select
                className="demo__select"
                value={input.sampleId}
                disabled={busy}
                onChange={(e) => void handleSampleChange(e.target.value as SampleId)}
              >
                {sampleOptions.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
          ) : (
            <div className="demo__field">
              <button
                type="button"
                className="demo__btn--secondary"
                disabled={busy}
                onClick={() => void requestMic()}
              >
                {input.microphoneReady ? t('common.micReady') : t('common.enableMic')}
              </button>
            </div>
          )}

          <div className="demo__field">
            <span className="demo__label">{t('common.model')}</span>
            <select
              className="demo__select"
              value={modelSize}
              disabled={busy}
              onChange={(e) => void handleModelChange(e.target.value as ModelSize)}
            >
              {modelOptions.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="demo__actions">
          <button
            type="button"
            className="demo__btn demo__btn--start"
            disabled={busy || active}
            onClick={() => void startProcessing()}
          >
            {busy ? t('common.loading') : t('common.start')}
          </button>
          <button
            type="button"
            className="demo__btn demo__btn--stop"
            disabled={!active}
            onClick={() => void stopProcessing()}
          >
            {t('common.stop')}
          </button>
          <label className="demo__toggle">
            <input type="checkbox" checked={bypass} onChange={(e) => setBypass(e.target.checked)} />
            {t('common.bypass')}
          </label>
          <label className="demo__volume">
            {t('common.vol')}
            <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(e) => setVolume(Number(e.target.value))} />
          </label>
        </div>
      </div>

      <div className="demo__metrics">
        <div className="demo__metric">
          <span className="demo__metric-label">{t('frame.lastFrame')}</span>
          <span className="demo__metric-value">{stats.lastMs.toFixed(2)} ms</span>
        </div>
        <div className="demo__metric">
          <span className="demo__metric-label">{t('frame.average')}</span>
          <span className="demo__metric-value">{stats.avgMs.toFixed(2)} ms</span>
        </div>
        <div className="demo__metric">
          <span className="demo__metric-label">{t('frame.p99')}</span>
          <span className="demo__metric-value">{stats.p99Ms.toFixed(2)} ms</span>
        </div>
        <div className="demo__metric">
          <span className="demo__metric-label">{t('frame.frames')}</span>
          <span className="demo__metric-value">{stats.totalFrames}</span>
        </div>
        <div className="demo__metric">
          <span className="demo__metric-label">{t('frame.dropped')}</span>
          <span className="demo__metric-value">{stats.droppedFrames}</span>
        </div>
      </div>

      <div className="demo__waves">
        <WaveformCanvas audioSource={inputNode} label={t('common.inputNoisy')} />
        <WaveformCanvas audioSource={outputNode} label={t('frame.outputDenoised')} />
      </div>

      <details className="demo__inspector">
        <summary>{t('frame.frameInspector')}</summary>
        <div className="demo__inspector-grid">
          {stats.inputPreview.map((v, i) => (
            <span key={`i${i}`} className="demo__inspector-cell">in[{i}] {v.toFixed(4)}</span>
          ))}
        </div>
        <div className="demo__inspector-grid" style={{ marginTop: '0.5rem' }}>
          {stats.outputPreview.map((v, i) => (
            <span key={`o${i}`} className="demo__inspector-cell">out[{i}] {v.toFixed(4)}</span>
          ))}
        </div>
      </details>

      <details className="demo__code">
        <summary>{t('common.codeExample')}</summary>
        <CodeBlock code={codeExample} language="ts" />
      </details>
    </div>
  )
}
