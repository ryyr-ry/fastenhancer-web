import { useCallback, useEffect, useRef, useState } from 'react'
import { loadModel, type StreamDenoiser } from 'fastenhancer-web'
import { CodeBlock } from '../components/CodeBlock'
import { StatusBadge } from '../components/StatusBadge'
import { WaveformCanvas } from '../components/WaveformCanvas'
import { useT, renderCode } from '../i18n'
import {
  createAudioPlayback,
  formatDemoError,
  getSampleOptions,
  getModelOptions,
  type AudioPlayback,
  type ModelSize,
  type SampleId,
} from '../lib/demo-media'
import { useDemoInputSource } from '../lib/useDemoInputSource'

const codeExample = `import { loadModel } from 'fastenhancer-web';

// 1. Load the model (cached after first call)
const model = await loadModel('small');

// 2. Get microphone access
const mic = await navigator.mediaDevices.getUserMedia({ audio: true });

// 3. Create a stream denoiser — returns a clean MediaStream
const denoiser = await model.createStreamDenoiser(mic);

// Attach the clean output to an <audio> element or send via WebRTC
const audio = document.querySelector('audio');
audio.srcObject = denoiser.outputStream;

// Toggle bypass to compare raw vs denoised audio
denoiser.bypass = true;

// Clean up when done
denoiser.destroy();`

export function VanillaPage() {
  const t = useT()
  const input = useDemoInputSource()
  const [modelSize, setModelSize] = useState<ModelSize>('small')
  const [volume, setVolume] = useState(0.8)
  const [bypass, setBypass] = useState(false)
  const [keepAlive, setKeepAlive] = useState(false)
  const [status, setStatus] = useState<'idle' | 'loading' | 'processing' | 'error'>('idle')
  const [warning, setWarning] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [outputStream, setOutputStream] = useState<MediaStream | null>(null)

  const streamRef = useRef<StreamDenoiser | null>(null)
  const playbackRef = useRef<AudioPlayback | null>(null)

  const cleanup = useCallback(async () => {
    if (playbackRef.current) {
      await playbackRef.current.cleanup()
      playbackRef.current = null
    }
    if (streamRef.current) {
      try { streamRef.current.destroy() } catch { /* already destroyed */ }
      streamRef.current = null
    }
    setOutputStream(null)
    await input.resetAfterStop()
  }, [input])

  useEffect(() => {
    playbackRef.current?.setVolume(volume)
  }, [volume])

  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.bypass = bypass
    }
  }, [bypass])

  const latestCleanupRef = useRef(cleanup)
  latestCleanupRef.current = cleanup
  const latestInputRef = useRef(input)
  latestInputRef.current = input

  useEffect(() => {
    return () => {
      void latestCleanupRef.current()
      void latestInputRef.current.cleanupAll()
    }
  }, [])

  const start = useCallback(async () => {
    await cleanup()
    setStatus('loading')
    setWarning(null)
    setErrorMessage(null)

    try {
      const session = await input.prepareInputSession()
      const model = await loadModel(modelSize, { baseUrl: './assets/wasm/' })
      const sd = await model.createStreamDenoiser(session.stream, {
        workletUrl: './assets/worklet/processor.js',
        onWarning: (msg) => setWarning(msg),
        keepAliveInBackground: keepAlive,
      })

      sd.bypass = bypass
      streamRef.current = sd
      setOutputStream(sd.outputStream)
      playbackRef.current = createAudioPlayback(sd.outputStream, volume)
      setStatus('processing')
    } catch (err) {
      setStatus('error')
      setErrorMessage(formatDemoError(err, t))
      await cleanup()
    }
  }, [bypass, cleanup, input, modelSize, volume, keepAlive, t])

  const stop = useCallback(async () => {
    await cleanup()
    setStatus('idle')
    setWarning(null)
    setErrorMessage(null)
  }, [cleanup])

  const handleSourceChange = useCallback(
    async (mode: 'sample' | 'microphone') => {
      if (status === 'processing' || status === 'loading') await stop()
      input.setSourceMode(mode)
    },
    [input, status, stop],
  )

  const handleSampleChange = useCallback(
    async (id: SampleId) => {
      if (status === 'processing' || status === 'loading') await stop()
      input.setSampleId(id)
    },
    [input, status, stop],
  )

  const handleModelChange = useCallback(
    async (size: ModelSize) => {
      if (status === 'processing' || status === 'loading') await stop()
      setModelSize(size)
    },
    [status, stop],
  )

  const busy = status === 'loading'
  const active = status === 'processing'
  const modelOptions = getModelOptions(t)
  const sampleOptions = getSampleOptions(t)

  return (
    <div className="demo">
      <header className="demo__header">
        <div>
          <span className="demo__tag">{t('vanilla.tag')}</span>
          <h1 className="demo__title">{t('vanilla.title')}</h1>
          <p className="demo__subtitle">{renderCode(t('vanilla.subtitle'))}</p>
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

          {input.sourceMode === 'sample' && (
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
            onClick={() => void start()}
          >
            {busy ? t('common.loading') : t('common.start')}
          </button>
          <button
            type="button"
            className="demo__btn demo__btn--stop"
            disabled={!active}
            onClick={() => void stop()}
          >
            {t('common.stop')}
          </button>
          <label className="demo__toggle">
            <input
              type="checkbox"
              checked={bypass}
              onChange={(e) => setBypass(e.target.checked)}
            />
            {t('common.bypass')}
          </label>
          <label className="demo__toggle">
            <input
              type="checkbox"
              checked={keepAlive}
              disabled={active}
              onChange={(e) => setKeepAlive(e.target.checked)}
            />
            {t('common.keepAlive')}
          </label>
          <label className="demo__volume">
            {t('common.vol')}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="demo__waves">
        <WaveformCanvas
          audioSource={input.inputStream}
          label={t('common.inputNoisy')}
        />
        <WaveformCanvas
          audioSource={outputStream}
          label={t('common.outputClean')}
        />
      </div>

      <details className="demo__code">
        <summary>{t('common.codeExample')}</summary>
        <CodeBlock code={codeExample} language="ts" />
      </details>
    </div>
  )
}
