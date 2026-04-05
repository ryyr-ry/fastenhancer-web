import { CodeBlock } from '../components/CodeBlock'
import { StatusBadge } from '../components/StatusBadge'
import { WaveformCanvas } from '../components/WaveformCanvas'
import { useT, renderCode } from '../i18n'
import { useHookDemoController } from '../lib/useHookDemoController'
import { getModelOptions, getSampleOptions, type ModelSize, type SampleId } from '../lib/demo-media'

const codeExample = `import { useDenoiser } from 'fastenhancer-web/react';

const { outputStream, start, stop } = useDenoiser('small');

<button onClick={start}>Start</button>
<button onClick={stop}>Stop</button>`

export function HomePage() {
  const t = useT()
  const c = useHookDemoController()
  const busy = c.state === 'loading'
  const active = c.state === 'processing'
  const modelOptions = getModelOptions(t)
  const sampleOptions = getSampleOptions(t)

  return (
    <div className="demo">
      <header className="demo__header">
        <div>
          <span className="demo__tag">{t('home.tag')}</span>
          <h1 className="demo__title">{t('home.title')}</h1>
          <p className="demo__subtitle">{renderCode(t('home.subtitle'))}</p>
        </div>
        <StatusBadge status={c.state} />
      </header>

      {c.errorMessage && <div className="demo__alert demo__alert--error">{c.errorMessage}</div>}
      {c.warning && <div className="demo__alert demo__alert--warn">{c.warning}</div>}

      <div className="demo__controls">
        <div className="demo__row">
          <div className="demo__field">
            <span className="demo__label">{t('common.source')}</span>
            <div className="seg" role="tablist">
              <button
                type="button"
                className={`seg__btn${c.sourceMode === 'sample' ? ' seg__btn--on' : ''}`}
                disabled={busy}
                onClick={() => void c.setSourceMode('sample')}
              >
                {t('common.sample')}
              </button>
              <button
                type="button"
                className={`seg__btn${c.sourceMode === 'microphone' ? ' seg__btn--on' : ''}`}
                disabled={busy}
                onClick={() => void c.setSourceMode('microphone')}
              >
                {t('common.mic')}
              </button>
            </div>
          </div>

          {c.sourceMode === 'sample' && (
            <div className="demo__field">
              <span className="demo__label">{t('common.clip')}</span>
              <select
                className="demo__select"
                value={c.sampleId}
                disabled={busy}
                onChange={(e) => void c.setSampleId(e.target.value as SampleId)}
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
              value={c.modelSize}
              disabled={busy}
              onChange={(e) => void c.setModelSize(e.target.value as ModelSize)}
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
            onClick={() => void c.startProcessing()}
          >
            {busy ? t('common.loading') : t('common.start')}
          </button>
          <button
            type="button"
            className="demo__btn demo__btn--stop"
            disabled={!active}
            onClick={() => void c.stopProcessing()}
          >
            {t('common.stop')}
          </button>
          <label className="demo__toggle">
            <input
              type="checkbox"
              checked={c.bypass}
              onChange={(e) => c.setBypass(e.target.checked)}
            />
            {t('common.bypass')}
          </label>
          <label className="demo__volume">
            {t('common.vol')}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={c.volume}
              onChange={(e) => c.setVolume(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="demo__waves">
        <WaveformCanvas audioSource={c.inputStream} label={t('common.inputNoisy')} />
        <WaveformCanvas audioSource={c.outputStream} label={t('common.outputClean')} />
      </div>

      <details className="demo__code">
        <summary>{t('common.codeExample')}</summary>
        <CodeBlock code={codeExample} language="tsx" />
      </details>
    </div>
  )
}
