import { useEffect, useRef, useState } from 'react'
import { enhanceAudio } from 'fastenhancer-web'
import { CodeBlock } from '../components/CodeBlock'
import { useT } from '../i18n'

function useBlobUrl(blob: Blob | null) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!blob) { setUrl(null); return }
    const next = URL.createObjectURL(blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return url
}

export function FilePage() {
  const t = useT()
  const [file, setFile] = useState<File | null>(null)
  const [clean, setClean] = useState<Blob | null>(null)
  const [progress, setProgress] = useState(0)
  const [running, setRunning] = useState(false)
  const [parallel, setParallel] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const inputUrl = useBlobUrl(file)
  const outputUrl = useBlobUrl(clean)

  useEffect(() => () => {
    controllerRef.current?.abort()
    controllerRef.current = null
  }, [])

  const process = async () => {
    if (!file || controllerRef.current) return
    const controller = new AbortController()
    controllerRef.current = controller
    setRunning(true)
    setClean(null)
    setError(null)
    setProgress(0)
    try {
      const result = await enhanceAudio(file, {
        parallel: parallel ? { maxWorkers: 4 } : false,
        signal: controller.signal,
        onProgress: value => {
          if (controllerRef.current === controller) setProgress(value)
        },
      })
      if (controllerRef.current === controller) setClean(result)
    } catch (cause) {
      if (controllerRef.current === controller && !controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null
        setRunning(false)
      }
    }
  }

  return (
    <div className="demo">
      <header className="demo__header">
        <div>
          <span className="demo__tag">Offline</span>
          <h1 className="demo__title">{t('file.title')}</h1>
          <p className="demo__subtitle">{t('file.subtitle')}</p>
        </div>
      </header>
      <p>{t('file.formats')}</p>
      <p>{t('file.stereoNote')}</p>
      {error && <div role="alert" className="demo__alert demo__alert--error">{error}</div>}
      <div className="demo__controls">
        <div className="demo__row">
          <label className="demo__field">
            <span className="demo__label">{t('file.select')}</span>
            <input type="file" accept="audio/*,.wav" disabled={running} onChange={event => {
              setFile(event.target.files?.[0] ?? null)
              setClean(null)
              setError(null)
              setProgress(0)
            }} />
          </label>
          <label className="demo__field">
            <span><input type="checkbox" checked={parallel} disabled={running} onChange={event => setParallel(event.target.checked)} /> {t('file.parallel')}</span>
          </label>
          <button className="demo__btn demo__btn--start" disabled={!file || running} onClick={() => void process()}>
            {t('file.process')}
          </button>
          {running && <button className="demo__btn demo__btn--stop" onClick={() => controllerRef.current?.abort()}>
            {t('file.cancel')}
          </button>}
        </div>
        {running && <div role="status">
          <progress value={progress} max={1} aria-label={t('file.progress')} /> {Math.round(progress * 100)}%
        </div>}
      </div>
      <p>{t('file.parallelNote')}</p>
      {inputUrl && <section>
        <h2>{t('file.original')}</h2>
        <audio controls src={inputUrl} aria-label={t('file.original')} style={{ width: '100%' }} />
      </section>}
      {outputUrl && <section>
        <h2>{t('file.result')}</h2>
        <audio controls src={outputUrl} aria-label={t('file.result')} style={{ width: '100%' }} />
        <a className="demo__btn" href={outputUrl} download={`${file?.name.replace(/\.[^.]+$/, '') || 'audio'}-enhanced.wav`}>
          {t('file.download')}
        </a>
      </section>}
      <details className="demo__code" open>
        <summary>{t('common.codeExample')}</summary>
        <CodeBlock code={parallel ? "import { enhanceAudio } from 'fastenhancer-web';\n\nconst cleanWav = await enhanceAudio(file, { parallel: { maxWorkers: 4 } });" : "import { enhanceAudio } from 'fastenhancer-web';\n\nconst cleanWav = await enhanceAudio(file);"} language="typescript" />
      </details>
    </div>
  )
}
