import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { useT, renderCode } from '../i18n'

const quickStart = `import { useDenoiser } from 'fastenhancer-web/react';

function App() {
  const { outputStream, start, stop } = useDenoiser('small');
  return (
    <>
      <button onClick={start}>Start</button>
      <button onClick={stop}>Stop</button>
      {outputStream && <audio autoPlay ref={el => el && (el.srcObject = outputStream)} />}
    </>
  );
}`

interface DemoCardProps {
  to: string
  layer: string
  title: string
  description: string
  api: string
}

function DemoCard({ to, layer, title, description, api }: DemoCardProps) {
  return (
    <Link to={to} className="landing__card">
      <span className="landing__card-layer">{layer}</span>
      <h3 className="landing__card-title">{title}</h3>
      <p className="landing__card-desc">{description}</p>
      <code className="landing__card-api">{api}</code>
    </Link>
  )
}

export function HomePage() {
  const t = useT()

  return (
    <div className="landing">
      <header className="landing__hero">
        <h1 className="landing__title">{t('home.title')}</h1>
        <p className="landing__subtitle">{renderCode(t('home.subtitle'))}</p>
      </header>

      <div className="landing__stats">
        <div className="landing__stat">
          <span className="landing__stat-value">~1.8 MB</span>
          <span className="landing__stat-label">{t('home.statBundle')}</span>
        </div>
        <div className="landing__stat">
          <span className="landing__stat-value">48 kHz</span>
          <span className="landing__stat-label">{t('home.statSampleRate')}</span>
        </div>
        <div className="landing__stat">
          <span className="landing__stat-value">WASM SIMD</span>
          <span className="landing__stat-label">{t('home.statEngine')}</span>
        </div>
        <div className="landing__stat">
          <span className="landing__stat-value">3 {t('home.statModelsUnit')}</span>
          <span className="landing__stat-label">{t('home.statModels')}</span>
        </div>
      </div>

      <section className="landing__features">
        <h2 className="landing__section-title">{t('home.featuresTitle')}</h2>
        <ul className="landing__feature-list">
          <li>{renderCode(t('home.feature1'))}</li>
          <li>{renderCode(t('home.feature2'))}</li>
          <li>{renderCode(t('home.feature3'))}</li>
          <li>{renderCode(t('home.feature4'))}</li>
          <li>{renderCode(t('home.feature5'))}</li>
        </ul>
      </section>

      <section className="landing__demos">
        <h2 className="landing__section-title">{t('home.demosTitle')}</h2>
        <div className="landing__cards">
          <DemoCard
            to="/react"
            layer="Layer 3"
            title={t('react.title')}
            description={t('home.demoReactDesc')}
            api="useDenoiser()"
          />
          <DemoCard
            to="/vanilla"
            layer="Layer 2"
            title={t('vanilla.title')}
            description={t('home.demoVanillaDesc')}
            api="createStreamDenoiser()"
          />
          <DemoCard
            to="/frame"
            layer="Layer 1"
            title={t('frame.title')}
            description={t('home.demoFrameDesc')}
            api="processFrame()"
          />
        </div>
      </section>

      <details className="demo__code">
        <summary>{t('home.quickStart')}</summary>
        <CodeBlock code={quickStart} language="tsx" />
      </details>
    </div>
  )
}
