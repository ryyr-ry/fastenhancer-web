import { NavLink, Outlet } from 'react-router-dom'
import { useI18n, type TranslationKey } from '../i18n'

const links: { to: string; labelKey: TranslationKey; end?: boolean }[] = [
  { to: '/', labelKey: 'nav.home', end: true },
  { to: '/react', labelKey: 'nav.react' },
  { to: '/vanilla', labelKey: 'nav.vanilla' },
  { to: '/frame', labelKey: 'nav.frame' },
]

export function Layout() {
  const { locale, setLocale, t, availableLocales, localeLabel } = useI18n()

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__inner">
          <div className="brand">
            <span className="brand__mark" aria-hidden="true" />
            <span>fastenhancer-web</span>
          </div>
          <nav className="nav-links" aria-label={t('nav.aria')}>
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) =>
                  `nav-link${isActive ? ' nav-link--active' : ''}`
                }
              >
                {t(link.labelKey)}
              </NavLink>
            ))}
          </nav>
          <select
            className="locale-switcher"
            value={locale}
            onChange={(e) => setLocale(e.target.value as typeof locale)}
          >
            {availableLocales.map((l) => (
              <option key={l} value={l}>{localeLabel(l)}</option>
            ))}
          </select>
        </div>
      </header>
      <main className="page-shell">
        <Outlet />
      </main>
    </div>
  )
}
