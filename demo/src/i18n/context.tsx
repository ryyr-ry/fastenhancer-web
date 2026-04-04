import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { en } from './locales/en'
import { ja } from './locales/ja'

export type Locale = 'en' | 'ja'
export type TranslationKey = keyof typeof en
export type TFunction = (key: TranslationKey) => string

const locales: Record<Locale, Record<string, string>> = { en, ja }

const LOCALE_LABELS: Record<Locale, string> = {
  en: 'English',
  ja: '日本語',
}

const STORAGE_KEY = 'fastenhancer-locale'

function detectLocale(): Locale {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && saved in locales) return saved as Locale
  } catch {
    /* SSR or blocked localStorage */
  }
  const browserLang = navigator.language.split('-')[0]
  if (browserLang in locales) return browserLang as Locale
  return 'en'
}

interface I18nContextValue {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: TFunction
  availableLocales: readonly Locale[]
  localeLabel: (locale: Locale) => string
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleRaw] = useState<Locale>(detectLocale)

  const setLocale = useCallback((next: Locale) => {
    setLocaleRaw(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      /* blocked localStorage */
    }
    document.documentElement.lang = next
  }, [])

  const t: TFunction = useCallback(
    (key) => locales[locale]?.[key] ?? locales.en[key] ?? key,
    [locale],
  )

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale,
      t,
      availableLocales: Object.keys(locales) as Locale[],
      localeLabel: (l) => LOCALE_LABELS[l] ?? l,
    }),
    [locale, setLocale, t],
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used within <I18nProvider>')
  return ctx
}

export function useT(): TFunction {
  return useI18n().t
}

/**
 * Renders a translated string, converting backtick-wrapped text to <code> elements.
 * Example: "Powered by `useDenoiser` hook." → ["Powered by ", <code>useDenoiser</code>, " hook."]
 */
export function renderCode(text: string): ReactNode {
  const parts = text.split(/`([^`]+)`/)
  if (parts.length === 1) return text
  return parts.map((segment, i) =>
    i % 2 === 0 ? segment : <code key={i}>{segment}</code>,
  )
}
