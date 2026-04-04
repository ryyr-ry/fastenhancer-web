import { useT, type TranslationKey } from '../i18n'

type StatusValue = 'idle' | 'loading' | 'processing' | 'error' | 'destroyed'

interface StatusBadgeProps {
  status: StatusValue
}

const statusKeys: Record<StatusValue, TranslationKey> = {
  idle: 'status.idle',
  loading: 'status.loading',
  processing: 'status.processing',
  error: 'status.error',
  destroyed: 'status.destroyed',
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const t = useT()
  return (
    <span className={`status-badge status-badge--${status}`}>
      {t(statusKeys[status])}
    </span>
  )
}
