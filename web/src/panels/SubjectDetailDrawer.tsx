import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import {
  PURPOSE_LABEL, STATUS_LABEL, TERMINAL_OUTCOME_STATUS,
} from '@/types'

// ── Subject Detail Drawer ─────────────────────────────────────────────────────
// Shows complete identity for any clicked subject (capture scope, segment, window,
// inference, plan, terminal closure, or finalization).

export default function SubjectDetailDrawer() {
  const focusedKey = usePipelineStore((s) => s.focusedSubjectKey)
  const focusedType = usePipelineStore((s) => s.focusedSubjectType)
  const setFocused = usePipelineStore((s) => s.setFocusedSubject)
  const streamView = usePipelineStore((s) => s.streamView)

  const visible = focusedKey !== null

  // Resolve the focused subject
  const subject = useMemo(() => {
    if (!focusedKey) return null

    switch (focusedType) {
      case 'capture_scope':
        return streamView.capture_scope ? { type: 'Capture Scope', data: streamView.capture_scope } : null
      case 'segment':
        return streamView.segments.get(focusedKey)
          ? { type: 'Segment', data: streamView.segments.get(focusedKey)! }
          : null
      case 'window':
        return streamView.windows.get(focusedKey)
          ? { type: 'Window', data: streamView.windows.get(focusedKey)! }
          : null
      case 'inference':
        return streamView.inferences.get(focusedKey)
          ? { type: 'Inference', data: streamView.inferences.get(focusedKey)! }
          : null
      default:
        // Try to find by key in any collection
        if (streamView.capture_scope?.capture_scope_key === focusedKey) {
          return { type: 'Capture Scope', data: streamView.capture_scope }
        }
        const segment = streamView.segments.get(focusedKey)
        if (segment) return { type: 'Segment', data: segment }
        const window = streamView.windows.get(focusedKey)
        if (window) return { type: 'Window', data: window }
        const inference = streamView.inferences.get(focusedKey)
        if (inference) return { type: 'Inference', data: inference }
        return null
    }
  }, [focusedKey, focusedType, streamView])

  return (
    <>
      {/* Backdrop */}
      {visible && (
        <div
          onClick={() => setFocused(null)}
          style={{
            position: 'fixed', inset: 0, zIndex: 40,
            background: 'rgba(26,23,20,0.08)',
          }}
        />
      )}

      {/* Drawer */}
      <div style={{
        position: 'fixed',
        top: 0, right: 0, bottom: 0,
        width: 400,
        zIndex: 50,
        background: '#FDFAF5',
        borderLeft: '1px solid rgba(26,23,20,0.10)',
        boxShadow: '-4px 0 24px rgba(26,23,20,0.10)',
        transform: visible ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 0.22s cubic-bezier(0.4,0,0.2,1)',
        display: 'flex', flexDirection: 'column',
        overflowY: 'hidden',
      }}>
        {subject && (
          <>
            {/* Header */}
            <div style={{
              padding: '24px 28px 20px',
              borderBottom: '1px solid rgba(26,23,20,0.08)',
              background: '#F8F3E8',
            }}>
              <div style={{
                display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12,
              }}>
                <div>
                  <span style={{
                    fontSize: 10, fontWeight: 600, fontFamily: 'Inter, sans-serif',
                    color: '#A89B93', textTransform: 'uppercase', letterSpacing: '0.07em',
                    display: 'block', marginBottom: 4,
                  }}>
                    {subject.type}
                  </span>
                  <h2 style={{
                    fontFamily: 'Lora, serif', fontSize: 18, fontWeight: 600,
                    color: '#1A1714', margin: 0, lineHeight: 1.25,
                  }}>
                    {getSubjectTitle(subject)}
                  </h2>
                </div>
                <button
                  onClick={() => setFocused(null)}
                  style={{
                    width: 28, height: 28, borderRadius: 8,
                    border: '1px solid rgba(26,23,20,0.12)', background: 'rgba(26,23,20,0.04)',
                    cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: '#6B5E55', flexShrink: 0,
                  }}
                >
                  <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                    <path d="M1 1l9 9M10 1l-9 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </button>
              </div>
            </div>

            {/* Content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px' }}>
              <SubjectContent subject={subject} />
            </div>
          </>
        )}
      </div>
    </>
  )
}

// ── Subject content renderer ──────────────────────────────────────────────────

function SubjectContent({ subject }: { subject: { type: string; data: unknown } }) {
  const data = subject.data as Record<string, unknown>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {Object.entries(data).map(([key, value]) => {
        // Skip internal/private fields
        if (key.startsWith('_')) return null

        const displayValue = formatValue(value)
        const isComplex = typeof value === 'object' && value !== null

        return (
          <div key={key}>
            <span style={{
              fontSize: 10, fontWeight: 600, fontFamily: 'Inter, sans-serif',
              color: '#A89B93', textTransform: 'uppercase', letterSpacing: '0.07em',
              display: 'block', marginBottom: 6,
            }}>
              {key}
            </span>
            {isComplex ? (
              <div style={{
                background: '#F8F3E8', border: '1px solid rgba(26,23,20,0.06)',
                borderRadius: 8, padding: '10px 14px',
                fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
                color: '#6B5E55', overflow: 'auto',
              }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  {displayValue}
                </pre>
              </div>
            ) : (
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
                color: '#3D3530', wordBreak: 'break-all',
              }}>
                {displayValue}
              </span>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getSubjectTitle(subject: { type: string; data: unknown }): string {
  const data = subject.data as Record<string, unknown>
  switch (subject.type) {
    case 'Capture Scope':
      return data.capture_authority_id as string ?? 'Capture Scope'
    case 'Segment':
      return `${data.camera_id as string} — ${formatInterval(data.effective_interval)}`
    case 'Window':
      return `${(PURPOSE_LABEL as Record<string, string>)[data.purpose as string] ?? data.purpose} — ${formatInterval(data.effective_interval)}`
    case 'Inference':
      return `${data.purpose as string} — ${(STATUS_LABEL as Record<string, string>)[(TERMINAL_OUTCOME_STATUS as Record<string, string>)[data.terminal_outcome as string]] ?? data.terminal_outcome}`
    default:
      return subject.type
  }
}

function formatValue(value: unknown): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'bigint') return value.toString()
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, (_, v) => typeof v === 'bigint' ? v.toString() : v, 2)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

function formatInterval(interval: unknown): string {
  if (!interval || typeof interval !== 'object') return '—'
  const iv = interval as { start_ns: bigint; end_ns: bigint }
  const start = Number(iv.start_ns) / 1e9
  const end = Number(iv.end_ns) / 1e9
  return `[${start.toFixed(2)}s, ${end.toFixed(2)}s)`
}
