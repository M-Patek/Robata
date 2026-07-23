import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { PURPOSE_LABEL, PURPOSE_COLORS, TERMINAL_OUTCOME_STATUS, STATUS_STYLE, STATUS_LABEL } from '@/types'
import type { CaptureScope, IncrementalWindow, StreamInference } from '@/types'

// ── Plane A: Media + Inference (replaceable execution plane) ─────────────────
// Shows the live execution view: capture scope → segments → windows → inferences

export default function PlaneAView() {
  const streamView = usePipelineStore((s) => s.streamView)
  const captureScope = streamView.capture_scope
  const windows = Array.from(streamView.windows.values())
  const inferences = Array.from(streamView.inferences.values())

  // Group inferences by window
  const inferencesByWindow = useMemo(() => {
    const map = new Map<string, StreamInference[]>()
    for (const inf of inferences) {
      const existing = map.get(inf.window_key) ?? []
      existing.push(inf)
      map.set(inf.window_key, existing)
    }
    return map
  }, [inferences])

  if (!captureScope) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', padding: 40,
      }}>
        <div style={{ textAlign: 'center' }}>
          <p style={{
            fontFamily: 'Lora, serif', fontSize: 16, color: '#A89B93',
            marginBottom: 12,
          }}>
            Plane A: Media + Inference
          </p>
          <p style={{
            fontFamily: 'Inter, sans-serif', fontSize: 13, color: '#8A7D74',
          }}>
            Click "Start Demo" to see the streaming pipeline in action.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px 28px', overflow: 'auto', height: '100%' }}>
      {/* Section header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{
          fontFamily: 'Lora, serif', fontSize: 15, fontWeight: 600,
          color: '#1A1714', margin: 0, marginBottom: 4,
        }}>
          Plane A — Media + Inference
        </h2>
        <p style={{
          fontFamily: 'Inter, sans-serif', fontSize: 12, color: '#A89B93',
          margin: 0,
        }}>
          Replaceable execution plane. Same window identities, different adapters.
        </p>
      </div>

      {/* Capture Scope Card */}
      <CaptureScopeCard scope={captureScope} />

      {/* Windows */}
      {windows.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{
            fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 600,
            color: '#6B5E55', textTransform: 'uppercase', letterSpacing: '0.07em',
            margin: '0 0 12px',
          }}>
            Windows ({windows.length} total)
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {windows.map((w) => (
              <WindowCard
                key={w.window_key}
                window={w}
                inferences={inferencesByWindow.get(w.window_key) ?? []}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Capture Scope Card ──────────────────────────────────────────────────────

function CaptureScopeCard({ scope }: { scope: CaptureScope }) {
  return (
    <div style={{
      background: '#FDFAF5',
      border: '1px solid rgba(26,23,20,0.08)',
      borderRadius: 12,
      padding: '16px 20px',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <h3 style={{
          fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 600,
          color: '#6B5E55', textTransform: 'uppercase', letterSpacing: '0.07em',
          margin: 0,
        }}>
          Capture Scope
        </h3>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#A89B93',
        }}>
          {scope.capture_scope_key.slice(0, 24)}…
        </span>
      </div>

      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12,
      }}>
        <KV label="Authority" value={scope.capture_authority_id} />
        <KV label="Epoch" value={String(scope.capture_authority_epoch)} />
        <KV label="Policy" value={scope.capture_assignment_policy_version} />
        <KV label="Acquisition" value={scope.acquisition_id} />
        <KV label="Acq. Epoch" value={String(scope.acquisition_epoch)} />
        <KV label="Channels" value={`${scope.channel_bindings.length}`} />
      </div>

      <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {scope.channel_bindings.map((ch: { camera_id: string; source_channel_id: string }) => (
          <span key={ch.camera_id} style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 10,
            padding: '3px 8px', borderRadius: 6,
            background: '#F0EBE1', color: '#6B5E55',
          }}>
            {ch.camera_id} → {ch.source_channel_id}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Window Card ─────────────────────────────────────────────────────────────

function WindowCard({ window, inferences }: { window: IncrementalWindow; inferences: StreamInference[] }) {
  const setFocusedSubject = usePipelineStore((s) => s.setFocusedSubject)

  const startSec = Number(window.effective_interval.start_ns) / 1e9
  const endSec = Number(window.effective_interval.end_ns) / 1e9
  const purposeColor = PURPOSE_COLORS[window.purpose]

  return (
    <div
      onClick={() => setFocusedSubject(window.window_key, 'window')}
      style={{
        background: '#FDFAF5',
        border: '1px solid rgba(26,23,20,0.08)',
        borderRadius: 10,
        padding: '12px 16px',
        cursor: 'pointer',
        transition: 'box-shadow 0.15s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 2px 12px rgba(26,23,20,0.08)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = 'none'
      }}
    >
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontSize: 10, fontWeight: 600, fontFamily: 'Inter, sans-serif',
            padding: '2px 8px', borderRadius: 4,
            background: purposeColor + '18', color: purposeColor,
            textTransform: 'uppercase', letterSpacing: '0.05em',
          }}>
            {PURPOSE_LABEL[window.purpose]}
          </span>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#A89B93',
          }}>
            [{startSec.toFixed(1)}s, {endSec.toFixed(1)}s)
          </span>
        </div>
        {window.refinement_role && (
          <span style={{
            fontSize: 10, fontFamily: 'Inter, sans-serif',
            padding: '2px 6px', borderRadius: 4,
            background: '#EBE0F5', color: '#7A4AA8',
          }}>
            {window.refinement_role} (gen {window.refinement_generation})
          </span>
        )}
      </div>

      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#C4B59E',
        marginBottom: 8, overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {window.window_key}
      </div>

      {/* Inferences for this window */}
      {inferences.length > 0 && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {inferences.map((inf) => {
            const status = TERMINAL_OUTCOME_STATUS[inf.terminal_outcome]
            const style = STATUS_STYLE[status]
            return (
              <span key={inf.inference_key} style={{
                fontSize: 10, fontFamily: 'Inter, sans-serif',
                padding: '2px 8px', borderRadius: 6,
                background: style.bg, color: style.text,
                display: 'flex', alignItems: 'center', gap: 4,
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: style.dot, display: 'inline-block',
                }} />
                {inf.purpose} — {STATUS_LABEL[status]}
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── KV helper ─────────────────────────────────────────────────────────────────

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span style={{
        fontSize: 10, color: '#A89B93', fontFamily: 'Inter, sans-serif',
        display: 'block', marginBottom: 2,
      }}>
        {label}
      </span>
      <span style={{
        fontSize: 12, color: '#3D3530', fontFamily: 'Inter, sans-serif',
      }}>
        {value}
      </span>
    </div>
  )
}
