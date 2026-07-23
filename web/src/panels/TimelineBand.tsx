import { useState, useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { PURPOSE_COLORS, TERMINAL_OUTCOME_STATUS, STATUS_STYLE, STATUS_LABEL } from '@/types'
import type { IncrementalWindow, StreamSegment, StreamInference } from '@/types'

// ── Timeline Band: Event-time flame graph ────────────────────────────────────
// All entities (segments, windows, inferences) laid out on a shared time axis.
// Click any window bar to expand its lifecycle detail.

const TOTAL_DURATION_SEC = 40.890455
const LANE_HEIGHT = 24
const GAP = 2

export default function TimelineBand() {
  const streamView = usePipelineStore((s) => s.streamView)
  const captureScope = streamView.capture_scope
  const segments = Array.from(streamView.segments.values())
  const windows = Array.from(streamView.windows.values())
  const inferences = Array.from(streamView.inferences.values())
  const watermarkNs = streamView.watermark_ns

  const [selectedWindowKey, setSelectedWindowKey] = useState<string | null>(null)

  if (!captureScope) {
    return (
      <div style={{
        height: 40, background: '#FDFAF5',
        borderBottom: '1px solid rgba(26,23,20,0.06)',
        display: 'flex', alignItems: 'center', padding: '0 28px',
      }}>
        <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 11, color: '#C4B59E' }}>
          Timeline — waiting for capture scope…
        </span>
      </div>
    )
  }

  // Group segments by camera
  const segmentsByCamera = useMemo(() => {
    const map = new Map<string, StreamSegment[]>()
    for (const seg of segments) {
      const existing = map.get(seg.camera_id) ?? []
      existing.push(seg)
      map.set(seg.camera_id, existing)
    }
    for (const [cameraId, segs] of map) {
      segs.sort((a, b) => Number(a.effective_interval.start_ns - b.effective_interval.start_ns))
      map.set(cameraId, segs)
    }
    return map
  }, [segments])

  // Group windows by purpose for lane display
  const windowsByPurpose = useMemo(() => {
    const map = new Map<string, IncrementalWindow[]>()
    for (const w of windows) {
      const existing = map.get(w.purpose) ?? []
      existing.push(w)
      map.set(w.purpose, existing)
    }
    return map
  }, [windows])

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

  const cameraIds = ['cam_01', 'cam_02', 'cam_03', 'cam_04', 'cam_05', 'cam_06']
  const purposeOrder = ['QA_COARSE', 'QA_DENSE', 'EVENT_PROPOSAL', 'ACTION_DENSE', 'BOUNDARY_REFINEMENT']
  const activePurposes = purposeOrder.filter((p) => windowsByPurpose.has(p))

  // Calculate positions
  const scale = (ns: bigint) => {
    const pct = (Number(ns) / 1e9 / TOTAL_DURATION_SEC) * 100
    return Math.max(0, Math.min(100, pct))
  }

  const watermarkPct = scale(watermarkNs)

  // Selected window details
  const selectedWindow = selectedWindowKey ? windows.find((w) => w.window_key === selectedWindowKey) ?? null : null

  return (
    <div style={{
      background: '#FDFAF5',
      borderBottom: '1px solid rgba(26,23,20,0.08)',
      padding: '12px 28px',
      position: 'relative',
    }}>
      {/* Time axis labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, paddingLeft: 80 }}>
        {Array.from({ length: 5 }).map((_, i) => {
          const sec = (TOTAL_DURATION_SEC / 4) * i
          return (
            <span key={i} style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#C4B59E' }}>
              {sec.toFixed(1)}s
            </span>
          )
        })}
      </div>

      {/* Camera segment lanes */}
      {cameraIds.map((cameraId) => {
        const segs = segmentsByCamera.get(cameraId) ?? []
        return (
          <Lane key={cameraId} label={cameraId}>
            {segs.map((seg) => {
              const left = scale(seg.effective_interval.start_ns)
              const right = scale(seg.effective_interval.end_ns)
              const hasIssue = seg.quality_observations.some(
                (q) => q.kind === 'FREEZE' || (q.kind === 'LUMINANCE' && q.value < 20),
              )
              return (
                <Bar
                  key={seg.segment_key}
                  left={left}
                  width={Math.max(right - left, 0.3)}
                  color={hasIssue ? '#C96442' : '#4A7A5A'}
                  title={`${cameraId}: ${Number(seg.effective_interval.start_ns) / 1e9}s - ${Number(seg.effective_interval.end_ns) / 1e9}s`}
                />
              )
            })}
          </Lane>
        )
      })}

      {/* Window lanes by purpose */}
      {activePurposes.map((purpose) => {
        const purposeWindows = windowsByPurpose.get(purpose) ?? []
        return (
          <Lane key={purpose} label={purpose.replace('_', ' ')}>
            {purposeWindows.map((w) => {
              const left = scale(w.effective_interval.start_ns)
              const right = scale(w.effective_interval.end_ns)
              const infs = inferencesByWindow.get(w.window_key) ?? []
              const hasFailed = infs.some((inf) => inf.terminal_outcome === 'FAILED')
              const isSelected = selectedWindowKey === w.window_key
              return (
                <Bar
                  key={w.window_key}
                  left={left}
                  width={Math.max(right - left, 0.5)}
                  color={hasFailed ? '#C96442' : PURPOSE_COLORS[purpose as keyof typeof PURPOSE_COLORS]}
                  title={`${w.purpose}: [${Number(w.effective_interval.start_ns) / 1e9}s, ${Number(w.effective_interval.end_ns) / 1e9}s)`}
                  onClick={() => setSelectedWindowKey(isSelected ? null : w.window_key)}
                  isSelected={isSelected}
                />
              )
            })}
          </Lane>
        )
      })}

      {/* Inference outcome markers */}
      <Lane label="outcomes">
        {inferences.map((inf) => {
          const window = windows.find((w) => w.window_key === inf.window_key)
          if (!window) return null
          const left = scale(window.effective_interval.start_ns)
          const status = TERMINAL_OUTCOME_STATUS[inf.terminal_outcome]
          const style = STATUS_STYLE[status]
          return (
            <Dot
              key={inf.inference_key}
              left={left + 0.2}
              color={style.dot}
              title={`${inf.purpose}: ${inf.terminal_outcome}`}
            />
          )
        })}
      </Lane>

      {/* Watermark line */}
      <div style={{
        position: 'absolute',
        left: `${watermarkPct}%`,
        top: 0, bottom: 0,
        width: 2,
        background: '#C96442',
        zIndex: 10,
        pointerEvents: 'none',
      }}>
        <div style={{
          position: 'absolute', top: -2, left: -30,
          background: '#C96442', color: '#F8F3E8',
          fontSize: 9, fontFamily: 'JetBrains Mono',
          padding: '2px 6px', borderRadius: 4,
          whiteSpace: 'nowrap',
        }}>
          {(Number(watermarkNs) / 1e9).toFixed(1)}s
        </div>
      </div>

      {/* Selected window detail panel */}
      {selectedWindow && (
        <WindowDetailPanel
          window={selectedWindow}
          inferences={inferencesByWindow.get(selectedWindow.window_key) ?? []}
          segments={segments}
          onClose={() => setSelectedWindowKey(null)}
        />
      )}
    </div>
  )
}

// ── Lane component ────────────────────────────────────────────────────────────

function Lane({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      height: LANE_HEIGHT, marginBottom: GAP,
    }}>
      <span style={{
        fontFamily: 'JetBrains Mono', fontSize: 9,
        color: '#A89B93', width: 72, textAlign: 'right',
        flexShrink: 0, textTransform: 'uppercase',
      }}>
        {label}
      </span>
      <div style={{
        flex: 1, height: LANE_HEIGHT - 4, position: 'relative',
        background: 'rgba(26,23,20,0.03)', borderRadius: 2,
      }}>
        {children}
      </div>
    </div>
  )
}

// ── Bar component ─────────────────────────────────────────────────────────────

function Bar({
  left,
  width,
  color,
  title,
  onClick,
  isSelected,
}: {
  left: number
  width: number
  color: string
  title: string
  onClick?: () => void
  isSelected?: boolean
}) {
  return (
    <div
      onClick={onClick}
      title={title}
      style={{
        position: 'absolute',
        left: `${left}%`,
        width: `${width}%`,
        height: '100%',
        background: color,
        borderRadius: 2,
        opacity: 0.8,
        cursor: onClick ? 'pointer' : 'default',
        border: isSelected ? '2px solid #1A1714' : 'none',
        boxSizing: 'border-box',
      }}
    />
  )
}

// ── Dot component ───────────────────────────────────────────────────────────────

function Dot({ left, color, title }: { left: number; color: string; title: string }) {
  return (
    <div
      title={title}
      style={{
        position: 'absolute',
        left: `${left}%`,
        top: '50%',
        transform: 'translateY(-50%)',
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: color,
      }}
    />
  )
}

// ── Window Detail Panel ───────────────────────────────────────────────────────

function WindowDetailPanel({
  window,
  inferences,
  segments,
  onClose,
}: {
  window: IncrementalWindow
  inferences: StreamInference[]
  segments: StreamSegment[]
  onClose: () => void
}) {
  const startSec = Number(window.effective_interval.start_ns) / 1e9
  const endSec = Number(window.effective_interval.end_ns) / 1e9

  // Find segments for this window's six-slot closure
  const windowSegments = window.ordered_six_slot_closure
    .map((slot) => {
      if ('segment_key' in slot) {
        return segments.find((s) => s.segment_key === slot.segment_key)
      }
      return null
    })
    .filter(Boolean) as StreamSegment[]

  return (
    <div style={{
      position: 'absolute',
      left: 28, right: 28,
      top: '100%',
      marginTop: 8,
      background: '#FDFAF5',
      border: '1px solid rgba(26,23,20,0.10)',
      borderRadius: 12,
      boxShadow: '0 8px 32px rgba(26,23,20,0.12)',
      zIndex: 50,
      padding: '16px 20px',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontSize: 10, fontWeight: 600, fontFamily: 'Inter',
            padding: '2px 8px', borderRadius: 4,
            background: PURPOSE_COLORS[window.purpose] + '18',
            color: PURPOSE_COLORS[window.purpose],
            textTransform: 'uppercase',
          }}>
            {window.purpose}
          </span>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: '#1A1714' }}>
            [{startSec.toFixed(1)}s, {endSec.toFixed(1)}s)
          </span>
          {window.refinement_role && (
            <span style={{
              fontSize: 10, fontFamily: 'Inter',
              padding: '2px 6px', borderRadius: 4,
              background: '#EBE0F5', color: '#7A4AA8',
            }}>
              {window.refinement_role} (gen {window.refinement_generation})
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          style={{
            width: 24, height: 24, borderRadius: 6,
            border: '1px solid rgba(26,23,20,0.12)', background: 'transparent',
            cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: '#6B5E55',
          }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M1 1l8 8M9 1L1 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      {/* Lifecycle swimlane */}
      <div style={{
        display: 'flex', gap: 8, alignItems: 'stretch',
      }}>
        {/* Segments */}
        <LifecycleStage title="Segments" count={windowSegments.length}>
          {windowSegments.map((seg) => (
            <div key={seg.segment_key} style={{
              fontSize: 10, fontFamily: 'JetBrains Mono', color: '#6B5E55',
              padding: '2px 0',
            }}>
              {seg.camera_id}: {seg.segment_key.slice(0, 20)}…
            </div>
          ))}
        </LifecycleStage>

        {/* Window */}
        <LifecycleStage title="Window" count={1}>
          <div style={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: '#6B5E55', wordBreak: 'break-all' }}>
            {window.window_key}
          </div>
          <div style={{ fontSize: 9, color: '#A89B93', marginTop: 4 }}>
            semantic: {window.window_semantic_sha256.slice(0, 16)}…
          </div>
        </LifecycleStage>

        {/* Inferences */}
        <LifecycleStage title="Inferences" count={inferences.length}>
          {inferences.map((inf) => {
            const status = TERMINAL_OUTCOME_STATUS[inf.terminal_outcome]
            const style = STATUS_STYLE[status]
            return (
              <div key={inf.inference_key} style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '3px 6px', borderRadius: 4,
                background: style.bg,
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', background: style.dot,
                }} />
                <span style={{ fontSize: 10, color: style.text }}>
                  {inf.purpose} — {STATUS_LABEL[status]}
                </span>
              </div>
            )
          })}
        </LifecycleStage>

        {/* Terminal Outcome */}
        <LifecycleStage title="Outcome" count={1}>
          {inferences.length > 0 ? (
            inferences.map((inf) => {
              const status = TERMINAL_OUTCOME_STATUS[inf.terminal_outcome]
              const style = STATUS_STYLE[status]
              return (
                <div key={inf.inference_key} style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '4px 8px', borderRadius: 4,
                  background: style.bg,
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', background: style.dot,
                  }} />
                  <span style={{ fontSize: 11, color: style.text, fontWeight: 500 }}>
                    {inf.terminal_outcome}
                  </span>
                </div>
              )
            })
          ) : (
            <span style={{ fontSize: 10, color: '#A89B93' }}>Pending</span>
          )}
        </LifecycleStage>
      </div>
    </div>
  )
}

// ── Lifecycle Stage component ─────────────────────────────────────────────────

function LifecycleStage({ title, count, children }: {
  title: string
  count: number
  children: React.ReactNode
}) {
  return (
    <div style={{
      flex: 1,
      background: '#F8F3E8',
      borderRadius: 8,
      padding: '10px 12px',
      minWidth: 0,
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <span style={{
          fontSize: 10, fontWeight: 600, fontFamily: 'Inter',
          color: '#A89B93', textTransform: 'uppercase', letterSpacing: '0.07em',
        }}>
          {title}
        </span>
        <span style={{
          fontSize: 10, fontFamily: 'JetBrains Mono', color: '#C4B59E',
        }}>
          {count}
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {children}
      </div>
    </div>
  )
}
