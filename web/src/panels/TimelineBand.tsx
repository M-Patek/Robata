import { useMemo } from 'react'
import { usePipelineStore } from '@/store'

// ── Timeline Band: visual event-time progression ──────────────────────────────
// Shows capture scope, segments, windows, and watermark along a shared time axis.

export default function TimelineBand() {
  const streamView = usePipelineStore((s) => s.streamView)
  const captureScope = streamView.capture_scope
  const segments = Array.from(streamView.segments.values())
  const windows = Array.from(streamView.windows.values())
  const watermarkNs = streamView.watermark_ns

  if (!captureScope) {
    return (
      <div style={{
        height: 40, background: '#FDFAF5',
        borderBottom: '1px solid rgba(26,23,20,0.06)',
        display: 'flex', alignItems: 'center', padding: '0 28px',
      }}>
        <span style={{
          fontFamily: 'Inter, sans-serif', fontSize: 11, color: '#C4B59E',
        }}>
          Timeline — waiting for capture scope…
        </span>
      </div>
    )
  }

  // Calculate total duration from segments or windows
  const totalDurationNs = useMemo(() => {
    if (windows.length > 0) {
      const lastWindow = windows[windows.length - 1]
      return lastWindow.effective_interval.end_ns
    }
    if (segments.length > 0) {
      const lastSegment = segments[segments.length - 1]
      return lastSegment.effective_interval.end_ns
    }
    return 40_890_455_000n // 40.89s in ns
  }, [segments, windows])

  const totalDurationSec = Number(totalDurationNs) / 1e9
  const watermarkSec = Number(watermarkNs) / 1e9

  // Group segments by camera for stacked display
  const cameraIds = useMemo(() => {
    const ids = new Set<string>()
    for (const seg of segments) ids.add(seg.camera_id)
    return Array.from(ids).sort()
  }, [segments])

  const segmentsByCamera = useMemo(() => {
    const map = new Map<string, typeof segments>()
    for (const seg of segments) {
      const existing = map.get(seg.camera_id) ?? []
      existing.push(seg)
      map.set(seg.camera_id, existing)
    }
    return map
  }, [segments])

  const scale = (ns: bigint) => {
    const pct = (Number(ns) / 1e9 / totalDurationSec) * 100
    return Math.max(0, Math.min(100, pct))
  }

  return (
    <div style={{
      background: '#FDFAF5',
      borderBottom: '1px solid rgba(26,23,20,0.08)',
      padding: '12px 28px',
    }}>
      {/* Time axis labels */}
      <div style={{
        display: 'flex', justifyContent: 'space-between',
        marginBottom: 6, paddingLeft: 80,
      }}>
        {Array.from({ length: 5 }).map((_, i) => {
          const sec = (totalDurationSec / 4) * i
          return (
            <span key={i} style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 9, color: '#C4B59E',
            }}>
              {sec.toFixed(1)}s
            </span>
          )
        })}
      </div>

      {/* Camera segment lanes */}
      {cameraIds.map((cameraId) => {
        const segs = segmentsByCamera.get(cameraId) ?? []
        return (
          <div key={cameraId} style={{
            display: 'flex', alignItems: 'center', gap: 8,
            height: 18, marginBottom: 2,
          }}>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 9,
              color: '#A89B93', width: 72, textAlign: 'right',
              flexShrink: 0,
            }}>
              {cameraId}
            </span>
            <div style={{
              flex: 1, height: 12, position: 'relative',
              background: 'rgba(26,23,20,0.04)', borderRadius: 2,
            }}>
              {segs.map((seg) => {
                const left = scale(seg.effective_interval.start_ns)
                const right = scale(seg.effective_interval.end_ns)
                const width = right - left
                const hasIssue = seg.quality_observations.some(
                  (q) => q.kind === 'FREEZE' || (q.kind === 'LUMINANCE' && q.value < 20),
                )
                return (
                  <div
                    key={seg.segment_key}
                    title={`${cameraId}: ${Number(seg.effective_interval.start_ns) / 1e9}s - ${Number(seg.effective_interval.end_ns) / 1e9}s`}
                    style={{
                      position: 'absolute',
                      left: `${left}%`, width: `${Math.max(width, 0.5)}%`,
                      height: '100%', top: 0,
                      background: hasIssue ? '#C96442' : '#4A7A5A',
                      borderRadius: 2, opacity: 0.7,
                    }}
                  />
                )
              })}
            </div>
          </div>
        )
      })}

      {/* Window lane */}
      {windows.length > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          height: 18, marginTop: 4,
        }}>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 9,
            color: '#A89B93', width: 72, textAlign: 'right',
            flexShrink: 0,
          }}>
            windows
          </span>
          <div style={{
            flex: 1, height: 12, position: 'relative',
            background: 'rgba(26,23,20,0.04)', borderRadius: 2,
          }}>
            {windows.map((w) => {
              const left = scale(w.effective_interval.start_ns)
              const right = scale(w.effective_interval.end_ns)
              const width = right - left
              const purposeColors: Record<string, string> = {
                QA_COARSE: '#4A7FA8',
                QA_DENSE: '#6B5EA8',
                EVENT_PROPOSAL: '#A87A2A',
                ACTION_DENSE: '#4A7A5A',
                BOUNDARY_REFINEMENT: '#A84A7A',
              }
              return (
                <div
                  key={w.window_key}
                  title={`${w.purpose}: ${Number(w.effective_interval.start_ns) / 1e9}s - ${Number(w.effective_interval.end_ns) / 1e9}s`}
                  style={{
                    position: 'absolute',
                    left: `${left}%`, width: `${Math.max(width, 0.5)}%`,
                    height: '100%', top: 0,
                    background: purposeColors[w.purpose] ?? '#A89B93',
                    borderRadius: 2, opacity: 0.6,
                  }}
                />
              )
            })}
          </div>
        </div>
      )}

      {/* Watermark indicator */}
      <div style={{
        position: 'relative', height: 20, marginTop: 4,
      }}>
        <div style={{
          position: 'absolute',
          left: `${scale(watermarkNs)}%`,
          top: -40, bottom: 0,
          width: 2,
          background: '#C96442',
          zIndex: 5,
        }}>
          <div style={{
            position: 'absolute', top: -20, left: -30,
            background: '#C96442', color: '#F8F3E8',
            fontSize: 9, fontFamily: 'JetBrains Mono, monospace',
            padding: '2px 6px', borderRadius: 4,
            whiteSpace: 'nowrap',
          }}>
            watermark {watermarkSec.toFixed(1)}s
          </div>
        </div>
      </div>
    </div>
  )
}
