import { usePipelineStore } from '@/store'
import { STATUS_STYLE } from '@/types'

// ── Watermark & Backpressure Bar ──────────────────────────────────────────────
// Shows current watermark position, backpressure state, and queue health.

export default function WatermarkBar() {
  const streamView = usePipelineStore((s) => s.streamView)
  const watermarkNs = streamView.watermark_ns
  const backpressure = streamView.backpressure
  const isSimulating = streamView.is_simulating

  const watermarkSec = Number(watermarkNs) / 1e9

  // Count active items
  const windowCount = streamView.windows.size
  const inferenceCount = streamView.inferences.size
  const segmentCount = streamView.segments.size

  const bpStyle = backpressure.level === 'CRITICAL' ? STATUS_STYLE.FAILED
    : backpressure.level === 'ELEVATED' ? STATUS_STYLE.WAITING_REVIEW
    : STATUS_STYLE.COMPLETE

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '8px 28px',
      background: '#FDFAF5',
      borderBottom: '1px solid rgba(26,23,20,0.06)',
      fontFamily: 'Inter, sans-serif', fontSize: 11,
    }}>
      {/* Left: Watermark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: '#A89B93', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Watermark
          </span>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#C96442', fontWeight: 600,
          }}>
            {watermarkSec.toFixed(3)}s
          </span>
        </div>

        {isSimulating && (
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 4,
            background: '#DDEAF5', color: '#2E5F82',
          }}>
            Simulating
          </span>
        )}
      </div>

      {/* Center: Active counts */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <CountBadge label="Segments" count={segmentCount} />
        <CountBadge label="Windows" count={windowCount} />
        <CountBadge label="Inferences" count={inferenceCount} />
      </div>

      {/* Right: Backpressure */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ color: '#A89B93', fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            Backpressure
          </span>
          <span style={{
            fontSize: 10, padding: '2px 8px', borderRadius: 4,
            background: bpStyle.bg, color: bpStyle.text, fontWeight: 500,
          }}>
            {backpressure.level} (Class {backpressure.bpClass})
          </span>
        </div>

        {backpressure.queue_depth > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ color: '#A89B93', fontSize: 10 }}>Queue</span>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#6B5E55',
            }}>
              {backpressure.queue_depth}
            </span>
          </div>
        )}

        {backpressure.oldest_required_age_ms > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ color: '#A89B93', fontSize: 10 }}>Oldest</span>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#6B5E55',
            }}>
              {backpressure.oldest_required_age_ms.toFixed(0)}ms
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Count badge ──────────────────────────────────────────────────────────────

function CountBadge({ label, count }: { label: string; count: number }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <span style={{ color: '#A89B93', fontSize: 10 }}>{label}</span>
      <span style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
        color: count > 0 ? '#4A7A5A' : '#A89B93', fontWeight: 500,
      }}>
        {count}
      </span>
    </div>
  )
}
