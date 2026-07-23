import { useState } from 'react'
import { usePipelineStore } from '@/store'
import { useWebSocket } from '@/hooks/useWebSocket'
import PlaneAView from '@/panels/PlaneAView'
import PlaneBView from '@/panels/PlaneBView'
import TimelineBand from '@/panels/TimelineBand'
import WatermarkBar from '@/panels/WatermarkBar'
import SubjectDetailDrawer from '@/panels/SubjectDetailDrawer'

export default function App() {
  const streamView = usePipelineStore((s) => s.streamView)
  const activePlane = usePipelineStore((s) => s.activePlane)
  const setActivePlane = usePipelineStore((s) => s.setActivePlane)
  const wsConnected = usePipelineStore((s) => s.wsConnected)

  const { startSimulation, stopSimulation } = useWebSocket()

  const [showEventLog, setShowEventLog] = useState(false)

  const hasData = streamView.capture_scope !== null

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100vh', width: '100vw',
      overflow: 'hidden', background: '#F8F3E8', color: '#1A1714',
    }}>
      {/* ── Top bar ───────────────────────────────────────────────────────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 28px', height: 56,
        background: '#FDFAF5',
        borderBottom: '1px solid rgba(26,23,20,0.09)',
        boxShadow: '0 1px 3px rgba(26,23,20,0.04)',
        flexShrink: 0, zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{
            fontFamily: 'Lora, serif', fontSize: 18, fontWeight: 600,
            color: '#1A1714', letterSpacing: '-0.01em',
          }}>
            Robata
          </span>
          <div style={{ width: 1, height: 18, background: 'rgba(26,23,20,0.12)' }} />
          <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 13, color: '#A89B93' }}>
            Streaming Pipeline
          </span>
          {hasData && streamView.capture_scope && (
            <>
              <div style={{ width: 1, height: 18, background: 'rgba(26,23,20,0.10)' }} />
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#C4B59E',
                maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {streamView.capture_scope.capture_scope_key.slice(0, 32)}…
              </span>
            </>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Plane toggle */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 4,
            padding: '4px 6px', borderRadius: 8,
            border: '1px solid rgba(26,23,20,0.10)', background: '#F8F3E8',
          }}>
            {(['A', 'B', 'both'] as const).map((plane) => (
              <button
                key={plane}
                onClick={() => setActivePlane(plane)}
                style={{
                  padding: '4px 12px', borderRadius: 6, border: 'none',
                  background: activePlane === plane ? '#1A1714' : 'transparent',
                  color: activePlane === plane ? '#F8F3E8' : '#6B5E55',
                  fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 500,
                  cursor: 'pointer', transition: 'all 0.12s',
                }}
              >
                {plane === 'A' ? 'Plane A' : plane === 'B' ? 'Plane B' : 'Both'}
              </button>
            ))}
          </div>

          {/* Connection status */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '5px 13px', borderRadius: 20,
            border: '1px solid rgba(26,23,20,0.10)', background: '#F8F3E8',
            fontFamily: 'Inter, sans-serif', fontSize: 12,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: wsConnected ? '#4A7A5A' : '#C4B59E',
              display: 'inline-block',
            }} />
            <span style={{ color: wsConnected ? '#4A7A5A' : '#A89B93' }}>
              {wsConnected ? 'Live' : 'Demo'}
            </span>
          </div>

          {/* Event log toggle */}
          <button
            onClick={() => setShowEventLog(!showEventLog)}
            style={{
              padding: '6px 14px', borderRadius: 8,
              border: '1px solid rgba(26,23,20,0.12)', background: 'transparent',
              fontFamily: 'Inter, sans-serif', fontSize: 12, color: '#6B5E55',
              cursor: 'pointer',
            }}
          >
            {showEventLog ? 'Hide Log' : 'Event Log'}
          </button>

          {/* Simulation controls */}
          {streamView.is_simulating ? (
            <button
              onClick={stopSimulation}
              style={{
                padding: '7px 18px', borderRadius: 8, border: 'none',
                background: '#C96442', color: '#F8F3E8',
                fontFamily: 'Inter, sans-serif', fontSize: 13, fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              Stop
            </button>
          ) : (
            <button
              onClick={startSimulation}
              style={{
                padding: '7px 18px', borderRadius: 8, border: 'none',
                background: '#1A1714', color: '#F8F3E8',
                fontFamily: 'Inter, sans-serif', fontSize: 13, fontWeight: 500,
                cursor: 'pointer', transition: 'background 0.12s',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = '#3D3530')}
              onMouseLeave={(e) => (e.currentTarget.style.background = '#1A1714')}
            >
              {hasData ? 'Replay' : 'Start Demo'}
            </button>
          )}
        </div>
      </header>

      {/* ── Timeline band ───────────────────────────────────────────────────── */}
      <div style={{ flexShrink: 0 }}>
        <TimelineBand />
      </div>

      {/* ── Watermark & backpressure bar ──────────────────────────────────── */}
      <div style={{ flexShrink: 0 }}>
        <WatermarkBar />
      </div>

      {/* ── Main content: two planes ────────────────────────────────────────── */}
      <div style={{
        flex: 1, minHeight: 0,
        display: 'flex', flexDirection: 'row',
        overflow: 'hidden',
      }}>
        {/* Plane A: Media + Inference */}
        {(activePlane === 'A' || activePlane === 'both') && (
          <div style={{
            flex: activePlane === 'both' ? 1 : undefined,
            width: activePlane === 'A' ? '100%' : undefined,
            minWidth: 0,
            borderRight: activePlane === 'both' ? '1px solid rgba(26,23,20,0.08)' : undefined,
            overflow: 'auto',
          }}>
            <PlaneAView />
          </div>
        )}

        {/* Plane B: Durable Window DAG */}
        {(activePlane === 'B' || activePlane === 'both') && (
          <div style={{
            flex: activePlane === 'both' ? 1 : undefined,
            width: activePlane === 'B' ? '100%' : undefined,
            minWidth: 0,
            overflow: 'auto',
          }}>
            <PlaneBView />
          </div>
        )}
      </div>

      {/* ── Subject detail drawer ─────────────────────────────────────────── */}
      <SubjectDetailDrawer />

      {/* ── Event log panel (optional) ────────────────────────────────────── */}
      {showEventLog && <EventLogPanel onClose={() => setShowEventLog(false)} />}
    </div>
  )
}

// ── Event log panel ──────────────────────────────────────────────────────────

function EventLogPanel({ onClose }: { onClose: () => void }) {
  const streamView = usePipelineStore((s) => s.streamView)

  // Build a simple event log from current state
  const events = [
    streamView.capture_scope && { type: 'CAPTURE_SCOPE', key: streamView.capture_scope.capture_scope_key, time: 0 },
    ...Array.from(streamView.segments.values()).map((s, i) => ({ type: 'SEGMENT', key: s.segment_key, time: i })),
    ...Array.from(streamView.windows.values()).map((w, i) => ({ type: 'WINDOW', key: w.window_key, time: i })),
    ...Array.from(streamView.inferences.values()).map((inf, i) => ({ type: 'INFERENCE', key: inf.inference_key, time: i })),
    streamView.plan && { type: 'PLAN', key: streamView.plan.plan_key, time: 0 },
    ...Array.from(streamView.terminal_closures.values()).map((c, i) => ({ type: 'TERMINAL_CLOSURE', key: c.closure_key, time: i })),
    streamView.finalization && { type: 'FINALIZATION', key: streamView.finalization.finalization_key, time: 0 },
  ].filter(Boolean) as { type: string; key: string; time: number }[]

  return (
    <div style={{
      position: 'fixed', bottom: 0, left: 0, right: 0,
      height: 200, zIndex: 90,
      background: '#FDFAF5',
      borderTop: '1px solid rgba(26,23,20,0.10)',
      boxShadow: '0 -4px 24px rgba(26,23,20,0.08)',
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '8px 20px',
        borderBottom: '1px solid rgba(26,23,20,0.06)',
      }}>
        <span style={{
          fontFamily: 'Inter, sans-serif', fontSize: 12, fontWeight: 600,
          color: '#6B5E55', textTransform: 'uppercase', letterSpacing: '0.07em',
        }}>
          Event Log ({events.length} events)
        </span>
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
      <div style={{
        flex: 1, overflow: 'auto', padding: '8px 20px',
        fontFamily: 'JetBrains Mono, monospace', fontSize: 11,
      }}>
        {events.map((ev, i) => (
          <div key={i} style={{
            display: 'flex', gap: 12, padding: '3px 0',
            borderBottom: '1px solid rgba(26,23,20,0.04)',
          }}>
            <span style={{ color: '#A89B93', flexShrink: 0, width: 80 }}>{ev.type}</span>
            <span style={{ color: '#6B5E55', overflow: 'hidden', textOverflow: 'ellipsis' }}>{ev.key}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
