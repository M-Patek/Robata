import { usePipelineStore } from '@/store'
import {
  STATUS_STYLE, STATUS_LABEL, TERMINAL_OUTCOME_STATUS,
  PURPOSE_LABEL, PURPOSE_COLORS,
} from '@/types'
import type { ExpectedWindowPlan, WindowTerminalClosure, RecordingFinalizationMap } from '@/types'

// ── Plane B: Durable Window DAG (persistent, cross-recording) ─────────────────
// Shows the append-only expected-window plan and its terminal closure.
// This plane survives restarts and is the authority for recording finalization.

export default function PlaneBView() {
  const streamView = usePipelineStore((s) => s.streamView)
  const plan = streamView.plan
  const terminalClosures = Array.from(streamView.terminal_closures.values())
  const finalization = streamView.finalization

  const hasData = plan !== null || terminalClosures.length > 0 || finalization !== null

  if (!hasData) {
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
            Plane B: Durable Window DAG
          </p>
          <p style={{
            fontFamily: 'Inter, sans-serif', fontSize: 13, color: '#8A7D74',
          }}>
            The append-only plan and terminal closure will appear here during simulation.
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
          Plane B — Durable Window DAG
        </h2>
        <p style={{
          fontFamily: 'Inter, sans-serif', fontSize: 12, color: '#A89B93',
          margin: 0,
        }}>
          Persistent plan + terminal closure. Survives restarts. Authority for finalization.
        </p>
      </div>

      {/* Expected Window Plan */}
      {plan && <ExpectedWindowPlanCard plan={plan} />}

      {/* Terminal Closure */}
      {terminalClosures.map((closure) => (
        <TerminalClosureCard key={closure.closure_key} closure={closure} />
      ))}

      {/* Recording Finalization */}
      {finalization && <FinalizationCard finalization={finalization} />}
    </div>
  )
}

// ── Expected Window Plan Card ────────────────────────────────────────────────

function ExpectedWindowPlanCard({ plan }: { plan: ExpectedWindowPlan }) {
  const windows = usePipelineStore((s) => s.streamView.windows)
  const inferences = usePipelineStore((s) => s.streamView.inferences)
  const isSealed = plan.sealed_manifest !== null

  return (
    <div style={{
      background: '#FDFAF5',
      border: '1px solid rgba(26,23,20,0.08)',
      borderRadius: 12,
      padding: '16px 20px',
      marginBottom: 16,
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
          Expected Window Plan
        </h3>
        <span style={{
          fontSize: 10, fontFamily: 'Inter, sans-serif',
          padding: '2px 8px', borderRadius: 4,
          background: isSealed ? '#D6EAD9' : '#DDEAF5',
          color: isSealed ? '#2E5E38' : '#2E5F82',
        }}>
          {isSealed ? 'Sealed' : 'Appending'}
        </span>
      </div>

      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#C4B59E',
        marginBottom: 12, overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {plan.plan_key}
      </div>

      {/* Declarations table */}
      <div style={{
        maxHeight: 300, overflow: 'auto',
        border: '1px solid rgba(26,23,20,0.06)', borderRadius: 8,
      }}>
        <table style={{
          width: '100%', borderCollapse: 'collapse',
          fontFamily: 'Inter, sans-serif', fontSize: 11,
        }}>
          <thead>
            <tr style={{
              borderBottom: '1px solid rgba(26,23,20,0.08)',
              background: '#F8F3E8',
            }}>
              <th style={{ padding: '6px 10px', textAlign: 'left', color: '#A89B93', fontWeight: 500 }}>#</th>
              <th style={{ padding: '6px 10px', textAlign: 'left', color: '#A89B93', fontWeight: 500 }}>Window Key</th>
              <th style={{ padding: '6px 10px', textAlign: 'left', color: '#A89B93', fontWeight: 500 }}>Interval</th>
              <th style={{ padding: '6px 10px', textAlign: 'left', color: '#A89B93', fontWeight: 500 }}>Purpose</th>
              <th style={{ padding: '6px 10px', textAlign: 'left', color: '#A89B93', fontWeight: 500 }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {plan.declarations.map((decl) => {
              const window = windows.get(decl.window_key)
              const purpose = window?.purpose ?? 'QA_COARSE'
              const purposeColor = PURPOSE_COLORS[purpose]
              const windowInferences = Array.from(inferences.values()).filter((inf) => inf.window_key === decl.window_key)
              const hasSucceeded = windowInferences.some((inf) => inf.terminal_outcome === 'SUCCEEDED')
              const hasFailed = windowInferences.some((inf) => inf.terminal_outcome === 'FAILED')
              const status = hasFailed ? 'FAILED' : hasSucceeded ? 'COMPLETE' : 'PENDING'
              const style = STATUS_STYLE[status]

              return (
                <tr key={decl.expected_ordinal} style={{
                  borderBottom: '1px solid rgba(26,23,20,0.04)',
                }}>
                  <td style={{ padding: '6px 10px', color: '#A89B93', fontFamily: 'JetBrains Mono, monospace' }}>
                    {decl.expected_ordinal}
                  </td>
                  <td style={{ padding: '6px 10px', color: '#6B5E55', fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }}>
                    {decl.window_key.slice(0, 32)}…
                  </td>
                  <td style={{ padding: '6px 10px', color: '#6B5E55' }}>
                    [{Number(decl.requested_interval.start_ns) / 1e9}s, {Number(decl.requested_interval.end_ns) / 1e9}s)
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={{
                      fontSize: 10, padding: '1px 6px', borderRadius: 4,
                      background: purposeColor + '18', color: purposeColor,
                    }}>
                      {PURPOSE_LABEL[purpose]}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={{
                      fontSize: 10, padding: '2px 8px', borderRadius: 4,
                      background: style.bg, color: style.text,
                    }}>
                      {STATUS_LABEL[status]}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {isSealed && plan.sealed_manifest && (
        <div style={{
          marginTop: 12, padding: '8px 12px',
          background: '#E2EFE4', borderRadius: 6,
          fontSize: 11, color: '#2E5E38', fontFamily: 'Inter, sans-serif',
        }}>
          Sealed at {new Date(plan.sealed_manifest.sealed_at).toLocaleString()} — {plan.sealed_manifest.ordered_members.length} members
        </div>
      )}
    </div>
  )
}

// ── Terminal Closure Card ─────────────────────────────────────────────────────

function TerminalClosureCard({ closure }: { closure: WindowTerminalClosure }) {
  const succeeded = closure.members.filter((m) => m.terminal_outcome === 'SUCCEEDED').length
  const failed = closure.members.filter((m) => m.terminal_outcome === 'FAILED').length
  const total = closure.members.length

  return (
    <div style={{
      background: '#FDFAF5',
      border: '1px solid rgba(26,23,20,0.08)',
      borderRadius: 12,
      padding: '16px 20px',
      marginBottom: 16,
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
          Terminal Closure
        </h3>
        <span style={{
          fontSize: 10, fontFamily: 'Inter, sans-serif',
          padding: '2px 8px', borderRadius: 4,
          background: failed === 0 ? '#D6EAD9' : '#F5DADA',
          color: failed === 0 ? '#2E5E38' : '#7A2A2A',
        }}>
          {succeeded}/{total} succeeded
        </span>
      </div>

      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#C4B59E',
        marginBottom: 12,
      }}>
        {closure.closure_key}
      </div>

      {/* Member summary */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {closure.members.slice(0, 20).map((member) => {
          const style = STATUS_STYLE[TERMINAL_OUTCOME_STATUS[member.terminal_outcome]]
          return (
            <div key={member.expected_ordinal} style={{
              width: 20, height: 20, borderRadius: 4,
              background: style.dot, opacity: 0.8,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }} title={`#${member.expected_ordinal}: ${member.terminal_outcome}`}>
              <span style={{ fontSize: 8, color: '#fff', fontFamily: 'Inter, sans-serif' }}>
                {member.expected_ordinal}
              </span>
            </div>
          )
        })}
        {closure.members.length > 20 && (
          <span style={{ fontSize: 10, color: '#A89B93', alignSelf: 'center' }}>
            +{closure.members.length - 20} more
          </span>
        )}
      </div>
    </div>
  )
}

// ── Finalization Card ─────────────────────────────────────────────────────────

function FinalizationCard({ finalization }: { finalization: RecordingFinalizationMap }) {
  const totalMappings = finalization.incremental_to_final_mappings.length
  const succeeded = finalization.incremental_to_final_mappings.filter(
    (m) => m.terminal_outcome === 'SUCCEEDED',
  ).length

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
          Recording Finalization
        </h3>
        <span style={{
          fontSize: 10, fontFamily: 'Inter, sans-serif',
          padding: '2px 8px', borderRadius: 4,
          background: finalization.finalized_at ? '#D6EAD9' : '#DDEAF5',
          color: finalization.finalized_at ? '#2E5E38' : '#2E5F82',
        }}>
          {finalization.finalized_at ? 'Finalized' : 'Pending'}
        </span>
      </div>

      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12,
      }}>
        <KV label="Duration" value={`${Number(finalization.duration_ns) / 1e9}s`} />
        <KV label="Mappings" value={`${succeeded}/${totalMappings}`} />
        <KV label="Source Digest" value={finalization.source_digest.slice(0, 16) + '…'} />
      </div>

      <div style={{
        fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#C4B59E',
      }}>
        {finalization.finalization_key}
      </div>
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
