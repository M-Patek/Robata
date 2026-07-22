import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { INITIAL_NODES } from '@/data/pipeline'
import { RobataNodeData, NodeKind, STATUS_LABEL, STATUS_STYLE } from '@/types'

const KIND_DOCS: Record<NodeKind, { section: string; file: string; description: string }> = {
  source:              { section: '25.6',  file: 'canonical/mcap_source.py',               description: 'Six-stream MCAP decoder with V2 validation, frame indexing and alignment.' },
  media_quality:       { section: '25.3',  file: 'canonical/media_quality.py',              description: 'Per-frame luma, edge energy, freeze, cadence, sequence gap, cross-camera skew analysis.' },
  adaptive_sampler:    { section: '25.3',  file: 'sampling/adaptive.py',                    description: 'Deterministic rational grid selection; resolves frozen trigger artifacts to integer-ns targets.' },
  qa_coarse:           { section: '25.10', file: 'qa_pipeline/coarse.py',                   description: 'Validates authoritative QA_COARSE enriched coverage; all results non-production.' },
  qa_dense:            { section: '25.10', file: 'qa_pipeline/dense.py',                    description: 'Executes planned dense work for degraded or unusable coordinates.' },
  qa_gate:             { section: '25.10', file: 'qa_pipeline/completion.py',               description: 'Three-state deterministic gate: QA_COMPLETE, QA_INCOMPLETE, or explicit zero dense outcome.' },
  event_proposal:      { section: '25.10', file: 'event_pipeline/proposer.py',              description: 'Normalises authoritative EVENT_PROPOSAL outputs; emits stable CLAIMS or NO_EVENTS identities.' },
  candidate_reducer:   { section: '25.10', file: 'event_pipeline/candidate.py',             description: 'Deterministically merges connected proposal intervals; binds policy and result digests.' },
  action_evidence:     { section: '25.10', file: 'event_pipeline/evidence.py',              description: 'Validates candidate-scoped ACTION_DENSE lineage; emits SUPPORTED / NO_ACTION / INDETERMINATE.' },
  provisional_fusion:  { section: '25.10', file: 'event_pipeline/provisional_fusion.py',    description: 'Validates exact candidate/evidence closure; emits ordered 0/1/N coarse physical actions.' },
  boundary_refinement: { section: '25.10', file: 'event_pipeline/boundary_refinement.py',   description: 'Separate ONSET/OFFSET windows across six cameras; one refined result per provisional action.' },
  final_fusion:        { section: '25.10', file: 'canonical/runner.py',                     description: 'Binds versioned context to all refined actions; requires exact 1:1 or explicit zero coverage.' },
  primary_completion:  { section: '25.9',  file: 'canonical/primary_completion.py',         description: 'Atomically commits identity, ActionEvent genesis, completion and pending outbox.' },
  outbox_relay:        { section: '25.9',  file: 'adapters/sqlite_outbox.py',               description: 'At-least-once relay with fencing; PENDING → LEASED → DELIVERED; DLQ on failure.' },
  review_queue:        { section: '25.8',  file: 'adapters/sqlite_review_queue.py',         description: 'Nonblocking routing for five Section 25.8 triggers; priority/SLA ordering; immutable tasks.' },
  work_scheduler:      { section: '25.2',  file: 'adapters/sqlite_work_scheduler.py',       description: 'Durable job DAG with lease/fence/deadline; crash recovery; ACTION_PUBLISH coordination.' },
}

export default function NodeDetailDrawer() {
  const focusedNodeId  = usePipelineStore((s) => s.focusedNodeId)
  const activeRun      = usePipelineStore((s) => s.activeRun)
  const setFocused     = usePipelineStore((s) => s.setFocusedNodeId)

  const node = useMemo(() =>
    focusedNodeId ? INITIAL_NODES.find((n) => n.id === focusedNodeId) ?? null : null,
    [focusedNodeId],
  )

  const visible = !!node

  return (
    <>
      {/* Backdrop — clicking closes the drawer */}
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
        width: 360,
        zIndex: 50,
        background: '#FDFAF5',
        borderLeft: '1px solid rgba(26,23,20,0.10)',
        boxShadow: '-4px 0 24px rgba(26,23,20,0.10)',
        transform: visible ? 'translateX(0)' : 'translateX(100%)',
        transition: 'transform 0.22s cubic-bezier(0.4,0,0.2,1)',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'hidden',
      }}>
        {node && (() => {
          const data = node.data as RobataNodeData
          const liveStatus = activeRun?.node_statuses?.[node.id] ?? data.status
          const style      = STATUS_STYLE[liveStatus]
          const docs       = KIND_DOCS[data.kind]

          return (
            <>
              {/* Header */}
              <div style={{
                padding: '24px 28px 20px',
                borderBottom: '1px solid rgba(26,23,20,0.08)',
                background: style.header,
              }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
                  <div>
                    <h2 style={{
                      fontFamily: 'Lora, serif',
                      fontSize: 20,
                      fontWeight: 600,
                      color: '#1A1714',
                      lineHeight: 1.25,
                      margin: 0,
                    }}>
                      {data.label}
                    </h2>
                    <span style={{
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: 11,
                      color: '#A89B93',
                      marginTop: 4,
                      display: 'block',
                    }}>
                      {node.id}
                    </span>
                  </div>
                  <button
                    onClick={() => setFocused(null)}
                    style={{
                      width: 28, height: 28, borderRadius: 8,
                      border: '1px solid rgba(26,23,20,0.12)',
                      background: 'rgba(26,23,20,0.04)',
                      cursor: 'pointer',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#6B5E55', flexShrink: 0,
                    }}
                  >
                    <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
                      <path d="M1 1l9 9M10 1l-9 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    </svg>
                  </button>
                </div>
                <div style={{ marginTop: 12 }}>
                  <span className="status-pill" style={{ background: style.bg, color: style.text, fontSize: 11 }}>
                    {STATUS_LABEL[liveStatus]}
                  </span>
                </div>
              </div>

              {/* Content */}
              <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px' }}>
                {/* Description */}
                <p style={{
                  fontFamily: 'Inter, sans-serif',
                  fontSize: 14,
                  color: '#3D3530',
                  lineHeight: 1.65,
                  marginBottom: 24,
                }}>
                  {docs.description}
                </p>

                {/* Architecture */}
                <Section title="Architecture">
                  <KV label="Section" value={`V1.1 §${docs.section}`} />
                  <KV label="Source file" value={docs.file} mono accent="blue" />
                </Section>

                {/* Metrics */}
                {data.metrics && Object.keys(data.metrics).length > 0 && (
                  <Section title="Metrics">
                    {data.metrics.camera_count !== undefined &&
                      <KV label="Cameras" value={`${data.metrics.camera_count}×`} mono />}
                    {data.metrics.duration_ms !== undefined &&
                      <KV label="Duration" value={`${data.metrics.duration_ms} ms`} mono />}
                    {data.metrics.attempt !== undefined &&
                      <KV label="Attempt" value={`#${data.metrics.attempt}`} mono />}
                    {data.metrics.schema_version &&
                      <KV label="Schema" value={data.metrics.schema_version} mono accent="purple" />}
                    {data.metrics.sha256 &&
                      <KV label="SHA-256" value={`${data.metrics.sha256.slice(0, 18)}…`} mono />}
                  </Section>
                )}

                {/* Instance */}
                {data.instance_id && (
                  <Section title="Instance">
                    <KV label="ID" value={`${data.instance_id}`} mono />
                    {data.instance_count && <KV label="Total" value={`${data.instance_count}`} mono />}
                  </Section>
                )}

                {/* Evidence class */}
                <Section title="Evidence">
                  <KV label="Class" value="LOCAL_CONFORMANCE" mono accent="amber" />
                  <KV label="Production eligible" value="false" mono accent="coral" />
                </Section>
              </div>
            </>
          )
        })()}
      </div>
    </>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p style={{
        fontSize: 10,
        fontWeight: 600,
        letterSpacing: '0.09em',
        textTransform: 'uppercase',
        color: '#A89B93',
        fontFamily: 'Inter, sans-serif',
        marginBottom: 8,
      }}>
        {title}
      </p>
      <div style={{
        background: '#F8F3E8',
        border: '1px solid rgba(26,23,20,0.08)',
        borderRadius: 10,
        overflow: 'hidden',
      }}>
        {children}
      </div>
    </div>
  )
}

function KV({ label, value, mono, accent }: {
  label: string; value: string; mono?: boolean; accent?: 'blue' | 'purple' | 'amber' | 'coral'
}) {
  const accentMap: Record<string, string> = {
    blue:   '#4A7FA8',
    purple: '#6A4AA8',
    amber:  '#A87A2A',
    coral:  '#C96442',
  }
  const accentColor = accent ? accentMap[accent] ?? '#3D3530' : '#3D3530'

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      padding: '10px 14px',
      borderBottom: '1px solid rgba(26,23,20,0.06)',
      gap: 12,
    }}>
      <span style={{
        fontSize: 12,
        color: '#8A7D74',
        fontFamily: 'Inter, sans-serif',
        flexShrink: 0,
        paddingTop: 1,
      }}>
        {label}
      </span>
      <span style={{
        fontSize: mono ? 11 : 13,
        fontFamily: mono ? 'JetBrains Mono, monospace' : 'Inter, sans-serif',
        color: accentColor,
        textAlign: 'right',
        wordBreak: 'break-all',
      }}>
        {value}
      </span>
    </div>
  )
}
