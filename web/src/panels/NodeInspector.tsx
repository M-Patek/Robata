import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { INITIAL_NODES } from '@/data/pipeline'
import { RobataNodeData, NodeKind, STATUS_LABEL, STATUS_STYLE } from '@/types'

const KIND_DOCS: Record<NodeKind, { section: string; file: string }> = {
  source:              { section: '25.6',  file: 'canonical/mcap_source.py' },
  media_quality:       { section: '25.3',  file: 'canonical/media_quality.py' },
  adaptive_sampler:    { section: '25.3',  file: 'sampling/adaptive.py' },
  qa_coarse:           { section: '25.10', file: 'qa_pipeline/coarse.py' },
  qa_dense:            { section: '25.10', file: 'qa_pipeline/dense.py' },
  qa_gate:             { section: '25.10', file: 'qa_pipeline/completion.py' },
  event_proposal:      { section: '25.10', file: 'event_pipeline/proposer.py' },
  candidate_reducer:   { section: '25.10', file: 'event_pipeline/candidate.py' },
  action_evidence:     { section: '25.10', file: 'event_pipeline/evidence.py' },
  provisional_fusion:  { section: '25.10', file: 'event_pipeline/provisional_fusion.py' },
  boundary_refinement: { section: '25.10', file: 'event_pipeline/boundary_refinement.py' },
  final_fusion:        { section: '25.10', file: 'canonical/runner.py' },
  primary_completion:  { section: '25.9',  file: 'canonical/primary_completion.py' },
  outbox_relay:        { section: '25.9',  file: 'adapters/sqlite_outbox.py' },
  review_queue:        { section: '25.8',  file: 'adapters/sqlite_review_queue.py' },
  work_scheduler:      { section: '25.2',  file: 'adapters/sqlite_work_scheduler.py' },
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3 py-1.5"
      style={{ borderBottom: '1px solid rgba(26,23,20,0.06)' }}>
      <span className="label-muted flex-shrink-0 pt-0.5">{label}</span>
      <span className="text-[11px] font-mono text-right break-all" style={{ color: '#3D3530' }}>
        {children}
      </span>
    </div>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <p className="label-muted mb-2">{title}</p>
      <div className="rounded-lg px-3 py-0.5"
        style={{ background: '#F8F3E8', border: '1px solid rgba(26,23,20,0.07)' }}>
        {children}
      </div>
    </div>
  )
}

export default function NodeInspector() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId)
  const activeRun      = usePipelineStore((s) => s.activeRun)
  const setSelected    = usePipelineStore((s) => s.setSelectedNodeId)

  const node = useMemo(() =>
    selectedNodeId ? INITIAL_NODES.find((n) => n.id === selectedNodeId) ?? null : null,
    [selectedNodeId],
  )

  if (!node) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3 p-6 text-center"
        style={{ color: '#A89B93' }}>
        <div className="w-8 h-8 rounded-full flex items-center justify-center"
          style={{ background: '#EDE4D3' }}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="5.5" stroke="#A89B93" strokeWidth="1.25"/>
            <circle cx="7" cy="7" r="1.5" fill="#A89B93"/>
          </svg>
        </div>
        <span className="text-[12px]" style={{ fontFamily: 'Inter, sans-serif' }}>
          Select a node to inspect
        </span>
      </div>
    )
  }

  const data = node.data as RobataNodeData
  const liveStatus = activeRun?.node_statuses?.[node.id] ?? data.status
  const statusStyle = STATUS_STYLE[liveStatus]
  const docs = KIND_DOCS[data.kind]

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid rgba(26,23,20,0.08)' }}>
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold leading-snug"
              style={{ fontFamily: 'Lora, serif', color: '#1A1714' }}>
              {data.label}
            </h3>
            <p className="text-[10px] font-mono mt-0.5" style={{ color: '#A89B93' }}>
              {node.id}
            </p>
          </div>
          <button onClick={() => setSelected(null)}
            className="flex-shrink-0 w-5 h-5 flex items-center justify-center rounded"
            style={{ color: '#A89B93' }}>
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
        {/* Status pill */}
        <div className="mt-2">
          <span className="status-pill" style={{ background: statusStyle.bg, color: statusStyle.text }}>
            {STATUS_LABEL[liveStatus]}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <Block title="Identity">
          <Row label="kind">{data.kind}</Row>
          <Row label="node id">{node.id}</Row>
          {data.instance_id && (
            <Row label="instance">{data.instance_id} of {data.instance_count}</Row>
          )}
        </Block>

        {data.metrics && Object.keys(data.metrics).length > 0 && (
          <Block title="Metrics">
            {data.metrics.camera_count !== undefined &&
              <Row label="cameras">{data.metrics.camera_count} ×</Row>}
            {data.metrics.duration_ms !== undefined &&
              <Row label="duration">{data.metrics.duration_ms} ms</Row>}
            {data.metrics.attempt !== undefined &&
              <Row label="attempt">#{data.metrics.attempt}</Row>}
            {data.metrics.schema_version &&
              <Row label="schema">{data.metrics.schema_version}</Row>}
            {data.metrics.sha256 &&
              <Row label="sha256">{data.metrics.sha256.slice(0, 16)}…</Row>}
          </Block>
        )}

        <Block title="Architecture">
          <Row label="section">V1.1 §{docs.section}</Row>
          <Row label="source">
            <span style={{ color: '#4A7FA8' }}>{docs.file}</span>
          </Row>
        </Block>

        <Block title="Evidence">
          <Row label="class">LOCAL_CONFORMANCE</Row>
          <Row label="production">
            <span style={{ color: '#C96442' }}>false</span>
          </Row>
        </Block>
      </div>
    </div>
  )
}
