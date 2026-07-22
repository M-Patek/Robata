import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { INITIAL_NODES } from '@/data/pipeline'
import { RobataNodeData, NodeKind } from '@/types'
import { clsx } from 'clsx'

const KIND_DOCS: Record<NodeKind, { section: string; file: string }> = {
  source:              { section: '25.6', file: 'canonical/mcap_source.py' },
  media_quality:       { section: '25.3', file: 'canonical/media_quality.py' },
  adaptive_sampler:    { section: '25.3', file: 'sampling/adaptive.py' },
  qa_coarse:           { section: '25.10', file: 'qa_pipeline/coarse.py' },
  qa_dense:            { section: '25.10', file: 'qa_pipeline/dense.py' },
  qa_gate:             { section: '25.10', file: 'qa_pipeline/completion.py' },
  event_proposal:      { section: '25.10', file: 'event_pipeline/proposer.py' },
  candidate_reducer:   { section: '25.10', file: 'event_pipeline/candidate.py' },
  action_evidence:     { section: '25.10', file: 'event_pipeline/evidence.py' },
  provisional_fusion:  { section: '25.10', file: 'event_pipeline/provisional_fusion.py' },
  boundary_refinement: { section: '25.10', file: 'event_pipeline/boundary_refinement.py' },
  final_fusion:        { section: '25.10', file: 'canonical/runner.py' },
  primary_completion:  { section: '25.9',  file: 'application/canonical/primary_completion.py' },
  outbox_relay:        { section: '25.9',  file: 'adapters/sqlite_outbox.py' },
  review_queue:        { section: '25.8',  file: 'adapters/sqlite_review_queue.py' },
  work_scheduler:      { section: '25.2',  file: 'adapters/sqlite_work_scheduler.py' },
}

const VALUE_COLOR = 'text-amber-200'

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-start gap-3 py-1 border-b border-canvas-border/40 last:border-0">
      <span className={clsx('text-[10px] text-gray-500 flex-shrink-0 pt-0.5')}>{label}</span>
      <span className={clsx('text-[11px] font-mono text-right break-all', VALUE_COLOR)}>
        {value}
      </span>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <div className="text-[9px] font-semibold text-gray-500 uppercase tracking-widest mb-2 px-1">
        {title}
      </div>
      <div className="rounded-md bg-canvas-bg border border-canvas-border px-3 py-1">
        {children}
      </div>
    </div>
  )
}

export default function NodeInspector() {
  const selectedNodeId = usePipelineStore((s) => s.selectedNodeId)
  const activeRun      = usePipelineStore((s) => s.activeRun)
  const setSelected    = usePipelineStore((s) => s.setSelectedNodeId)

  const node = useMemo(() => {
    if (!selectedNodeId) return null
    return INITIAL_NODES.find((n) => n.id === selectedNodeId) ?? null
  }, [selectedNodeId])

  if (!node) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-600 text-sm gap-2 p-6 text-center">
        <span className="text-3xl">🔎</span>
        <span>Click any node to inspect</span>
      </div>
    )
  }

  const data = node.data as RobataNodeData
  const liveStatus = activeRun?.node_statuses?.[node.id] ?? data.status
  const docs = KIND_DOCS[data.kind]

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-canvas-border flex-shrink-0">
        <div>
          <h3 className="text-sm font-semibold text-white">{data.label}</h3>
          <p className="text-[10px] text-gray-500 mt-0.5 font-mono">{node.id}</p>
        </div>
        <button
          onClick={() => setSelected(null)}
          className="text-gray-500 hover:text-white text-lg leading-none px-1"
        >
          ✕
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-1">
        <Section title="Identity">
          <Field label="kind"       value={data.kind} />
          <Field label="node id"    value={node.id} />
          <Field label="status"     value={
            <span className={clsx(
              'px-1.5 py-0.5 rounded text-[9px] uppercase font-semibold',
              liveStatus === 'RUNNING'        && 'bg-blue-900 text-blue-200',
              liveStatus === 'COMPLETE'       && 'bg-green-900 text-green-200',
              liveStatus === 'FAILED'         && 'bg-red-900 text-red-200',
              liveStatus === 'PENDING'        && 'bg-gray-700 text-gray-300',
              liveStatus === 'WAITING_REVIEW' && 'bg-yellow-900 text-yellow-200',
              liveStatus === 'BLOCKED'        && 'bg-purple-900 text-purple-200',
              liveStatus === 'NO_EVENTS'      && 'bg-slate-700 text-slate-300',
            )}>
              {liveStatus}
            </span>
          } />
          {data.instance_id && (
            <Field label="instance"   value={`#${data.instance_id} of ${data.instance_count}`} />
          )}
        </Section>

        {data.metrics && Object.keys(data.metrics).length > 0 && (
          <Section title="Metrics">
            {data.metrics.camera_count !== undefined && (
              <Field label="cameras"     value={`${data.metrics.camera_count} ×`} />
            )}
            {data.metrics.duration_ms !== undefined && (
              <Field label="duration"    value={`${data.metrics.duration_ms} ms`} />
            )}
            {data.metrics.attempt !== undefined && (
              <Field label="attempt"     value={`#${data.metrics.attempt}`} />
            )}
            {data.metrics.schema_version && (
              <Field label="schema"      value={data.metrics.schema_version} />
            )}
            {data.metrics.sha256 && (
              <Field label="sha256"      value={data.metrics.sha256.slice(0, 16) + '…'} />
            )}
          </Section>
        )}

        <Section title="Architecture Reference">
          <Field label="section"   value={`V1.1 §${docs.section}`} />
          <Field label="source"    value={
            <span className={clsx('text-cyan-400 text-[10px]')}>{docs.file}</span>
          } />
        </Section>

        <Section title="Evidence Class">
          <Field label="class"              value="LOCAL_CONFORMANCE" />
          <Field label="production eligible" value={
            <span className="text-red-400 text-[10px]">false</span>
          } />
        </Section>
      </div>
    </div>
  )
}
