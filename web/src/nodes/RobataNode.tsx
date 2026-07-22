import { memo, useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { clsx } from 'clsx'
import { RobataNodeData, STATUS_STYLE, STATUS_LABEL, NodeKind } from '@/types'
import { usePipelineStore } from '@/store'

const KIND_ABBR: Record<NodeKind, string> = {
  source:               'SRC',
  media_quality:        'MQ',
  adaptive_sampler:     'AS',
  qa_coarse:            'QAc',
  qa_dense:             'QAd',
  qa_gate:              'QA',
  event_proposal:       'EP',
  candidate_reducer:    'CR',
  action_evidence:      'AE',
  provisional_fusion:   'PF',
  boundary_refinement:  'BR',
  final_fusion:         'FF',
  primary_completion:   'PC',
  outbox_relay:         'OB',
  review_queue:         'RQ',
  work_scheduler:       'WS',
}

const KIND_GROUP: Record<NodeKind, string> = {
  source:               'Ingestion',
  media_quality:        'Ingestion',
  adaptive_sampler:     'Sampling',
  qa_coarse:            'Quality',
  qa_dense:             'Quality',
  qa_gate:              'Quality',
  event_proposal:       'Events',
  candidate_reducer:    'Events',
  action_evidence:      'Events',
  provisional_fusion:   'Fusion',
  boundary_refinement:  'Fusion',
  final_fusion:         'Fusion',
  primary_completion:   'Completion',
  outbox_relay:         'Delivery',
  review_queue:         'Delivery',
  work_scheduler:       'Delivery',
}

function RunningDot() {
  return (
    <span className="relative flex h-2 w-2 flex-shrink-0">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-60"
        style={{ background: '#4A7FA8' }} />
      <span className="relative inline-flex rounded-full h-2 w-2"
        style={{ background: '#4A7FA8' }} />
    </span>
  )
}

function RobataNode({ id, data, selected }: NodeProps<RobataNodeData>) {
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId)
  const activeRun = usePipelineStore((s) => s.activeRun)
  const liveStatus = activeRun?.node_statuses?.[id] ?? data.status
  const style = STATUS_STYLE[liveStatus]

  const handleClick = useCallback(() => setSelectedNodeId(id), [id, setSelectedNodeId])

  return (
    <div
      onClick={handleClick}
      className={clsx('node-base cursor-pointer', selected && 'node-selected')}
      style={{
        minWidth: 180,
        maxWidth: 220,
        borderColor: selected ? undefined : style.nodeBorder,
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: '#F8F3E8', borderColor: '#C4B59E' }}
      />

      {/* Header band */}
      <div
        className="px-3 py-2 rounded-t-lg flex items-center justify-between gap-2"
        style={{ background: style.header, borderBottom: '1px solid rgba(26,23,20,0.07)' }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded"
            style={{ background: 'rgba(26,23,20,0.07)', color: '#6B5E55' }}
          >
            {KIND_ABBR[data.kind]}
          </span>
          <span className="text-[11px] font-semibold text-ink-900 truncate leading-tight"
            style={{ fontFamily: 'Inter, sans-serif' }}>
            {data.label}
          </span>
        </div>
        {liveStatus === 'RUNNING'
          ? <RunningDot />
          : <span className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ background: style.dot }} />
        }
      </div>

      {/* Body */}
      <div className="px-3 py-2.5 space-y-1.5">
        {/* Status pill */}
        <div className="flex items-center justify-between">
          <span className="label-muted">status</span>
          <span
            className="status-pill"
            style={{ background: style.bg, color: style.text }}
          >
            {STATUS_LABEL[liveStatus]}
          </span>
        </div>

        {/* Group */}
        <div className="flex items-center justify-between">
          <span className="label-muted">group</span>
          <span className="text-[10px] text-ink-500">{KIND_GROUP[data.kind]}</span>
        </div>

        {/* Instance */}
        {data.instance_id && (
          <div className="flex items-center justify-between">
            <span className="label-muted">instance</span>
            <span className="text-[10px] font-mono text-ink-500">
              {data.instance_id}
            </span>
          </div>
        )}

        {/* Metrics */}
        {data.metrics?.camera_count !== undefined && (
          <div className="flex items-center justify-between">
            <span className="label-muted">cameras</span>
            <span className="text-[10px] font-mono" style={{ color: '#4A7FA8' }}>
              {data.metrics.camera_count}×
            </span>
          </div>
        )}
        {data.metrics?.attempt !== undefined && (
          <div className="flex items-center justify-between">
            <span className="label-muted">attempt</span>
            <span className="text-[10px] font-mono" style={{ color: '#A87A2A' }}>
              #{data.metrics.attempt}
            </span>
          </div>
        )}
        {data.metrics?.duration_ms !== undefined && (
          <div className="flex items-center justify-between">
            <span className="label-muted">duration</span>
            <span className="text-[10px] font-mono text-ink-500">
              {data.metrics.duration_ms}ms
            </span>
          </div>
        )}
        {data.metrics?.schema_version && (
          <div className="flex items-center justify-between gap-2">
            <span className="label-muted flex-shrink-0">schema</span>
            <span className="text-[9px] font-mono truncate text-right"
              style={{ color: '#6A4AA8' }}>
              {data.metrics.schema_version}
            </span>
          </div>
        )}
      </div>

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: '#F8F3E8', borderColor: '#C4B59E' }}
      />
    </div>
  )
}

export default memo(RobataNode)
