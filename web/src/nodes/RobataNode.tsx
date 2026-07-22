import { memo, useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
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

function RobataNode({ id, data }: NodeProps<RobataNodeData>) {
  const activeRun      = usePipelineStore((s) => s.activeRun)
  const focusedNodeId  = usePipelineStore((s) => s.focusedNodeId)
  const setFocusedNodeId = usePipelineStore((s) => s.setFocusedNodeId)

  const liveStatus = activeRun?.node_statuses?.[id] ?? data.status
  const style      = STATUS_STYLE[liveStatus]
  const isFocused  = focusedNodeId === id

  const onClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    setFocusedNodeId(id)
  }, [id, setFocusedNodeId])

  return (
    <div
      onClick={onClick}
      className="cursor-pointer select-none"
      style={{
        width: 220,
        background: '#FDFAF5',
        border: `1.5px solid ${isFocused ? '#C96442' : style.nodeBorder}`,
        borderRadius: 10,
        boxShadow: isFocused
          ? '0 4px 20px rgba(201,100,66,0.18), 0 0 0 3px rgba(201,100,66,0.12)'
          : '0 1px 6px rgba(26,23,20,0.07)',
        transition: 'all 0.15s ease',
      }}
      onMouseEnter={(e) => {
        if (!isFocused) e.currentTarget.style.boxShadow = '0 4px 14px rgba(26,23,20,0.10)'
      }}
      onMouseLeave={(e) => {
        if (!isFocused) e.currentTarget.style.boxShadow = '0 1px 6px rgba(26,23,20,0.07)'
      }}
    >
      <Handle type="target" position={Position.Left}
        style={{ background: '#F8F3E8', borderColor: '#C4B59E', width: 9, height: 9 }} />

      {/* Header */}
      <div style={{
        background: style.header,
        borderBottom: '1px solid rgba(26,23,20,0.07)',
        borderRadius: '8px 8px 0 0',
        padding: '10px 14px 9px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <span style={{
            fontSize: 10,
            fontFamily: 'JetBrains Mono, monospace',
            fontWeight: 600,
            padding: '1px 6px',
            borderRadius: 4,
            background: 'rgba(26,23,20,0.07)',
            color: '#6B5E55',
            flexShrink: 0,
          }}>
            {KIND_ABBR[data.kind]}
          </span>
          <span style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: 13,
            fontWeight: 600,
            color: '#1A1714',
            lineHeight: 1.2,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
          }}>
            {data.label}
          </span>
        </div>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: style.dot, flexShrink: 0,
          boxShadow: liveStatus === 'RUNNING' ? `0 0 0 3px ${style.dot}33` : 'none',
        }} />
      </div>

      {/* Body */}
      <div style={{ padding: '10px 14px 12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: 10, color: '#A89B93', fontFamily: 'Inter', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            status
          </span>
          <span className="status-pill" style={{ background: style.bg, color: style.text, fontSize: 10 }}>
            {STATUS_LABEL[liveStatus]}
          </span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <span style={{ fontSize: 10, color: '#A89B93', fontFamily: 'Inter', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
            group
          </span>
          <span style={{ fontSize: 11, color: '#6B5E55', fontFamily: 'Inter' }}>
            {KIND_GROUP[data.kind]}
          </span>
        </div>

        {data.instance_id && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <span style={{ fontSize: 10, color: '#A89B93', fontFamily: 'Inter', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              instance
            </span>
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#6B5E55' }}>
              {data.instance_id}
            </span>
          </div>
        )}

        {data.metrics?.camera_count !== undefined && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <span style={{ fontSize: 10, color: '#A89B93', fontFamily: 'Inter', textTransform: 'uppercase', letterSpacing: '0.07em' }}>cameras</span>
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#4A7FA8' }}>{data.metrics.camera_count}×</span>
          </div>
        )}
        {data.metrics?.attempt !== undefined && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
            <span style={{ fontSize: 10, color: '#A89B93', fontFamily: 'Inter', textTransform: 'uppercase', letterSpacing: '0.07em' }}>attempt</span>
            <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#A87A2A' }}>#{data.metrics.attempt}</span>
          </div>
        )}
        {data.metrics?.schema_version && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 10, color: '#A89B93', fontFamily: 'Inter', textTransform: 'uppercase', letterSpacing: '0.07em', flexShrink: 0 }}>schema</span>
            <span style={{ fontSize: 10, fontFamily: 'JetBrains Mono', color: '#6A4AA8', textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {data.metrics.schema_version}
            </span>
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Right}
        style={{ background: '#F8F3E8', borderColor: '#C4B59E', width: 9, height: 9 }} />
    </div>
  )
}

export default memo(RobataNode)
