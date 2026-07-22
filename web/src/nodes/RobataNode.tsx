import { memo, useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { clsx } from 'clsx'
import { RobataNodeData, STATUS_RING, NodeKind } from '@/types'
import { usePipelineStore } from '@/store'

// ── Icon map per node kind ────────────────────────────────────────────────────
const KIND_ICON: Record<NodeKind, string> = {
  source:               '📼',
  media_quality:        '🎞',
  adaptive_sampler:     '🔢',
  qa_coarse:            '🔍',
  qa_dense:             '🔬',
  qa_gate:              '🚦',
  event_proposal:       '💡',
  candidate_reducer:    '🔀',
  action_evidence:      '📸',
  provisional_fusion:   '🔗',
  boundary_refinement:  '📐',
  final_fusion:         '✅',
  primary_completion:   '💾',
  outbox_relay:         '📤',
  review_queue:         '👁',
  work_scheduler:       '🗓',
}

const STATUS_BG: Record<string, string> = {
  PENDING:        'bg-gray-700/60',
  RUNNING:        'bg-blue-900/60',
  COMPLETE:       'bg-green-900/60',
  FAILED:         'bg-red-900/60',
  WAITING_REVIEW: 'bg-yellow-900/60',
  BLOCKED:        'bg-purple-900/60',
  NO_EVENTS:      'bg-slate-700/60',
}

const STATUS_HEADER: Record<string, string> = {
  PENDING:        'bg-gray-600/40 text-gray-300',
  RUNNING:        'bg-blue-700/60 text-blue-100',
  COMPLETE:       'bg-green-800/60 text-green-100',
  FAILED:         'bg-red-800/60 text-red-100',
  WAITING_REVIEW: 'bg-yellow-800/60 text-yellow-100',
  BLOCKED:        'bg-purple-800/60 text-purple-100',
  NO_EVENTS:      'bg-slate-600/60 text-slate-200',
}

const STATUS_DOT: Record<string, string> = {
  PENDING:        'bg-gray-400',
  RUNNING:        'bg-blue-400 animate-pulse-fast',
  COMPLETE:       'bg-green-400',
  FAILED:         'bg-red-400',
  WAITING_REVIEW: 'bg-yellow-400 animate-pulse',
  BLOCKED:        'bg-purple-400',
  NO_EVENTS:      'bg-slate-400',
}

function RobataNode({ id, data, selected }: NodeProps<RobataNodeData>) {
  const setSelectedNodeId = usePipelineStore((s) => s.setSelectedNodeId)
  const activeRun = usePipelineStore((s) => s.activeRun)

  const liveStatus = activeRun?.node_statuses?.[id] ?? data.status

  const handleClick = useCallback(() => {
    setSelectedNodeId(id)
  }, [id, setSelectedNodeId])

  return (
    <div
      onClick={handleClick}
      className={clsx(
        'node-base min-w-[200px] max-w-[260px] cursor-pointer',
        STATUS_RING[liveStatus],
        STATUS_BG[liveStatus],
        selected && 'ring-2 ring-blue-400 ring-offset-1 ring-offset-canvas-bg',
      )}
    >
      {/* Handles */}
      <Handle type="target" position={Position.Top}
        style={{ background: '#6b7280', border: '2px solid #9ca3af' }} />

      {/* Header */}
      <div className={clsx('node-header rounded-t-md', STATUS_HEADER[liveStatus])}>
        <div className="flex items-center gap-2">
          <span className="text-base">{KIND_ICON[data.kind]}</span>
          <span className="text-[11px] font-semibold leading-tight">
            {data.label}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {data.instance_id && (
            <span className="text-[9px] px-1 py-0.5 rounded bg-black/30 font-mono">
              #{data.instance_id}
            </span>
          )}
          <span className={clsx('w-2 h-2 rounded-full flex-shrink-0', STATUS_DOT[liveStatus])} />
        </div>
      </div>

      {/* Body */}
      <div className="node-body text-gray-300">
        {/* Status */}
        <div className="flex items-center justify-between">
          <span className="text-gray-500 uppercase text-[9px] tracking-widest">status</span>
          <span className={clsx(
            'status-badge text-[9px]',
            liveStatus === 'RUNNING' && 'bg-blue-900 text-blue-200',
            liveStatus === 'COMPLETE' && 'bg-green-900 text-green-200',
            liveStatus === 'FAILED' && 'bg-red-900 text-red-200',
            liveStatus === 'PENDING' && 'bg-gray-700 text-gray-300',
            liveStatus === 'WAITING_REVIEW' && 'bg-yellow-900 text-yellow-200',
            liveStatus === 'BLOCKED' && 'bg-purple-900 text-purple-200',
            liveStatus === 'NO_EVENTS' && 'bg-slate-700 text-slate-300',
          )}>
            {liveStatus}
          </span>
        </div>

        {/* Metrics */}
        {data.metrics?.camera_count !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-500 text-[9px]">cameras</span>
            <span className="font-mono text-[10px] text-cyan-300">
              {data.metrics.camera_count} ×
            </span>
          </div>
        )}
        {data.metrics?.duration_ms !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-500 text-[9px]">duration</span>
            <span className="font-mono text-[10px] text-cyan-300">
              {data.metrics.duration_ms}ms
            </span>
          </div>
        )}
        {data.metrics?.attempt !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-500 text-[9px]">attempt</span>
            <span className="font-mono text-[10px] text-amber-300">
              #{data.metrics.attempt}
            </span>
          </div>
        )}
        {data.metrics?.schema_version && (
          <div className="flex justify-between gap-2 min-w-0">
            <span className="text-gray-500 text-[9px] flex-shrink-0">schema</span>
            <span className="font-mono text-[9px] text-violet-300 truncate text-right">
              {data.metrics.schema_version}
            </span>
          </div>
        )}
        {data.metrics?.sha256 && (
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 text-[9px]">sha256</span>
            <span className="font-mono text-[9px] text-slate-400 truncate">
              {data.metrics.sha256.slice(0, 10)}…
            </span>
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom}
        style={{ background: '#6b7280', border: '2px solid #9ca3af' }} />
    </div>
  )
}

export default memo(RobataNode)
