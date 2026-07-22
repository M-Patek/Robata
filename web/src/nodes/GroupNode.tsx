import { memo, useCallback } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { PipelineGroup, getGroupStatus } from '@/data/groups'
import { STATUS_STYLE, STATUS_LABEL } from '@/types'
import { usePipelineStore } from '@/store'

interface GroupNodeData {
  group: PipelineGroup
}

function GroupNode({ data }: NodeProps<GroupNodeData>) {
  const { group } = data
  const activeRun        = usePipelineStore((s) => s.activeRun)
  const setExpandedGroup = usePipelineStore((s) => s.setExpandedGroup)

  const status = getGroupStatus(group.id, activeRun?.node_statuses ?? {})
  const style  = STATUS_STYLE[status]

  const onClick = useCallback(
    () => setExpandedGroup(group.id, 'enter'),
    [group.id, setExpandedGroup],
  )

  return (
    <div
      onClick={onClick}
      className="cursor-pointer select-none"
      style={{
        width: 260,
        background: '#FDFAF5',
        border: `1.5px solid ${style.nodeBorder}`,
        borderRadius: 14,
        boxShadow: '0 2px 12px rgba(26,23,20,0.07), 0 1px 3px rgba(26,23,20,0.05)',
        transition: 'box-shadow 0.18s ease, border-color 0.18s ease, transform 0.15s ease',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 6px 24px rgba(26,23,20,0.11), 0 2px 6px rgba(26,23,20,0.07)'
        e.currentTarget.style.transform = 'translateY(-1px)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 2px 12px rgba(26,23,20,0.07), 0 1px 3px rgba(26,23,20,0.05)'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      <Handle type="target" position={Position.Left}
        style={{ background: '#F8F3E8', borderColor: '#C4B59E', width: 10, height: 10 }} />

      {/* Header */}
      <div style={{
        background: style.header,
        borderBottom: '1px solid rgba(26,23,20,0.07)',
        borderRadius: '12px 12px 0 0',
        padding: '14px 18px 12px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
          <span style={{
            fontFamily: 'Lora, serif', fontSize: 16, fontWeight: 600,
            color: '#1A1714', lineHeight: 1.2,
          }}>
            {group.label}
          </span>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            {status === 'RUNNING' && (
              <span style={{
                position: 'absolute', inset: 0, borderRadius: '50%',
                background: style.dot, opacity: 0.4,
                animation: 'ping 1s cubic-bezier(0,0,0.2,1) infinite',
              }} />
            )}
            <span style={{
              width: 10, height: 10, borderRadius: '50%',
              background: style.dot, display: 'inline-block', flexShrink: 0,
            }} />
          </div>
        </div>
        <span style={{
          fontSize: 11, color: '#A89B93',
          fontFamily: 'Inter, sans-serif', marginTop: 2, display: 'block',
        }}>
          §{group.section}
        </span>
      </div>

      {/* Body */}
      <div style={{ padding: '12px 18px 14px' }}>
        <p style={{
          fontSize: 13, color: '#6B5E55',
          fontFamily: 'Inter, sans-serif', lineHeight: 1.5, margin: '0 0 12px',
        }}>
          {group.description}
        </p>

        {/* Node pills */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          {group.nodeIds.map((nid) => {
            const nStatus = activeRun?.node_statuses?.[nid] ?? 'PENDING'
            const ns = STATUS_STYLE[nStatus]
            return (
              <span key={nid} style={{
                fontSize: 10, fontFamily: 'JetBrains Mono, monospace',
                padding: '2px 7px', borderRadius: 100,
                background: ns.bg, color: ns.text,
              }}>
                {nid.replace(/_\d+$/, '')}
              </span>
            )
          })}
        </div>

        {/* CTA */}
        <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span className="status-pill" style={{ background: style.bg, color: style.text }}>
            {STATUS_LABEL[status]}
          </span>
          <span style={{ fontSize: 11, color: '#C4B59E', fontFamily: 'Inter, sans-serif' }}>
            Expand →
          </span>
        </div>
      </div>

      <Handle type="source" position={Position.Right}
        style={{ background: '#F8F3E8', borderColor: '#C4B59E', width: 10, height: 10 }} />
    </div>
  )
}

export default memo(GroupNode)
