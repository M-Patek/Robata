import { useCallback, useEffect, useRef } from 'react'
import ReactFlow, {
  Background, Controls, BackgroundVariant,
  Node, Edge, NodeChange, EdgeChange,
  applyNodeChanges, applyEdgeChanges,
  useReactFlow,
  ReactFlowProvider,
} from 'reactflow'
import 'reactflow/dist/style.css'

import RobataNode from '@/nodes/RobataNode'
import SchemaEdge from '@/edges/SchemaEdge'
import NodeDetailDrawer from '@/panels/NodeDetailDrawer'
import { usePipelineStore } from '@/store'
import {
  getGroupById, getGroupStatus, PIPELINE_GROUPS, GROUP_INTERNAL_EDGES,
} from '@/data/groups'
import { INITIAL_NODES } from '@/data/pipeline'
import { STATUS_STYLE, STATUS_LABEL } from '@/types'
import { useState } from 'react'

const NODE_TYPES = { robata: RobataNode }
const EDGE_TYPES = { schema: SchemaEdge }

// ── Invisible gateway anchor ──────────────────────────────────────────────────
const GATEWAY_NODE: Node = {
  id: '__gateway__',
  type: 'default',
  position: { x: 0, y: 120 },
  data: { label: '' },
  draggable: false, selectable: false, connectable: false,
  style: { opacity: 0, width: 1, height: 1, pointerEvents: 'none' },
}

function defaultPositions(groupId: string): Node[] {
  const group = getGroupById(groupId)
  if (!group) return [GATEWAY_NODE]
  const count = group.nodeIds.length
  const cols  = Math.ceil(count / 2)
  const colW  = 300
  const rowH  = 200
  const nodes: Node[] = [GATEWAY_NODE]
  group.nodeIds.forEach((nid, i) => {
    const src = INITIAL_NODES.find((n) => n.id === nid)
    if (!src) return
    nodes.push({
      ...src,
      position: { x: 180 + (i % cols) * colW, y: Math.floor(i / cols) * rowH + 30 },
    })
  })
  return nodes
}

function buildEdges(groupId: string): Edge[] {
  return (GROUP_INTERNAL_EDGES[groupId] ?? []).map((d, i) => ({
    id: `ge-${groupId}-${i}`,
    source: d.source, target: d.target,
    type: 'schema', data: { schema: d.schema, label: d.label },
  }))
}

function restoreNodes(
  groupId: string,
  saved: { id: string; position: { x: number; y: number } }[] | undefined,
): Node[] {
  const base = defaultPositions(groupId)
  if (!saved || !saved.length) return base
  return base.map((n) => {
    const s = saved.find((x) => x.id === n.id)
    return s ? { ...n, position: s.position } : n
  })
}

// ── Chain rail ────────────────────────────────────────────────────────────────
function ChainRail({ activeGroupId }: { activeGroupId: string }) {
  const activeRun     = usePipelineStore((s) => s.activeRun)
  const setExpanded   = usePipelineStore((s) => s.setExpandedGroup)
  const expandedGroup = usePipelineStore((s) => s.expandedGroup)

  const currentIdx = PIPELINE_GROUPS.findIndex((x) => x.id === expandedGroup)

  return (
    <div className="chain-rail">
      {PIPELINE_GROUPS.map((g, i) => {
        const status   = getGroupStatus(g.id, activeRun?.node_statuses ?? {})
        const st       = STATUS_STYLE[status]
        const isActive = g.id === activeGroupId

        return (
          <div key={g.id} style={{ display: 'flex', alignItems: 'center' }}>
            {i > 0 && (
              <div className="chain-arrow">
                <svg width="14" height="10" viewBox="0 0 14 10" fill="none">
                  <path d="M1 5h11M8 1l4 4-4 4" stroke="#D9CCBA" strokeWidth="1.5"
                    strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
            )}
            <button
              className={`chain-node${isActive ? ' active' : ''}`}
              style={{
                borderColor: isActive ? st.nodeBorder : 'transparent',
                background:  isActive ? st.header    : 'transparent',
                color:       isActive ? '#1A1714'    : '#8A7D74',
              }}
              onClick={() => {
                if (g.id !== expandedGroup) {
                  setExpanded(g.id, i > currentIdx ? 'forward' : 'back')
                }
              }}
            >
              <span
                style={{ width: 7, height: 7, borderRadius: '50%', background: st.dot,
                  flexShrink: 0, display: 'inline-block' }}
                className={status === 'RUNNING' ? 'dot-running' : ''}
              />
              {g.label}
            </button>
          </div>
        )
      })}
    </div>
  )
}

// ── Ghost edge card ───────────────────────────────────────────────────────────
function GhostCard({
  group, side, onClick,
}: {
  group: NonNullable<ReturnType<typeof getGroupById>>
  side: 'left' | 'right'
  onClick: () => void
}) {
  const activeRun = usePipelineStore((s) => s.activeRun)
  const status    = getGroupStatus(group.id, activeRun?.node_statuses ?? {})
  const st        = STATUS_STYLE[status]

  return (
    <div onClick={onClick} style={{
      position: 'absolute', top: '50%', [side]: 0,
      transform: 'translateY(-50%)', zIndex: 5, cursor: 'pointer',
    }}>
      <div
        style={{
          width: 120, padding: '12px 16px',
          borderRadius: side === 'left' ? '0 10px 10px 0' : '10px 0 0 10px',
          background: '#FDFAF5',
          border: `1px dashed ${st.nodeBorder}`,
          borderLeft:  side === 'right' ? undefined : 'none',
          borderRight: side === 'left'  ? undefined : 'none',
          opacity: 0.75, transition: 'opacity 0.15s',
          boxShadow: '0 2px 8px rgba(26,23,20,0.06)',
        }}
        onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.opacity = '1')}
        onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.opacity = '0.75')}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%',
            background: st.dot, flexShrink: 0, display: 'inline-block' }} />
          <span style={{ fontFamily: 'Lora, serif', fontSize: 12, fontWeight: 600, color: '#3D3530' }}>
            {group.label}
          </span>
        </div>
        <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, color: '#C4B59E', display: 'block' }}>
          {side === 'left' ? '← ' : ''}§{group.section}{side === 'right' ? ' →' : ''}
        </span>
      </div>
    </div>
  )
}

// ── Inner canvas — stable, never re-mounts ────────────────────────────────────
function StableCanvas() {
  const expandedGroup   = usePipelineStore((s) => s.expandedGroup)
  const groupLayouts    = usePipelineStore((s) => s.groupLayouts)
  const saveGroupLayout = usePipelineStore((s) => s.saveGroupLayout)
  const setFocused      = usePipelineStore((s) => s.setFocusedNodeId)

  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const loadedGroup = useRef<string | null>(null)
  const { fitView } = useReactFlow()

  // Load group data when group changes — never re-mounts the ReactFlow instance
  useEffect(() => {
    if (!expandedGroup) return
    if (expandedGroup === loadedGroup.current) return
    const isFirst = !groupLayouts[expandedGroup]
    loadedGroup.current = expandedGroup
    setNodes(restoreNodes(expandedGroup, groupLayouts[expandedGroup]))
    setEdges(buildEdges(expandedGroup))
    // fitView only when no saved layout, short delay for DOM
    if (isFirst) setTimeout(() => fitView({ padding: 0.18, maxZoom: 1.0, duration: 300 }), 50)
  }, [expandedGroup, groupLayouts, fitView])

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((nds) => {
        const updated = applyNodeChanges(changes, nds)
        if (expandedGroup && changes.some((c) => c.type === 'position' && c.dragging === false)) {
          saveGroupLayout(expandedGroup, updated)
        }
        return updated
      })
    },
    [expandedGroup, saveGroupLayout],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  )

  const group     = expandedGroup ? getGroupById(expandedGroup) : null
  const prevGroup = group?.prevGroupId ? getGroupById(group.prevGroupId) : null
  const nextGroup = group?.nextGroupId ? getGroupById(group.nextGroupId) : null
  const setExpanded = usePipelineStore((s) => s.setExpandedGroup)

  return (
    <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
      <ReactFlow
        nodes={nodes} edges={edges}
        nodeTypes={NODE_TYPES} edgeTypes={EDGE_TYPES}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
        fitView={false}
        minZoom={0.25} maxZoom={2}
        onPaneClick={() => setFocused(null)}
        defaultEdgeOptions={{ type: 'schema' }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1.1} color="#C4B59E" />
        <Controls showInteractive={false} />
      </ReactFlow>

      {prevGroup && (
        <GhostCard group={prevGroup} side="left"
          onClick={() => setExpanded(prevGroup.id, 'back')} />
      )}
      {nextGroup && (
        <GhostCard group={nextGroup} side="right"
          onClick={() => setExpanded(nextGroup.id, 'forward')} />
      )}
    </div>
  )
}

// ── Main exported component ───────────────────────────────────────────────────
export default function GroupExpandedView() {
  const expandedGroup = usePipelineStore((s) => s.expandedGroup)
  const setExpanded   = usePipelineStore((s) => s.setExpandedGroup)
  const activeRun     = usePipelineStore((s) => s.activeRun)

  const group     = expandedGroup ? getGroupById(expandedGroup) : null
  const prevGroup = group?.prevGroupId ? getGroupById(group.prevGroupId) : null
  const nextGroup = group?.nextGroupId ? getGroupById(group.nextGroupId) : null
  const status    = group ? getGroupStatus(group.id, activeRun?.node_statuses ?? {}) : 'PENDING'
  const style     = STATUS_STYLE[status]

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      background: '#F8F3E8',
    }}>
      {/* ── Sub-header (stable chrome, no re-mount) ──────────────────── */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 28px', height: 52,
        background: '#FDFAF5',
        borderBottom: '1px solid rgba(26,23,20,0.08)',
        flexShrink: 0,
      }}>
        {/* Left: back button (stable) + crossfading title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            onClick={() => setExpanded(null)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '6px 14px', borderRadius: 8,
              border: '1px solid rgba(26,23,20,0.12)',
              background: 'transparent', cursor: 'pointer',
              fontFamily: 'Inter, sans-serif', fontSize: 13, color: '#6B5E55',
              transition: 'background 0.12s',
              flexShrink: 0,
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#F0EBE1')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M9 2L4 7l5 5" stroke="currentColor" strokeWidth="1.6"
                strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Overview
          </button>

          <div style={{ width: 1, height: 18, background: 'rgba(26,23,20,0.12)', flexShrink: 0 }} />

          {/* Title zone: crossfades on group change via key */}
          <div
            key={expandedGroup ?? '__none__'}
            className="header-content-fade"
            style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}
          >
            <h2 style={{
              fontFamily: 'Lora, serif', fontSize: 17, fontWeight: 600,
              color: '#1A1714', margin: 0, lineHeight: 1,
            }}>
              {group?.label ?? ''}
            </h2>
            <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 11, color: '#A89B93' }}>
              §{group?.section ?? ''}
            </span>
          </div>
        </div>

        {/* Right: status pill + prev/next arrows — crossfade on group change */}
        <div
          key={`ctrl-${expandedGroup ?? '__none__'}`}
          className="header-content-fade"
          style={{ display: 'flex', alignItems: 'center', gap: 10 }}
        >
          <span className="status-pill"
            style={{ background: style.bg, color: style.text, fontSize: 11 }}>
            {STATUS_LABEL[status]}
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            {(['back', 'forward'] as const).map((dir) => {
              const target = dir === 'back' ? prevGroup : nextGroup
              const arrow  = dir === 'back'
                ? 'M8 1L3 6l5 5'
                : 'M4 1l5 5-5 5'
              return (
                <button key={dir}
                  disabled={!target}
                  onClick={() => target && setExpanded(target.id, dir)}
                  style={{
                    width: 32, height: 32, borderRadius: 8,
                    cursor: target ? 'pointer' : 'default',
                    border: '1px solid rgba(26,23,20,0.12)',
                    background: 'transparent',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: target ? '#6B5E55' : '#D9CCBA',
                    transition: 'background 0.12s',
                  }}
                  onMouseEnter={(e) => target && (e.currentTarget.style.background = '#F0EBE1')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d={arrow} stroke="currentColor" strokeWidth="1.5"
                      strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* ── Chain rail (pills slide via CSS transition, no re-mount) ──── */}
      {group && <ChainRail activeGroupId={group.id} />}

      {/* ── Stable canvas + ghost cards ──────────────────────────────── */}
      <ReactFlowProvider>
        <StableCanvas />
      </ReactFlowProvider>

      {/* ── Node detail drawer ──────────────────────────────────────── */}
      <NodeDetailDrawer />
    </div>
  )
}
