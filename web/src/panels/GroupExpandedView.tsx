import { useCallback, useMemo, useState } from 'react'
import ReactFlow, {
  Background, Controls, BackgroundVariant,
  Node, Edge, NodeChange, EdgeChange,
  applyNodeChanges, applyEdgeChanges,
} from 'reactflow'
import 'reactflow/dist/style.css'

import RobataNode from '@/nodes/RobataNode'
import SchemaEdge from '@/edges/SchemaEdge'
import NodeDetailDrawer from '@/panels/NodeDetailDrawer'
import { usePipelineStore } from '@/store'
import { getGroupById, getGroupStatus, PIPELINE_GROUPS, GROUP_INTERNAL_EDGES } from '@/data/groups'
import { INITIAL_NODES } from '@/data/pipeline'
import { STATUS_STYLE } from '@/types'

const NODE_TYPES = { robata: RobataNode }
const EDGE_TYPES = { schema: SchemaEdge }

// Gateway stub node (non-interactive — represents the previous group)
const GATEWAY_NODE: Node = {
  id: '__gateway__',
  type: 'default',
  position: { x: 0, y: 120 },
  data: { label: '' },
  draggable: false,
  selectable: false,
  style: { opacity: 0 },
}

function buildGroupNodes(groupId: string): Node[] {
  const group = getGroupById(groupId)
  if (!group) return []

  const nodes: Node[] = [GATEWAY_NODE]
  const count = group.nodeIds.length
  const colW  = 280
  const rowH  = 180

  group.nodeIds.forEach((nid, i) => {
    const sourceNode = INITIAL_NODES.find((n) => n.id === nid)
    if (!sourceNode) return
    // lay out in a 2-column grid starting at x=200
    const col = Math.floor(i / Math.ceil(count / 2))
    const row = i % Math.ceil(count / 2)
    nodes.push({
      ...sourceNode,
      position: { x: 200 + col * colW, y: row * rowH + 20 },
    })
  })
  return nodes
}

function buildGroupEdges(groupId: string): Edge[] {
  const defs = GROUP_INTERNAL_EDGES[groupId] ?? []
  return defs.map((d, i) => ({
    id: `ge-${groupId}-${i}`,
    source: d.source,
    target: d.target,
    type: 'schema',
    data: { schema: d.schema, label: d.label },
  }))
}

export default function GroupExpandedView() {
  const expandedGroup  = usePipelineStore((s) => s.expandedGroup)
  const setExpanded    = usePipelineStore((s) => s.setExpandedGroup)
  const activeRun      = usePipelineStore((s) => s.activeRun)
  const setFocused     = usePipelineStore((s) => s.setFocusedNodeId)

  const group = useMemo(() =>
    expandedGroup ? getGroupById(expandedGroup) : null,
    [expandedGroup],
  )

  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])

  // Rebuild nodes/edges whenever group changes
  useMemo(() => {
    if (!expandedGroup) return
    setNodes(buildGroupNodes(expandedGroup))
    setEdges(buildGroupEdges(expandedGroup))
  }, [expandedGroup])

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  )

  if (!group) return null

  const prevGroup  = group.prevGroupId ? getGroupById(group.prevGroupId) : null
  const nextGroup  = group.nextGroupId ? getGroupById(group.nextGroupId) : null
  const status     = getGroupStatus(group.id, activeRun?.node_statuses ?? {})
  const style      = STATUS_STYLE[status]

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 30,
      background: '#F8F3E8',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* ── Top breadcrumb bar ─────────────────────────────────────────── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '14px 28px',
        background: '#FDFAF5',
        borderBottom: '1px solid rgba(26,23,20,0.09)',
        flexShrink: 0,
        zIndex: 10,
      }}>
        {/* Back to overview */}
        <button
          onClick={() => setExpanded(null)}
          style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '6px 14px',
            borderRadius: 8,
            border: '1px solid rgba(26,23,20,0.12)',
            background: 'transparent',
            cursor: 'pointer',
            fontFamily: 'Inter, sans-serif',
            fontSize: 13,
            color: '#6B5E55',
            transition: 'background 0.12s',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = '#F8F3E8')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M9 2L4 7l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Overview
        </button>

        {/* Breadcrumb divider */}
        <span style={{ color: '#D9CCBA', fontFamily: 'Inter', fontSize: 16, userSelect: 'none' }}>/</span>

        {/* Group breadcrumb pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, overflowX: 'auto' }}>
          {PIPELINE_GROUPS.map((g) => {
            const isActive = g.id === group.id
            const gs = getGroupStatus(g.id, activeRun?.node_statuses ?? {})
            const gStyle = STATUS_STYLE[gs]
            return (
              <button
                key={g.id}
                onClick={() => setExpanded(g.id)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '5px 12px',
                  borderRadius: 20,
                  border: isActive
                    ? `1.5px solid ${gStyle.nodeBorder}`
                    : '1px solid transparent',
                  background: isActive ? gStyle.header : 'transparent',
                  cursor: 'pointer',
                  fontFamily: 'Inter, sans-serif',
                  fontSize: 13,
                  fontWeight: isActive ? 600 : 400,
                  color: isActive ? '#1A1714' : '#8A7D74',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.12s',
                }}
              >
                <span style={{
                  width: 7, height: 7, borderRadius: '50%',
                  background: gStyle.dot, flexShrink: 0,
                }} />
                {g.label}
              </button>
            )
          })}
        </div>

        <div style={{ flex: 1 }} />

        {/* Group status */}
        <span className="status-pill"
          style={{ background: style.bg, color: style.text, fontSize: 11 }}>
          {status}
        </span>
      </div>

      {/* ── Group header ───────────────────────────────────────────────── */}
      <div style={{
        padding: '18px 32px 16px',
        borderBottom: '1px solid rgba(26,23,20,0.07)',
        background: '#FDFAF5',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div>
          <h1 style={{
            fontFamily: 'Lora, serif',
            fontSize: 22,
            fontWeight: 600,
            color: '#1A1714',
            margin: 0,
            lineHeight: 1.2,
          }}>
            {group.label}
          </h1>
          <p style={{
            fontFamily: 'Inter, sans-serif',
            fontSize: 13,
            color: '#8A7D74',
            marginTop: 4,
          }}>
            {group.description} &nbsp;·&nbsp; §{group.section}
          </p>
        </div>

        {/* Prev / Next navigation */}
        <div style={{ display: 'flex', gap: 8 }}>
          {prevGroup && (
            <button
              onClick={() => setExpanded(prevGroup.id)}
              style={{
                padding: '7px 16px',
                borderRadius: 8,
                border: '1px solid rgba(26,23,20,0.12)',
                background: 'transparent',
                cursor: 'pointer',
                fontFamily: 'Inter, sans-serif',
                fontSize: 13,
                color: '#6B5E55',
              }}
            >
              ← {prevGroup.label}
            </button>
          )}
          {nextGroup && (
            <button
              onClick={() => setExpanded(nextGroup.id)}
              style={{
                padding: '7px 16px',
                borderRadius: 8,
                border: '1.5px solid rgba(201,100,66,0.35)',
                background: '#FBF0EB',
                cursor: 'pointer',
                fontFamily: 'Inter, sans-serif',
                fontSize: 13,
                fontWeight: 500,
                color: '#C96442',
              }}
            >
              {nextGroup.label} →
            </button>
          )}
        </div>
      </div>

      {/* ── Node canvas ────────────────────────────────────────────────── */}
      <div style={{ flex: 1, minHeight: 0 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          fitViewOptions={{ padding: 0.15, maxZoom: 1.0 }}
          minZoom={0.3}
          maxZoom={2}
          onPaneClick={() => setFocused(null)}
          defaultEdgeOptions={{ type: 'schema' }}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#C4B59E" />
          <Controls showInteractive={false} />

          {/* Left gateway: ghost of previous group */}
          {prevGroup && (
            <div style={{
              position: 'absolute',
              left: 16, top: '50%',
              transform: 'translateY(-50%)',
              zIndex: 5,
              pointerEvents: 'none',
            }}>
              <div style={{
                width: 140,
                padding: '12px 16px',
                borderRadius: 10,
                background: '#FDFAF5',
                border: '1px dashed rgba(26,23,20,0.18)',
                opacity: 0.65,
              }}>
                <p style={{ fontFamily: 'Lora, serif', fontSize: 13, color: '#8A7D74', margin: 0 }}>
                  {prevGroup.label}
                </p>
                <p style={{ fontFamily: 'Inter, sans-serif', fontSize: 10, color: '#C4B59E', marginTop: 3 }}>
                  §{prevGroup.section}
                </p>
              </div>
            </div>
          )}
        </ReactFlow>
      </div>

      {/* Node detail drawer */}
      <NodeDetailDrawer />
    </div>
  )
}
