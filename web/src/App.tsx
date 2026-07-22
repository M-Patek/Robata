import { useCallback, useState } from 'react'
import ReactFlow, {
  Background, Controls, MiniMap, BackgroundVariant,
  NodeChange, EdgeChange,
  applyNodeChanges, applyEdgeChanges,
  Node, Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'

import GroupNode from '@/nodes/GroupNode'
import SchemaEdge from '@/edges/SchemaEdge'
import GroupExpandedView from '@/panels/GroupExpandedView'
import { usePipelineStore } from '@/store'
import { useWebSocket } from '@/hooks/useWebSocket'
import { OVERVIEW_NODES, OVERVIEW_EDGES } from '@/data/overview'
import { EDGE_COLORS, EdgeSchema } from '@/types'

const NODE_TYPES = { group: GroupNode }
const EDGE_TYPES = { schema: SchemaEdge }

const SCHEMA_LEGEND: { schema: EdgeSchema; label: string }[] = [
  { schema: 'SOURCE',     label: 'Source' },
  { schema: 'QA',         label: 'QA' },
  { schema: 'PROPOSAL',   label: 'Proposal' },
  { schema: 'CANDIDATE',  label: 'Candidate' },
  { schema: 'EVIDENCE',   label: 'Evidence' },
  { schema: 'FUSION',     label: 'Fusion' },
  { schema: 'COMPLETION', label: 'Completion' },
  { schema: 'REVIEW',     label: 'Review' },
]

export default function App() {
  const [nodes, setNodes] = useState<Node[]>(OVERVIEW_NODES)
  const [edges, setEdges] = useState<Edge[]>(OVERVIEW_EDGES)

  const expandedGroup = usePipelineStore((s) => s.expandedGroup)
  const wsConnected   = usePipelineStore((s) => s.wsConnected)
  const activeRun     = usePipelineStore((s) => s.activeRun)

  const { startDemo } = useWebSocket()

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  )

  const isExpanded = expandedGroup !== null

  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      height: '100vh', width: '100vw',
      overflow: 'hidden', background: '#F8F3E8', color: '#1A1714',
    }}>
      {/* ── Persistent top bar (always visible, never re-mounts) ───────── */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 28px', height: 56,
        background: '#FDFAF5',
        borderBottom: '1px solid rgba(26,23,20,0.09)',
        boxShadow: '0 1px 3px rgba(26,23,20,0.04)',
        flexShrink: 0, zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{
            fontFamily: 'Lora, serif', fontSize: 18, fontWeight: 600,
            color: '#1A1714', letterSpacing: '-0.01em',
          }}>
            Robata
          </span>
          <div style={{ width: 1, height: 18, background: 'rgba(26,23,20,0.12)' }} />
          <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 13, color: '#A89B93' }}>
            Pipeline Workflow
          </span>
          {activeRun && (
            <>
              <div style={{ width: 1, height: 18, background: 'rgba(26,23,20,0.10)' }} />
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#C4B59E',
                maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {activeRun.run_id}
              </span>
            </>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '5px 13px', borderRadius: 20,
            border: '1px solid rgba(26,23,20,0.10)', background: '#F8F3E8',
            fontFamily: 'Inter, sans-serif', fontSize: 12,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: wsConnected ? '#4A7A5A' : '#C4B59E',
              display: 'inline-block',
            }} />
            <span style={{ color: wsConnected ? '#4A7A5A' : '#A89B93' }}>
              {wsConnected ? 'Live' : 'Demo mode'}
            </span>
          </div>
          <button
            onClick={startDemo}
            style={{
              padding: '7px 18px', borderRadius: 8, border: 'none',
              background: '#1A1714', color: '#F8F3E8',
              fontFamily: 'Inter, sans-serif', fontSize: 13, fontWeight: 500,
              cursor: 'pointer', transition: 'background 0.12s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#3D3530')}
            onMouseLeave={(e) => (e.currentTarget.style.background = '#1A1714')}
          >
            Replay Demo
          </button>
        </div>
      </header>

      {/* ── Main canvas area — two layers, crossfade via opacity ───────── */}
      <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>

        {/* ── Overview layer ─────────────────────────────────────────── */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          opacity: isExpanded ? 0 : 1,
          pointerEvents: isExpanded ? 'none' : 'all',
          transition: 'opacity 0.25s ease',
          zIndex: 1,
        }}>
          {/* Schema legend */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 24,
            padding: '0 28px', height: 38,
            background: '#FDFAF5', borderBottom: '1px solid rgba(26,23,20,0.07)',
            flexShrink: 0, overflowX: 'auto',
          }}>
            <span style={{
              fontSize: 10, fontWeight: 600, letterSpacing: '0.09em', textTransform: 'uppercase',
              color: '#C4B59E', fontFamily: 'Inter, sans-serif', flexShrink: 0,
            }}>
              Schema types
            </span>
            {SCHEMA_LEGEND.map(({ schema, label }) => (
              <div key={schema} style={{ display: 'flex', alignItems: 'center', gap: 7, flexShrink: 0 }}>
                <span style={{
                  display: 'inline-block', width: 22, height: 2, borderRadius: 2,
                  background: EDGE_COLORS[schema],
                }} />
                <span style={{ fontSize: 12, color: EDGE_COLORS[schema], fontFamily: 'Inter, sans-serif' }}>
                  {label}
                </span>
              </div>
            ))}
          </div>

          {/* Overview ReactFlow */}
          <div style={{ flex: 1, minHeight: 0 }}>
            <ReactFlow
              nodes={nodes} edges={edges}
              nodeTypes={NODE_TYPES} edgeTypes={EDGE_TYPES}
              onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
              fitView fitViewOptions={{ padding: 0.20, maxZoom: 0.9 }}
              minZoom={0.15} maxZoom={2}
              defaultEdgeOptions={{ type: 'schema' }}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={24} size={1.2} color="#C4B59E" />
              <Controls showInteractive={false} />
              <MiniMap nodeColor={() => '#C4B59E'} maskColor="rgba(248,243,232,0.7)" />
            </ReactFlow>

            {!activeRun && (
              <div style={{
                position: 'absolute', bottom: 28, left: '50%',
                transform: 'translateX(-50%)',
                pointerEvents: 'none',
                padding: '9px 20px', borderRadius: 20,
                background: 'rgba(253,250,245,0.92)',
                border: '1px solid rgba(26,23,20,0.10)',
                backdropFilter: 'blur(6px)',
              }}>
                <span style={{ fontFamily: 'Inter, sans-serif', fontSize: 13, color: '#8A7D74' }}>
                  Click a stage card to drill in &nbsp;·&nbsp; Press Replay Demo to animate
                </span>
              </div>
            )}
          </div>
        </div>

        {/* ── Expanded detail layer (always mounted, fades in/out) ────── */}
        <div style={{
          position: 'absolute', inset: 0,
          opacity: isExpanded ? 1 : 0,
          pointerEvents: isExpanded ? 'all' : 'none',
          transition: 'opacity 0.25s ease',
          zIndex: 2,
        }}>
          <GroupExpandedView />
        </div>

      </div>
    </div>
  )
}
