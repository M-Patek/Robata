import { useCallback, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  NodeChange,
  EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'

import RobataNode from '@/nodes/RobataNode'
import SchemaEdge from '@/edges/SchemaEdge'
import NodeInspector from '@/panels/NodeInspector'
import SixCameraPanel from '@/panels/SixCameraPanel'
import RunSummaryPanel from '@/panels/RunSummaryPanel'
import { usePipelineStore } from '@/store'
import { useWebSocket } from '@/hooks/useWebSocket'
import { INITIAL_NODES, INITIAL_EDGES } from '@/data/pipeline'
import { EDGE_COLORS, EdgeSchema } from '@/types'
import { clsx } from 'clsx'

const NODE_TYPES = { robata: RobataNode }
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
  const [nodes, setNodes] = useState<Node[]>(INITIAL_NODES)
  const [edges, setEdges] = useState<Edge[]>(INITIAL_EDGES)

  const showSixCameraPanel    = usePipelineStore((s) => s.showSixCameraPanel)
  const toggleSixCameraPanel  = usePipelineStore((s) => s.toggleSixCameraPanel)
  const showInspector         = usePipelineStore((s) => s.showInspector)
  const toggleInspector       = usePipelineStore((s) => s.toggleInspector)
  const wsConnected           = usePipelineStore((s) => s.wsConnected)
  const activeRun             = usePipelineStore((s) => s.activeRun)

  const { startDemo } = useWebSocket()

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((nds) => applyNodeChanges(changes, nds)),
    [],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    [],
  )

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden"
      style={{ background: '#F8F3E8', color: '#1A1714' }}>

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-6 py-3 flex-shrink-0 z-20"
        style={{
          background: '#FDFAF5',
          borderBottom: '1px solid rgba(26,23,20,0.09)',
          boxShadow: '0 1px 3px rgba(26,23,20,0.05)',
        }}>
        <div className="flex items-center gap-4">
          <div>
            <span className="text-[15px] font-semibold tracking-tight"
              style={{ fontFamily: 'Lora, serif', color: '#1A1714' }}>
              Robata
            </span>
            <span className="ml-2 text-[11px] font-medium"
              style={{ color: '#A89B93' }}>
              Pipeline Workflow
            </span>
          </div>
          <div className="h-4 w-px" style={{ background: 'rgba(26,23,20,0.12)' }} />
          {activeRun ? (
            <span className="text-[10px] font-mono hidden sm:block truncate max-w-[220px]"
              style={{ color: '#A89B93' }}>
              {activeRun.run_id}
            </span>
          ) : (
            <span className="text-[10px]" style={{ color: '#C4B59E' }}>
              No active run
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Connection status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px]"
            style={{
              background: '#F8F3E8',
              border: '1px solid rgba(26,23,20,0.10)',
            }}>
            <span className={clsx(
              'w-1.5 h-1.5 rounded-full',
              wsConnected ? 'animate-pulse' : '',
            )} style={{ background: wsConnected ? '#4A7A5A' : '#C4B59E' }} />
            <span style={{ color: wsConnected ? '#4A7A5A' : '#A89B93' }}>
              {wsConnected ? 'Live' : 'Demo'}
            </span>
          </div>

          <button
            onClick={startDemo}
            className="px-3 py-1.5 rounded-md text-[11px] font-medium transition-colors"
            style={{
              background: '#1A1714',
              color: '#F8F3E8',
              fontFamily: 'Inter, sans-serif',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = '#3D3530')}
            onMouseLeave={(e) => (e.currentTarget.style.background = '#1A1714')}
          >
            Replay Demo
          </button>

          <button
            onClick={toggleSixCameraPanel}
            className="px-3 py-1.5 rounded-md text-[11px] font-medium transition-colors"
            style={{
              background: showSixCameraPanel ? '#DDEAF5' : '#F8F3E8',
              color: showSixCameraPanel ? '#2E5F82' : '#6B5E55',
              border: `1px solid ${showSixCameraPanel ? 'rgba(74,127,168,0.35)' : 'rgba(26,23,20,0.10)'}`,
            }}
          >
            6-Camera
          </button>

          <button
            onClick={toggleInspector}
            className="px-3 py-1.5 rounded-md text-[11px] font-medium transition-colors"
            style={{
              background: showInspector ? '#EBE0F5' : '#F8F3E8',
              color: showInspector ? '#5A2A8A' : '#6B5E55',
              border: `1px solid ${showInspector ? 'rgba(106,74,168,0.35)' : 'rgba(26,23,20,0.10)'}`,
            }}
          >
            Inspector
          </button>
        </div>
      </header>

      {/* ── Schema legend ────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-5 px-6 py-2 flex-shrink-0 overflow-x-auto"
        style={{
          background: '#FDFAF5',
          borderBottom: '1px solid rgba(26,23,20,0.07)',
        }}>
        <span className="label-muted flex-shrink-0">Schema types</span>
        {SCHEMA_LEGEND.map(({ schema, label }) => (
          <div key={schema} className="flex items-center gap-1.5 flex-shrink-0">
            <span className="w-5 h-px inline-block rounded-full"
              style={{ background: EDGE_COLORS[schema] }} />
            <span className="text-[10px]" style={{ color: EDGE_COLORS[schema] }}>
              {label}
            </span>
          </div>
        ))}
      </div>

      {/* ── Main layout ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0">
        {/* Canvas — horizontal flow */}
        <div className="flex-1 min-w-0 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            fitViewOptions={{ padding: 0.12, maxZoom: 0.85 }}
            minZoom={0.12}
            maxZoom={2}
            defaultEdgeOptions={{ type: 'schema', style: { strokeWidth: 1.75 } }}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="#C4B59E"
            />
            <Controls showInteractive={false} />
            <MiniMap
              nodeColor={(node) => {
                const s = node.data?.status ?? 'PENDING'
                const map: Record<string, string> = {
                  PENDING: '#C4B59E', RUNNING: '#4A7FA8', COMPLETE: '#4A7A5A',
                  FAILED: '#C96442', WAITING_REVIEW: '#A87A2A',
                  BLOCKED: '#7A4AA8', NO_EVENTS: '#A89B93',
                }
                return map[s] ?? '#C4B59E'
              }}
              maskColor="rgba(248,243,232,0.65)"
            />
          </ReactFlow>
        </div>

        {/* Right side panels */}
        <div className="flex flex-shrink-0"
          style={{ borderLeft: '1px solid rgba(26,23,20,0.09)' }}>
          {/* Run summary — always visible */}
          <div className="w-52 overflow-hidden panel-surface">
            <RunSummaryPanel />
          </div>

          {/* Inspector — toggleable */}
          {showInspector && (
            <div className="w-60 overflow-hidden panel-surface">
              <NodeInspector />
            </div>
          )}

          {/* Six-camera — toggleable */}
          {showSixCameraPanel && (
            <div className="w-80 overflow-hidden panel-surface">
              <SixCameraPanel />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
