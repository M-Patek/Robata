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

  const showSixCameraPanel = usePipelineStore((s) => s.showSixCameraPanel)
  const toggleSixCameraPanel = usePipelineStore((s) => s.toggleSixCameraPanel)
  const showInspector = usePipelineStore((s) => s.showInspector)
  const toggleInspector = usePipelineStore((s) => s.toggleInspector)
  const wsConnected = usePipelineStore((s) => s.wsConnected)
  const activeRun = usePipelineStore((s) => s.activeRun)

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
    <div className="flex flex-col h-screen w-screen bg-canvas-bg text-white overflow-hidden">
      {/* ── Top bar ── */}
      <header className="flex items-center justify-between px-5 py-2.5 border-b border-canvas-border bg-canvas-panel flex-shrink-0 z-20">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold tracking-tight text-white">🤖 Robata</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-900/60 text-violet-300 font-mono border border-violet-700/40">
            Pipeline Workflow
          </span>
          {activeRun && (
            <span className="text-[10px] font-mono text-gray-500 hidden sm:block truncate max-w-[200px]">
              {activeRun.run_id}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* WS Status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-canvas-bg border border-canvas-border text-[10px]">
            <span className={clsx(
              'w-1.5 h-1.5 rounded-full',
              wsConnected ? 'bg-green-400 animate-pulse' : 'bg-gray-600',
            )} />
            <span className={wsConnected ? 'text-green-400' : 'text-gray-500'}>
              {wsConnected ? 'Live' : 'Demo'}
            </span>
          </div>

          <button
            onClick={startDemo}
            className="px-3 py-1.5 rounded-md bg-blue-700 hover:bg-blue-600 text-white text-[11px] font-semibold transition-colors"
          >
            ▶ Replay Demo
          </button>

          <button
            onClick={toggleSixCameraPanel}
            className={clsx(
              'px-3 py-1.5 rounded-md text-[11px] font-semibold transition-colors border',
              showSixCameraPanel
                ? 'bg-cyan-900/60 text-cyan-300 border-cyan-700/40'
                : 'bg-canvas-bg text-gray-400 border-canvas-border hover:text-white',
            )}
          >
            📷 6-Cam
          </button>

          <button
            onClick={toggleInspector}
            className={clsx(
              'px-3 py-1.5 rounded-md text-[11px] font-semibold transition-colors border',
              showInspector
                ? 'bg-violet-900/60 text-violet-300 border-violet-700/40'
                : 'bg-canvas-bg text-gray-400 border-canvas-border hover:text-white',
            )}
          >
            🔎 Inspector
          </button>
        </div>
      </header>

      {/* ── Schema legend ── */}
      <div className="flex items-center gap-3 px-5 py-1.5 border-b border-canvas-border bg-canvas-panel/70 flex-shrink-0 overflow-x-auto">
        <span className="text-[9px] text-gray-600 uppercase tracking-widest flex-shrink-0">Schema types</span>
        {SCHEMA_LEGEND.map(({ schema, label }) => (
          <div key={schema} className="flex items-center gap-1 flex-shrink-0">
            <span
              className="w-5 h-0.5 rounded-full inline-block"
              style={{ background: EDGE_COLORS[schema] }}
            />
            <span className="text-[10px]" style={{ color: EDGE_COLORS[schema] }}>{label}</span>
          </div>
        ))}
      </div>

      {/* ── Main layout ── */}
      <div className="flex flex-1 min-h-0">
        {/* Canvas */}
        <div className="flex-1 min-w-0 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            edgeTypes={EDGE_TYPES}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            fitViewOptions={{ padding: 0.15, maxZoom: 0.9 }}
            minZoom={0.15}
            maxZoom={2}
            defaultEdgeOptions={{
              type: 'schema',
              style: { strokeWidth: 2 },
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={24}
              size={1}
              color="#1e2130"
            />
            <Controls
              showInteractive={false}
              className="bg-canvas-panel border-canvas-border"
            />
            <MiniMap
              nodeColor={(node) => {
                const status = node.data?.status ?? 'PENDING'
                const colors: Record<string, string> = {
                  PENDING: '#374151', RUNNING: '#1d4ed8', COMPLETE: '#15803d',
                  FAILED: '#b91c1c', WAITING_REVIEW: '#a16207', BLOCKED: '#6b21a8',
                  NO_EVENTS: '#475569',
                }
                return colors[status] ?? '#374151'
              }}
              maskColor="rgba(15, 17, 23, 0.7)"
              style={{ background: '#16181f' }}
            />
          </ReactFlow>
        </div>

        {/* Right panels */}
        <div className="flex flex-shrink-0 border-l border-canvas-border">
          {/* Run summary — always visible */}
          <div className="w-52 border-r border-canvas-border overflow-hidden">
            <RunSummaryPanel />
          </div>

          {/* Inspector — toggleable */}
          {showInspector && (
            <div className="w-64 overflow-hidden">
              <NodeInspector />
            </div>
          )}

          {/* Six-camera — toggleable */}
          {showSixCameraPanel && (
            <div className="w-80 overflow-hidden">
              <SixCameraPanel />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
