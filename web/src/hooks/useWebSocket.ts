import { useEffect, useRef, useCallback } from 'react'
import { usePipelineStore } from '@/store'
import { NodeStatus, RobataRun, ReviewTask } from '@/types'

interface WsMessage {
  type: 'run_update' | 'node_status' | 'review_tasks' | 'ping'
  run?: RobataRun
  node_id?: string
  status?: NodeStatus
  tasks?: ReviewTask[]
}

// Simulates live status progression for demo purposes (no backend needed)
function simulateRun(
  onNodeStatus: (nodeId: string, status: NodeStatus) => void,
  onRunUpdate: (run: RobataRun) => void,
) {
  const demoRun: RobataRun = {
    run_id: 'run-demo-' + Date.now().toString(36),
    recording_id: 'rec-fixture-001',
    status: 'RUNNING',
    started_at: new Date().toISOString(),
    evidence_class: 'LOCAL_CONFORMANCE',
    production_eligible: false,
    node_statuses: {},
  }
  onRunUpdate(demoRun)

  // Sequential stage progression
  const stages: [string, NodeStatus, number][] = [
    ['source',               'RUNNING',  400],
    ['source',               'COMPLETE', 600],
    ['media_quality',        'RUNNING',  200],
    ['adaptive_sampler',     'RUNNING',  200],
    ['media_quality',        'COMPLETE', 700],
    ['adaptive_sampler',     'COMPLETE', 300],
    ['qa_coarse',            'RUNNING',  400],
    ['qa_coarse',            'COMPLETE', 900],
    ['qa_dense',             'RUNNING',  300],
    ['qa_gate',              'RUNNING',  100],
    ['qa_dense',             'COMPLETE', 1000],
    ['qa_gate',              'COMPLETE', 200],
    ['event_proposal',       'RUNNING',  400],
    ['event_proposal',       'COMPLETE', 800],
    ['candidate_reducer',    'RUNNING',  200],
    ['candidate_reducer',    'COMPLETE', 500],
    ['action_evidence_0',    'RUNNING',  300],
    ['action_evidence_1',    'RUNNING',  100],
    ['action_evidence_0',    'COMPLETE', 900],
    ['action_evidence_1',    'COMPLETE', 700],
    ['provisional_fusion',   'RUNNING',  300],
    ['provisional_fusion',   'COMPLETE', 600],
    ['boundary_onset_0',     'RUNNING',  200],
    ['boundary_offset_0',    'RUNNING',  150],
    ['boundary_onset_1',     'RUNNING',  250],
    ['boundary_offset_1',    'RUNNING',  100],
    ['boundary_onset_0',     'COMPLETE', 900],
    ['boundary_offset_0',    'COMPLETE', 700],
    ['boundary_onset_1',     'COMPLETE', 800],
    ['boundary_offset_1',    'COMPLETE', 600],
    ['final_fusion',         'RUNNING',  300],
    ['final_fusion',         'COMPLETE', 700],
    ['primary_completion',   'RUNNING',  400],
    ['primary_completion',   'COMPLETE', 600],
    ['outbox_relay',         'RUNNING',  200],
    ['review_queue',         'RUNNING',  100],
    ['outbox_relay',         'COMPLETE', 800],
    ['review_queue',         'WAITING_REVIEW', 400],
  ]

  let cumulative = 500
  for (const [nodeId, status, delay] of stages) {
    cumulative += delay
    setTimeout(() => onNodeStatus(nodeId, status), cumulative)
  }
}

export function useWebSocket() {
  const setWsConnected  = usePipelineStore((s) => s.setWsConnected)
  const setActiveRun    = usePipelineStore((s) => s.setActiveRun)
  const updateNodeStatus = usePipelineStore((s) => s.updateNodeStatus)
  const setReviewTasks  = usePipelineStore((s) => s.setReviewTasks)

  const wsRef = useRef<WebSocket | null>(null)

  const connectWs = useCallback(() => {
    try {
      const ws = new WebSocket('ws://localhost:8000/ws/pipeline')
      wsRef.current = ws

      ws.onopen = () => setWsConnected(true)
      ws.onclose = () => {
        setWsConnected(false)
        // Retry after 3s
        setTimeout(connectWs, 3000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (ev) => {
        try {
          const msg: WsMessage = JSON.parse(ev.data)
          if (msg.type === 'run_update' && msg.run) setActiveRun(msg.run)
          if (msg.type === 'node_status' && msg.node_id && msg.status)
            updateNodeStatus(msg.node_id, msg.status)
          if (msg.type === 'review_tasks' && msg.tasks) setReviewTasks(msg.tasks)
        } catch {/* ignore malformed messages */}
      }
    } catch {
      // Backend not available — fall back to demo simulation
      setTimeout(startDemo, 1000)
    }
  }, [setWsConnected, setActiveRun, updateNodeStatus, setReviewTasks])

  const startDemo = useCallback(() => {
    simulateRun(
      (nodeId, status) => updateNodeStatus(nodeId, status),
      (run) => setActiveRun(run),
    )
  }, [updateNodeStatus, setActiveRun])

  useEffect(() => {
    // Try WS first; if it fails immediately, run demo
    let ws: WebSocket | null = null
    try {
      ws = new WebSocket('ws://localhost:8000/ws/pipeline')
      wsRef.current = ws
      ws.onopen = () => setWsConnected(true)
      ws.onerror = () => {
        ws?.close()
        startDemo()
      }
      ws.onclose = () => setWsConnected(false)
      ws.onmessage = (ev) => {
        try {
          const msg: WsMessage = JSON.parse(ev.data)
          if (msg.type === 'run_update' && msg.run) setActiveRun(msg.run)
          if (msg.type === 'node_status' && msg.node_id && msg.status)
            updateNodeStatus(msg.node_id, msg.status)
          if (msg.type === 'review_tasks' && msg.tasks) setReviewTasks(msg.tasks)
        } catch {/* ignore */}
      }
      // If still CONNECTING after 1.5s, assume backend not present
      setTimeout(() => {
        if (ws?.readyState === WebSocket.CONNECTING) {
          ws.close()
          startDemo()
        }
      }, 1500)
    } catch {
      startDemo()
    }

    return () => { ws?.close() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return { startDemo }
}
