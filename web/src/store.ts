import { create } from 'zustand'
import { RobataRun, NodeStatus, ReviewTask } from '@/types'
import { Node } from 'reactflow'

interface PipelineStore {
  // Active run
  activeRun: RobataRun | null
  setActiveRun: (run: RobataRun | null) => void
  updateNodeStatus: (nodeId: string, status: NodeStatus) => void

  // Navigation — which group is expanded (null = overview)
  expandedGroup: string | null
  // direction: 'forward' = going right, 'back' = going left, 'enter' = first entry
  expandTransition: 'forward' | 'back' | 'enter' | null
  setExpandedGroup: (id: string | null, dir?: 'forward' | 'back' | 'enter') => void

  // Focused node — shows detail drawer
  focusedNodeId: string | null
  setFocusedNodeId: (id: string | null) => void

  // Persisted node layouts keyed by group id
  groupLayouts: Record<string, { id: string; position: { x: number; y: number } }[]>
  saveGroupLayout: (groupId: string, nodes: Node[]) => void

  // Review queue
  reviewTasks: ReviewTask[]
  setReviewTasks: (tasks: ReviewTask[]) => void

  // WebSocket
  wsConnected: boolean
  setWsConnected: (v: boolean) => void
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  activeRun: null,
  setActiveRun: (run) => set({ activeRun: run }),
  updateNodeStatus: (nodeId, status) =>
    set((state) => {
      if (!state.activeRun) return state
      return {
        activeRun: {
          ...state.activeRun,
          node_statuses: { ...state.activeRun.node_statuses, [nodeId]: status },
        },
      }
    }),

  expandedGroup: null,
  expandTransition: null,
  setExpandedGroup: (id, dir = 'enter') =>
    set({ expandedGroup: id, expandTransition: id ? dir : null, focusedNodeId: null }),

  focusedNodeId: null,
  setFocusedNodeId: (id) =>
    set((state) => ({ focusedNodeId: state.focusedNodeId === id ? null : id })),

  groupLayouts: {},
  saveGroupLayout: (groupId, nodes) =>
    set((state) => ({
      groupLayouts: {
        ...state.groupLayouts,
        [groupId]: nodes
          .filter((n) => n.id !== '__gateway__')
          .map((n) => ({ id: n.id, position: { ...n.position } })),
      },
    })),

  reviewTasks: [],
  setReviewTasks: (tasks) => set({ reviewTasks: tasks }),

  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),
}))
