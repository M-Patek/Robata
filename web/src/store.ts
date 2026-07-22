import { create } from 'zustand'
import { RobataRun, NodeStatus, ReviewTask } from '@/types'

interface PipelineStore {
  // Active run
  activeRun: RobataRun | null
  setActiveRun: (run: RobataRun | null) => void
  updateNodeStatus: (nodeId: string, status: NodeStatus) => void

  // Navigation — which group is expanded (null = overview)
  expandedGroup: string | null
  setExpandedGroup: (id: string | null) => void

  // Focused node — shows detail drawer
  focusedNodeId: string | null
  setFocusedNodeId: (id: string | null) => void

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
  setExpandedGroup: (id) => set({ expandedGroup: id, focusedNodeId: null }),

  focusedNodeId: null,
  setFocusedNodeId: (id) =>
    set((state) => ({ focusedNodeId: state.focusedNodeId === id ? null : id })),

  reviewTasks: [],
  setReviewTasks: (tasks) => set({ reviewTasks: tasks }),

  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),
}))
