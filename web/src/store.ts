import { create } from 'zustand'
import { RobataRun, NodeStatus, ReviewTask } from '@/types'

interface PipelineStore {
  // Active run
  activeRun: RobataRun | null
  setActiveRun: (run: RobataRun | null) => void
  updateNodeStatus: (nodeId: string, status: NodeStatus) => void

  // Selected node for inspector panel
  selectedNodeId: string | null
  setSelectedNodeId: (id: string | null) => void

  // Review queue
  reviewTasks: ReviewTask[]
  setReviewTasks: (tasks: ReviewTask[]) => void

  // WebSocket connection status
  wsConnected: boolean
  setWsConnected: (v: boolean) => void

  // UI state
  showSixCameraPanel: boolean
  toggleSixCameraPanel: () => void
  showInspector: boolean
  toggleInspector: () => void
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

  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  reviewTasks: [],
  setReviewTasks: (tasks) => set({ reviewTasks: tasks }),

  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),

  showSixCameraPanel: false,
  toggleSixCameraPanel: () =>
    set((state) => ({ showSixCameraPanel: !state.showSixCameraPanel })),

  showInspector: true,
  toggleInspector: () =>
    set((state) => ({ showInspector: !state.showInspector })),
}))
