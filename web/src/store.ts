import { create } from 'zustand'
import type {
  StreamViewState,
  StreamEvent,
  ExpectedWindowPlan,
  BackpressureState,
} from '@/types'

interface PipelineStore {
  // ── Stream view state (replaces activeRun) ────────────────────────────────
  streamView: StreamViewState
  resetStreamView: () => void

  // ── Event ingestion ──────────────────────────────────────────────────────
  ingestEvent: (event: StreamEvent) => void
  ingestEvents: (events: StreamEvent[]) => void

  // ── Simulation control ───────────────────────────────────────────────────
  setSimulating: (v: boolean) => void
  setSimulationSpeed: (speed: number) => void

  // ── Focused subject (replaces focusedNodeId) ─────────────────────────────
  focusedSubjectKey: string | null
  focusedSubjectType: string | null
  setFocusedSubject: (key: string | null, subjectType?: string | null) => void

  // ── Plane visibility ─────────────────────────────────────────────────────
  activePlane: 'A' | 'B' | 'both'
  setActivePlane: (plane: 'A' | 'B' | 'both') => void

  // ── Timeline zoom ────────────────────────────────────────────────────────
  timelineZoom: number // 1.0 = full recording width
  setTimelineZoom: (zoom: number) => void
  timelinePanNs: bigint // pan offset in nanoseconds
  setTimelinePanNs: (panNs: bigint) => void

  // ── WebSocket (still mock for now) ────────────────────────────────────────
  wsConnected: boolean
  setWsConnected: (v: boolean) => void
}

const INITIAL_BACKPRESSURE: BackpressureState = {
  level: 'NORMAL',
  bpClass: 'B',
  oldest_required_age_ms: 0,
  queue_depth: 0,
}

const INITIAL_STATE: StreamViewState = {
  capture_scope: null,
  segments: new Map(),
  windows: new Map(),
  inferences: new Map(),
  plan: null,
  terminal_closures: new Map(),
  finalization: null,
  watermark_ns: 0n,
  backpressure: INITIAL_BACKPRESSURE,
  is_simulating: false,
  simulation_speed: 2.0, // 2x speed by default
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  streamView: INITIAL_STATE,

  resetStreamView: () =>
    set({ streamView: INITIAL_STATE, focusedSubjectKey: null, focusedSubjectType: null }),

  ingestEvent: (event) =>
    set((state) => {
      const view = state.streamView
      let newView: StreamViewState

      switch (event.type) {
        case 'CAPTURE_SCOPE':
          newView = { ...view, capture_scope: event.scope }
          break

        case 'SEGMENT':
          newView = {
            ...view,
            segments: new Map(view.segments).set(event.segment.segment_key, event.segment),
          }
          break

        case 'WINDOW':
          newView = {
            ...view,
            windows: new Map(view.windows).set(event.window.window_key, event.window),
          }
          break

        case 'INFERENCE':
          newView = {
            ...view,
            inferences: new Map(view.inferences).set(event.inference.inference_key, event.inference),
          }
          break

        case 'PLAN_APPEND': {
          const existingPlan = view.plan
          const newPlan: ExpectedWindowPlan = existingPlan
            ? {
                ...existingPlan,
                declarations: [...existingPlan.declarations, event.declaration],
              }
            : {
                plan_key: event.plan.plan_key,
                capture_scope_digest: event.plan.capture_scope_digest,
                declarations: [event.declaration],
                sealed_manifest: null,
              }
          newView = { ...view, plan: newPlan }
          break
        }

        case 'PLAN_SEAL': {
          const existingPlan = view.plan
          const newPlan: ExpectedWindowPlan = existingPlan
            ? { ...existingPlan, sealed_manifest: event.seal }
            : {
                plan_key: event.plan.plan_key,
                capture_scope_digest: event.plan.capture_scope_digest,
                declarations: event.seal.ordered_members,
                sealed_manifest: event.seal,
              }
          newView = { ...view, plan: newPlan }
          break
        }

        case 'TERMINAL_CLOSURE':
          newView = {
            ...view,
            terminal_closures: new Map(view.terminal_closures).set(
              event.closure.closure_key,
              event.closure,
            ),
          }
          break

        case 'FINALIZATION':
          newView = { ...view, finalization: event.finalization }
          break

        case 'WATERMARK':
          newView = { ...view, watermark_ns: event.watermark_ns }
          break

        case 'BACKPRESSURE':
          newView = {
            ...view,
            backpressure: {
              level: event.level,
              bpClass: event.bpClass,
              oldest_required_age_ms: event.oldest_required_age_ms,
              queue_depth: event.queue_depth,
            },
          }
          break

        default:
          newView = view
      }

      return { streamView: newView }
    }),

  ingestEvents: (events) =>
    set((state) => {
      let view = state.streamView
      for (const event of events) {
        // Apply each event sequentially using the same logic as ingestEvent
        switch (event.type) {
          case 'CAPTURE_SCOPE':
            view = { ...view, capture_scope: event.scope }
            break
          case 'SEGMENT':
            view = {
              ...view,
              segments: new Map(view.segments).set(event.segment.segment_key, event.segment),
            }
            break
          case 'WINDOW':
            view = {
              ...view,
              windows: new Map(view.windows).set(event.window.window_key, event.window),
            }
            break
          case 'INFERENCE':
            view = {
              ...view,
              inferences: new Map(view.inferences).set(event.inference.inference_key, event.inference),
            }
            break
          case 'PLAN_APPEND': {
            const existingPlan = view.plan
            const newPlan: ExpectedWindowPlan = existingPlan
              ? {
                  ...existingPlan,
                  declarations: [...existingPlan.declarations, event.declaration],
                }
              : {
                  plan_key: event.plan.plan_key,
                  capture_scope_digest: event.plan.capture_scope_digest,
                  declarations: [event.declaration],
                  sealed_manifest: null,
                }
            view = { ...view, plan: newPlan }
            break
          }
          case 'PLAN_SEAL': {
            const existingPlan = view.plan
            const newPlan: ExpectedWindowPlan = existingPlan
              ? { ...existingPlan, sealed_manifest: event.seal }
              : {
                  plan_key: event.plan.plan_key,
                  capture_scope_digest: event.plan.capture_scope_digest,
                  declarations: event.seal.ordered_members,
                  sealed_manifest: event.seal,
                }
            view = { ...view, plan: newPlan }
            break
          }
          case 'TERMINAL_CLOSURE':
            view = {
              ...view,
              terminal_closures: new Map(view.terminal_closures).set(
                event.closure.closure_key,
                event.closure,
              ),
            }
            break
          case 'FINALIZATION':
            view = { ...view, finalization: event.finalization }
            break
          case 'WATERMARK':
            view = { ...view, watermark_ns: event.watermark_ns }
            break
          case 'BACKPRESSURE':
            view = {
              ...view,
              backpressure: {
                level: event.level,
                bpClass: event.bpClass,
                oldest_required_age_ms: event.oldest_required_age_ms,
                queue_depth: event.queue_depth,
              },
            }
            break
        }
      }
      return { streamView: view }
    }),

  setSimulating: (v) =>
    set((state) => ({
      streamView: { ...state.streamView, is_simulating: v },
    })),

  setSimulationSpeed: (speed) =>
    set((state) => ({
      streamView: { ...state.streamView, simulation_speed: speed },
    })),

  focusedSubjectKey: null,
  focusedSubjectType: null,
  setFocusedSubject: (key, subjectType = null) =>
    set((state) => ({
      focusedSubjectKey: state.focusedSubjectKey === key ? null : key,
      focusedSubjectType: state.focusedSubjectKey === key ? null : subjectType,
    })),

  activePlane: 'both',
  setActivePlane: (plane) => set({ activePlane: plane }),

  timelineZoom: 1.0,
  setTimelineZoom: (zoom) => set({ timelineZoom: Math.max(0.1, Math.min(10.0, zoom)) }),

  timelinePanNs: 0n,
  setTimelinePanNs: (panNs) => set({ timelinePanNs: panNs }),

  wsConnected: false,
  setWsConnected: (v) => set({ wsConnected: v }),
}))
