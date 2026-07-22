// ── Core domain types mirroring the Python architecture ──────────────────────

export type NodeStatus =
  | 'PENDING'
  | 'RUNNING'
  | 'COMPLETE'
  | 'FAILED'
  | 'WAITING_REVIEW'
  | 'BLOCKED'
  | 'NO_EVENTS'

export type EdgeSchema =
  | 'SOURCE'
  | 'QA'
  | 'PROPOSAL'
  | 'CANDIDATE'
  | 'EVIDENCE'
  | 'FUSION'
  | 'COMPLETION'
  | 'REVIEW'

export type NodeKind =
  | 'source'
  | 'media_quality'
  | 'adaptive_sampler'
  | 'qa_coarse'
  | 'qa_dense'
  | 'qa_gate'
  | 'event_proposal'
  | 'candidate_reducer'
  | 'action_evidence'
  | 'provisional_fusion'
  | 'boundary_refinement'
  | 'final_fusion'
  | 'primary_completion'
  | 'outbox_relay'
  | 'review_queue'
  | 'work_scheduler'

export interface StageMetrics {
  duration_ms?: number
  attempt?: number
  camera_count?: number
  sha256?: string
  schema_version?: string
}

export interface RobataNodeData {
  kind: NodeKind
  label: string
  status: NodeStatus
  metrics?: StageMetrics
  detail?: Record<string, unknown>
  // For multi-instance nodes (per candidate, per action)
  instance_id?: string
  instance_count?: number
}

export interface RobataRun {
  run_id: string
  recording_id: string
  status: NodeStatus
  started_at: string
  evidence_class: 'LOCAL_CONFORMANCE' | 'MEASURED' | 'SYNTHETIC_LOCAL'
  production_eligible: boolean
  node_statuses: Record<string, NodeStatus>
}

export interface ReviewTask {
  task_id: string
  subject_type: string
  subject_id: string
  priority: number
  sla_deadline_ns: number
  status: 'PENDING' | 'LEASED' | 'ANNOTATED' | 'OVERDUE'
  trigger: string
}

// Edge color mapping
export const EDGE_COLORS: Record<EdgeSchema, string> = {
  SOURCE:     '#06b6d4',
  QA:         '#6366f1',
  PROPOSAL:   '#f59e0b',
  CANDIDATE:  '#10b981',
  EVIDENCE:   '#f97316',
  FUSION:     '#ec4899',
  COMPLETION: '#8b5cf6',
  REVIEW:     '#14b8a6',
}

// Node status color mapping
export const STATUS_COLORS: Record<NodeStatus, string> = {
  PENDING:        '#374151',
  RUNNING:        '#1d4ed8',
  COMPLETE:       '#15803d',
  FAILED:         '#b91c1c',
  WAITING_REVIEW: '#a16207',
  BLOCKED:        '#6b21a8',
  NO_EVENTS:      '#475569',
}

export const STATUS_RING: Record<NodeStatus, string> = {
  PENDING:        'border-gray-600',
  RUNNING:        'border-blue-500',
  COMPLETE:       'border-green-600',
  FAILED:         'border-red-600',
  WAITING_REVIEW: 'border-yellow-600',
  BLOCKED:        'border-purple-700',
  NO_EVENTS:      'border-slate-500',
}
