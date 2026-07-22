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

export const EDGE_COLORS: Record<EdgeSchema, string> = {
  SOURCE:     '#4A7FA8',
  QA:         '#6B5EA8',
  PROPOSAL:   '#A87A2A',
  CANDIDATE:  '#4A7A5A',
  EVIDENCE:   '#A85A3A',
  FUSION:     '#A84A7A',
  COMPLETION: '#6A4AA8',
  REVIEW:     '#3A8A88',
}

export const STATUS_LABEL: Record<NodeStatus, string> = {
  PENDING:        'Pending',
  RUNNING:        'Running',
  COMPLETE:       'Complete',
  FAILED:         'Failed',
  WAITING_REVIEW: 'Review',
  BLOCKED:        'Blocked',
  NO_EVENTS:      'No Events',
}

// Pill bg / text combinations (light theme)
export const STATUS_STYLE: Record<NodeStatus, { bg: string; text: string; dot: string; header: string; nodeBorder: string }> = {
  PENDING:        { bg: '#EDE4D3', text: '#6B5E55', dot: '#A89B93',  header: '#EDE4D3', nodeBorder: 'rgba(26,23,20,0.10)' },
  RUNNING:        { bg: '#DDEAF5', text: '#2E5F82', dot: '#4A7FA8',  header: '#E8F1F9', nodeBorder: '#4A7FA8' },
  COMPLETE:       { bg: '#D6EAD9', text: '#2E5E38', dot: '#4A7A5A',  header: '#E2EFE4', nodeBorder: '#4A7A5A' },
  FAILED:         { bg: '#F5DADA', text: '#7A2A2A', dot: '#C96442',  header: '#F5E2E2', nodeBorder: '#C96442' },
  WAITING_REVIEW: { bg: '#F5ECD8', text: '#7A5A1A', dot: '#A87A2A',  header: '#F5EFE0', nodeBorder: '#A87A2A' },
  BLOCKED:        { bg: '#EBE0F5', text: '#5A2A8A', dot: '#7A4AA8',  header: '#EDE8F5', nodeBorder: '#7A4AA8' },
  NO_EVENTS:      { bg: '#EDE4D3', text: '#8A7D74', dot: '#A89B93',  header: '#EDE4D3', nodeBorder: 'rgba(26,23,20,0.12)' },
}
