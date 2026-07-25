// Core domain types mirroring the Python streaming contracts.
//
// Contract sources are the registered schemas and `src/robata/contracts/stream_*.py`.
// This UI remains mock-driven until a supported backend API is connected.
export type OpaqueUuid = string
export type Sha256Digest = string

export interface NanosecondInterval {
  start_ns: bigint
  end_ns: bigint
}

// ── Stream subject types (mirrors StreamSubjectType) ────────────────────────

export type StreamSubjectType =
  | 'PRE_EOS_CAPTURE'
  | 'STREAM_SEGMENT'
  | 'INCREMENTAL_WINDOW'
  | 'STREAM_INFERENCE'
  | 'WINDOW_RESULT'
  | 'STREAM_WORK'
  | 'EXPECTED_WINDOW_PLAN'
  | 'EXPECTED_WINDOW_DECLARATION'
  | 'EXPECTED_WINDOW_PLAN_SEAL'
  | 'WINDOW_TERMINAL_CLOSURE'
  | 'RECORDING_FINALIZATION'

// ── Execution stages (mirrors StreamStage) ──────────────────────────────────

export type StreamStage =
  | 'SEGMENT'
  | 'WINDOW'
  | 'QA_COARSE'
  | 'QA_DENSE'
  | 'EVENT_PROPOSAL'
  | 'ACTION_DENSE'
  | 'BOUNDARY_REFINEMENT'
  | 'WINDOW_REDUCTION'
  | 'FINALIZATION'

// ── Terminal outcomes (mirrors TerminalOutcome) ─────────────────────────────

export type TerminalOutcome =
  | 'SUCCEEDED'
  | 'SKIPPED_POLICY'
  | 'SKIPPED_NOT_NEEDED'
  | 'FAILED'
  | 'CANCELLED'
  | 'EXPIRED'
  | 'QUARANTINED'
  | 'LATE_INPUT'
  | 'INCOMPLETE'
  | 'ABSTAINED'
  | 'NO_EVENTS'
  | 'INVALIDATED'

// ── Quality observation kinds ───────────────────────────────────────────────

export type QualityObservationKind =
  | 'LUMINANCE'
  | 'EDGE_ENERGY'
  | 'FREEZE'
  | 'CADENCE'
  | 'SEQUENCE_GAP'
  | 'CROSS_CAMERA_SKEW'

// ── Camera absence reasons ──────────────────────────────────────────────────

export type CameraAbsenceReason =
  | 'ABSENT'
  | 'LATE'
  | 'BLACK'
  | 'FROZEN'
  | 'DEGRADED'
  | 'CORRUPT'
  | 'UNAVAILABLE'
  | 'GAP'
  | 'UNKNOWN'

// ── Refinement role ─────────────────────────────────────────────────────────

export type RefinementRole = 'ONSET' | 'OFFSET'

// ── Stream purpose ──────────────────────────────────────────────────────────

export type StreamPurpose =
  | 'QA_COARSE'
  | 'QA_DENSE'
  | 'EVENT_PROPOSAL'
  | 'ACTION_DENSE'
  | 'BOUNDARY_REFINEMENT'

// ── Backpressure level ──────────────────────────────────────────────────────

export type BackpressureLevel = 'NORMAL' | 'ELEVATED' | 'CRITICAL'

// ── Backpressure class ─────────────────────────────────────────────────────

export type BackpressureClass = 'A' | 'B' | 'C' | 'D'

// ── Capture scope (immutable, pre-EOS) ──────────────────────────────────────

export interface ChannelBinding {
  camera_id: string
  source_channel_id: string
  source_channel_epoch: number
  channel_binding_semantic_sha256: Sha256Digest
}

export interface AuthorityBinding {
  authority_id: string
  authority_epoch: number
  policy_version: string
  initial_binding_semantic_sha256: Sha256Digest
}

export interface CaptureScope {
  capture_scope_id: OpaqueUuid
  capture_scope_key: string
  capture_scope_digest: Sha256Digest
  capture_authority_id: string
  capture_authority_epoch: number
  capture_assignment_policy_version: string
  acquisition_id: string
  acquisition_epoch: number
  channel_bindings: ChannelBinding[]
  mapping_authority: AuthorityBinding
  clock_authority: AuthorityBinding
}

// ── Quality observation ─────────────────────────────────────────────────────

export interface QualityObservation {
  kind: QualityObservationKind
  value: number
  interval: NanosecondInterval
}

// ── Stream segment (per-camera, bounded encoded ring closure) ──────────────

export interface StreamSegment {
  segment_id: OpaqueUuid
  segment_key: string
  segment_semantic_sha256: Sha256Digest
  camera_id: string
  requested_interval: NanosecondInterval
  effective_interval: NanosecondInterval
  content_digest: Sha256Digest
  mapping_semantic_sha256: Sha256Digest
  clock_alignment_semantic_sha256: Sha256Digest
  quality_observations: QualityObservation[]
}

// ── Segment reference or explicit absence ───────────────────────────────────

export interface StreamSegmentRef {
  segment_key: string
  segment_semantic_sha256: Sha256Digest
}

export interface CameraAbsence {
  reason: CameraAbsenceReason
  camera_id: string
}

export type SixSlotClosure = (StreamSegmentRef | CameraAbsence)[]

// ── Incremental window (immutable, with lineage) ────────────────────────────

export interface IncrementalWindow {
  window_id: OpaqueUuid
  window_key: string
  window_semantic_sha256: Sha256Digest
  capture_scope_digest: Sha256Digest
  purpose: StreamPurpose
  requested_interval: NanosecondInterval
  effective_interval: NanosecondInterval
  ordered_six_slot_closure: SixSlotClosure
  mapping_semantic_sha256: Sha256Digest
  clock_alignment_semantic_sha256: Sha256Digest
  parent_window_key: string | null
  refinement_role: RefinementRole | null
  refinement_generation: number
  window_policy_version: string
  semantic_projection_version: string
  identity_policy_version: string
}

// ── Artifact evidence reference ───────────────────────────────────────────────

export interface ArtifactEvidenceRef {
  artifact_id: string
  digest: Sha256Digest
  byte_count: number
  media_type: string
}

// ── Stream inference (bound to window + input plan) ──────────────────────────

export interface StreamInference {
  inference_id: OpaqueUuid
  inference_key: string
  window_key: string
  purpose: StreamPurpose
  input_plan_digest: Sha256Digest
  attempt_id: OpaqueUuid
  attempt_number: number
  terminal_outcome: TerminalOutcome
  evidence_ref: ArtifactEvidenceRef | null
  inference_policy_version: string
  semantic_projection_version: string
}

// ── Expected window plan (append-only, sealed at EOS) ──────────────────────────

export interface ExpectedWindowDeclaration {
  expected_ordinal: number
  window_key: string
  requested_interval: NanosecondInterval
  planning_policy_digest: Sha256Digest
}

export interface ExpectedWindowPlanSeal {
  sealed_at: string // ISO timestamp
  ordered_members: ExpectedWindowDeclaration[]
}

export interface ExpectedWindowPlan {
  plan_key: string
  capture_scope_digest: Sha256Digest
  declarations: ExpectedWindowDeclaration[]
  sealed_manifest: ExpectedWindowPlanSeal | null
}

// ── Window terminal closure (reconciled per expected member) ──────────────────

export interface WindowTerminalMember {
  expected_ordinal: number
  window_key: string
  window_semantic_sha256: Sha256Digest
  terminal_outcome: TerminalOutcome
  terminal_work_item_id: OpaqueUuid
  terminal_work_logical_key: string
  terminal_evidence_ref: ArtifactEvidenceRef
}

export interface WindowTerminalClosure {
  closure_key: string
  plan_key: string
  members: WindowTerminalMember[]
}

// ── Recording finalization (EOS mapping) ────────────────────────────────────

export interface IncrementalToFinalMapping {
  incremental_window_key: string
  final_window_identity: Sha256Digest
  terminal_outcome: TerminalOutcome
}

export interface RecordingFinalizationMap {
  finalization_key: string
  capture_scope_digest: Sha256Digest
  source_digest: Sha256Digest
  duration_ns: bigint
  incremental_to_final_mappings: IncrementalToFinalMapping[]
  primary_completion_ref: ArtifactEvidenceRef
  finalized_at: string | null
}

// ── Unified stream event (what the frontend consumes) ────────────────────────

export type StreamEvent =
  | { type: 'CAPTURE_SCOPE'; scope: CaptureScope }
  | { type: 'SEGMENT'; segment: StreamSegment }
  | { type: 'WINDOW'; window: IncrementalWindow }
  | { type: 'INFERENCE'; inference: StreamInference }
  | { type: 'PLAN_APPEND'; plan: ExpectedWindowPlan; declaration: ExpectedWindowDeclaration }
  | { type: 'PLAN_SEAL'; plan: ExpectedWindowPlan; seal: ExpectedWindowPlanSeal }
  | { type: 'TERMINAL_CLOSURE'; closure: WindowTerminalClosure }
  | { type: 'FINALIZATION'; finalization: RecordingFinalizationMap }
  | { type: 'WATERMARK'; watermark_ns: bigint; idle_cameras: string[] }
  | { type: 'BACKPRESSURE'; level: BackpressureLevel; bpClass: BackpressureClass; oldest_required_age_ms: number; queue_depth: number }

// ── Backpressure state ──────────────────────────────────────────────────────

export interface BackpressureState {
  level: BackpressureLevel
  bpClass: BackpressureClass
  oldest_required_age_ms: number
  queue_depth: number
}

// ── View state (replaces RobataRun) ─────────────────────────────────────────

export interface StreamViewState {
  capture_scope: CaptureScope | null
  segments: Map<string, StreamSegment>
  windows: Map<string, IncrementalWindow>
  inferences: Map<string, StreamInference>
  plan: ExpectedWindowPlan | null
  terminal_closures: Map<string, WindowTerminalClosure>
  finalization: RecordingFinalizationMap | null
  watermark_ns: bigint
  backpressure: BackpressureState
  is_simulating: boolean
  simulation_speed: number // 1.0 = real-time, 2.0 = 2x, etc.
}

// ── Legacy types (retained for migration compatibility) ──────────────────────

export type NodeStatus =
  | 'PENDING'
  | 'RUNNING'
  | 'COMPLETE'
  | 'FAILED'
  | 'WAITING_REVIEW'
  | 'BLOCKED'
  | 'NO_EVENTS'

// Map terminal outcomes to display status
export const TERMINAL_OUTCOME_STATUS: Record<TerminalOutcome, NodeStatus> = {
  SUCCEEDED: 'COMPLETE',
  SKIPPED_POLICY: 'COMPLETE',
  SKIPPED_NOT_NEEDED: 'COMPLETE',
  FAILED: 'FAILED',
  CANCELLED: 'FAILED',
  EXPIRED: 'FAILED',
  QUARANTINED: 'FAILED',
  LATE_INPUT: 'FAILED',
  INCOMPLETE: 'FAILED',
  ABSTAINED: 'NO_EVENTS',
  NO_EVENTS: 'NO_EVENTS',
  INVALIDATED: 'FAILED',
}

// ── UI styling constants (retained from existing design) ─────────────────────

export const STATUS_LABEL: Record<NodeStatus, string> = {
  PENDING: 'Pending',
  RUNNING: 'Running',
  COMPLETE: 'Complete',
  FAILED: 'Failed',
  WAITING_REVIEW: 'Review',
  BLOCKED: 'Blocked',
  NO_EVENTS: 'No Events',
}

export const STATUS_STYLE: Record<NodeStatus, { bg: string; text: string; dot: string; header: string; nodeBorder: string }> = {
  PENDING:        { bg: '#EDE4D3', text: '#6B5E55', dot: '#A89B93',  header: '#EDE4D3', nodeBorder: 'rgba(26,23,20,0.10)' },
  RUNNING:        { bg: '#DDEAF5', text: '#2E5F82', dot: '#4A7FA8',  header: '#E8F1F9', nodeBorder: '#4A7FA8' },
  COMPLETE:       { bg: '#D6EAD9', text: '#2E5E38', dot: '#4A7A5A',  header: '#E2EFE4', nodeBorder: '#4A7A5A' },
  FAILED:         { bg: '#F5DADA', text: '#7A2A2A', dot: '#C96442',  header: '#F5E2E2', nodeBorder: '#C96442' },
  WAITING_REVIEW: { bg: '#F5ECD8', text: '#7A5A1A', dot: '#A87A2A',  header: '#F5EFE0', nodeBorder: '#A87A2A' },
  BLOCKED:        { bg: '#EBE0F5', text: '#5A2A8A', dot: '#7A4AA8',  header: '#EDE8F5', nodeBorder: '#7A4AA8' },
  NO_EVENTS:      { bg: '#EDE4D3', text: '#8A7D74', dot: '#A89B93',  header: '#EDE4D3', nodeBorder: 'rgba(26,23,20,0.12)' },
}

// Terminal outcome display colors
export const OUTCOME_COLORS: Record<TerminalOutcome, string> = {
  SUCCEEDED: '#4A7A5A',
  SKIPPED_POLICY: '#6B5EA8',
  SKIPPED_NOT_NEEDED: '#6B5EA8',
  FAILED: '#C96442',
  CANCELLED: '#C96442',
  EXPIRED: '#C96442',
  QUARANTINED: '#C96442',
  LATE_INPUT: '#C96442',
  INCOMPLETE: '#C96442',
  ABSTAINED: '#A89B93',
  NO_EVENTS: '#A89B93',
  INVALIDATED: '#C96442',
}

// Purpose display labels
export const PURPOSE_LABEL: Record<StreamPurpose, string> = {
  QA_COARSE: 'QA Coarse',
  QA_DENSE: 'QA Dense',
  EVENT_PROPOSAL: 'Event Proposal',
  ACTION_DENSE: 'Action Dense',
  BOUNDARY_REFINEMENT: 'Boundary Refinement',
}

// Purpose display colors
export const PURPOSE_COLORS: Record<StreamPurpose, string> = {
  QA_COARSE: '#4A7FA8',
  QA_DENSE: '#6B5EA8',
  EVENT_PROPOSAL: '#A87A2A',
  ACTION_DENSE: '#4A7A5A',
  BOUNDARY_REFINEMENT: '#A84A7A',
}

// Stage abbreviations for compact display
export const STAGE_ABBR: Record<StreamStage, string> = {
  SEGMENT: 'SEG',
  WINDOW: 'WIN',
  QA_COARSE: 'QA⊕',
  QA_DENSE: 'QA⊕⊕',
  EVENT_PROPOSAL: 'PROP',
  ACTION_DENSE: 'ACT',
  BOUNDARY_REFINEMENT: 'BND',
  WINDOW_REDUCTION: 'RED',
  FINALIZATION: 'FIN',
}
