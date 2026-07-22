import { Node, Edge } from 'reactflow'
import { RobataNodeData, EdgeSchema } from '@/types'

// ── Mock data representing a canonical pipeline run ───────────────────────────

export const INITIAL_NODES: Node<RobataNodeData>[] = [
  // Row 0 — Source
  {
    id: 'source',
    type: 'robata',
    position: { x: 400, y: 20 },
    data: {
      kind: 'source',
      label: 'MCAP Source',
      status: 'COMPLETE',
      metrics: { camera_count: 6, schema_version: 'v2' },
    },
  },
  // Row 1 — Media quality + Adaptive sampler
  {
    id: 'media_quality',
    type: 'robata',
    position: { x: 100, y: 160 },
    data: {
      kind: 'media_quality',
      label: 'Media Quality',
      status: 'COMPLETE',
      metrics: { camera_count: 6, duration_ms: 230 },
    },
  },
  {
    id: 'adaptive_sampler',
    type: 'robata',
    position: { x: 700, y: 160 },
    data: {
      kind: 'adaptive_sampler',
      label: 'Adaptive Sampler',
      status: 'COMPLETE',
      metrics: { schema_version: 'rational-grid-v1' },
    },
  },
  // Row 2 — QA Coarse
  {
    id: 'qa_coarse',
    type: 'robata',
    position: { x: 400, y: 310 },
    data: {
      kind: 'qa_coarse',
      label: 'QA Coarse',
      status: 'COMPLETE',
      metrics: { attempt: 1, camera_count: 6 },
    },
  },
  // Row 3 — QA Dense (conditional)
  {
    id: 'qa_dense',
    type: 'robata',
    position: { x: 200, y: 450 },
    data: {
      kind: 'qa_dense',
      label: 'QA Dense',
      status: 'COMPLETE',
      metrics: { attempt: 1, camera_count: 6 },
    },
  },
  // Row 3 — QA Gate
  {
    id: 'qa_gate',
    type: 'robata',
    position: { x: 600, y: 450 },
    data: {
      kind: 'qa_gate',
      label: 'QA Gate',
      status: 'COMPLETE',
    },
  },
  // Row 4 — Event Proposal
  {
    id: 'event_proposal',
    type: 'robata',
    position: { x: 400, y: 590 },
    data: {
      kind: 'event_proposal',
      label: 'Event Proposal',
      status: 'COMPLETE',
      metrics: { attempt: 1 },
    },
  },
  // Row 5 — Candidate Reducer
  {
    id: 'candidate_reducer',
    type: 'robata',
    position: { x: 400, y: 720 },
    data: {
      kind: 'candidate_reducer',
      label: 'Candidate Reducer',
      status: 'COMPLETE',
    },
  },
  // Row 6 — Action Evidence (per candidate)
  {
    id: 'action_evidence_0',
    type: 'robata',
    position: { x: 100, y: 860 },
    data: {
      kind: 'action_evidence',
      label: 'Action Evidence',
      status: 'COMPLETE',
      instance_id: '0',
      instance_count: 2,
      metrics: { camera_count: 6 },
    },
  },
  {
    id: 'action_evidence_1',
    type: 'robata',
    position: { x: 700, y: 860 },
    data: {
      kind: 'action_evidence',
      label: 'Action Evidence',
      status: 'COMPLETE',
      instance_id: '1',
      instance_count: 2,
      metrics: { camera_count: 6 },
    },
  },
  // Row 7 — Provisional Fusion
  {
    id: 'provisional_fusion',
    type: 'robata',
    position: { x: 400, y: 1000 },
    data: {
      kind: 'provisional_fusion',
      label: 'Provisional Fusion',
      status: 'COMPLETE',
    },
  },
  // Row 8 — Boundary Refinement (per action, ONSET + OFFSET)
  {
    id: 'boundary_onset_0',
    type: 'robata',
    position: { x: 50, y: 1140 },
    data: {
      kind: 'boundary_refinement',
      label: 'Boundary ONSET',
      status: 'COMPLETE',
      instance_id: 'action-0-onset',
      metrics: { camera_count: 6 },
    },
  },
  {
    id: 'boundary_offset_0',
    type: 'robata',
    position: { x: 350, y: 1140 },
    data: {
      kind: 'boundary_refinement',
      label: 'Boundary OFFSET',
      status: 'RUNNING',
      instance_id: 'action-0-offset',
      metrics: { camera_count: 6 },
    },
  },
  {
    id: 'boundary_onset_1',
    type: 'robata',
    position: { x: 650, y: 1140 },
    data: {
      kind: 'boundary_refinement',
      label: 'Boundary ONSET',
      status: 'PENDING',
      instance_id: 'action-1-onset',
    },
  },
  {
    id: 'boundary_offset_1',
    type: 'robata',
    position: { x: 950, y: 1140 },
    data: {
      kind: 'boundary_refinement',
      label: 'Boundary OFFSET',
      status: 'PENDING',
      instance_id: 'action-1-offset',
    },
  },
  // Row 9 — Final Fusion
  {
    id: 'final_fusion',
    type: 'robata',
    position: { x: 400, y: 1290 },
    data: {
      kind: 'final_fusion',
      label: 'Final Fusion',
      status: 'PENDING',
    },
  },
  // Row 10 — Primary Completion
  {
    id: 'primary_completion',
    type: 'robata',
    position: { x: 400, y: 1420 },
    data: {
      kind: 'primary_completion',
      label: 'Primary Completion',
      status: 'PENDING',
      metrics: { schema_version: 'completion-detail-v4' },
    },
  },
  // Row 11 — Post-completion
  {
    id: 'outbox_relay',
    type: 'robata',
    position: { x: 150, y: 1560 },
    data: {
      kind: 'outbox_relay',
      label: 'Outbox Relay',
      status: 'PENDING',
    },
  },
  {
    id: 'review_queue',
    type: 'robata',
    position: { x: 650, y: 1560 },
    data: {
      kind: 'review_queue',
      label: 'Review Queue',
      status: 'PENDING',
    },
  },
]

type EdgeDef = { id: string; source: string; target: string; schema: EdgeSchema; label?: string }

const EDGE_DEFS: EdgeDef[] = [
  { id: 'e-src-mq',   source: 'source',            target: 'media_quality',       schema: 'SOURCE' },
  { id: 'e-src-as',   source: 'source',            target: 'adaptive_sampler',    schema: 'SOURCE' },
  { id: 'e-mq-qac',   source: 'media_quality',     target: 'qa_coarse',           schema: 'QA',    label: 'quality triggers' },
  { id: 'e-as-qac',   source: 'adaptive_sampler',  target: 'qa_coarse',           schema: 'QA',    label: 'frame budget' },
  { id: 'e-qac-qad',  source: 'qa_coarse',         target: 'qa_dense',            schema: 'QA',    label: 'degraded / unusable' },
  { id: 'e-qac-gate', source: 'qa_coarse',         target: 'qa_gate',             schema: 'QA' },
  { id: 'e-qad-gate', source: 'qa_dense',          target: 'qa_gate',             schema: 'QA' },
  { id: 'e-gate-ep',  source: 'qa_gate',           target: 'event_proposal',      schema: 'PROPOSAL', label: 'QA_COMPLETE' },
  { id: 'e-ep-cr',    source: 'event_proposal',    target: 'candidate_reducer',   schema: 'CANDIDATE' },
  { id: 'e-cr-ae0',   source: 'candidate_reducer', target: 'action_evidence_0',   schema: 'EVIDENCE' },
  { id: 'e-cr-ae1',   source: 'candidate_reducer', target: 'action_evidence_1',   schema: 'EVIDENCE' },
  { id: 'e-ae0-pf',   source: 'action_evidence_0', target: 'provisional_fusion',  schema: 'FUSION' },
  { id: 'e-ae1-pf',   source: 'action_evidence_1', target: 'provisional_fusion',  schema: 'FUSION' },
  { id: 'e-pf-bon0',  source: 'provisional_fusion',target: 'boundary_onset_0',    schema: 'EVIDENCE', label: 'action 0' },
  { id: 'e-pf-bof0',  source: 'provisional_fusion',target: 'boundary_offset_0',   schema: 'EVIDENCE' },
  { id: 'e-pf-bon1',  source: 'provisional_fusion',target: 'boundary_onset_1',    schema: 'EVIDENCE', label: 'action 1' },
  { id: 'e-pf-bof1',  source: 'provisional_fusion',target: 'boundary_offset_1',   schema: 'EVIDENCE' },
  { id: 'e-bon0-ff',  source: 'boundary_onset_0',  target: 'final_fusion',        schema: 'FUSION' },
  { id: 'e-bof0-ff',  source: 'boundary_offset_0', target: 'final_fusion',        schema: 'FUSION' },
  { id: 'e-bon1-ff',  source: 'boundary_onset_1',  target: 'final_fusion',        schema: 'FUSION' },
  { id: 'e-bof1-ff',  source: 'boundary_offset_1', target: 'final_fusion',        schema: 'FUSION' },
  { id: 'e-ff-pc',    source: 'final_fusion',      target: 'primary_completion',  schema: 'COMPLETION' },
  { id: 'e-pc-or',    source: 'primary_completion',target: 'outbox_relay',        schema: 'COMPLETION', label: 'ActionPublish' },
  { id: 'e-pc-rq',    source: 'primary_completion',target: 'review_queue',        schema: 'REVIEW', label: 'nonblocking' },
]

export const INITIAL_EDGES: Edge[] = EDGE_DEFS.map((def) => ({
  id: def.id,
  source: def.source,
  target: def.target,
  type: 'schema',
  data: { schema: def.schema, label: def.label },
  animated: false,
}))
