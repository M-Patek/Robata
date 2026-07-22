import { EdgeSchema, NodeStatus } from '@/types'
import { INITIAL_NODES } from './pipeline'

export interface PipelineGroup {
  id: string
  label: string
  description: string
  section: string
  nodeIds: string[]
  // Which group comes before this one (null = first)
  prevGroupId: string | null
  nextGroupId: string | null
}

export const PIPELINE_GROUPS: PipelineGroup[] = [
  {
    id: 'ingestion',
    label: 'Ingestion',
    description: 'Source decoding, media quality analysis, adaptive frame grid',
    section: '25.3 / 25.6',
    nodeIds: ['source', 'media_quality', 'adaptive_sampler'],
    prevGroupId: null,
    nextGroupId: 'quality',
  },
  {
    id: 'quality',
    label: 'Quality Assurance',
    description: 'Coarse and dense QA across six cameras; three-state gate',
    section: '25.10',
    nodeIds: ['qa_coarse', 'qa_dense', 'qa_gate'],
    prevGroupId: 'ingestion',
    nextGroupId: 'events',
  },
  {
    id: 'events',
    label: 'Event Processing',
    description: 'Proposal, candidate reduction, per-candidate action evidence',
    section: '25.10',
    nodeIds: ['event_proposal', 'candidate_reducer', 'action_evidence_0', 'action_evidence_1'],
    prevGroupId: 'quality',
    nextGroupId: 'fusion',
  },
  {
    id: 'fusion',
    label: 'Fusion',
    description: 'Provisional fusion, ONSET/OFFSET boundary refinement, final fusion',
    section: '25.10',
    nodeIds: [
      'provisional_fusion',
      'boundary_onset_0', 'boundary_offset_0',
      'boundary_onset_1', 'boundary_offset_1',
      'final_fusion',
    ],
    prevGroupId: 'events',
    nextGroupId: 'completion',
  },
  {
    id: 'completion',
    label: 'Completion',
    description: 'Atomic primary completion, outbox relay, nonblocking review routing',
    section: '25.8 / 25.9',
    nodeIds: ['primary_completion', 'outbox_relay', 'review_queue'],
    prevGroupId: 'fusion',
    nextGroupId: null,
  },
]

export function getGroupById(id: string) {
  return PIPELINE_GROUPS.find((g) => g.id === id) ?? null
}

export function getGroupStatus(groupId: string, nodeStatuses: Record<string, NodeStatus>): NodeStatus {
  const group = getGroupById(groupId)
  if (!group) return 'PENDING'
  const statuses = group.nodeIds.map((nid) => {
    const node = INITIAL_NODES.find((n) => n.id === nid)
    return nodeStatuses[nid] ?? node?.data?.status ?? 'PENDING'
  })
  if (statuses.some((s) => s === 'FAILED')) return 'FAILED'
  if (statuses.some((s) => s === 'RUNNING')) return 'RUNNING'
  if (statuses.some((s) => s === 'WAITING_REVIEW')) return 'WAITING_REVIEW'
  if (statuses.every((s) => s === 'COMPLETE' || s === 'NO_EVENTS')) return 'COMPLETE'
  if (statuses.some((s) => s === 'BLOCKED')) return 'BLOCKED'
  return 'PENDING'
}

// Internal edges per group for the expanded canvas
export const GROUP_INTERNAL_EDGES: Record<string, { source: string; target: string; schema: EdgeSchema; label?: string }[]> = {
  ingestion: [
    { source: '__gateway__', target: 'source',          schema: 'SOURCE' },
    { source: 'source',      target: 'media_quality',   schema: 'SOURCE' },
    { source: 'source',      target: 'adaptive_sampler',schema: 'SOURCE' },
  ],
  quality: [
    { source: '__gateway__', target: 'qa_coarse',  schema: 'QA' },
    { source: 'qa_coarse',   target: 'qa_dense',   schema: 'QA', label: 'degraded' },
    { source: 'qa_coarse',   target: 'qa_gate',    schema: 'QA' },
    { source: 'qa_dense',    target: 'qa_gate',    schema: 'QA' },
  ],
  events: [
    { source: '__gateway__',      target: 'event_proposal',   schema: 'PROPOSAL', label: 'QA_COMPLETE' },
    { source: 'event_proposal',   target: 'candidate_reducer',schema: 'CANDIDATE' },
    { source: 'candidate_reducer',target: 'action_evidence_0',schema: 'EVIDENCE' },
    { source: 'candidate_reducer',target: 'action_evidence_1',schema: 'EVIDENCE' },
  ],
  fusion: [
    { source: '__gateway__',       target: 'provisional_fusion', schema: 'FUSION' },
    { source: 'provisional_fusion',target: 'boundary_onset_0',   schema: 'EVIDENCE', label: 'action 0' },
    { source: 'provisional_fusion',target: 'boundary_offset_0',  schema: 'EVIDENCE' },
    { source: 'provisional_fusion',target: 'boundary_onset_1',   schema: 'EVIDENCE', label: 'action 1' },
    { source: 'provisional_fusion',target: 'boundary_offset_1',  schema: 'EVIDENCE' },
    { source: 'boundary_onset_0',  target: 'final_fusion',       schema: 'FUSION' },
    { source: 'boundary_offset_0', target: 'final_fusion',       schema: 'FUSION' },
    { source: 'boundary_onset_1',  target: 'final_fusion',       schema: 'FUSION' },
    { source: 'boundary_offset_1', target: 'final_fusion',       schema: 'FUSION' },
  ],
  completion: [
    { source: '__gateway__',      target: 'primary_completion', schema: 'COMPLETION' },
    { source: 'primary_completion',target: 'outbox_relay',      schema: 'COMPLETION', label: 'ActionPublish' },
    { source: 'primary_completion',target: 'review_queue',      schema: 'REVIEW', label: 'nonblocking' },
  ],
}
