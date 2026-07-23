// ── Mock stream event data for the streaming architecture demo ───────────────
//
// This generates a deterministic mock event stream for a 40.89-second six-camera
// recording, matching the fixture-backed sample used in the backend baseline.
//
// Window parameters: W=2s, H=1s, initial allowed lateness=300ms
// Segments: 1-second logical chunks per camera

import type {
  CaptureScope,
  StreamSegment,
  IncrementalWindow,
  StreamInference,
  ExpectedWindowPlan,
  WindowTerminalClosure,
  RecordingFinalizationMap,
  StreamEvent,
  NanosecondInterval,
  QualityObservation,
  Sha256Digest,
  OpaqueUuid,
  TerminalOutcome,
  StreamPurpose,
} from '@/types'

// ── Helpers ───────────────────────────────────────────────────────────────────

function interval(startSec: number, endSec: number): NanosecondInterval {
  return { start_ns: BigInt(Math.round(startSec * 1e9)), end_ns: BigInt(Math.round(endSec * 1e9)) }
}

function mockSha256(seed: string): Sha256Digest {
  // Deterministic mock SHA-256 — not cryptographically valid, but stable for demo
  const chars = '0123456789abcdef'
  let hash = ''
  for (let i = 0; i < 64; i++) {
    hash += chars[(seed.charCodeAt(i % seed.length) + i * 7) % 16]
  }
  return hash
}

function mockUuid(seed: string): OpaqueUuid {
  const chars = '0123456789abcdef'
  const parts = [8, 4, 4, 4, 12]
  let uuid = ''
  let idx = 0
  for (let p = 0; p < parts.length; p++) {
    if (p > 0) uuid += '-'
    for (let i = 0; i < parts[p]; i++) {
      uuid += chars[(seed.charCodeAt(idx % seed.length) + idx * 13) % 16]
      idx++
    }
  }
  return uuid
}

// ── Capture scope ───────────────────────────────────────────────────────────

export const MOCK_CAPTURE_SCOPE: CaptureScope = {
  capture_scope_id: mockUuid('capture-scope-001'),
  capture_scope_key: 'pre-eos-capture-v1:' + mockSha256('capture-scope-001'),
  capture_scope_digest: mockSha256('capture-scope-001'),
  capture_authority_id: 'local-authority-001',
  capture_authority_epoch: 1,
  capture_assignment_policy_version: 'capture-assignment-v1',
  acquisition_id: 'acq-fixture-001',
  acquisition_epoch: 1,
  channel_bindings: [
    { camera_id: 'cam_01', source_channel_id: 'ch_01', source_channel_epoch: 1, channel_binding_semantic_sha256: mockSha256('ch_01') },
    { camera_id: 'cam_02', source_channel_id: 'ch_02', source_channel_epoch: 1, channel_binding_semantic_sha256: mockSha256('ch_02') },
    { camera_id: 'cam_03', source_channel_id: 'ch_03', source_channel_epoch: 1, channel_binding_semantic_sha256: mockSha256('ch_03') },
    { camera_id: 'cam_04', source_channel_id: 'ch_04', source_channel_epoch: 1, channel_binding_semantic_sha256: mockSha256('ch_04') },
    { camera_id: 'cam_05', source_channel_id: 'ch_05', source_channel_epoch: 1, channel_binding_semantic_sha256: mockSha256('ch_05') },
    { camera_id: 'cam_06', source_channel_id: 'ch_06', source_channel_epoch: 1, channel_binding_semantic_sha256: mockSha256('ch_06') },
  ],
  mapping_authority: {
    authority_id: 'mapping-auth-001',
    authority_epoch: 1,
    policy_version: 'mapping-policy-v1',
    initial_binding_semantic_sha256: mockSha256('mapping-auth'),
  },
  clock_authority: {
    authority_id: 'clock-auth-001',
    authority_epoch: 1,
    policy_version: 'clock-policy-v1',
    initial_binding_semantic_sha256: mockSha256('clock-auth'),
  },
}

const CAPTURE_SCOPE_DIGEST = MOCK_CAPTURE_SCOPE.capture_scope_digest

// ── Segments ──────────────────────────────────────────────────────────────────

const SOURCE_DURATION_SEC = 40.890455
const CHUNK_DURATION_SEC = 1.0
const NUM_CHUNKS = Math.ceil(SOURCE_DURATION_SEC / CHUNK_DURATION_SEC)

function generateQualityObservations(cameraId: string, chunkIdx: number): QualityObservation[] {
  const baseTime = chunkIdx * CHUNK_DURATION_SEC
  const observations: QualityObservation[] = []

  // Luminance — mostly normal, occasional low
  const lumValue = cameraId === 'cam_03' && chunkIdx >= 15 && chunkIdx <= 17 ? 12.5 : 145.0
  observations.push({
    kind: 'LUMINANCE',
    value: lumValue,
    interval: interval(baseTime, baseTime + CHUNK_DURATION_SEC),
  })

  // Edge energy
  observations.push({
    kind: 'EDGE_ENERGY',
    value: cameraId === 'cam_03' && chunkIdx >= 15 && chunkIdx <= 17 ? 0.8 : 42.5,
    interval: interval(baseTime, baseTime + CHUNK_DURATION_SEC),
  })

  // Freeze detection — cam_05 has a brief freeze at chunk 22
  if (cameraId === 'cam_05' && chunkIdx === 22) {
    observations.push({
      kind: 'FREEZE',
      value: 0.97,
      interval: interval(baseTime, baseTime + CHUNK_DURATION_SEC),
    })
  }

  // Cadence
  observations.push({
    kind: 'CADENCE',
    value: 30.0,
    interval: interval(baseTime, baseTime + CHUNK_DURATION_SEC),
  })

  return observations
}

export const MOCK_SEGMENTS: StreamSegment[] = []

for (let chunkIdx = 0; chunkIdx < NUM_CHUNKS; chunkIdx++) {
  const baseTime = chunkIdx * CHUNK_DURATION_SEC
  const endTime = Math.min(baseTime + CHUNK_DURATION_SEC, SOURCE_DURATION_SEC)

  for (const cameraId of ['cam_01', 'cam_02', 'cam_03', 'cam_04', 'cam_05', 'cam_06']) {
    // cam_04 has a gap at chunk 30
    if (cameraId === 'cam_04' && chunkIdx === 30) continue

    const segmentKey = `stream-segment-v1:${cameraId}:${chunkIdx}`
    const segment: StreamSegment = {
      segment_id: mockUuid(`segment-${cameraId}-${chunkIdx}`),
      segment_key: segmentKey,
      segment_semantic_sha256: mockSha256(`segment-semantic-${cameraId}-${chunkIdx}`),
      camera_id: cameraId,
      requested_interval: interval(baseTime, baseTime + CHUNK_DURATION_SEC),
      effective_interval: interval(baseTime, endTime),
      content_digest: mockSha256(`segment-content-${cameraId}-${chunkIdx}`),
      mapping_semantic_sha256: mockSha256(`segment-mapping-${cameraId}-${chunkIdx}`),
      clock_alignment_semantic_sha256: mockSha256(`segment-clock-${cameraId}-${chunkIdx}`),
      quality_observations: generateQualityObservations(cameraId, chunkIdx),
    }
    MOCK_SEGMENTS.push(segment)
  }
}

// ── Windows ───────────────────────────────────────────────────────────────────

const WINDOW_WIDTH_SEC = 2.0
const WINDOW_HOP_SEC = 1.0
const NUM_WINDOWS = Math.ceil((SOURCE_DURATION_SEC - WINDOW_WIDTH_SEC) / WINDOW_HOP_SEC) + 1

function getSegmentRef(cameraId: string, chunkIdx: number) {
  const segment = MOCK_SEGMENTS.find(
    (s) => s.camera_id === cameraId &&
      Math.floor(Number(s.requested_interval.start_ns) / 1e9 / CHUNK_DURATION_SEC) === chunkIdx,
  )
  if (!segment) {
    return { reason: 'GAP' as const, camera_id: cameraId }
  }
  return {
    segment_key: segment.segment_key,
    segment_semantic_sha256: segment.segment_semantic_sha256,
  }
}

function buildSixSlotClosure(windowIdx: number): IncrementalWindow['ordered_six_slot_closure'] {
  const chunkIdx = windowIdx // Each window starts at its index * hop
  return [
    getSegmentRef('cam_01', chunkIdx),
    getSegmentRef('cam_02', chunkIdx),
    getSegmentRef('cam_03', chunkIdx),
    getSegmentRef('cam_04', chunkIdx),
    getSegmentRef('cam_05', chunkIdx),
    getSegmentRef('cam_06', chunkIdx),
  ]
}

function determinePurpose(windowIdx: number): StreamPurpose {
  // Every window gets QA_COARSE
  // Every 5th window gets QA_DENSE
  // Windows 8, 15, 22 get EVENT_PROPOSAL
  // Windows with proposals that have actions get ACTION_DENSE
  // Refinement windows (ONSET/OFFSET) follow their parent

  if (windowIdx % 5 === 0) return 'QA_DENSE'
  if ([8, 15, 22].includes(windowIdx)) return 'EVENT_PROPOSAL'
  if ([9, 16, 23].includes(windowIdx)) return 'ACTION_DENSE'
  if ([10, 11, 17, 18, 24, 25].includes(windowIdx)) return 'BOUNDARY_REFINEMENT'
  return 'QA_COARSE'
}

function determineParent(windowIdx: number): { parent_key: string | null; role: 'ONSET' | 'OFFSET' | null } {
  // ONSET refinement follows EVENT_PROPOSAL by 1 hop
  // OFFSET refinement follows ONSET by 1 hop
  if ([10, 17, 24].includes(windowIdx)) {
    return { parent_key: `incremental-window-v1:${mockSha256(`window-${windowIdx - 2}`)}`, role: 'ONSET' }
  }
  if ([11, 18, 25].includes(windowIdx)) {
    return { parent_key: `incremental-window-v1:${mockSha256(`window-${windowIdx - 1}`)}`, role: 'OFFSET' }
  }
  return { parent_key: null, role: null }
}

export const MOCK_WINDOWS: IncrementalWindow[] = []

for (let w = 0; w < NUM_WINDOWS; w++) {
  const startSec = w * WINDOW_HOP_SEC
  const endSec = Math.min(startSec + WINDOW_WIDTH_SEC, SOURCE_DURATION_SEC)
  const purpose = determinePurpose(w)
  const parent = determineParent(w)

  const window: IncrementalWindow = {
    window_id: mockUuid(`window-${w}`),
    window_key: `incremental-window-v1:${mockSha256(`window-${w}`)}`,
    window_semantic_sha256: mockSha256(`window-${w}`),
    capture_scope_digest: CAPTURE_SCOPE_DIGEST,
    purpose,
    requested_interval: interval(startSec, startSec + WINDOW_WIDTH_SEC),
    effective_interval: interval(startSec, endSec),
    ordered_six_slot_closure: buildSixSlotClosure(w),
    mapping_semantic_sha256: mockSha256(`window-mapping-${w}`),
    clock_alignment_semantic_sha256: mockSha256(`window-clock-${w}`),
    parent_window_key: parent.parent_key,
    refinement_role: parent.role,
    refinement_generation: parent.role ? 1 : 0,
    window_policy_version: 'incremental-window-identity-v1',
    semantic_projection_version: 'incremental-window-semantic-v1',
    identity_policy_version: 'incremental-window-identity-v1',
  }
  MOCK_WINDOWS.push(window)
}

// ── Inferences ────────────────────────────────────────────────────────────────

export const MOCK_INFERENCES: StreamInference[] = []

for (let w = 0; w < NUM_WINDOWS; w++) {
  const window = MOCK_WINDOWS[w]
  const purposes: StreamPurpose[] = []

  // Base QA_COARSE for all windows
  purposes.push('QA_COARSE')

  // QA_DENSE for every 5th window
  if (w % 5 === 0) purposes.push('QA_DENSE')

  // EVENT_PROPOSAL at specific windows
  if ([8, 15, 22].includes(w)) purposes.push('EVENT_PROPOSAL')

  // ACTION_DENSE following proposals
  if ([9, 16, 23].includes(w)) purposes.push('ACTION_DENSE')

  // BOUNDARY_REFINEMENT for refinement windows
  if (window.refinement_role) purposes.push('BOUNDARY_REFINEMENT')

  for (const purpose of purposes) {
    const inferenceKey = `stream-inference-v1:${mockSha256(`inference-${w}-${purpose}`)}`
    const attemptId = mockUuid(`attempt-${w}-${purpose}`)

    // Simulate some failures for demonstration
    let outcome: TerminalOutcome = 'SUCCEEDED'
    if (w === 12 && purpose === 'QA_COARSE') outcome = 'FAILED'
    if (w === 18 && purpose === 'BOUNDARY_REFINEMENT') outcome = 'ABSTAINED'
    if (w === 25 && purpose === 'BOUNDARY_REFINEMENT') outcome = 'INCOMPLETE'

    const inference: StreamInference = {
      inference_id: mockUuid(`inference-${w}-${purpose}`),
      inference_key: inferenceKey,
      window_key: window.window_key,
      purpose,
      input_plan_digest: mockSha256(`input-plan-${w}-${purpose}`),
      attempt_id: attemptId,
      attempt_number: 0,
      terminal_outcome: outcome,
      evidence_ref: outcome === 'SUCCEEDED'
        ? {
            artifact_id: `artifact-${w}-${purpose}`,
            digest: mockSha256(`evidence-${w}-${purpose}`),
            byte_count: 4096 + w * 100,
            media_type: 'application/json',
          }
        : null,
      inference_policy_version: 'stream-inference-identity-v1',
      semantic_projection_version: 'stream-inference-semantic-v1',
    }
    MOCK_INFERENCES.push(inference)
  }
}

// ── Expected window plan ────────────────────────────────────────────────────

export const MOCK_PLAN: ExpectedWindowPlan = {
  plan_key: 'expected-window-plan-v1:' + mockSha256('plan-001'),
  capture_scope_digest: CAPTURE_SCOPE_DIGEST,
  declarations: MOCK_WINDOWS.map((w, i) => ({
    expected_ordinal: i,
    window_key: w.window_key,
    requested_interval: w.requested_interval,
    planning_policy_digest: mockSha256('planning-policy-v1'),
  })),
  sealed_manifest: {
    sealed_at: new Date().toISOString(),
    ordered_members: MOCK_WINDOWS.map((w, i) => ({
      expected_ordinal: i,
      window_key: w.window_key,
      requested_interval: w.requested_interval,
      planning_policy_digest: mockSha256('planning-policy-v1'),
    })),
  },
}

// ── Terminal closure ────────────────────────────────────────────────────────

export const MOCK_TERMINAL_CLOSURE: WindowTerminalClosure = {
  closure_key: 'window-terminal-closure-v1:' + mockSha256('closure-001'),
  plan_key: MOCK_PLAN.plan_key,
  members: MOCK_WINDOWS.map((w, i) => {
    const inference = MOCK_INFERENCES.find((inf) => inf.window_key === w.window_key)
    const outcome = inference?.terminal_outcome ?? 'NO_EVENTS'

    return {
      expected_ordinal: i,
      window_key: w.window_key,
      window_semantic_sha256: w.window_semantic_sha256,
      terminal_outcome: outcome,
      terminal_work_item_id: inference?.attempt_id ?? mockUuid(`no-work-${i}`),
      terminal_work_logical_key: inference
        ? `stream-work-v1:${mockSha256(`work-${i}`)}`
        : `stream-work-v1:${mockSha256(`no-work-${i}`)}`,
      terminal_evidence_ref: inference?.evidence_ref ?? {
        artifact_id: `no-evidence-${i}`,
        digest: mockSha256(`no-evidence-${i}`),
        byte_count: 0,
        media_type: 'application/octet-stream',
      },
    }
  }),
}

// ── Recording finalization ────────────────────────────────────────────────────

export const MOCK_FINALIZATION: RecordingFinalizationMap = {
  finalization_key: 'recording-finalization-map-v1:' + mockSha256('finalization-001'),
  capture_scope_digest: CAPTURE_SCOPE_DIGEST,
  source_digest: mockSha256('complete-source-digest'),
  duration_ns: BigInt(Math.round(SOURCE_DURATION_SEC * 1e9)),
  incremental_to_final_mappings: MOCK_WINDOWS.map((w) => ({
    incremental_window_key: w.window_key,
    final_window_identity: mockSha256(`final-${w.window_key}`),
    terminal_outcome: MOCK_INFERENCES.find((inf) => inf.window_key === w.window_key)?.terminal_outcome ?? 'NO_EVENTS',
  })),
  primary_completion_ref: {
    artifact_id: 'primary-completion-001',
    digest: mockSha256('primary-completion'),
    byte_count: 8192,
    media_type: 'application/json',
  },
  finalized_at: new Date().toISOString(),
}

// ── Ordered event stream ──────────────────────────────────────────────────────

export function buildMockEventStream(): StreamEvent[] {
  const events: StreamEvent[] = []

  // 1. Capture scope (first)
  events.push({ type: 'CAPTURE_SCOPE', scope: MOCK_CAPTURE_SCOPE })

  // 2. Segments in time order
  for (let chunkIdx = 0; chunkIdx < NUM_CHUNKS; chunkIdx++) {
    for (const cameraId of ['cam_01', 'cam_02', 'cam_03', 'cam_04', 'cam_05', 'cam_06']) {
      const segment = MOCK_SEGMENTS.find(
        (s) => s.camera_id === cameraId &&
          Math.floor(Number(s.requested_interval.start_ns) / 1e9 / CHUNK_DURATION_SEC) === chunkIdx,
      )
      if (segment) {
        events.push({ type: 'SEGMENT', segment })
      }
    }
  }

  // 3. Windows (as they are planned)
  for (let w = 0; w < NUM_WINDOWS; w++) {
    events.push({ type: 'WINDOW', window: MOCK_WINDOWS[w] })
  }

  // 4. Plan append events (one per declaration)
  for (const decl of MOCK_PLAN.declarations) {
    events.push({ type: 'PLAN_APPEND', plan: MOCK_PLAN, declaration: decl })
  }

  // 5. Inferences (as they complete)
  for (const inference of MOCK_INFERENCES) {
    events.push({ type: 'INFERENCE', inference })
  }

  // 6. Plan seal
  if (MOCK_PLAN.sealed_manifest) {
    events.push({ type: 'PLAN_SEAL', plan: MOCK_PLAN, seal: MOCK_PLAN.sealed_manifest })
  }

  // 7. Terminal closure
  events.push({ type: 'TERMINAL_CLOSURE', closure: MOCK_TERMINAL_CLOSURE })

  // 8. Finalization
  events.push({ type: 'FINALIZATION', finalization: MOCK_FINALIZATION })

  // 9. Watermarks (interspersed — we'll add them during simulation)
  // 10. Backpressure (interspersed)

  return events
}

export const MOCK_EVENT_STREAM = buildMockEventStream()

// ── Timeline bands ────────────────────────────────────────────────────────────

export interface TimelineBand {
  start_ns: bigint
  end_ns: bigint
  label: string
  kind: 'segment' | 'window' | 'inference' | 'watermark'
  color: string
  dataId: string
}

export function buildTimelineBands(): TimelineBand[] {
  const bands: TimelineBand[] = []

  // Segment bands (one per camera, stacked)
  for (const segment of MOCK_SEGMENTS) {
    const cameraIdx = parseInt(segment.camera_id.replace('cam_', '')) - 1
    bands.push({
      start_ns: segment.effective_interval.start_ns,
      end_ns: segment.effective_interval.end_ns,
      label: segment.camera_id,
      kind: 'segment',
      color: `hsl(${cameraIdx * 60}, 45%, 55%)`,
      dataId: segment.segment_key,
    })
  }

  // Window bands
  for (const window of MOCK_WINDOWS) {
    bands.push({
      start_ns: window.effective_interval.start_ns,
      end_ns: window.effective_interval.end_ns,
      label: window.purpose,
      kind: 'window',
      color: window.purpose === 'QA_COARSE' ? '#4A7FA8'
        : window.purpose === 'QA_DENSE' ? '#6B5EA8'
        : window.purpose === 'EVENT_PROPOSAL' ? '#A87A2A'
        : window.purpose === 'ACTION_DENSE' ? '#4A7A5A'
        : '#A84A7A',
      dataId: window.window_key,
    })
  }

  return bands
}

export const MOCK_TIMELINE_BANDS = buildTimelineBands()
