# Web Frontend Streaming-Architecture Migration Plan V1

- Status: non-normative local engineering plan
- Date: 2026-07-22
- Governing authority: Architecture V1.1 Sections 17, 19, 20, 25; ADR 0015; and the streaming-throughput next-iteration guide
- Evidence boundary: current facts are `LOCAL_CONFORMANCE`; the frontend remains `production_eligible=false`
- Supersedes: the existing web/README.md "Next Steps" section and the `NodeKind` / `RobataRun` domain model

## 1. Authority and purpose

This guide translates the streaming-throughput rearchitecture (WP0–WP7) into a frontend domain-model migration. It is subordinate to:

1. the [streaming-throughput next-iteration guide](streaming-throughput-next-iteration-v1.md);
2. [ADR 0015](../adr/0015-pre-eos-stream-identity-and-planning.md) pre-EOS stream identity;
3. the accepted stream contract family (`stream_common`, `stream_source`, `stream_window`, `stream_planning`, `stream_finalization`); and
4. the existing web frontend in `web/src/`.

It does not add a new backend endpoint, change a phase gate, or claim production capacity. The frontend remains a mock-driven visualization that validates contract shapes before the backend is fully wired.

The goal is to replace the current whole-recording fixed-DAG frontend model with a continuous two-plane model that reflects the new streaming architecture: a replaceable media+inference plane and a durable window DAG plane. The target experience is a live-updating view of capture scopes, segments, windows, inferences, and finalization that a human operator can read and trust.

## 2. Why migrate now

The backend contracts (`bb348e7`) already freeze:

- `PreEosCaptureSubject` — immutable capture scope before first segment
- `StreamSegment` — per-camera bounded encoded ring closure
- `IncrementalWindow` — window identity with semantic SHA-256, parent lineage, and refinement role
- `StreamInference` — inference identity bound to window and input plan
- `ExpectedWindowPlan` / `ExpectedWindowPlanSeal` — append-before-publish planner
- `WindowTerminalClosure` — reconciled terminal outcome per expected member
- `RecordingFinalizationMap` — EOS mapping from pre-EOS subjects to recording-scoped final identities

These are data shapes, not runtime dependencies. The frontend can model them now with mock data and discover contract usability issues before the backend window scheduler (WP3) and finalization (WP5) are implemented. Waiting until WP6 would delay frontend contract validation and create a late-integration bottleneck.

## 3. Current frontend model (to be retired)

The existing domain in `web/src/types.ts` encodes a fixed recording-scoped DAG:

```
NodeKind = source | media_quality | adaptive_sampler | qa_coarse | qa_dense | qa_gate |
           event_proposal | candidate_reducer | action_evidence | provisional_fusion |
           boundary_refinement | final_fusion | primary_completion | outbox_relay |
           review_queue | work_scheduler

RobataRun = { run_id, recording_id, status, started_at, evidence_class,
              production_eligible, node_statuses: Record<string, NodeStatus> }
```

This model assumes:
- One run = one recording
- Nodes are fixed pipeline stages
- Status is per-node and per-run
- There is no concept of window, segment, capture scope, or incremental identity

The `useWebSocket.ts` hook falls back to `simulateRun()`, which plays a hard-coded stage sequence. This is the old batch mental model.

## 4. Target frontend model

The new model has two visual planes that can be toggled or stacked:

### 4.1 Plane A: Media + Inference (replaceable, per-window)

This plane shows the live execution of one window. It is replaceable because the same window identity can be produced by PyAV (local) or DeepStream/Triton (accelerated).

```
CaptureScope
  ├── Segment[0..5] (one per camera, or explicit absence)
  │     ├── Segment interval (requested + effective)
  │     ├── Content digest (SHA-256)
  │     └── Quality observations (black, frozen, gap, degraded)
  ├── Window (purpose, interval, semantic SHA-256)
  │     ├── Parent window (if refinement)
  │     ├── Refinement role (ONSET / OFFSET / none)
  │     └── Six-slot closure (segment refs or explicit absence)
  ├── Inference (purpose, input plan digest)
  │     ├── Attempt identity
       └── Terminal outcome (SUCCEEDED, SKIPPED, FAILED, etc.)
  └── WindowResult (evidence ref, terminal closure)
```

### 4.2 Plane B: Durable Window DAG (persistent, cross-recording)

This plane shows the append-only expected-window plan and its terminal closure. It survives restarts and is the authority for recording finalization.

```
ExpectedWindowPlan
  ├── Declarations (appended before child publication)
  │     ├── Window key
  │     ├── Requested interval
  │     └── Planning policy digest
  ├── Sealed manifest (at EOS)
  │     └── Ordered expected members
  └── Terminal closure (reconciled after execution)
        ├── Per-member outcome
        └── Evidence reference

RecordingFinalizationMap
  ├── Capture scope → Final source identity
  ├── Incremental windows → Recording-scoped window identities
  └── Primary completion (only after all barriers)
```

### 4.3 Frontend types (new)

```typescript
// ── Identity primitives ───────────────────────────────────────────────────
export type OpaqueUuid = string
export type Sha256Digest = string
export type NanosecondInterval = { start_ns: bigint; end_ns: bigint }

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

// ── Terminal outcomes (mirrors TerminalOutcome) ────────────────────────────
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

// ── Capture scope (immutable, pre-EOS) ────────────────────────────────────
export interface CaptureScope {
  capture_scope_id: OpaqueUuid
  capture_scope_key: string
  capture_scope_digest: Sha256Digest
  capture_authority_id: string
  capture_authority_epoch: number
  acquisition_id: string
  acquisition_epoch: number
  channel_bindings: ChannelBinding[]
}

export interface ChannelBinding {
  camera_id: string
  source_channel_id: string
  source_channel_epoch: number
}

// ── Segment (per-camera, bounded encoded ring closure) ────────────────────
export interface StreamSegment {
  segment_id: OpaqueUuid
  camera_id: string
  requested_interval: NanosecondInterval
  effective_interval: NanosecondInterval
  content_digest: Sha256Digest
  mapping_semantic_sha256: Sha256Digest
  clock_alignment_semantic_sha256: Sha256Digest
  quality_observations: QualityObservation[]
}

export interface QualityObservation {
  kind: 'LUMINANCE' | 'EDGE_ENERGY' | 'FREEZE' | 'CADENCE' | 'SEQUENCE_GAP' | 'CROSS_CAMERA_SKEW'
  value: number
  interval: NanosecondInterval
}

// ── Incremental window (immutable, with lineage) ─────────────────────────────
export interface IncrementalWindow {
  window_id: OpaqueUuid
  window_key: string
  window_semantic_sha256: Sha256Digest
  capture_scope_digest: Sha256Digest
  purpose: 'QA_COARSE' | 'QA_DENSE' | 'EVENT_PROPOSAL' | 'ACTION_DENSE' | 'BOUNDARY_REFINEMENT'
  requested_interval: NanosecondInterval
  effective_interval: NanosecondInterval
  six_slot_closure: (StreamSegmentRef | CameraAbsence)[]
  parent_window_key: string | null
  refinement_role: 'ONSET' | 'OFFSET' | null
  refinement_generation: number
}

export interface StreamSegmentRef {
  segment_key: string
  segment_semantic_sha256: Sha256Digest
}

export interface CameraAbsence {
  reason: 'ABSENT' | 'LATE' | 'BLACK' | 'FROZEN' | 'DEGRADED' | 'CORRUPT' | 'UNAVAILABLE' | 'GAP' | 'UNKNOWN'
  camera_id: string
}

// ── Stream inference (bound to window + input plan) ─────────────────────────
export interface StreamInference {
  inference_id: OpaqueUuid
  window_key: string
  purpose: string
  input_plan_digest: Sha256Digest
  attempt_id: OpaqueUuid
  terminal_outcome: TerminalOutcome
  evidence_ref: ArtifactEvidenceRef | null
}

export interface ArtifactEvidenceRef {
  artifact_id: string
  digest: Sha256Digest
  byte_count: number
  media_type: string
}

// ── Expected window plan (append-only, sealed at EOS) ─────────────────────────
export interface ExpectedWindowPlan {
  plan_key: string
  capture_scope_digest: Sha256Digest
  declarations: ExpectedWindowDeclaration[]
  sealed_manifest: ExpectedWindowPlanSeal | null
}

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

// ── Window terminal closure (reconciled per expected member) ─────────────────
export interface WindowTerminalClosure {
  closure_key: string
  plan_key: string
  members: WindowTerminalMember[]
}

export interface WindowTerminalMember {
  expected_ordinal: number
  window_key: string
  window_semantic_sha256: Sha256Digest
  terminal_outcome: TerminalOutcome
  terminal_work_item_id: OpaqueUuid
  terminal_evidence_ref: ArtifactEvidenceRef
}

// ── Recording finalization (EOS mapping) ────────────────────────────────────
export interface RecordingFinalizationMap {
  finalization_key: string
  capture_scope_digest: Sha256Digest
  source_digest: Sha256Digest
  duration_ns: bigint
  incremental_to_final_mappings: IncrementalToFinalMapping[]
  primary_completion_ref: ArtifactEvidenceRef
}

export interface IncrementalToFinalMapping {
  incremental_window_key: string
  final_window_identity: Sha256Digest
  terminal_outcome: TerminalOutcome
}

// ── Unified stream event (what the frontend consumes) ───────────────────────
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
  | { type: 'BACKPRESSURE'; level: 'NORMAL' | 'ELEVATED' | 'CRITICAL'; class: 'A' | 'B' | 'C' | 'D' }

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
}

export interface BackpressureState {
  level: 'NORMAL' | 'ELEVATED' | 'CRITICAL'
  class: 'A' | 'B' | 'C' | 'D'
  oldest_required_age_ms: number
  queue_depth: number
}
```

## 5. Component architecture

### 5.1 Top-level layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Robata  │  Streaming Pipeline  │  [Plane A] [Plane B]  │  Live │  Demo      │
─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────┐   │
│  │  TIMELINE (event time, not wall time)                               │   │
│  │  0s    2s    4s    6s    8s    10s   12s   14s   16s   18s   20s   │   │
│  │  │─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│       │   │
│  │  [W0] [W1] [W2] [W3] [W4] [W5] [W6] [W7] [W8] [W9]                  │   │
│  │   ✓    ✓        ✓    ✓    ✓        ✓    ✓                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │  PLANE A: Media + Inference │  │  PLANE B: Durable Window DAG        │  │
│  │  (per-window live view)     │  │  (append-only plan + terminal)      │  │
│  │                             │  │                                     │  │
│  │  ┌─────────────────────┐   │  │  ExpectedWindowPlan                 │  │
│  │  │ CaptureScope        │   │  │  ├── [0] W0: QA_COARSE  ✓ SUCCEEDED │  │
│  │  │ ├── cam_01: SEG_01  │   │  │  ├── [1] W1: QA_COARSE  ✓ SUCCEEDED │  │
│  │  │ ├── cam_02: SEG_02  │   │  │  ├── [2] W2: QA_DENSE   ✓ SUCCEEDED │  │
│  │  │ ├── cam_03: SEG_03  │   │  │  ├── [3] W3: EVENT_PRO... ✓ SUCCEEDED │  │
│  │  │ ├── cam_04: SEG_04  │   │  │  ├── [4] W4: QA_COARSE  ✓ FAILED    │  │
│  │  │ ├── cam_05: SEG_05  │   │  │  ├── [5] W5: QA_COARSE  ○ RUNNING   │  │
│  │  │ └── cam_06: SEG_06  │   │  │  └── ...                            │  │
│  │  └─────────────────────┘   │  │                                     │  │
│  │                             │  │  WindowTerminalClosure              │  │
│  │  ┌─────────────────────┐   │  │  ├── [0] W0 → SUCCEEDED            │  │
│  │  │ Window: W5          │   │  │  ├── [1] W1 → SUCCEEDED            │  │
│  │  │ Purpose: QA_COARSE  │   │  │  ├── [2] W2 → SUCCEEDED            │  │
│  │  │ Interval: [10s,12s) │   │  │  └── ...                            │  │
│  │  │ Semantic: abc123... │   │  │                                     │  │
│  │  │ Parent: null        │   │  │  RecordingFinalizationMap           │  │
│  │  │ Refinement: none    │   │  │  ├── Status: NOT_FINALIZED          │  │
│  │  │ Six-slot: [6 refs]  │   │  │  └── Barrier: 7/10 windows closed   │  │
│  │  └─────────────────────┘   │  │                                     │  │
│  │                             │  └─────────────────────────────────────┘  │
│  │  ┌─────────────────────   │                                         │
│  │  │ Inference: INF_05   │   │                                         │
│  │  │ Attempt: ATT_05_0   │   │                                         │
│  │  │ Outcome: SUCCEEDED  │   │                                         │
│  │  │ Evidence: [ref]     │   │                                         │
│  │  └─────────────────────┘   │                                         │
│  │                             │                                         │
│  └─────────────────────────────┘  ───────────────────────────────────────┘
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WATERMARK & BACKPRESSURE                                           │   │
│  │  Watermark: 18.3s  │  Queue depth: 3  │  Oldest: 1.2s  │  Level: NORMAL │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Component map

| Component | Current | New | File |
|---|---|---|---|
| `App.tsx` | Overview + expanded group view | Two-plane layout with timeline | `App.tsx` |
| `types.ts` | `NodeKind`, `RobataRun`, fixed DAG | `StreamEvent`, `CaptureScope`, `IncrementalWindow`, etc. | `types.ts` |
| `store.ts` | `activeRun: RobataRun \| null` | `streamView: StreamViewState` | `store.ts` |
| `useWebSocket.ts` | `simulateRun()` with hard-coded stages | `simulateStream()` emitting `StreamEvent[]` | `hooks/useWebSocket.ts` |
| `GroupNode.tsx` | Group card with node pills | **Removed** — no longer groups stages | — |
| `RobataNode.tsx` | Fixed pipeline stage node | `WindowNode`, `SegmentNode`, `InferenceNode`, `CaptureScopeNode` | `nodes/` |
| `SchemaEdge.tsx` | Colored edge by schema type | `EventEdge` — shows event-time/watermark flow | `edges/EventEdge.tsx` |
| `GroupExpandedView.tsx` | Expanded group with internal DAG | **Removed** — replaced by plane views | — |
| `NodeDetailDrawer.tsx` | Node detail with metrics | `SubjectDetailDrawer` — shows full subject with semantic identity | `panels/SubjectDetailDrawer.tsx` |
| `data/pipeline.ts` | `INITIAL_NODES` fixed DAG | `mock_stream_events.ts` — mock `StreamEvent[]` | `data/mock_stream_events.ts` |
| `data/groups.ts` | `PIPELINE_GROUPS` | **Removed** — no longer stage groups | — |
| `data/overview.ts` | `OVERVIEW_NODES` / `OVERVIEW_EDGES` | `TIMELINE_BANDS` — event-time bands | `data/timeline.ts` |

## 6. Mock data strategy

The mock data must reflect the new two-plane architecture while remaining deterministic and inspectable.

### 6.1 Mock capture scope

One `CaptureScope` with six `ChannelBinding`s, matching the fixture-backed six-camera sample.

### 6.2 Mock segments

For a 40.89-second source at 1-second logical chunks, produce approximately 40 segments per camera (some may be explicit absence with `CameraAbsence` reason). Segments carry:
- Requested vs effective interval (may differ at source boundaries)
- Content digest (deterministic SHA-256 from mock content)
- Quality observations (luminance, edge energy, freeze, cadence)

### 6.3 Mock windows

Produce windows with:
- 2-second width, 1-second hop → approximately 39 windows
- Purposes: QA_COARSE (all), QA_DENSE (every 5th), EVENT_PROPOSAL (selective), ACTION_DENSE (conditional), BOUNDARY_REFINEMENT (with ONSET/OFFSET refinement)
- Parent lineage for refinement windows
- Six-slot closure referencing segments or explicit absence

### 6.4 Mock inferences

Each window has 0–3 inferences (coarse, dense, action/boundary). Inferences carry:
- Attempt identity
- Terminal outcome (mostly SUCCEEDED, some SKIPPED_POLICY, one FAILED for demonstration)
- Evidence reference (mock artifact ID + digest)

### 6.5 Mock plan and closure

- `ExpectedWindowPlan` with all declarations appended before children
- `ExpectedWindowPlanSeal` at EOS (after last window)
- `WindowTerminalClosure` with reconciled outcomes per member
- `RecordingFinalizationMap` mapping incremental to final identities

### 6.6 Mock event stream

A single ordered `StreamEvent[]` array that replays in event-time order. The `simulateStream()` function emits events with realistic delays (not wall-time accurate, but perceptually plausible).

## 7. Work packages (frontend-only)

### WP-F0: Replace domain types

Deliver:
- New `types.ts` with all stream contract types
- Remove `NodeKind`, `RobataRun`, `StageMetrics`, `PipelineGroup`
- Add `StreamEvent`, `CaptureScope`, `StreamSegment`, `IncrementalWindow`, `StreamInference`, `ExpectedWindowPlan`, `WindowTerminalClosure`, `RecordingFinalizationMap`
- Keep `NodeStatus` and `STATUS_STYLE` for UI rendering, but map them to `TerminalOutcome`

Accept when TypeScript compiles with no errors and the new types cover all fields from the Python contracts.

### WP-F1: Rewrite store and hooks

Deliver:
- New `store.ts` with `StreamViewState`
- New `useWebSocket.ts` with `simulateStream()` emitting `StreamEvent[]`
- Event-time timeline state (watermark position, current window)
- Backpressure state (level, queue depth, oldest age)

Accept when mock events flow through the store and update UI reactively.

### WP-F2: Build two-plane layout

Deliver:
- `App.tsx` with two-pane layout (Plane A + Plane B)
- `TimelineBand` component showing event-time progression
- `WatermarkBar` showing current watermark and backpressure
- Plane toggle or stacked view

Accept when the layout renders without scroll issues and both planes are visible.

### WP-F3: Build plane components

Deliver:
- `CaptureScopeCard` — shows six channel bindings, authority IDs, policy versions
- `SegmentRow` — per-camera segment timeline with quality indicators
- `WindowCard` — shows window identity, interval, semantic SHA-256, parent lineage
- `InferenceCard` — shows attempt, outcome, evidence reference
- `PlanPanel` — append-only declarations with seal marker
- `TerminalClosurePanel` — reconciled outcomes per expected member
- `FinalizationPanel` — EOS mapping, barrier status

Accept when all components render mock data correctly and show semantic identity fields (SHA-256, keys, digests).

### WP-F4: Build subject detail drawer

Deliver:
- Replace `NodeDetailDrawer` with `SubjectDetailDrawer`
- Shows full subject with all identity fields
- Copy-to-clipboard for SHA-256 digests
- Links to governing ADR/schema sections

Accept when clicking any subject in any plane opens the drawer with complete identity.

### WP-F5: Add streaming simulation

Deliver:
- `simulateStream()` plays mock events with configurable speed
- Pause / resume / step controls
- Event log panel showing raw `StreamEvent` JSON
- Watermark animation

Accept when a full 40.89-second mock recording plays through in under 30 seconds of wall time.

### WP-F6: Document and freeze

Deliver:
- Updated `web/README.md` with new architecture mapping
- Component diagram in `web/docs/` (if not already present)
- Type-to-contract cross-reference table

Accept when a new developer can understand the two-plane model from README alone.

## 8. Files to modify, create, and delete

### Modify
- `web/src/types.ts` — replace domain model
- `web/src/store.ts` — replace store shape
- `web/src/App.tsx` — replace layout
- `web/src/hooks/useWebSocket.ts` — replace simulation
- `web/README.md` — update documentation

### Create
- `web/src/types/stream.ts` — new types (or inline in `types.ts`)
- `web/src/data/mock_stream_events.ts` — mock event stream
- `web/src/data/timeline.ts` — timeline bands
- `web/src/nodes/CaptureScopeNode.tsx`
- `web/src/nodes/SegmentNode.tsx`
- `web/src/nodes/WindowNode.tsx`
- `web/src/nodes/InferenceNode.tsx`
- `web/src/edges/EventEdge.tsx`
- `web/src/panels/SubjectDetailDrawer.tsx`
- `web/src/panels/TimelineBand.tsx`
- `web/src/panels/WatermarkBar.tsx`
- `web/src/panels/PlaneAView.tsx`
- `web/src/panels/PlaneBView.tsx`

### Delete
- `web/src/data/pipeline.ts`
- `web/src/data/groups.ts`
- `web/src/data/overview.ts`
- `web/src/nodes/GroupNode.tsx`
- `web/src/nodes/RobataNode.tsx`
- `web/src/edges/SchemaEdge.tsx`
- `web/src/panels/GroupExpandedView.tsx`
- `web/src/panels/NodeDetailDrawer.tsx`

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Backend contracts change before WP3/WP5 complete | Frontend types are mock-only; changes are TypeScript-level, not runtime. Track contract versions explicitly. |
| Two-plane layout is confusing | Provide clear labels, toggle, and a "what am I seeing?" tooltip. Default to Plane A (live view). |
| Mock data becomes stale | Generate mock data from a deterministic script that mirrors the Python contract validators. |
| Timeline with 39 windows is cluttered | Collapsible bands, zoom, and filter by purpose/outcome. |
| Semantic identity fields (SHA-256) overwhelm UI | Show truncated with full copy-to-clipboard. Use monospace font and muted color. |

## 10. Acceptance criteria

1. TypeScript compiles with zero errors
2. All new types map 1:1 to Python contract fields (no missing fields, no extra fields)
3. Mock data produces at least 10 windows, 6 segments per window, and 1 finalization
4. Two-plane layout renders without horizontal scroll on 1440px width
5. Simulation plays through a full mock recording in under 30 seconds
6. Subject detail drawer shows complete identity for any clicked subject
7. README documents the two-plane model for new developers

## 11. Next steps after this plan

1. **Get user approval** on this plan
2. **Implement WP-F0** (types) — smallest change, highest impact on contract validation
3. **Implement WP-F1** (store + hooks) — enables reactive mock data flow
4. **Implement WP-F2** (layout) — visual foundation
5. **Implement WP-F3** (plane components) — content
6. **Implement WP-F4** (detail drawer) — inspection
7. **Implement WP-F5** (simulation) — demo experience
8. **Implement WP-F6** (documentation) — freeze

The frontend remains `production_eligible=false` and `LOCAL_CONFORMANCE` throughout. It does not connect to a real backend until WP3–WP6 are complete and the WebSocket contract is frozen.
