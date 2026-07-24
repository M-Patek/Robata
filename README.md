# Robata

A contract-first streaming pipeline for six-camera video QA and physical-action event extraction.

> **Environment Notice**
> - **Production**: The `src/` directory contains all executable code. No dependency on `docs/`, `archive/`, or external documentation.
> - **Development**: Comprehensive design documents, ADRs, and WP progress are available in `docs/`.

---

## Overview

Robata implements a deterministic, replayable streaming architecture for processing egocentric six-camera video streams. The current implementation completes WP6 local conformance validation with the following characteristics:

| Attribute | Value |
|-----------|-------|
| Processing Model | Window-based streaming (2s windows, 1s hop) |
| Throughput | 2.162 rec-sec/wall-sec (WP6 smoke, fixture-backed) |
| Latency | p95 3.908s end-to-end |
| State Management | Durable Window DAG + SQLite work scheduler |
| Evidence Chain | 74 registered schemas, content-addressed |
| Replay | Exact replay verified (deterministic) |
| Evidence Class | `LOCAL_CONFORMANCE`, `NOT_PRODUCTION_QUALIFIED` |

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       Stream Ingestion                          │
│    Single-pass MCAP → Segment Manifest → Window Declaration     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Window DAG Pipeline                        │
│                                                                 │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                   │
│   │ Window  │────→│QA_COARSE│────→│QA_DENSE │────────┐          │
│   └─────────┘     └─────────┘     └─────────┘        │          │
│        │                              │              │          │
│        │                              └────────┐     │          │
│        │                                       ↓     ↓          │
│        │                              ┌───────────────────┐     │
│        │                              │   EVENT_PROPOSAL  │     │
│        │                              └─────────┬─────────┘     │
│        │                                        │               │
│        └────────────────────────────────────────┘               │
│                                                 ↓               │
│                                      ┌───────────────────┐      │
│                                      │ WINDOW_REDUCTION  │──────┼──→ Terminal
│                                      └───────────────────┘      │     Member
│                                                 │               │
│   After EOS + Export Barrier Complete:          ↓               │
│                                      ┌───────────────────┐      │
│                                      │    FINALIZATION   │──────┼──→ Recording
│                                      └───────────────────┘      │     V4
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Evidence & Persistence                     │
│                                                                 │
│          ┌──────────────────┐    ┌─────────────────────┐        │
│          │  Work Scheduler  │    │ Inference Ledger    │        │
│          │  (SQLite + DAG)  │    │ (Content-addressed) │        │
│          └──────────────────┘    └─────────────────────┘        │
│                   │                       │                     │
│                   └───────────┬───────────┘                     │
│                               ↓                                 │
│                   ┌───────────────────────┐                     │
│                   │  Terminal Closure     │                     │
│                   │  (Deterministic ID)   │                     │
│                   └───────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Window Model** | Fixed 2s windows, 1s hop | Bounded latency, incremental output |
| **DAG Topology** | 5+1 stage pipeline with explicit dependencies | Deterministic partial order, replay safety |
| **State Recovery** | Checkpoint at window boundary | Resume from last terminal, no re-dispatch |
| **Evidence Storage** | SQLite + content-addressed blobs | Local durability, exact replay |
| **Schema Governance** | 74 pinned schemas, atomic registration | Wire compatibility, evolution safety |
| **Lease Fencing** | Epoch + token per work item | Prevents split-brain in recovery |

### Determinism and Replay

The pipeline guarantees exact replay through the following mechanisms:

**Content-Addressed Inputs**
- Every window carries `window_semantic_sha256` computed from ordered six-camera segments
- Work items include `input_semantic_sha256` covering window hash + upstream dependencies
- Config SHA-256 pins the DAG topology and policy versions

**Dependency Tracking**
```
QA_COARSE: depends on WINDOW
QA_DENSE:  depends on QA_COARSE
EVENT_PROPOSAL: depends on [QA_COARSE, QA_DENSE]
WINDOW_REDUCTION: depends on [WINDOW, QA_COARSE, QA_DENSE, EVENT_PROPOSAL]
FINALIZATION: depends on all WINDOW_REDUCTION terminals
```

**Terminal Evidence Acceptance**
- Worker completes work → stores `pending_terminal_json` with lease fence
- Execution scheduler commits SUCCEEDED state
- Stream scheduler accepts terminal → binds `terminal_member_json`
- Two-phase commit ensures exactly-once terminal semantics

### Backpressure and Flow Control

Admission control prevents unbounded queue growth:

| Threshold | Trigger | Action |
|-----------|---------|--------|
| queue_depth > 256 | Too many windows in flight | Reject new window declarations |
| oldest_age > 30s | Stalled work detected | Signal upstream throttle |
| backlog_slope > 128 | Arrivals exceed throughput | Trigger admission denial |

Backpressure propagates upstream: window declaration rejected → planner retains watermark → source ingestion pauses.

### Causal Consistency (RR4)

Window reduction produces causally consistent output through dependency ordering:

1. **QA_COARSE** and **QA_DENSE** run independently once inputs ready
2. **EVENT_PROPOSAL** waits for both QA stages (join semantics)
3. **WINDOW_REDUCTION** aggregates all 4 upstream stages
4. No window N+1 reduction begins before window N reduction completes (monotonic chain)

This ensures `recording_v4` output maintains stream temporal order without cross-window anomalies.

### Execution Guarantees

- **Determinism**: Same input → same output (verified by exact replay)
- **Fail-closed**: Invalid evidence → `INDETERMINATE`, no downstream propagation
- **Idempotency**: Replay creates zero duplicate business facts or outbox rows
- **Fencing**: Lease epoch + token prevents stale work completion

---

## Repository Structure

```
robata/
├── src/robata/                 # Core implementation
│   ├── contracts/              # Schema definitions, wire types
│   ├── adapters/               # SQLite, MCAP, frame materialization
│   ├── application/canonical/  # Streaming pipeline, composition root
│   ├── inference/              # Provider-neutral inference boundary
│   ├── queue/                  # Work scheduler, backpressure
│   └── runtime/                # Observability, profiling
├── schemas/                    # JSON Schema catalog (74 entries)
├── tests/                      # Test suite (1,137 tests)
├── scripts/                    # CLI entry points
├── docs/                       # Development documentation
│   ├── architecture/           # WP progress, design records
│   ├── adr/                    # Architecture Decision Records
│   └── README.md               # Documentation index
├── archive/old_mvp/            # Historical (non-normative)
├── README.md                   # This file
├── REQUIREMENTS.md             # Product requirements (Chinese)
└── uv.lock                     # Locked dependencies
```

---

## Current Status

### Completed (WP0-WP6)

| WP | Description | Evidence |
|----|-------------|----------|
| WP0 | Instrumented baseline | Profile reconciliation |
| WP1 | Incremental identities | Schema registry, ADR 0015 |
| WP2 | Single-pass media ingest | Bounded rings, MP4 export |
| WP3 | Durable window scheduler | Window DAG, backpressure |
| WP4 | Batch inference persistence | Evidence ledger, barriers |
| WP5 | Incremental evidence commit | Recording finalization |
| WP6 | Streaming qualification | **Passed** (see below) |

### WP6 Acceptance Results

| Gate | Target | Actual | Status |
|------|--------|--------|--------|
| Fresh elapsed | <= 204.452s | 118.750s | PASS |
| Fresh RTF | <= 5.000 | 2.908 | PASS |
| 30-min capacity | >= 1.860 | 2.162 | PASS |
| p95 latency | <= 5.000s | 3.908s | PASS |
| Backlog growth | None | 0 | PASS |

**Test Coverage**: 1,800 windows, 9,001 work items, 6 injected failures recovered.

### Not Covered (External Dependencies)

| Item | Blocker | Impact |
|------|---------|--------|
| Real model (Qwen 7B + vLLM) | No qualified endpoint | Throughput unvalidated |
| Cloud deployment (R2/RunPod/Supabase) | No infrastructure | Production topology unknown |
| Long-soak stability | Time | Durability unproven beyond 30min |
| Failover design | O-14 policy | Recovery ownership undefined |

---

## Quick Start

### Prerequisites

- Python >=3.12, <3.14
- uv (dependency management)
- (Optional) FFmpeg, PyAV for MCAP processing

### Installation

```bash
# Sync locked dependencies
uv sync --locked

# Verify installation
python -m pytest -q --collect-only | tail -5
```

### Local Execution

Fixture-backed (no network):
```bash
python scripts/run_canonical_fixture.py \
    tests/fixtures/canonical/source-recording.json \
    --state-dir tmp/canonical-state \
    --run-key primary
```

Raw MCAP (authorized source required):
```bash
python scripts/run_canonical_mcap.py \
    data/source/sample.mcap \
    --mapping-config config/genrobot-observed-v0.json \
    --state-dir tmp/mcap-state \
    --run-key primary \
    --max-duration-seconds 180
```

### Verification

```bash
# Full validation (development)
python -m pytest -q -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
python -m mypy
python scripts/verify_schema_registry.py

# Minimal validation (production)
python -m pytest -q
```

---

## Schema Evolution

Schema publication commands (development):

```bash
# Single schema
python scripts/register_schema.py --schema-id my.schema --version 1.0.0

# Bundle (atomic)
python scripts/register_schema_bundle.py --manifest bundle.json

# Evolution (target + upcasters)
python scripts/register_schema_evolution.py --manifest evolution.json

# Verification
python scripts/verify_schema_registry.py
python scripts/check_schema_immutability.py --baseline-ref $SCHEMA_BASELINE_REF
```

**Constraints**:
- Published `(schema_id, version)` immutable
- `compatibility_mode=NONE` (no retroactive chains)
- 74 schemas registered, `upcasters=[]`

**Schema Categories**

| Category | Count | Purpose |
|----------|-------|---------|
| Stream contracts | 12 | Window declaration, work plans, terminal closure |
| QA inference | 18 | Coarse, dense, supplemental evidence |
| Event extraction | 22 | Candidate, proposal, boundary refinement |
| Persistence | 15 | Recording V4, outbox, evidence chain |
| Internal | 7 | Scheduler state, backpressure metrics |

## Storage Architecture

### SQLite Schema (Stream Work Ledger)

```sql
-- Expected window plan with EOS seal
stream_work_plan(plan_key, plan_json, seal_json, terminal_closure_json, ...)

-- Per-window declarations with chain SHA-256
expected_window(plan_key, ordinal, declaration_json, window_json, terminal_member_json)

-- Work items with publication state
stream_work_item(work_item_id, plan_key, stage, expected_ordinal, 
                 publication_state, terminal_evidence_json, pending_terminal_json)
```

### Content-Addressed Blobs

Artifacts stored by SHA-256 digest:
- Inference responses (QA results, event proposals)
- Terminal evidence (JSON with schema refs)
- Source segments (MCAP chunks)

Lookup: `semantic_sha256 → filesystem path or object storage URI`

### Recovery Procedure

1. Load all `stream_work_plan` rows for run_id
2. Verify contiguous window ordinals (0..N)
3. Replay internal execution projections to work scheduler
4. Reconcile pending terminals with execution state
5. Resume from earliest non-terminal work item

Recovery is idempotent: re-running on already-completed graph produces no changes.

---

## Development Documentation

| Document | Path | Audience |
|----------|------|----------|
| WP Progress & Metrics | `docs/architecture/streaming-throughput-next-iteration-v1.md` | Developers |
| Implementation Status | `docs/current-implementation-status.md` | Developers |
| Architecture Decisions | `docs/adr/` | Developers |
| Requirements (Chinese) | `REQUIREMENTS.md` | Product |

**Note**: Production deployments do not require these documents.

---

## Evidence Classes

| Class | Meaning | Production Eligible |
|-------|---------|---------------------|
| `LOCAL_CONFORMANCE` | Local mechanism verified | No |
| `SYNTHETIC_LOCAL` | Simulation-based | No |
| `NOT_MEASURED` | Awaiting representative load | No |
| `MEASURED` | Qualified workload evidence | Pending |

Current status: **LOCAL_CONFORMANCE**, **NOT_PRODUCTION_QUALIFIED**.

---

## License

Copyright (c) 2026 Robata Contributors.

This software is provided for evaluation and development purposes.
Production use requires explicit qualification of real models,
infrastructure, and operational policies not included in this repository.

See LICENSE for full terms (if present) or contact the maintainers.

---

## Contact

For production qualification inquiries: [maintainer contact]
For development issues: Open an issue with `LOCAL_CONFORMANCE` evidence.
