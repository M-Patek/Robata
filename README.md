# Robata

A contract-first streaming pipeline for six-camera video QA and physical-action event extraction.

> **Environment Notice**
> - **Executable contracts**: `schemas/schema-catalog.json` and `schemas/v*/` govern published wire contracts; `src/`, `tests/`, and `conformance/` govern executable behavior.
> - **Agent navigation**: Start at `AGENTS.md`, then use the blueprint template and module guides in `governance/`.
> - **Local archive**: `archive/` is non-authoritative historical context. It cannot define or override a contract.

> **Deployment posture**
> - This repository currently provides local-conformance behavior and production-shaped adapter boundaries. It is not a deployed production topology or a production-capacity claim.
> - `governance/REQUIREMENTS.md` records target architecture and qualification assumptions. It does not authorize a release or override schemas, source, tests, or conformance fixtures.
> - Read [Deployment Status and Target Topology](#deployment-status-and-target-topology) before provisioning cloud resources or exposing an endpoint.

---

## Overview

Robata implements a deterministic, replayable streaming architecture for processing egocentric six-camera video streams. The checked-in local-conformance slice has the following characteristics:

| Attribute | Value |
|-----------|-------|
| Processing Model | Window-based streaming (2s windows, 1s hop) |
| Throughput | 2.162 rec-sec/wall-sec (dated fixture-backed smoke snapshot; not capacity evidence) |
| Latency | p95 3.908s (dated fixture-backed smoke snapshot) |
| State Management | Durable Window DAG + SQLite work scheduler |
| Evidence Chain | 74 registered schemas, content-addressed |
| Replay | Exact replay verified (deterministic) |
| Evidence Class | `LOCAL_CONFORMANCE`, `NOT_PRODUCTION_QUALIFIED` |
| Deployment Scope | Local worker and local read-only workbench; optional cloud adapters are not yet composed into a service |

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

## Deployment Status and Target Topology

### Current Status

Robata currently has a local-conformance pipeline and production-shaped adapter
boundaries. It does not yet have a production composition root, a cloud deployment,
or qualified service-capacity evidence. Qualification artifacts deliberately retain
`production_eligible: false`; technical evidence cannot self-authorize a release.

| Boundary | Available in the repository | Still deployment-owned or unqualified |
|----------|-----------------------------|---------------------------------------|
| Canonical pipeline | Deterministic local worker, SQLite state, replay, recovery, and outbox proofs | Multi-host composition, durable operational ownership, and representative load |
| Cloudflare R2 | HTTPS S3-compatible adapter with versioned idempotent writes, exact-byte checks, range reads, and reconciliation logic | Bucket policy, lifecycle, real object I/O, cache retention, and R2-to-worker throughput |
| PostgreSQL / pgvector | Optional `psycopg` runtime, `verify-full` TLS, required RLS, and separate primary/worker roles | Reviewed DDL and policy deployment, tenant-claim integration, real database verification, and migrations |
| Redis / broker | Task-queue and idempotent-outbox adapter boundaries | A durable broker selection, composition, operations, and real fault evidence |
| RunPod | HTTPS provider transport with authentication, idempotency keys, bounded retries, concurrency controls, and qualification bindings | vLLM handler, model image, endpoint configuration, GPU topology, and real endpoint qualification |
| Web workbench | Local read-only REST/WebSocket explorer over committed SQLite completions | Authentication, tenant authorization, public API contract, production data source, and an Internet-facing deployment |
| Docker / Compose | Read-only, one-shot local CPU fixture and recovery workers | API, frontend, GPU service, broker, persistent application services, ingress, and production health checks |

The production targets in [governance/REQUIREMENTS.md](governance/REQUIREMENTS.md)
are useful deployment and qualification context. They are not a claim that the
target stack, 500 recording-hours/day, T+1/T+3, or two-H100 capacity is already
available.

### Target Topology (Operator-Owned, Not Yet Deployed)

```text
                         Browser
                            |
                TLS proxy + authenticated API
                            |
                 Web/API application service
                   |                    |
                   |                    +---- Supabase/PostgreSQL
                   |                           Auth, result data, pgvector/RLS
                   |
                   +---- durable broker / task authority
                            |
                     CPU/NVMe worker
                       |
                       +---- RunPod GPU endpoint
                       |       vLLM / model inference
                       |
                  Cloudflare R2
          source video, immutable artifacts, frame-cache objects
```

This is a target deployment boundary, not the current `compose.yaml`:

- **R2 is object storage only.** It cannot run the Docker worker, host the API, or expose an application port.
- **RunPod is a provider boundary.** A GPU pod does not supply the product API, authentication, durable queue, database policy, or release evidence.
- **The browser must not receive R2 administrative credentials, database passwords, or RunPod API keys.** Secrets stay in a deployment secret manager and are injected only into the required server-side process.
- **The local SQLite state and local Web API are not cluster authority.** Do not mount a developer state directory into a public service or treat the local explorer as the production API.

### Production Release Boundary

An environment is eligible for a governed release only after the target topology is
implemented and the representative external gates are evidenced. Local tests, adapter
construction, and a successful configuration preflight remain useful prerequisites,
but they do not establish storage durability, model quality, provider capacity, or
service-level compliance.

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
├── tests/                      # Test suite (run `pytest --collect-only` for the current count)
├── scripts/                    # CLI entry points
├── governance/                 # Agent-friendly blueprint template and module guides
├── README.md                   # This file
└── uv.lock                     # Locked dependencies
```

---

## Local Conformance Snapshot

This is dated local evidence, not a release or production-capacity claim. Re-run the documented checks to establish the current working-tree result.

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

### External and Production Qualification Status

The following table separates code that exists in this repository from evidence
that must be produced in a real staging or production-like environment.

| Area | Current repository evidence | Required before a governed release |
|------|-----------------------------|------------------------------------|
| Real model and vLLM | Provider-neutral and RunPod transport boundaries; no qualified endpoint | Pinned model/runtime/topology, response-contract check, quality review, saturation, latency, and cost evidence |
| Cloudflare R2 | Optional `boto3` adapter with local-double tests and configuration preflight | Bucket policy, lifecycle, real PUT/GET/range/reconcile faults, retention, and throughput evidence |
| PostgreSQL / pgvector | TLS/RLS/primary-worker adapter boundary and target-verification path | Reviewed DDL, tenant policy, migrations, real RLS isolation test, backup, and restore evidence |
| Redis / broker | Task-queue and idempotent-outbox adapter boundaries | Chosen durable service, composition, availability model, fault handling, and operator ownership |
| Web/API | Local committed-run explorer only | Authenticated, authorized, observable public service with a production data source and TLS ingress |
| Capacity and deadlines | Dated fixture-backed local snapshot | Representative 500 recording-hours/day equivalent, T+1/T+3, p95/p99, utilization, and cost evidence |
| Reliability and recovery | Local restart/replay and injected-fault coverage | Storage/provider faults, backup/restore, incident ownership, and the declared soak duration |

---

## Quick Start

### Prerequisites

- Python >=3.12, <3.14
- uv (dependency management)
- Node.js (required only for the local Vite workbench)
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
    /absolute/path/to/authorized-recording.mcap \
    --mapping-config config/genrobot-observed-v0.json \
    --state-dir tmp/mcap-state \
    --run-key primary \
    --max-duration-seconds 180
```

### Local Committed Run Workbench

The web workbench exposes only immutable, committed primary completions. It does
not control workers, open a writable repository, or synthesize in-flight
pipeline state. After producing local state with one of the commands above,
start the API in one terminal:

```bash
uv sync --locked --extra web
python scripts/run_web_api.py --state-dir tmp/canonical-state
```

Then start the Vite client in another terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies the versioned REST and WebSocket
paths to `http://127.0.0.1:8000`; a same-origin reverse proxy or the two
`VITE_ROBATA_*_BASE` variables can be used outside local development.

The local API defaults to a loopback listener and provides no authentication or
tenant authorization. A same-origin proxy or base URL is only a transport choice,
not a public-deployment security model. Do not expose this explorer to the Internet
or connect it to production data until a separate authenticated, authorized service
and production read model exist.

### Containerized Local Worker

The checked-in container is a one-shot CPU worker for the existing local-conformance
composition. It deliberately has no public HTTP port. The default service uses disposable
in-container state, so each invocation is a fresh fixture regression:

```bash
docker compose up --build canonical-fixture
```

An explicit recovery profile uses a persistent named volume and initializes its ownership
before the non-root worker starts:

```bash
docker compose --profile recovery up --build canonical-recovery
```

Use the recovery profile only when replaying the same durable state is intended. Remove its
project volume before an independent recovery scenario. Both workers run read-only as UID
`10001`; application layers are root-owned and only the declared state mount is writable.

This is local regression execution, not a production topology or qualification result.
R2, PostgreSQL/pgvector, Redis or another broker, and RunPod are outbound dependencies
that must be explicitly injected by a production composition root; they are not silently
created by Compose. Cloudflare R2 is object storage and cannot host this Docker worker
or expose an application port.

The checked-in image is not an API, frontend, GPU, or vLLM image. It does not package
the `web/` client, run a public service, declare GPU resources, or provision a database
or broker. Do not promote this Compose project to staging or production unchanged.

### Optional Adapter Preflight

The explicit R2 and pgvector variable names are listed in
[`.env.example`](.env.example). Copy it to a local ignored file, replace every placeholder,
then choose that file explicitly. The command never auto-loads `.env` or injects it into
Compose:

```bash
Copy-Item .env.example .env
uv sync --locked --extra r2 --extra pgvector
python scripts/preflight_optional_adapters.py --r2 --pgvector --env-file .env
```

The pgvector runtime accepts only `verify-full` TLS and requires separate primary and worker
CA certificate paths (`PGVECTOR_SSLROOTCERT` and `PGVECTOR_WORKER_SSLROOTCERT`) readable by
the process. The non-verifying `require` and hostname-skipping `verify-ca` modes are rejected.

`--verify-pgvector` is different: it connects to the configured target and verifies
pgvector, the configured vector dimension, RLS flags, policy presence, and the separate
worker role. Run it only after reviewed DDL and RLS policy have been applied.

The R2 path in this command validates configuration and constructs the optional SDK client;
it performs no R2 object read, write, range-read, or reconciliation operation. The repository
does not yet provide an equivalent RunPod deployment preflight. Real R2 faults, the RunPod
endpoint, and all provider/storage capacity claims remain separate external qualification work.
A successful preflight remains `NOT_MEASURED`.

### Pre-Production Inputs

Provision the following outside the repository before building a staging composition. Use
separate staging and production resources. Do not put any secret in Git, a frontend build,
or a browser-accessible environment variable.

| Boundary | Operator-provided inputs | Minimum safety boundary |
|----------|--------------------------|-------------------------|
| Cloudflare R2 | Per-environment bucket and prefix, endpoint, scoped access key, lifecycle/retention policy | Separate prefixes; least-privilege server-side credentials; tested restore and deletion procedure |
| PostgreSQL / Supabase | Project/database endpoint, CA certificate, app and worker roles, RLS tenant policy, pgvector dimension/index choice | `verify-full` TLS; distinct app/worker identities; policy review before target verification |
| Redis or broker | Managed endpoint, TLS/authentication, retention/dead-letter and availability settings | No implicit local queue; ownership, retry, and incident behavior defined |
| RunPod | API credential, HTTPS endpoint, handler image, model/version/precision, topology, concurrency and timeout limits | Secret remains server-side; endpoint contract, model license, and failure behavior are pinned |
| Application edge | CPU/NVMe host, image registry, domain/DNS, TLS proxy, secret manager, logging/metrics and alert ownership | Public API is authenticated and authorized; no public SQLite explorer or direct admin credential exposure |
| Qualification corpus | Representative recordings, governed labels, workload manifest, acceptance thresholds, and cost constraints | Data lineage and tenant/data handling are reviewed before testing |

### Staging Verification Sequence

1. Build separate application, worker, and GPU/handler deployment artifacts. Do not reuse the
   checked-in local Compose project as a staging manifest.
2. Apply reviewed PostgreSQL/pgvector DDL and RLS policy, then run
   `python scripts/preflight_optional_adapters.py --pgvector --verify-pgvector --env-file <secure-env-file>`.
3. Exercise an isolated R2 prefix with real immutable write, HEAD, full GET, range GET,
   reconciliation, retry, and deletion/retention scenarios.
4. Pin a RunPod model/runtime/topology and verify request/response, idempotency, timeout,
   partial failure, concurrency, and evidence capture against the real endpoint.
5. Run a representative one-hour workload before estimating capacity, then execute the declared
   soak, recovery, backlog-drain, and cost measurements under the frozen scope.

### Release Qualification Gates

The [P15 external gates](governance/BLUEPRINT.md#p15---run-the-representative-pareto-and-external-qualification-gates)
are the release-evidence checklist. An unexecuted gate remains `NOT_MEASURED`; a measured
failure remains recorded as a failure; no technical artifact can self-promote the release.

| Gate | Required decision evidence |
|------|----------------------------|
| E0 | Frozen code, schema, workload, policy, hardware, provider, storage, and cost scope |
| E1 | Governed QA/event/boundary quality and calibration evidence on frozen labels |
| E2 | Target media/storage parity, durability, and object-store fault/reconciliation evidence |
| E3 | Real model/runtime/hardware correctness, saturation, latency, retry, and cost evidence |
| E4 | Representative arrivals, recovery/fault injection, backlog drain, and declared soak evidence |
| E5 | 500 recording-hours/day equivalent, T+1/T+3, utilization, p95/p99, and unit-cost evidence |
| E6 | Independent go/no-go review with security, retention, incident, and unresolved-risk evidence |

### Verification

```bash
# Full validation (development)
python -m pytest -q -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
python -m mypy
python scripts/verify_schema_registry.py

# Minimum local regression baseline (not production qualification)
python -m pytest -q
uv lock --check
docker compose config --quiet
```

The checked-in CI validates the Python tree, schemas, documentation, and a tracked-source
archive. It does not build the Vite client, build or publish a container image, deploy an
environment, or run cloud qualification. Add those controls to a separate deployment pipeline
before treating a Git push as a release.

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

An object-storage URI is a contract abstraction, not proof that the local canonical
composition has written to Cloudflare R2. The checked-in local path uses local state and
does not inject an R2-backed frame cache or object-store composition. A staging deployment
must provide that composition, configure bucket/prefix ownership, and qualify real storage
durability and reconciliation before treating R2 as authoritative.

Lookup: `semantic_sha256 → filesystem path or object storage URI`

### Recovery Procedure

1. Load all `stream_work_plan` rows for run_id
2. Verify contiguous window ordinals (0..N)
3. Replay internal execution projections to work scheduler
4. Reconcile pending terminals with execution state
5. Resume from earliest non-terminal work item

Recovery is idempotent: re-running on already-completed graph produces no changes.

---

## Agent Development Guide

| Entry point | Path | Audience |
|---|---|---|
| Start guide | `AGENTS.md` | Any development agent |
| Architecture blueprint template | `governance/BLUEPRINT_TEMPLATE.md` | Architecture agents |
| Actual blueprint (created on demand) | `governance/BLUEPRINT.md` | Architecture and phase agents |
| Module guides | `governance/modules/` | Phase implementation agents |

Local architecture, requirement, and historical materials remain optional context. They cannot be required to navigate or validate a clean checkout.

---

## Evidence and Qualification Status

Evidence class and measurement status are separate. Neither a local result nor a
successful adapter configuration automatically grants release authority.

### Evidence Classes

| Class | Meaning | Release implication |
|-------|---------|---------------------|
| `LOCAL_CONFORMANCE` | Deterministic local behavior and conformance fixtures | Not production evidence |
| `LOCAL_BENCHMARK` | Measured local benchmark under an explicit scope | Not representative or production evidence |
| `REPRESENTATIVE_BENCHMARK` | Representative workload measurement without external release qualification | Requires the remaining external gates |
| `EXTERNAL_QUALIFICATION` | Evidence from a real provider, storage target, hardware, or governed dataset | Pending gate review and release decision |
| `PRODUCTION_QUALIFIED` | Classification available only after governed qualification evidence | Does not replace the independent E6 release decision |

### Measurement Status

| Status | Meaning |
|--------|---------|
| `NOT_MEASURED` | The required representative or external observation has not been run |
| `MEASURED` | An observation exists with a bound scope; it can still fail its threshold or remain pending review |

Current repository status: `LOCAL_CONFORMANCE`; real hardware, external storage/provider,
representative labels, long soak, and production capacity remain `NOT_MEASURED`.
The repository is `NOT_PRODUCTION_QUALIFIED`.

---

## License

Copyright (c) 2026 Robata Contributors.

This software is provided for evaluation and development purposes.
Production use requires explicit qualification of real models,
infrastructure, and operational policies not included in this repository.

See [LICENSE](LICENSE) for full terms, or contact the maintainers via a tracked GitHub issue.

---

## Contact

For production qualification inquiries: open a GitHub issue on
[M-Patek/Robata](https://github.com/M-Patek/Robata) with the `production-qualification`
label and the relevant evidence class.
For development issues: open an issue and attach your `LOCAL_CONFORMANCE` evidence
(commit, run key, and reproduction command).
