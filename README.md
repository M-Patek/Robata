# Robata

**Robata is an evidence-first robotics video intelligence platform.** It turns synchronized
robot recordings into durable, replayable, reviewable facts while keeping model inference,
media storage, scheduling, and publication behind explicit contracts.

> **Current status (August 10, 2026)**
> - The repository is **runnable and locally qualified for bounded conformance/benchmark
>   scopes**.
> - The default new-perception route is `mage_stream_vnext_v1`; the historical Qwen/window
>   route is retained only through an explicit `legacy_window_v1` selection.
> - PostgreSQL/Supabase, R2, pgvector, and RunPod production boundaries are implemented as
>   a canonical composition and admission-gate sequence, but they are **not a deployed
>   service and do not constitute production qualification**.
> - No route currently has `production_eligible=true`. Real H100/RunPod execution, real R2
>   I/O, representative labels, long-soak reliability, and 500 recording-hours/day capacity
>   remain `NOT_MEASURED`.

## What Robata provides

- **Evidence-first ingestion:** source identity, timestamp alignment, segment/window manifests,
  and immutable input references.
- **Provider-neutral perception:** a Mage stream route and an explicitly admitted legacy Qwen
  window route can produce model artifacts without changing the downstream fact contracts.
- **Deterministic projection:** QA, event, evidence, temporal reconciliation, fusion, and
  review decisions are derived from persisted model observations rather than hidden prompt
  cascades.
- **Durable execution:** leases, epochs, fencing, idempotency, retries, completion barriers,
  and outbox delivery prevent duplicate or stale work from becoming published facts.
- **Content-addressed evidence:** raw responses and derived artifacts are addressed by exact
  SHA-256 bytes; a storage location is never treated as the identity of its contents.
- **Qualification boundaries:** local conformance, local benchmarks, external qualification,
  canary/shadow routing, and production release decisions are represented separately.

## Current status at a glance

| Area | Status | What that means |
|---|---|---|
| Local canonical conformance | **Runnable** | SQLite/MCAP or fixture execution, deterministic replay, recovery, and read-only committed-run exploration are checked in. |
| Mage stream vNext | **Local benchmark / HOLD** | Native video-segment execution, observation projection, durable vNext scheduling, and bounded qualification artifacts exist; semantic production admission is not granted. |
| Qwen serial control | **Local control baseline** | Stable serial behavior and exact control parity are available for regression and rollback comparison. |
| Qwen Hybrid Batch4 | **Local throughput candidate / HOLD** | Strongest local QA-only throughput result; it is not a labeled semantic-quality or production-capacity proof. |
| PostgreSQL/Supabase canonical authority | **Implemented composition** | Migrations, adapters, roles, RLS checks, completion/evidence/outbox boundaries, and verification commands exist. A real deployment and restore exercise are still required. |
| Cloudflare R2 | **Implemented adapter boundary** | Immutable-write, exact-byte verification, range-read, and reconciliation logic exist. A real bucket, lifecycle, retention, and network test are still required. |
| pgvector | **Derived projection** | It supports retrieval/read-model use; it is not the canonical scheduler, completion, evidence, or outbox authority. |
| RunPod | **Pinned provider boundary** | Primary/candidate bindings, request identity, retries, and preflight checks exist; no real endpoint/H100 qualification is recorded here. |
| Web workbench | **Local read-only** | It explores committed local runs over loopback. It has no production authentication, tenant authorization, or public-service claim. |
| Production release | **Not qualified** | A successful local test, Docker build, or adapter preflight cannot self-authorize release. |

## Architecture at a glance

Robata has two intentionally different execution routes and one production composition boundary.
They share contracts, identity, evidence, and release rules; they do not silently share model
assumptions.

```text
LOCAL / QUALIFICATION ROUTES

  MCAP or fixture
       |
       v
  ingest + timestamp alignment + immutable media manifest
       |
       +--> mage_stream_vnext_v1 (default for new perception runs)
       |       non-overlap causal segments
       |          -> Mage native video endpoint
       |          -> one MageObservation per context
       |          -> deterministic QA / event / evidence projectors
       |          -> temporal reconcile -> multi-view fusion -> review/refine request
       |          -> content-addressed artifacts and completion evidence
       |
       +--> legacy_window_v1 (explicit compatibility/control route only)
               2 s windows / 1 s hop
                  -> Qwen QA/event/evidence stages
                  -> window reduction -> completion/evidence/outbox

PRODUCTION ADMISSION COMPOSITION (operator-owned; not a worker loop)

  PostgreSQL/Supabase canonical authority
       +-- scheduler, stream state, completion, evidence, barriers, review, outbox
  R2 immutable artifact mirror
       +-- verified raw-provider bytes and replay/reconciliation receipts
  pgvector derived projection
       +-- retrieval/read model only; never canonical truth
  RunPod primary/candidate binding
       +-- model endpoint transport, identity, retry, and qualification boundary
```

### The stable local route

The historical local conformance path remains valuable because it is deterministic and easy to
replay. It uses a bounded window DAG, SQLite work state, MCAP/fixture adapters, and an offline
vision adapter. The route is now selected explicitly as `legacy_window_v1`; new callers should
not infer that a Qwen stage name is the product contract.

### The Mage stream route

`mage_stream_vnext_v1` is provider-neutral at the scheduler boundary. Its physical model call is
an observation step over a causal video context. QA, event, and evidence are deterministic
projections of that observation. A refinement request is a bounded exception handoff; it is not
an implicit second model cascade. Mage recurrent/codec state is an optimization cache, not
authoritative durable state: recovery replays from the explicit segment/context manifest.

### The production boundary

The production composition constructs and verifies the canonical PostgreSQL/R2/pgvector/RunPod
graph. It does **not** automatically ingest a source, run a generic source-to-publication worker,
serve a public API, or prove provider quality/capacity. A source-specific worker must use the
shared production root after all admission gates pass.

## Local model qualification snapshot

The following figures are retained local measurements, not production promises. They were
collected on a single RTX 4060 Laptop unless the cited artifact states otherwise.

| Route | Workload | Recurring wall / result | Interpretation |
|---|---|---:|---|
| **Qwen serial control** | 6 cameras, about 40.8335 s each, 51 QA calls | 247.995 s; `0.988x` camera RTF (`0.165x` recording RTF) | Exact serial control baseline; not a semantic ground-truth score. |
| **Qwen Hybrid Batch4** | Same six-camera QA corpus | 65.588 s median; `3.728x` camera RTF; about `3.78x` serial speedup | Best local QA-only throughput candidate; still production HOLD. |
| **Mage Provider V2 / native DCVC stream** | One camera, 5 x 8 s contexts | 21.962 s; `1.821x` camera RTF | Native stream route is runnable; model load `38.368 s` is recorded separately from recurring wall. |
| **Mage traditional codec, 8 canvases** | One camera, 5 x 8 s contexts | 32.046 s; `1.248x` hot RTF | **HOLD:** retained output calls green fabric a “green book”. |
| **Mage traditional codec, 16 canvases** | One camera, 5 x 8 s contexts | Not decision-eligible | **STOP:** 256-token exhaustion and truncated non-JSON output. |
| **Mage fixed-frame control** | One camera, 5 x 8 s; 6 frames/context | 20.428 s; `1.958x` RTF; strict projection `5/5` | Diagnostic baseline only; codec and recurrent stream state are intentionally disabled. |

### How to read the numbers

- **Camera RTF** = total camera-seconds divided by recurring wall time.
- **Recording RTF** = recording duration divided by recurring wall time. For six cameras,
  camera-seconds are roughly six times recording-seconds.
- The Qwen figures and Mage figures above are **not directly comparable**: Qwen uses six
  cameras and a 51-call QA workload; the Mage routes use one camera and five 8-second contexts.
- Model load/cold-start time is shown separately when the report exposes it. It must be included
  in a cold-start SLA, but not silently added to a recurring warm-stream RTF.
- Strict JSON, raw/normalized parity, model-agreement F1/IoU, and downstream determinism are
  structural evidence. They are not human-labeled semantic accuracy. The common Qwen/Mage
  comparison is explicitly `UNLABELED_MODEL_AGREEMENT_ONLY`.

Authoritative local summaries:

- [`docs/local-model-qualification-summary-2026-08-10.md`](docs/local-model-qualification-summary-2026-08-10.md)
- [`docs/ROBATA_25X_ROUTE_DECISION_2026-08-09.md`](docs/ROBATA_25X_ROUTE_DECISION_2026-08-09.md)
- [`docs/qwen-native-batch-qualification-2026-08-09.json`](docs/qwen-native-batch-qualification-2026-08-09.json)
- [`docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json`](docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json)
- [`docs/mage-fixed-frame-native-prompt-qualification-2026-08-10.json`](docs/mage-fixed-frame-native-prompt-qualification-2026-08-10.json)

## Production composition and deployment boundary

`compose.production.yaml` is an **admission and verification sequence**, not a complete public
service deployment. It is deliberately split into attributable, one-shot gates:

1. `canonical-migrate` — apply the reviewed PostgreSQL migrations.
2. `canonical-postgres-verify` — verify database shape, roles, TLS/RLS expectations, and
   canonical tables.
3. `optional-adapter-preflight` — validate R2, pgvector, and paired RunPod configuration; it
   does not dispatch inference or perform real R2 object I/O.
4. `canonical-runtime-verify` — construct the exact production adapter graph and verify the
   mounted route/release artifacts.
5. `canonical-r2-reconcile` (operator profile) — bounded reconciliation of staged raw artifacts
   when required by the migration/recovery procedure.

Run the gates only with reviewed secrets and pinned qualification artifacts:

```powershell
docker compose --env-file .env.production -f compose.production.yaml build
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-migrate
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-postgres-verify
docker compose --env-file .env.production -f compose.production.yaml run --rm optional-adapter-preflight
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-runtime-verify
```

If migration `0005` leaves staged raw responses to reconcile:

```powershell
docker compose --env-file .env.production -f compose.production.yaml run --rm canonical-r2-reconcile
```

Passing these gates verifies wiring and admission prerequisites. It does **not** launch a
GPU worker, dispatch a model request, prove cloud durability, qualify semantic quality, or prove
500 recording-hours/day, T+1/T+3, 24-hour soak, or cost targets.

The intended target boundary is:

```text
Authenticated API / web service
          |
          +--> PostgreSQL/Supabase canonical authority
          +--> durable broker / task authority
          +--> CPU/NVMe source worker
                    +--> RunPod GPU endpoint
                    +--> Cloudflare R2 artifact mirror
```

R2 cannot run Docker or host the API. RunPod supplies a GPU/provider boundary, not product
authentication, canonical state, retention policy, or a public application port. Secrets stay
server-side; the browser must never receive database, R2, or RunPod credentials.

For the full operator procedure see
[`docs/operations/postgres-supabase-production.md`](docs/operations/postgres-supabase-production.md).

## Quick start: local conformance

### Prerequisites

- Python `>=3.12,<3.14`
- `uv`
- Node.js only for the Vite workbench
- FFmpeg/PyAV/MCAP extras when processing an authorized MCAP source
- Model weights are external inputs; they are not stored in this repository.

### Install and verify

```powershell
uv sync --locked --dev
uv run pytest -q --collect-only
uv run python scripts/verify_schema_registry.py
```

### Fixture-backed canonical run (offline)

```powershell
uv run python scripts/run_canonical_fixture.py `
  tests/fixtures/canonical/source-recording.json `
  --state-dir tmp/canonical-state `
  --run-key primary
```

### Authorized MCAP run

The historical Qwen/window graph is explicit and cannot be selected accidentally:

```powershell
uv run python scripts/run_canonical_mcap.py `
  /absolute/path/to/authorized-recording.mcap `
  --profile legacy_window_v1 `
  --allow-unapproved-profile `
  --mapping-config config/genrobot-observed-v0.json `
  --state-dir tmp/mcap-state `
  --run-key primary `
  --max-duration-seconds 180
```

For a new Mage stream plan or local execution, use the route-specific runner:

```powershell
uv run python scripts/run_local_mage_stream.py `
  --recording-key sample-run `
  --recording-start-ns 0 `
  --recording-end-ns 40000000000 `
  --dry-run
```

Use `--materialize` only with an authorized local video source and FFmpeg/FFprobe. Use
`--execute` only with a running, identity-pinned Mage endpoint and a dedicated artifact/state
directory. The default route is single-worker and fail-closed; it is not a capacity claim.

## Local read-only workbench

After a local canonical run, start the API with the web extra:

```powershell
uv sync --locked --extra web
uv run python scripts/run_web_api.py --state-dir tmp/canonical-state
```

In another terminal:

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. The API listens on loopback by default and serves only committed
primary completions. It has no authentication or tenant authorization; do not expose it to the
Internet or connect it to production data.

## Production qualification checklist

The following are still required before a governed production release:

| Gate | Evidence still required |
|---|---|
| Quality | Representative, human-labeled six-camera QA/event/boundary corpus; calibration and error taxonomy |
| Multi-view | Six-camera fusion and missing/contradictory-camera behavior on representative recordings |
| Model/runtime | Real Linux container, RunPod endpoint, pinned model revision, vLLM/BF16/precision profile, saturation and retry evidence |
| Storage | Real R2 PUT/GET/range/reconcile/retention test; PostgreSQL/Supabase TLS, RLS isolation, migration, backup and restore |
| Durable completion | One real source-to-publication run proving completion, evidence, outbox, delivery, and recovery |
| Capacity | Representative workload, 24-hour/declared soak, backlog drain, p95/p99, utilization, cost, and 500 recording-hours/day equivalent |
| Operations | Canary/shadow comparison, alerting, rollback, ownership, incident and data-retention procedures |

All unexecuted gates remain `NOT_MEASURED`. See
[`docs/operations/e2e-production-qualification.md`](docs/operations/e2e-production-qualification.md)
for the evidence model and release gates.

## Verification and CI

Local checks:

```powershell
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/verify_schema_registry.py
uv run python scripts/check_doc_links.py
uv lock --check
```

The required GitHub Actions workflow is `.github/workflows/quality.yml`. It runs eight isolated
Windows test shards, a PostgreSQL 16 integration job, schema/documentation/format/type checks, and
published-schema immutability checks. CI validates the tracked tree; it does not deploy Docker,
RunPod, R2, Supabase, or the web application, and a green workflow is not a production release.

## Contracts, schemas, and replay

- Published `(schema_id, version)` entries are immutable. The current catalog contains **77
  registered schemas** and no implicit upcaster chain for newly published perception families.
- Wire changes require a new schema/version or an explicit migration decision; never edit a
  published schema in place.
- `SCHEMA_BASELINE_REF` is required by CI for immutability checks. Use the repository's protected
  baseline tag/ref rather than a moving branch name.
- Business identity is separate from storage location. Exact bytes are verified by SHA-256;
  R2 object keys, filesystem paths, and database rows are references, not content authority.
- Artifact replay means reading the persisted raw artifact byte-for-byte. Re-running a GPU model is
  model recomputation and may differ across kernels, drivers, precision, or provider versions.
- Lease epochs, fencing tokens, idempotency keys, completion barriers, and outbox state are part
  of the correctness boundary, not optional observability.

Schema commands:

```powershell
uv run python scripts/verify_schema_registry.py
uv run python scripts/check_schema_immutability.py --baseline-ref $env:SCHEMA_BASELINE_REF
```

## Repository map

```text
src/robata/
  adapters/                 SQLite, PostgreSQL, R2, MCAP, frame/cache adapters
  application/canonical/    local and production composition roots; route selection
  contracts/                typed domain/wire contracts and schema bindings
  inference/                provider-neutral, Qwen, Mage, RunPod boundaries
  perception/               observation projectors, tracking, fusion, durable scheduler
  queue/                    legacy window scheduler, leases, backpressure, outbox
  runtime/                  capacity, traces, participation, and qualification telemetry
  web_api/                  local read-only committed-run API
schemas/                    published JSON Schema catalog and immutable bytes
db/migrations/              PostgreSQL canonical schema, RLS, and migration history
config/                     checked-in non-secret route and workload profiles
scripts/                    reproducible runners, qualification, verification, and preflight
tests/                      unit, contract, integration, and conformance tests
web/                        local Vite workbench (not a public production frontend)
docs/                       operations, qualification reports, and route decisions
governance/                 navigation, module cards, and on-demand blueprints
```

## Documentation

- [`AGENTS.md`](AGENTS.md) - repository development and contract rules.
- [`docs/operations/e2e-production-qualification.md`](docs/operations/e2e-production-qualification.md) -
  evidence model and external release gates.
- [`docs/operations/postgres-supabase-production.md`](docs/operations/postgres-supabase-production.md) -
  canonical PostgreSQL/Supabase, R2, pgvector, and production admission sequence.
- [`docs/local-model-qualification-summary-2026-08-10.md`](docs/local-model-qualification-summary-2026-08-10.md) -
  local Qwen/Mage benchmark and boundary summary.
- [`docs/ROBATA_25X_ROUTE_DECISION_2026-08-09.md`](docs/ROBATA_25X_ROUTE_DECISION_2026-08-09.md) -
  25x route decision and unresolved capacity assumptions.
- [`docs/mage-vl-vs-qwen3-vl-4b-selection-report.md`](docs/mage-vl-vs-qwen3-vl-4b-selection-report.md) -
  model and architecture comparison background.
- [`docs/operations/mage-h100-bf16-single-worker.md`](docs/operations/mage-h100-bf16-single-worker.md) -
  H100 qualification plan; it is a plan, not H100 evidence.
- [`governance/REQUIREMENTS.md`](governance/REQUIREMENTS.md) - migrated production requirements reference;
  it is not a release authorization or executable contract.

## Evidence classes and release boundary

Evidence and measurement status are separate:

| Class | Meaning |
|---|---|
| `LOCAL_CONFORMANCE` | Deterministic local behavior and conformance fixtures |
| `LOCAL_BENCHMARK` | Scoped local performance/qualification measurement |
| `REPRESENTATIVE_BENCHMARK` | Representative workload measurement without release authorization |
| `EXTERNAL_QUALIFICATION` | Evidence from real provider, storage target, hardware, or governed data |
| `PRODUCTION_QUALIFIED` | Only after the independent release decision and all required gates |

Current repository posture is `LOCAL_CONFORMANCE` plus scoped `LOCAL_BENCHMARK` artifacts. The
production posture is **not qualified**. Do not interpret a green test, Docker build, schema
check, model-agreement score, or adapter preflight as a release decision.

## License and contact

Copyright (c) 2026 Robata Contributors.

This software is provided for evaluation and development purposes. Production use requires
explicit qualification of models, infrastructure, data quality, and operational policy.
See [`LICENSE`](LICENSE) for terms. For qualification questions or reproducibility issues,
open a GitHub issue with the run key, commit, route profile, and evidence artifact reference.
