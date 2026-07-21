# Robata

Robata is a contract-first foundation for the native six-camera video QA and
physical-action event pipeline described by Architecture V1.1. The current repository is
an executable local foundation, not a production deployment or a model-quality claim.

## Authority

Read project inputs in this order:

1. [Large-Scale Agent Execution Specification](large_scale_6camera_video_agent_execution_spec.md)
   defines product intent, fixed functional goals, research priorities, and reporting
   expectations.
2. [Architecture Design V1.1](ARCHITECTURE_DESIGN_V1.md), especially normative Section 25,
   governs security, provider trust, exact time, identity, revisions, state accounting,
   and phase dependencies.
3. [Execution Specification V1 Overlay](docs/architecture/execution-spec-v1-overlay.md)
   normalizes the product instruction into testable requirements without turning examples
   into wire contracts.
4. Accepted [ADRs](docs/adr) and checked-in registered JSON Schemas govern implementation
   decisions and wire compatibility.
5. [Implementation Plan](IMPLEMENTATION_PLAN.md) defines execution order and evidence gates.

[Current implementation status](docs/current-implementation-status.md) maps the live code to
those authorities and lists every deferred gate. Content under
[`archive/old_mvp`](archive/old_mvp) is historical and non-normative; it does not prove that
the current tree satisfies a phase gate.

## Current Baseline

The live tree provides strict domain values and schemas, deterministic canonical hashing,
six-camera invariants, immutable artifact and revision primitives, local ingestion/alignment
services, provider-neutral inference orchestration and input-plan contracts, deterministic
barrier coordination with run-scoped local SQLite persistence, offline QA and event reduction
components, benchmark calculations, and structured retrieval primitives.

A canonical local-conformance command now accepts either one immutable six-camera fixture or one
explicitly authorized raw MCAP. The raw path inspects and hashes the exact source, binds exact
schema bytes, probes all six H.264 streams, registers MP4 and sidecar artifacts, constructs V2
admission/alignment evidence, and materializes a canonical six-camera frame index plus selected
PNG artifacts. Both sources then drive the same explicit processing run through exact coarse/dense QA and QA
completion; provider-neutral EVENT_PROPOSAL planning; deterministic candidate reduction;
per-candidate ACTION_EVIDENCE; provisional 0/1/N physical-action fusion; and separate padded
ONSET/OFFSET boundary passes. The runner builds one versioned final-fusion context from the exact
ordered refined-action closure, binds it to the input plan and adapter requests, and accepts only
explicit zero output or exact 1:1 final hypotheses. Every inference stage preserves separate
raw/parsed/enriched evidence, independently retried call parts, and an all-terminal barrier.
The application prepares stable recording-scoped identity and deterministic `ActionEvent` genesis
revision/selection/current facts before one SQLite transaction commits the terminal run, exact
detailed result, compact completion, and pending outbox rows. The SQLite inference-evidence ledger
and logical-node registry preserve reusable stage evidence across command processes without
redispatching selected calls. A run-scoped SQLite authority persists the generic barrier
definition/state/members and inference-call definition/completions/reduction. A fresh process can
therefore recover an interrupted same run after the reduction and inference evidence are durable
without redispatching the model calls.

The local composition still selects the offline fixture inference adapter and performs no network
calls. It supports `QA_COARSE`, `QA_DENSE`, `EVENT_PROPOSAL`, `ACTION_EVIDENCE`,
`BOUNDARY_REFINEMENT`, and `FUSION_ADJUDICATION` through the same provider-neutral request
boundary. Every local result is conformance evidence only and carries
`production_eligible=false`. The canonical runner
independently injects the provider-neutral model adapter, exact raw-byte store, and strict claim
parser, so replacing the fixture does not require changing its business control flow. The local
composition also injects
the run-scoped SQLite barrier authority; the runner retains in-memory reference storage only for
component use. This does not provide a durable work ledger, deadline/lease/fence recovery,
registered persisted-barrier wire contracts, Redis/broker integration, or a production recovery
topology. Pending outbox rows have no publisher, and every SQLite adapter remains local
conformance evidence rather than a production infrastructure decision. Every command receipt and
published local payload reports `evidence_class=LOCAL_CONFORMANCE` and
`production_eligible=false`.

Canonical implementation ownership is split under `robata.application.canonical`:
`models.py` owns status, error, root-window, part-result, and execution-policy models;
`projections.py` owns semantic projections and identity-policy namespaces;
`reduction.py` and `output_admission.py` own deterministic fusion and local output decisions;
`logical_nodes.py` owns typed logical-node producers; `runner_support.py` owns chain validation
and conversion helpers; `result_validation.py` owns the retained terminal run result;
`runner.py` owns stage progression; `mcap_source.py` owns the concrete raw-MCAP-to-canonical source
bridge; and `local_composition.py` owns local source selection, durable-adapter, identity, and
completion wiring. `canonical_offline.py` is the stable re-export facade and does not own those
implementations.

Schema-evolution conformance includes a registry-backed synthetic upcaster fixture with exact
source/target refs, catalog paths and byte pins for code/runtime/golden artifacts, golden endpoint
validation, repeated-execution determinism and input-mutation checks, and fail-closed graph
validation. The live schema catalog registers no domain upcaster; this fixture is local mechanism
evidence only and does not close Section 25.7 or any Architecture phase.

The former fake-model analysis runner is no longer part of the live package or CLI surface. Its
prior reports and older MVP material remain under `archive/old_mvp` for history only and are not a
supported execution path. Real Qwen/GPT adapters, governed approval of raw-MCAP admission policy,
durable work scheduling and lease/fence recovery, production database/Redis/broker/object storage,
governed ActionEvent contracts and successor publication, approved QA/event policy, quality
evidence, SLO evidence, and capacity qualification remain separate blocked work.

## Requirements

- Python `>=3.12,<3.14`
- The dependencies pinned by `uv.lock`
- Optional MCAP/media dependencies for canonical fixture or raw-MCAP execution and explicitly
  authorized source inspection or export

The code fails closed when an optional adapter dependency is absent. It does not install or
download provider SDKs, model checkpoints, databases, or services at runtime.

## Verification

Use the existing environment for local checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp D:\tmp\robata-tests
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe scripts\verify_schema_registry.py
.\.venv\Scripts\python.exe scripts\check_doc_links.py
```

Provisioning a clean environment is an explicit operator action, not part of runtime behavior:

```powershell
uv sync --locked --dev
```

## Local CLIs

The current CLI surface can be inspected without touching source media:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_mcap.py --help
.\.venv\Scripts\python.exe scripts\export_camera_videos.py --help
```

The canonical local fixture command is:

```powershell
.venv/Scripts/python.exe scripts/run_canonical_fixture.py tests/fixtures/canonical/source-recording.json --state-dir tmp/canonical-state --run-key primary
```

The matching raw-MCAP command is:

```powershell
.venv/Scripts/python.exe scripts/run_canonical_mcap.py data/source/sample-medium.mcap --mapping-config config/genrobot-observed-v0.json --allow-unapproved-profile --state-dir tmp/canonical-mcap-state --run-key primary
```

Repeating either exact command performs recovery/replay and does not duplicate the business result
or its outbox rows. These are local operator entry points, not production commands.
Local processing timestamps come from an explicit versioned deterministic execution clock; they
are not derived from the fixture's `recording_start_utc` source fact. Changing that clock policy
changes the local run namespace.
The standalone media inspection/export CLIs remain separate entry points; the raw canonical
command composes the same inspection and registered-export adapters internally.

Source inspection and local export require an explicitly authorized source path and mapping
decision. An unapproved mapping may be exercised only with the CLI's explicit local-development
override. That mode may derive local V2 evidence, but never publishes a governed READY decision,
makes a provider call, or sets `production_eligible=true`.

Schema publication has one repository command. It normalizes a new candidate to deterministic
UTF-8/LF bytes, derives its exact digest and artifact ID, validates the complete offline registry,
fsyncs staged file contents on supported platforms, fsyncs directory metadata on POSIX, and
replaces the catalog last as the publication commit point. Readers share the publication lock and
therefore cannot observe the artifact/catalog replacement window. A digest-bound transaction
marker makes the sole exact pre-commit orphan readable and recoverable after a hard interruption;
all other unregistered schemas remain fail-closed. Exact replay also removes a stale marker left
after the catalog commit:

```powershell
.\.venv\Scripts\python.exe scripts\register_schema.py --help
.\.venv\Scripts\python.exe scripts\verify_schema_registry.py
.\.venv\Scripts\python.exe scripts\check_schema_immutability.py --baseline-ref $env:SCHEMA_BASELINE_REF
```

This command currently publishes only schemas with `compatibility_mode=NONE`; governed
predecessor and upcaster publication remains a separate Section 25.7 boundary. On Windows the
tool guarantees atomic catalog visibility and retry after normal process interruption, but does
not claim power-loss durability for directory metadata. That production filesystem decision
remains part of O-14.

The exact `SchemaRef` stored in `schemas/schema-catalog.json` is the golden pin. Published
`(schema_id, version)` entries are immutable; any wire or formatting change requires a new
version. CI must set `SCHEMA_BASELINE_REF` to a protected prior release commit or tag. The
workflow fails closed when it is absent and checks both that protected baseline and the
pull-request/push event baseline; a missing or all-zero event baseline also fails closed.
Comparing only with the mutable current catalog does not prove
append-only history. The checker has one
non-bypassable repair case for an internally inconsistent historical tree: the catalog entry must
be unchanged, the baseline blob must not match its existing pin, and the candidate blob must match
that same pin exactly. It reports the reconciled schema labels and count. This is not a general
immutability waiver; a baseline that already matches its pin remains byte-frozen. A source release
must come from a clean tracked tree and pass:

```powershell
.\.venv\Scripts\python.exe scripts\check_release_hygiene.py --check-only
.\.venv\Scripts\python.exe scripts\check_release_hygiene.py --archive-output $env:TEMP\robata-source.tar --expected-commit HEAD > $env:TEMP\robata-source-manifest.json
```

`.github/workflows/quality.yml` applies the locked full test, lint, format, type, schema,
documentation, immutability, and release-hygiene gates to pull requests and `main`. It publishes
the exact archive byte stream validated by the release-hygiene command plus its deterministic
source manifest, both bound to the same checked and unchanged commit SHA.

## Historical Evidence

Prior MVP reports and runbooks remain available for traceability, for example the
[local fake-model report](archive/old_mvp/reports/local-full-mainline-fake-model-2026-07-19.md),
[artifact-registry report](archive/old_mvp/reports/local-artifact-registry-v2-2026-07-18.md),
and [old local runbook](archive/old_mvp/docs/operations/local-mainline-runbook.md). They are
inputs for comparison only. Current acceptance is based on the live code, current tests, and
the status matrix linked above.
