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
services, provider-neutral inference orchestration and input-plan contracts, local queue/barrier
logic, offline QA and event reduction components, benchmark calculations, and structured
retrieval primitives.

A canonical post-admission offline conformance slice now connects an explicit fresh or resumed
processing run and resolved `AdmittedRecordingContextV2` through root-window derivation,
materialized `TemporalPackageSet`, exact `InferenceInputPlan` and request catalog, independently
retried call parts, an all-terminal barrier, selected raw/parsed/enriched evidence, deterministic
fusion reduction, a local output decision, and immutable local event hypotheses. It deliberately
stops before stable event-identity assignment or outbox publication. Run stages are attached
through the local SQLite logical-node registry. An optional single SQLite inference-evidence
ledger preserves intents, exact pre-parse bytes, terminal attempts, typed raw artifacts,
selections, parsed claims, selected outputs, and enrichments across fresh adapter, ledger, and
pipeline instances without redispatching selected calls.

The canonical slice accepts only the offline fixture adapter, performs no network calls, supports
only `FUSION_ADJUDICATION`, and starts after admission. Processing-run/work lifecycle records,
barriers, output decisions, and run results are still in-process, and every SQLite adapter remains
local conformance evidence rather than a production infrastructure decision.

Canonical implementation ownership is split under `robata.application.canonical`:
`models.py` owns status, error, root-window, part-result, and execution-policy models;
`projections.py` owns semantic projections and identity-policy namespaces;
`reduction.py` and `output_admission.py` own deterministic fusion and local output decisions;
`logical_nodes.py` owns typed logical-node producers; `runner_support.py` owns chain validation
and conversion helpers; `result_validation.py` owns the retained terminal run result; and
`runner.py` owns state progression and port composition. `canonical_offline.py` is the stable
re-export facade and does not own those implementations.

Schema-evolution conformance includes a registry-backed synthetic upcaster fixture with exact
source/target refs, catalog paths and byte pins for code/runtime/golden artifacts, golden endpoint
validation, repeated-execution determinism and input-mutation checks, and fail-closed graph
validation. The live schema catalog registers no domain upcaster; this fixture is local mechanism
evidence only and does not close Section 25.7 or any Architecture phase.

The former fake-model analysis runner is no longer part of the live package or CLI surface. Its
prior reports and older MVP material remain under `archive/old_mvp` for history only and are not a
supported execution path. Real Qwen/GPT adapters, governed data admission, durable work/barrier
recovery, production database/broker/object storage, ActionEvent revision publication, approved
QA/event policy, quality evidence, SLO evidence, and capacity qualification remain separate
blocked work.

## Requirements

- Python `>=3.12,<3.14`
- The dependencies pinned by `uv.lock`
- Optional MCAP/media dependencies only for explicitly authorized source inspection or export

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

These media CLIs do not invoke the canonical offline conformance slice. That slice is currently
an application API exercised by integration tests, not a production operator entry point.

Source inspection and local export require an explicitly authorized source path and mapping
decision. An unapproved mapping may be exercised only with the CLI's explicit local-development
override. That mode never publishes a governed READY manifest and never makes a provider call.

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
