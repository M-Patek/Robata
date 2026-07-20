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

A canonical post-admission offline conformance slice now connects a resolved
`AdmittedRecordingContextV2` through root-window derivation, materialized
`TemporalPackageSet`, exact `InferenceInputPlan` and request catalog, single-part barrier and
attempt selection, raw-byte persistence before strict parsing, parsed provider claims,
orchestrator enrichment, output admission, and recording-scoped fenced identity/outbox
assignment. It accepts only the offline fixture adapter, performs no network calls, supports only
single-part `FUSION_ADJUDICATION`, starts after admission, and uses in-memory state. It is not a
durable production path or phase-promotion result.

The legacy fake-model mainline remains an isolated development smoke path. It performs zero
external provider requests and always records `production_eligible=false`; it has not been
rewired to the canonical slice above. Real Qwen/GPT adapters, governed data admission,
multi-part reduction, production Redis/database/object storage, approved QA/event policy,
quality evidence, SLO evidence, and capacity qualification remain separate blocked work.

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
.\.venv\Scripts\python.exe scripts\preflight_local_mainline.py --help
.\.venv\Scripts\python.exe scripts\run_local_mainline.py --help
.\.venv\Scripts\python.exe scripts\verify_local_mainline.py --help
```

These media and legacy-mainline CLIs do not invoke the canonical offline conformance slice. That
slice is currently an application API exercised by integration tests, not a production operator
entry point.

Source inspection and local export require an explicitly authorized source path and mapping
decision. An unapproved mapping may be exercised only with the CLI's explicit local-development
override. That mode never publishes a governed READY manifest and never makes a provider call.

Example development invocation after those prerequisites are satisfied:

```powershell
.\.venv\Scripts\python.exe scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-full-mainline `
  --allow-unapproved `
  --registry-root tmp/local-mainline-registry
```

After publication, the output can be verified offline:

```powershell
.\.venv\Scripts\python.exe scripts/verify_local_mainline.py tmp/local-full-mainline
```

## Historical Evidence

Prior MVP reports and runbooks remain available for traceability, for example the
[local fake-model report](archive/old_mvp/reports/local-full-mainline-fake-model-2026-07-19.md),
[artifact-registry report](archive/old_mvp/reports/local-artifact-registry-v2-2026-07-18.md),
and [old local runbook](archive/old_mvp/docs/operations/local-mainline-runbook.md). They are
inputs for comparison only. Current acceptance is based on the live code, current tests, and
the status matrix linked above.
