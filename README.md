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

The live tree provides strict domain values and 44 exact-pinned schemas, deterministic canonical
hashing, six-camera invariants, immutable artifact and revision primitives, local
inference/barrier evidence, a durable SQLite work scheduler, a local outbox relay, nonblocking
review routing, offline QA/event reduction, benchmark calculations, a synthetic capacity harness,
and structured retrieval primitives.

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
boundary. A separate credential-redacting `RunPodVisionAdapter` is mock-transport tested for
strict binding, bounded retries, timeout handling, and raw-byte preservation, but it is not
composed and has never used a real endpoint or credential. The canonical runner independently
injects the provider-neutral model adapter, exact raw-byte store, and strict claim parser.

The run-scoped SQLite barrier remains the canonical local recovery authority. Separate local
components now provide a durable work ledger with deadlines/leases/fences/invalidation, registered
work-message and persisted-barrier projections, at-least-once outbox relay with idempotent sink and
DLQ simulation, and a nonblocking priority/SLA review queue. Persisted and published event-identity
outbox payloads use an exact-pinned `EventIdentityOutboxWireRecord`; the embedded
`EventIdentityOutboxRecord` keeps its original frozen shape inside completion detail. The review
queue exact-validates registered task, annotation, and reopen-command payloads on write and read.
These components are not yet wired into one production topology and do not select Redis, a broker,
reconciliation ownership, or O-14 recovery policy. Every command receipt and published local
payload remains `LOCAL_CONFORMANCE` with `production_eligible=false`.

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
validation. A separate atomic evolution command can publish one new target plus one or more direct
incoming upcasters, including their exact code, runtime, and golden-vector artifacts. The live
schema catalog still registers no domain upcaster. Existing completion target entries were
published with `compatibility_mode=NONE` and empty predecessor sets, so retroactive
V1-to-V2-to-V3/V4 chains would violate published catalog immutability; detailed V1 also lacks facts
required by later versions. A governed chain must target new versions rather than rewriting these
entries.

The former fake-model analysis runner is no longer part of the live package or CLI surface. Its
prior reports and older MVP material remain under `archive/old_mvp` for history only and are not a
supported execution path. Real Qwen/GPT/RunPod qualification, governed raw-MCAP admission,
production database/broker/object storage and scheduler composition, governed ActionEvent
successors, approved QA/event/review policy, representative quality evidence, long-soak SLO data,
and measured capacity qualification remain blocked or external work.

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
.\.venv\Scripts\python.exe scripts\verify_rational_grid_vectors.py
node scripts\verify_rational_grid_vectors.mjs
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

Schema publication has two repository commands. `register_schema.py` publishes one new
`compatibility_mode=NONE` schema. `register_schema_evolution.py` publishes one new `BACKWARD`
target plus one or more direct incoming upcasters and their exact code, runtime, and paired golden
artifacts as one bundle. Both normalize candidates to deterministic UTF-8/LF bytes, derive exact
digests and artifact IDs, validate a complete temporary registry snapshot, and replace the catalog
last as the publication commit point. Evolution publication additionally constructs the full
upcaster graph and executes its golden validation before commit. Readers share the publication
lock and therefore cannot observe the artifact/catalog replacement window. A digest-bound
transaction marker covers every staged bundle artifact and supports exact recovery after an
interruption; other uncataloged Schema documents remain fail-closed. Code, runtime, and golden
artifacts become governed only when an exact catalog entry pins them. Exact replay also removes a
stale marker left after the catalog commit:

```powershell
.\.venv\Scripts\python.exe scripts\register_schema.py --help
.\.venv\Scripts\python.exe scripts\register_schema_evolution.py --help
.\.venv\Scripts\python.exe scripts\verify_schema_registry.py
.\.venv\Scripts\python.exe scripts\check_schema_immutability.py --baseline-ref $env:SCHEMA_BASELINE_REF
```

The evolution command deliberately does not edit an existing target or edge: its bundle must name
one previously unknown target and at least one direct predecessor. The live catalog still has
`upcasters=[]`; the command is executable publication infrastructure, not evidence that a business
migration has been approved or published. On Windows these tools guarantee atomic catalog
visibility and retry after normal process interruption, but do not claim power-loss durability for
directory metadata. That production filesystem decision remains part of O-14.

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
