# E2E Production Qualification Trace

`robata-e2e-trace-v1` is a versioned operational observation sidecar for
real-sample qualification. It is deliberately **not** a published schema,
canonical evidence, a selection input, or a production-promotion artifact. It
records observed runtime facts, preserves links to frozen inputs and reports,
and represents absent measurements explicitly.

## Authority Boundary

The trace is outside canonical identity, idempotency, selection, evidence
bytes, and production-route authority. It must never be used to:

- choose Mage or Qwen as a primary model;
- approve a production route or publication;
- replace PostgreSQL canonical evidence, an R2 receipt, or a release decision;
- add a trace identifier to a stream plan, request payload, object key, or
  provider wire contract.

The sidecar can link to immutable facts using digests and opaque identifiers. It
cannot turn an unobserved boundary into evidence. Retain it beside the
qualification report and exact input manifests; make any release decision
through the separate reviewed release process.

## P20 Paired Observation

The external paired launcher writes the original P20 report exactly as before.
Without `--trace-output`, it follows the existing report-only path and prints
only the report JSON. With the option, it writes the unchanged report plus a
separate atomic JSON sidecar:

```powershell
python scripts\run_external_paired_model_qualification.py `
  --env-file runtime.env `
  --capabilities-dir capabilities `
  --workload paired-workload.json `
  --evidence-dir evidence `
  --output observations\p20.json `
  --trace-output observations\p20.trace.json `
  --trace-id 11111111-1111-4111-8111-111111111111
```

`--trace-id` is optional when `--trace-output` is present; the launcher
creates one when it is omitted. Supplying `--trace-id` without
`--trace-output` is invalid. The trace output must differ from the workload,
capability, and report paths. A launch failure remains a failed observation, not
a fallback or model-routing signal.

The trace contains the SHA-256 of the serialized P20 report file. Verify that
digest before associating the two files. Keep the report's existing semantics:
it is `EXTERNAL_PROVIDER_OBSERVATION`, is not production eligible, and does
not promote either endpoint.

## Artifact Shape And Coverage

Each sidecar has `format_version: "robata-e2e-trace-v1"`,
`execution_class: "EXTERNAL_PAIRED_OBSERVATION"`,
`selection_eligible: false`, and `production_eligible: false`. Its
current P20 coverage is always `PARTIAL`.

Top-level correlations bind the paired report to the experiment contract, source
workload manifest, input identity, and each endpoint's deployment, endpoint
configuration, handler image, and capability snapshot. Terminal observations may
add opaque inference/request identifiers, retry count, provider request
identifier, latency, output validity, raw-response digests, and transport count.
They must not include secrets or raw request bodies.

The sidecar has three independent runtime fragments:

| Fragment | What it measures |
| --- | --- |
| `LAUNCHER` | The local paired-launcher orchestration. |
| `CONTROL` | The control endpoint adapter's local request/retry work. |
| `CANDIDATE` | The candidate endpoint adapter's local request/retry work. |

Each fragment uses `PROCESS_LOCAL_MONOTONIC` timing. Its offsets can only be
compared within that recorder's process. Do not subtract timestamps across
containers, hosts, RunPod endpoints, or handler processes. Cross-host analysis
uses bound reports, digests, identifiers, and separately clocked artifacts, not
a fabricated global timeline.

Stages are emitted in this stable order:

1. `ORCHESTRATION`
2. `SOURCE`
3. `SCHEDULING`
4. `INFERENCE`
5. `EVIDENCE`
6. `REDUCTION`
7. `PUBLICATION`

For a measured stage, `wall_time_union_ns` is the union of observed
intervals, so nested or concurrent spans are not double-counted.
`inclusive_span_time_ns` is retained separately to reveal nested work.
`unclassified_span_count` makes unknown instrumentation visible instead of
silently assigning it to a stage. The runtime profile also holds local process
CPU, RSS, and local I/O values where supported; it is not handler GPU telemetry.

`NOT_MEASURED` is an epistemic value, not zero, success, failure, or absence
of work. A `NOT_MEASURED` stage has no span count or timing value. The same
rule applies to quality-funnel steps, provider-cost inputs, handler telemetry,
and billed cost. Do not calculate averages, totals, or quality loss from it.

The current P20 trace measures dispatch, terminal, provider-success,
schema-valid, and pair-comparable funnel counts. Ground-truth quality remains
`NOT_MEASURED`. Provider token/image/frame usage or cost is only an input
observation; reconciled cloud billing remains `NOT_MEASURED` until a
separately retained billing artifact is bound. Handler GPU telemetry likewise
requires a hashed handler-side artifact and must never be inferred from client
latency.

## Production Observer Wiring

Use one `RuntimeProfileRecorder` per process and role. A recorder is local
and immutable after snapshot; flush it only after the process has finished its
bounded work. Do not share a recorder between independent control/candidate
processes or infer endpoint separation from identical provider names.

At a production worker boundary, construct a recorder and inject the same
process-local observer into every local dependency that can observe its work:

```python
from robata.adapters.r2_object_store import R2ObjectStore
from robata.application.canonical.production_runtime import build_production_canonical_runtime
from robata.inference.runpod import RunPodVisionAdapter
from robata.runtime.observability import RuntimeProfileRecorder, runtime_span

observer = RuntimeProfileRecorder()
r2_store = R2ObjectStore(r2_config, r2_client, runtime_observer=observer)

def primary_adapter_factory(evidence):
    return RunPodVisionAdapter(
        config=runpod_config,
        evidence_ledger=evidence,
        transport=runpod_transport,
        runtime_observer=observer,
    )

with runtime_span(observer, "runtime.real_sample"):
    runtime = build_production_canonical_runtime(
        # reviewed production arguments omitted
        r2_object_store=r2_store,
        primary_adapter_factory=primary_adapter_factory,
        runtime_observer=observer,
    )
    # Source-specific worker invokes canonical scheduling and completion here.

profile = observer.snapshot()
```

`build_production_canonical_runtime(..., runtime_observer=observer)` passes the
observer to the PostgreSQL authority and supported completion, evidence, barrier,
outbox, logical-node, and review adapters. The R2 store and RunPod factory are
explicit caller-owned dependencies, so construct both with that same observer.
Observation is fail-open: a recorder failure must not alter a canonical
transaction, R2 operation, or RunPod request.

Keep span attributes low-cardinality and non-secret. Allowed examples are a
fixed operation family, route role, retry class, or bounded boolean. Never place
raw IDs, tenant IDs, object keys, URLs containing credentials, prompts, media,
raw provider responses, API keys, database passwords, authorization headers, or
filesystem paths in attributes. Put large or sensitive data only in an approved
separately retained artifact, referenced by content digest where permitted.

## Real-Sample Capture Matrix

For every bounded real-sample run, retain the following facts in a controlled
operator evidence location. Record `NOT_MEASURED` when a row cannot be
collected; never backfill a guessed value.

| Boundary | Capture | Minimum interpretation |
| --- | --- | --- |
| Input | Source manifest/workload version, exact bytes SHA-256, media digest, capability snapshots | Proves both models used identical reviewed inputs. |
| Source | Fetch, decode, sampling, staging spans; frames/windows emitted | Separates source bottlenecks from model behavior. |
| Scheduler | Queue wait, claim/lease, dependency/backpressure state, attempt and retry counts | Locates pressure before inference without altering canonical identity. |
| RunPod | Endpoint/image/capability digests, request/retry timing, terminal state, response digest, provider usage | Compares endpoints under one frozen workload. |
| R2 | Stage/PUT/HEAD/GET verification time, receipt state, byte/digest outcome | Shows immutable artifact lifecycle; location is never identity. |
| PostgreSQL | Authority transaction timing, retry count, RLS/tenant check, completion/evidence writes | Identifies canonical persistence cost and fail-closed behavior. |
| Reduction | Barrier wait, fusion/reduction spans, accepted/rejected inputs | Locates quality or throughput loss after inference. |
| Publication | Completion seal, outbox, review/publication spans and terminal outcomes | Shows whether work reached its allowed output boundary. |
| Handler | Separately exported GPU utilization/memory, queueing, batch size, model/handler version, artifact digest | Never infer GPU saturation from launcher timing. |
| Cost | Provider usage, RunPod billing export/reference, R2/PostgreSQL cost-period inputs | Distinguishes provider fields from reconciled billed cost. |

Freeze the input manifest and endpoint/image/capability values before the run.
Capture repetitions sufficient to characterize warmup, steady state, and tails.
Retain start conditions, concurrency, batching policy, request limits, and
worker count. For paired tests, use the same declared workload and load shape;
record any asymmetry rather than normalizing it away.

## Qualification Rules

Accept a bounded observation for engineering analysis only when all of the
following are true:

- input, workload, capability, endpoint configuration, and handler-image
  digests are frozen and retained;
- the report and trace-file digests link correctly and control/candidate roles
  are correct;
- terminal and schema-valid counts reconcile with the retained report;
- every missing boundary is `NOT_MEASURED` and listed in limitations;
- fragments are analyzed only in their own clock domain;
- canonical PostgreSQL/R2 evidence, when used, passes its existing byte,
  receipt, transaction, and RLS invariants;
- GPU or billing claims have their own retained, hashed source artifact.

Reject or repeat the observation when frozen inputs differ across sides, the
report/trace linkage fails, a required terminal is absent, a provider or schema
failure occurs, a sidecar claims measured values without a source, a
canonical/R2 integrity check fails, or required instrumentation is missing.
Classify insufficient coverage as incomplete, not as a pass.

No P20 trace, including one with two successful endpoints, authorizes a
production model route. Promotion still requires the independent pinned
qualification and release-decision flow in
[PostgreSQL/Supabase Production Composition](postgres-supabase-production.md).

## Current Limitation

The current production composition builds and verifies concrete PostgreSQL, R2,
pgvector, and RunPod adapters, but it is an adapter composition rather than a
generic source-to-publication worker. The repository does not provide one worker
that universally executes source ingestion, scheduling, inference, reduction,
and publication for every source type. Therefore, do **not** claim a complete
E2E trace merely because `build_production_canonical_runtime` succeeds or a
P20 sidecar exists.

A source-specific production task process must own the real stage boundaries,
wrap them with the injected recorder, snapshot its process-local fragment, and
retain handler/billing artifacts. Until that process has executed and the
capture matrix is complete, the honest coverage state is `PARTIAL`.
