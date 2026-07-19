# Provider Adapter Readiness Boundary

The local pipeline has a fixed provider-neutral model port:

```python
VisionModelAdapter.infer(
    request,
    package: TemporalVisualPackage,
    artifact_root: Path,
)
```

Everything outside that adapter boundary is prepared and exercised locally. Replacing the fake
adapter is the final implementation task; it is deliberately not part of the current local
readiness slice.

## Already prepared

- strict six-camera source inspection and exact mapping policy;
- registered MP4 export, timestamp sidecars, V2 manifest, and content-addressed registry;
- coarse/dense temporal visual packages with PNG frame materialization;
- QA, proposal, action-evidence, boundary-refinement, and deterministic fusion contracts;
- complete request/outcome/bundle/run-report lineage validation;
- atomic `video + analysis` publication and failure cleanup;
- deterministic execution manifest with exact artifact hashes;
- append-only canonical NDJSON local audit evidence;
- offline preflight and no-event/replay paths;
- locked tests, Ruff, mypy, and offline schema verification.

## Adapter implementation checklist (not yet done)

1. Select and approve a concrete provider/model identity and output contract version.
2. Implement the existing `VisionModelAdapter` port without changing package or request
   contracts. The adapter must persist provider request/response evidence under the supplied
   `artifact_root` without storing credentials or hidden raw prompts in the public bundle.
3. Add explicit timeout, retry, rate-limit, and cancellation accounting. Retries must be visible
   in stage and provider counters; they must never be silently converted to fake success.
4. Add credential injection through the deployment secret manager. No credential may appear in
   the execution manifest, audit NDJSON, run report, exception text, or artifact path.
5. Add provider response validation, schema-version negotiation, and a deterministic redaction
   policy for provider diagnostics.
6. Add quality, latency, cost, capacity, and failure-budget measurements on representative
   six-camera recordings. A successful HTTP response is not a quality gate.
7. Add an approval record and a governed promotion path before setting any event
   `production_eligible` value to `true`.

## Non-goals of the fake adapter

The deterministic fake adapter proves graph connectivity, lineage, failure cleanup, and output
accounting. It does not estimate model quality, provider availability, alignment correctness,
production latency, cost, or safety. Its five-inference event path and two-inference no-event
path are development evidence only.

## Production admission blockers

The following remain open even after the local evidence work is complete: Phase 0 governance,
O-03/O-04/O-10 acceptance items, approved mapping/alignment, provider approval and credentials,
model quality evaluation, capacity/load testing, and Phase 1B promotion gates.
