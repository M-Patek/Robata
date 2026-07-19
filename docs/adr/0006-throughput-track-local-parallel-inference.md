# ADR 0006: Opt-In Deterministic Parallel Inference for Throughput Track T1

- Status: Accepted for local fake-model experimentation only
- Date: 2026-07-19
- Governing authority: Architecture V1.1, ADR 0001, and the local mainline contracts
- Scope: non-normative throughput track T1; does not redefine Architecture V1.1 Phase 1B/2

## Context

The local mainline has a serial inference path. The throughput roadmap identifies independent
work after event proposal, but the normative Architecture V1.1 phase gates and provider boundary
remain unchanged. In particular, a parallel experiment must not imply source admission, provider
approval, production capacity, or quality promotion.

`QA_DENSE` and `ACTION_EVIDENCE` both consume the same dense temporal package and candidate
identity. `BOUNDARY_REFINEMENT` consumes the action-evidence result and therefore remains a
downstream serial stage.

## Decision

1. Add an explicit `LocalMainlineConfig.parallel_independent_inference` feature flag. It is
   disabled by default.
2. Require an adapter capability declaration (`supports_parallel_inference`) before enabling
   the flag. The deterministic fake adapter declares the capability because it is stateless and
   network-free. A future real provider adapter must make an explicit, separately governed
   decision; the flag never enables provider traffic by itself.
3. Use a bounded `ThreadPoolExecutor` for the two independent calls. The executor is local and
   synchronous at the application boundary; no queue, broker, credential, or network dependency
   is introduced.
4. Submit tasks in canonical order (`QA_DENSE`, then `ACTION_EVIDENCE`) and merge results in that
   same order regardless of completion order. Any task failure propagates and publication remains
   atomic/fail-closed.
5. Keep operational worker-count/parallel flags out of `LocalMainlineConfig.semantic_projection`.
   They affect execution strategy, not the semantic output identity; serial and parallel replay of
   the same inputs must therefore preserve run-independent request, package, event, and bundle
   identity. Wall-clock accounting remains observational; exact bundle-byte equality requires a fixed
   logical clock in replay tests.
6. Keep `BOUNDARY_REFINEMENT` serial until a separate dependency/contract decision is recorded.

## Consequences

### Positive

- The T1 experiment is available without changing provider contracts or adding dependencies.
- Deterministic ordering and semantic identity are preserved.
- Capability gating prevents accidental concurrent calls on an adapter that has not declared
  thread safety.
- The default path remains byte-compatible with the existing local fake baseline.

### Negative / limits

- No throughput or latency claim follows from this ADR. Measurements require a governed benchmark
  corpus, CPU/RSS/disk instrumentation, and replay evidence.
- The current export and frame-materialization stages remain serial.
- ThreadPoolExecutor is not a decision for PyAV export/materialization or distributed workers.
- Provider retries, timeout budgets, rate limits, cost accounting, credentials, and network policy
  remain outside this ADR.

## Verification

The implementation is accepted only for local experimentation when:

- serial and opt-in parallel runs under a fixed logical clock produce the same canonical inference
  sequence, outcomes, event semantics, and bundle bytes;
- an adapter without the capability declaration is rejected before the parallel stage runs;
- failures leave no published output or partial staging tree; and
- provider request accounting remains zero for the fake adapter.

The ADR does not close Architecture V1.1 Phase 0, Phase 1B source/time admission, O-03, O-04, O-10,
or any production eligibility gate.
