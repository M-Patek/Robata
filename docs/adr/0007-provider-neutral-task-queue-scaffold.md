# ADR 0007: Provider-Neutral Task Queue Contract and In-Memory T2 Scaffold

- Status: Accepted for local contract experimentation only
- Date: 2026-07-19
- Governing authority: Architecture V1.1, ADR 0001, and the throughput-track governance review
- Scope: non-normative throughput Track T2; no external broker or durable production claim

## Context

The throughput roadmap requires queueing, leasing, retries, dead-letter handling, and backpressure,
but the repository has no accepted task/lease contract and infrastructure selection is unresolved.
The implementation-plan examples mix asynchronous scheduler methods, Redis-specific types, and symbols
that do not exist in the current ports. Adding Redis, Kafka, Celery, credentials, or network traffic
now would silently close an open infrastructure decision.

## Decision

1. Define a provider-neutral synchronous `TaskQueue` protocol in `robata.ports.task_queue`. The
   protocol owns only task identity, stage/payload metadata, priority, claim/lease, heartbeat,
   complete, fail, and status operations. It contains no Redis/Celery/HTTP types.
2. Define immutable `TaskId`, `LeaseId`, `PipelineTask`, and `TaskSnapshot` values with explicit
   timezone-aware timestamps and machine-readable `TaskQueueErrorCode` failures.
3. Provide `InMemoryTaskQueue` as a deterministic single-process scaffold for contract tests. It
   implements stable priority ordering, bounded active depth, lease expiry, exponential retry
   backoff, dead-letter transition, stale-lease rejection, and exact result retention.
4. Keep the production durability/atomicity requirement explicit in the protocol documentation. A
   future Redis/PostgreSQL/broker adapter must prove equivalent claim/heartbeat/complete/fail atomicity
   and durability behind this boundary. No external dependency is added by this ADR.
5. Keep this scaffold outside normative Architecture V1.1 Phase 2 promotion. It is T2 preparation
   only and does not enqueue governed production work.

## Consequences

### Positive

- Queue semantics can be tested and integrated without selecting infrastructure.
- Retry, lease, dead-letter, and backpressure behavior become concrete and deterministic.
- The application layer can target one stable port before a broker/worker ADR is accepted.

### Negative / limits

- In-memory state is lost on process restart and is not a production durability implementation.
- The synchronous protocol does not decide whether a future distributed scheduler will expose async
  wrappers, a broker, or a database-backed lease store.
- Payload bytes are opaque; schema/version and stage-specific contracts remain caller-owned.

## Verification

The scaffold is accepted for local preparation when tests cover:

- deterministic priority and tie ordering;
- lease metadata, heartbeat, completion, and exact result bytes;
- retry backoff and dead-letter transition;
- expiry requeue and stale-lease rejection; and
- bounded queue capacity and machine-readable invalid-operation errors.

These tests are implemented in `tests/unit/test_task_queue.py`. This ADR does not close Phase 0,
normative Phase 1B/2, O-01/O-03/O-04/O-10, infrastructure selection, or production readiness.
