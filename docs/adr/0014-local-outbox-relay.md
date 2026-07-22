# ADR 0014: Local Fenced Outbox Relay

- Status: Accepted for local conformance
- Date: 2026-07-21
- Amended: 2026-07-22
- Authority: Architecture V1.1 Sections 15.5, 25.2, 25.7, 25.9, and ADRs 0003 and 0012

## Context

The primary-completion transaction already appends immutable event-identity outbox facts with the
business result. A pending row was previously only recoverable data: no live component claimed,
published, retried, dead-lettered, or acknowledged it. Treating insertion as delivery would be a
false success, while coupling broker I/O to the primary transaction would make an external network
call part of the commit point.

## Decision

The local primary-completion SQLite schema is upgraded from internal schema version 1 to version 2.
The migration is additive and preserves every primary fact and outbox ID.

The primary_outbox table remains the immutable business fact. Only its nullable delivered_at
acknowledgement may transition, once, from null to a UTC timestamp. A separate
primary_outbox_deliveries table owns mutable operational state:

PENDING -> LEASED -> DELIVERED

PENDING/RETRY_WAIT -> LEASED -> RETRY_WAIT -> ... -> DEAD_LETTER

Each claim is serialized with BEGIN IMMEDIATE, increments a lease epoch, and derives a new
fencing token. Acknowledge and failure transitions require the current worker, epoch, and token.
Expired leases are recovered before the next claim. Retry parameters are versioned and copied onto
the delivery row when its immutable outbox fact is first discovered, so later configuration changes
cannot rewrite an in-flight retry schedule.

The relay is explicitly at-least-once. It publishes the exact stored payload bytes and exact
SHA-256, then acknowledges the source row. The local sink is independently durable and idempotent
by outbox_id; exact replay is a no-op and same-ID/different-bytes is an integrity error. A crash
after sink commit and before source acknowledgement therefore republishes under a new fence without
creating a second sink record.

The persisted and published payload boundary uses the registered, exact-pinned
`EventIdentityOutboxWireRecord`, including its full `SchemaRef`. The domain
`EventIdentityOutboxRecord` deliberately keeps its original shape because that record is also
embedded inside the already-published `canonical-primary-completion-detail@4.0.0` schema. SQLite
adapters project the domain record to the Wire record for persistence/publication and back on read;
they resolve and validate the exact schema pin at both boundaries. Adding `schema_ref` directly to
the domain record would mutate the frozen completion-detail Wire shape and is therefore prohibited.

Primary completion does not wait for delivery. A committed completion with pending, retrying, or
dead-lettered delivery remains a committed completion. The relay refuses to create a missing
primary database, preventing an incorrect path from appearing as an empty successful queue.

## Recovery Semantics

| Failure point | Recovery |
|---|---|
| Before claim commit | Row remains eligible. |
| After claim, before publish | Lease expiry returns it to retry or terminal DLQ state. |
| After publish, before acknowledgement | Lease expiry causes at-least-once replay; the sink deduplicates exact bytes. |
| After acknowledgement commit | The row is terminal and cannot be claimed again. |
| Stale worker acknowledgement | The epoch/token fence rejects it. |
| Repeated sink failure | Versioned retry delay is applied until max_attempts, then the row is retained in DLQ state. |

## Boundary

This is local executable recovery evidence, not the O-14 production broker or an exactly-once
network claim. Production still needs its approved database/broker topology, authentication,
monitoring, operator DLQ actions, retention policy, and reconciliation procedure. The local
SQLite sink is a broker substitute used to prove ordering, fencing, exact-byte idempotency, and
crash replay without installing infrastructure.
