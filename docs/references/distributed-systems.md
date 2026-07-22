# Distributed Systems References

## 1. Consensus and Leader Election (Paxos / Raft)

**References**:
- Lamport, L. "The Part-Time Parliament." *ACM Transactions on Computer Systems*, 16(2), 1998, pp. 133–169.
- Lamport, L. "Paxos Made Simple." *ACM SIGACT News*, 32(4), 2001, pp. 18–25.
- Ongaro, D. and Ousterhout, J. "In Search of an Understandable Consensus Algorithm (Extended Version)." *USENIX ATC*, 2014.

### Robata Application

Robata does not implement a consensus protocol directly — it runs as a
single-process local authority. However, the **lease epoch + fencing token**
idiom in `adapters/sqlite_outbox.py` and `adapters/sqlite_work_scheduler.py`
is the single-node equivalent of the epoch-increment safety property from
multi-Paxos: a worker whose lease has expired cannot commit state changes
because its fencing token no longer matches the current epoch.

---

## 2. Fencing Tokens

**Reference**: Kleppmann, M. "How to do distributed locking."
*martin.kleppmann.com*, 2016.  
**Book**: Kleppmann, M. *Designing Data-Intensive Applications*. O'Reilly, 2017.
Chapter 8 ("The Trouble with Distributed Systems"), pp. 295–300.

### Principle

When a process holds a lock, it receives a monotonically increasing fencing
token. Any write to the shared resource must include the token; the resource
rejects writes with a token lower than the last seen.

### Robata Application

- `queue/outbox.py` — `OutboxDeliveryClaim.fencing_token` is derived from
  `lease_epoch` via `uuid5`; each new claim increments the epoch.
- `adapters/sqlite_outbox.py` — the acknowledgement query requires the caller
  to supply the current worker ID, epoch, and token; a stale worker's
  `BEGIN IMMEDIATE` will see a mismatched epoch and fail closed.
- `adapters/sqlite_work_scheduler.py` — identical epoch/token pattern for
  durable work items.

---

## 3. Content-Addressable Storage (CAS)

**References**:
- Torvalds, L. *Git source code and design documentation*, 2005.
  (Git's object store is the canonical industrial CAS.)
- Benet, J. "IPFS — Content Addressed, Versioned, P2P File System."
  *arXiv:1407.3561*, 2014.

### Principle

Every object is identified by the hash of its content. Two objects with the
same hash are identical; changing one bit changes the identity. There is no
mutable pointer from a name to changing content.

### Robata Application

- `contracts/hashing.py` — `exact_bytes_sha256()` and `semantic_sha256()` are
  the two CAS primitives. Exact-byte identity ties to raw provider bytes;
  semantic identity ties to a canonicalized subset of business-meaningful fields.
- `adapters/local_artifact_registry.py` — artifact IDs are derived from content
  digests; mutation is rejected.
- `adapters/sqlite_inference_evidence.py` — `raw_sha256` is committed before
  the parse step; a subsequent `INVALID_OUTPUT` terminal still references the
  exact bytes.

---

## 4. Compare-and-Swap (CAS) Atomic Primitives

**Reference**: Herlihy, M. "Wait-free synchronization."
*ACM Transactions on Programming Languages and Systems*, 13(1), 1991, pp. 124–149.

### Principle

An atomic operation reads a value, compares it to an expected value, and writes
a new value only if the comparison succeeds. This is the foundation of
lock-free data structures and optimistic concurrency control.

### Robata Application

- `contracts/revisions.py` — `CurrentSelection` is updated only when the
  caller supplies the exact current revision ID; a stale expectation fails
  without partial state (`compare_and_swap`).
- `adapters/sqlite_primary_completion.py` — the run-version column acts as an
  optimistic generation counter; the completion transaction fails if another
  writer incremented it since the run was opened.

---

## 5. Exactly-Once and At-Least-Once Delivery

**References**:
- Kleppmann, M. *Designing Data-Intensive Applications*. O'Reilly, 2017.
  Chapter 11 ("Stream Processing"), pp. 476–479.
- Kreps, J., Narkhede, N. and Rao, J. "Kafka: a Distributed Messaging System
  for Log Processing." *NetDB Workshop*, 2011.

### Principle

Exactly-once delivery requires both idempotent producers and transactional
consumers. At-least-once delivery is easier to guarantee; the consumer must
be idempotent to achieve the same observable result.

### Robata Application

- `adapters/sqlite_outbox.py` — the relay is explicitly **at-least-once**:
  it publishes the stored payload, then acknowledges the source row. A crash
  after sink commit but before acknowledgement republishes under a new fence;
  the sink is idempotent by `outbox_id` and returns a no-op on replay.
- `docs/adr/0014-local-outbox-relay.md` — the ADR explicitly names this
  trade-off and the integrity error raised when the same `outbox_id` arrives
  with different bytes.

---

## 6. Logical Clocks and Causality

**References**:
- Lamport, L. "Time, Clocks, and the Ordering of Events in a Distributed
  System." *Communications of the ACM*, 21(7), 1978, pp. 558–565.
- Fidge, C. J. "Timestamps in message-passing systems that preserve the
  partial ordering." *Proceedings of the 11th Australian Computer Science
  Conference*, 1988.

### Robata Application

Robata does not use vector clocks, but the **semantic SHA-256 identity** of
each pipeline stage acts as a content-derived causal certificate: a downstream
stage's identity preimage includes the digest of its upstream inputs, so any
change to an upstream result produces a new identity for all descendants.
This is a deterministic, content-addressed substitute for causal timestamps.

---

## 7. Database Isolation and Serialisability

**References**:
- Gray, J. and Reuter, A. *Transaction Processing: Concepts and Techniques*.
  Morgan Kaufmann, 1992.
- Berenson, H. et al. "A Critique of ANSI SQL Isolation Levels."
  *ACM SIGMOD*, 1995.

### Robata Application

- All local SQLite writes use `BEGIN IMMEDIATE`, which acquires a write lock at
  the start of the transaction and provides **serialisable** isolation for the
  single-writer model.
- Immutable-table triggers enforce that serialisability is preserved even if
  a future migration accidentally issues an `UPDATE`.
