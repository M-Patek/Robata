# Architecture Patterns

## 1. Hexagonal Architecture (Ports and Adapters)

**Originator**: Alistair Cockburn (2005)  
**Reference**: Cockburn, A. "Hexagonal Architecture." *alistair.cockburn.us*, 2005.  
**Related**: Evans, E. *Domain-Driven Design: Tackling Complexity in the Heart of Software*. Addison-Wesley, 2003.

### Principle

The application core (domain + application services) is independent of all
infrastructure. External actors reach the core only through typed ports. Adapters
translate between the port contract and a specific technology.

### Robata Application

| Layer | Robata module | Role |
|---|---|---|
| Domain | `contracts/`, `alignment/`, `sampling/grid.py` | Invariants, value types, pure logic |
| Application | `application/canonical/runner.py`, `application/canonical/local_composition.py` | Use cases, transaction boundaries |
| Ports | `ports/` (VisionModelAdapter, ReviewQueue, TaskQueue, artifact store) | Technology-neutral interfaces |
| Adapters | `adapters/` (SQLite*, PyAV, RunPod, local artifact registry) | Concrete technology bindings |
| Entry points | `scripts/run_canonical_*.py` | CLI; calls application, never domain policy |

**Key invariant**: No file under `contracts/` or `application/canonical/runner.py`
imports a database library, HTTP client, or filesystem path. Measured bottlenecks
can be replaced in `adapters/` without touching domain contracts.

---

## 2. Event Sourcing

**Originator**: Greg Young, Martin Fowler (circa 2005–2010)  
**References**:
- Fowler, M. "Event Sourcing." *martinfowler.com*, 2005.
- Young, G. "CQRS and Event Sourcing." NDC Oslo, 2010.
- Vernon, V. *Implementing Domain-Driven Design*. Addison-Wesley, 2013. Chapter 8.

### Principle

State is never overwritten. Every change is an immutable event appended to a
log. Current state is derived by replaying the event history from the beginning
or from a snapshot.

### Robata Application

- `adapters/sqlite_inference_evidence.py` — append-only evidence graph; rows
  are never updated after commit.
- `event_pipeline/identity_registry.py` — immutable event identity assignments;
  `CREATED` / `REUSED` / `INVALIDATED` / `OBSERVED` memberships.
- `contracts/revisions.py` — `ImmutableNodeRevision`, `SelectionDecision`;
  current selection is a projection, never a mutable field.
- `adapters/sqlite_primary_completion.py` — outbox rows are append-only; the
  `delivered_at` acknowledgement is the only allowed transition.

**Key invariant**: No `UPDATE` or `DELETE` is issued against business fact tables.
Triggers enforce this at the SQLite level (see `_immutable_table_triggers`).

---

## 3. Command Query Responsibility Segregation (CQRS)

**Originator**: Greg Young (2010), extending Bertrand Meyer's CQS principle (1988)  
**References**:
- Young, G. "CQRS Documents." *cqrs.files.wordpress.com*, 2010.
- Fowler, M. "CQRS." *martinfowler.com*, 2011.

### Principle

Commands mutate state and return no data. Queries read state and have no side
effects. Write and read models are separate.

### Robata Application

- **Commands**: `PrimaryCompletionCommand`, `PrimaryCompletionCommitResult` —
  returned from the write path; carry no query results.
- **Queries / projections**: `CommittedPrimaryCompletion`,
  `EventRegistrySnapshot`, `current_selection` rebuild — read-only views derived
  from the append-only authority.
- `application/canonical/primary_completion.py` deliberately performs no writes.
  The aggregate repository (`sqlite_primary_completion.py`) owns the single
  `BEGIN IMMEDIATE` transaction.

---

## 4. Saga Pattern

**Originator**: Garcia-Molina, H. and Salem, K.  
**Reference**: Garcia-Molina, H. and Salem, K. "Sagas." *Proceedings of the 1987 ACM SIGMOD International Conference on Management of Data*, 1987, pp. 249–259.  
**Modern treatment**: Richardson, C. *Microservices Patterns*. Manning, 2018. Chapter 4.

### Principle

A long-lived transaction is decomposed into a sequence of local transactions,
each with a corresponding compensating transaction. If a step fails, already
committed steps are undone via compensations.

### Robata Application

- `adapters/sqlite_work_scheduler.py` — each `WorkItem` is one local transaction;
  the dependency graph (`WorkDependency`) encodes the saga choreography.
- `queue/models.py` — `REQUIRED` vs `OPTIONAL` dependency criticality; optional
  failures do not trigger compensation of the entire saga.
- Failure semantics: an `INDETERMINATE` action evidence result or an incomplete
  boundary-refinement closure causes the runner to stop explicitly rather than
  silently producing a partial result. This is an explicit abort, not a silent
  compensation.

---

## 5. Transactional Outbox Pattern

**Originator**: Chris Richardson (Eventuate.io, circa 2017)  
**References**:
- Richardson, C. "Transactional Outbox." *microservices.io/patterns/data/transactional-outbox.html*, 2017.
- Richardson, C. *Microservices Patterns*. Manning, 2018. Chapter 3.
- Kleppmann, M. *Designing Data-Intensive Applications*. O'Reilly, 2017. Chapter 11.

### Principle

A service atomically persists a business result and one or more outbox messages
in the same local transaction. A separate relay process reads the outbox and
publishes to the broker. This decouples the commit point from external I/O.

### Robata Application

- `adapters/sqlite_primary_completion.py` — `BEGIN IMMEDIATE` atomically commits:
  identity mutations, ActionEvent genesis, completion record, and pending outbox
  rows (`primary_outbox` table).
- `adapters/sqlite_outbox.py` — fenced relay; `PENDING → LEASED → DELIVERED`
  state machine with lease epoch and fencing token.
- `docs/adr/0014-local-outbox-relay.md` — full decision record.

**Key invariant**: The primary completion transaction never calls a network socket.
A committed completion with a pending outbox row is correct; delivery is a
separate, eventually-consistent concern.

---

## 6. Two-Phase Commit and Local Conformance Substitute

**Reference**: Gray, J. "Notes on Data Base Operating Systems." In *Operating Systems: An Advanced Course*, Lecture Notes in Computer Science, vol. 60. Springer, 1978. pp. 393–481.

### Principle

Distributed two-phase commit (2PC) guarantees atomic cross-node commitment at
the cost of availability and latency. For local conformance the same guarantee
is achieved with a single-node `BEGIN IMMEDIATE` transaction.

### Robata Application

The local aggregate transaction in `sqlite_primary_completion.py` achieves the
same atomicity guarantee as 2PC for the local conformance scope without a
distributed coordinator. The O-14 decision will choose the production-grade
equivalent (e.g., Postgres + outbox, or distributed saga with idempotent steps).
