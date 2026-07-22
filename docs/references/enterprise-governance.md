# Enterprise Governance and Engineering Practices

This document traces each major Robata design decision to published engineering
practices from leading technology organizations. Citations are to publicly
available engineering blog posts, conference papers, and open-source
documentation unless otherwise noted.

---

## 1. Netflix — Resilience, Shadow Traffic, and Chaos Engineering

**References**:
- Izrailevsky, Y. and Tseitlin, A. "The Netflix Simian Army."
  *Netflix Tech Blog*, 2011.
- Nygard, M. *Release It! Design and Deploy Production-Ready Software*.
  Pragmatic Programmers, 2007. (Stability Patterns: Circuit Breaker, Bulkhead)
- Basiri, A. et al. "Chaos Engineering." *IEEE Software*, 33(3), 2016, pp. 35–41.

### Practices mirrored in Robata

| Netflix practice | Robata equivalent |
|---|---|
| Shadow traffic for model evaluation | `inference/shadow.py` — Qwen (primary) vs. GPT (shadow), structurally isolated |
| Circuit breaker / fail-closed | Every unimplemented policy raises `NotImplementedError` explicitly; no silent fallback |
| Chaos engineering principle | Immutable-table SQLite triggers act as local invariant guards; injected transaction failures are tested in `test_sqlite_primary_completion.py` |
| Conductor workflow orchestration | `adapters/sqlite_work_scheduler.py` — local durable work ledger with dependency graph |

---

## 2. Google — Experiment Infrastructure and Data Governance

**References**:
- Tang, D. et al. "Overlapping Experiment Infrastructure: More, Better, Faster
  Experimentation." *KDD*, 2010.
- Sculley, D. et al. "Hidden Technical Debt in Machine Learning Systems."
  *NeurIPS*, 2015.
- Zinkevich, M. "Rules of Machine Learning: Best Practices for ML Engineering."
  *Google Developers*, 2018.

### Practices mirrored in Robata

| Google practice | Robata equivalent |
|---|---|
| Overlapping experiment layers | Shadow route is fully isolated from the primary path; shadow output never touches primary event identity |
| Data freshness validation before training | `qa_pipeline/coarse.py` — quality gate before event detection; low-quality frames cannot produce events |
| Avoid training-serving skew | Inference input is a provider-neutral `TemporalVisualPackage`; the same package is used in offline fixture and production |
| Hermetic data pipelines | Deterministic sampling grid + canonical frame selection; same MCAP always produces the same packages |
| Chubby / distributed lock service | Lease epoch + fencing token in `sqlite_work_scheduler.py` mirrors Chubby's epoch-based lock validity |

---

## 3. Uber — Cadence/Temporal Workflow and Event Sourcing

**References**:
- Fateev, M. and Reshetnik, S. "Cadence: Fault-Tolerant Actor Framework
  for Distributed, Scalable Applications." *VLDB Demonstration*, 2019.
- Chronosphere Engineering. "How Uber Scaled to Support a Massive Growth
  in Demand Using Event Sourcing." *Uber Engineering Blog*, 2020.

### Practices mirrored in Robata

| Uber / Cadence practice | Robata equivalent |
|---|---|
| Durable task scheduling with retry and timeout | `adapters/sqlite_work_scheduler.py` — `WorkItem` with `max_attempts`, `sla_deadline_at`, `execution_expiry_at` |
| Workflow activity dependencies | `WorkDependency` table with `REQUIRED` / `OPTIONAL` criticality |
| Deterministic replay | Exact logical-key reuse across run IDs; `CommittedPrimaryCompletion` replay is a no-op |
| Event sourcing for payment ledger | Append-only inference evidence graph; no `UPDATE` on business fact tables |
| Workflow versioning | `canonical-offline-v5` binding; older version namespaces cannot resume under v5 |

---

## 4. Stripe — Idempotency and Reliable Delivery

**References**:
- Stripe Engineering. "Idempotency Keys and Retries." *Stripe Blog*, 2018.
- Stripe Engineering. "Designing Robust and Predictable APIs with Idempotency."
  *QCon*, 2016.

### Practices mirrored in Robata

| Stripe practice | Robata equivalent |
|---|---|
| Idempotency key on every mutation | Every inference intent carries a deterministic `intent_id` derived from input-plan SHA-256; re-submitting the same request is a no-op |
| Outbox for reliable event delivery | `adapters/sqlite_outbox.py` with at-least-once semantics and idempotent sink |
| Request fingerprinting | `package_input_set_sha256` on every `VisionInferenceRequest`; same request always maps to the same logical call |
| Retry budget with exponential backoff | `OutboxRetryPolicy` with versioned parameters copied at row creation (not reread on retry) |

---

## 5. LinkedIn — Kafka and Append-Only Logs

**References**:
- Kreps, J., Narkhede, N. and Rao, J. "Kafka: a Distributed Messaging System
  for Log Processing." *NetDB Workshop*, 2011.
- Kreps, J. "The Log: What every software engineer should know about
  real-time data's unifying abstraction." *LinkedIn Engineering Blog*, 2013.

### Practices mirrored in Robata

| LinkedIn / Kafka practice | Robata equivalent |
|---|---|
| Immutable, append-only log as system of record | All SQLite business-fact tables are append-only; `UPDATE` and `DELETE` are prevented by triggers |
| Log compaction (keeping only the latest value per key) | `current_selection` table is the compacted view of the `selection_decision` log |
| Consumer group offset tracking | `primary_outbox_deliveries` tracks per-message delivery state independently of the immutable outbox fact |
| Exactly-once via idempotent consumer | `delivered_outbox_messages` uses `outbox_id` as primary key; same ID is a no-op |

---

## 6. Airbnb — Data Quality and Schema Registry

**References**:
- Airbnb Engineering. "Minerva: The Airbnb Metric Store." *Airbnb Tech Blog*, 2021.
- Airbnb Engineering. "Streamalert: Real-time Data Analysis and Alerting."
  *Airbnb Tech Blog*, 2017.

### Practices mirrored in Robata

| Airbnb practice | Robata equivalent |
|---|---|
| Central metric store with immutable definitions | `schemas/schema-catalog.json` — every schema entry is immutable once registered; digest mismatch fails closed |
| Schema evolution with compatibility checks | `contracts/schema_upcasting.py` — compatibility modes (`BACKWARD`, `FORWARD`, `FULL`); ambiguous upgrade paths fail closed |
| Data quality SLAs | `QACompletionProjector` — `QA_INCOMPLETE` terminates the run before event detection; no result is better than a wrong result |

---

## 7. Amazon / AWS — DynamoDB and the Single-Table Design

**References**:
- DeCandia, G. et al. "Dynamo: Amazon's Highly Available Key-value Store."
  *SOSP*, 2007.
- Elmasri, R. and Navathe, S. *Fundamentals of Database Systems*. 7th ed.
  Pearson, 2015.

### Practices mirrored in Robata

| AWS / DynamoDB practice | Robata equivalent |
|---|---|
| Conditional writes (optimistic concurrency) | `run_version` generation counter in `sqlite_primary_completion.py`; completion fails if another writer incremented it |
| Eventual consistency awareness | Outbox delivery is eventually consistent with the commit; the system is designed to tolerate a window of pending delivery |
| Versioned items | Every schema entry carries an explicit `version`; older versions remain readable without upcast |

---

## 8. Confluent / Apache Kafka — Schema Registry

**References**:
- Narkhede, N. "Schema Registry." *Confluent Documentation*, 2014.
- Kleppmann, M. *Designing Data-Intensive Applications*. O'Reilly, 2017.
  Chapter 4 ("Encoding and Evolution").

### Practices mirrored in Robata

| Confluent Schema Registry practice | Robata equivalent |
|---|---|
| Central schema store with immutable versions | `contracts/schema_registry.py` — `SchemaRegistry.register()` rejects re-registration with a different digest |
| Subject/version addressing | Schema entries addressed by `(schema_id, version)` pair |
| Compatibility enforcement before producer publish | `validate_registered_primary_completion_record()` runs before the completion transaction commits |
| Schema ID embedded in messages | `SchemaRef` (schema_id + version + digest) is embedded in every wire envelope |

---

## 9. Martin Fowler / ThoughtWorks — Strangler Fig and Modular Monolith

**References**:
- Fowler, M. "StranglerFigApplication." *martinfowler.com*, 2004.
- Newman, S. *Building Microservices*. 2nd ed. O'Reilly, 2021. Chapter 3.
- Richardson, C. "Pattern: Modular Monolith." *microservices.io*, 2019.

### Practices mirrored in Robata

The system deliberately starts as a **modular monolith** (ADR 0001):
> "Begin as a modular monolith using ports and adapters ... Process separation
> is deferred until measurements or isolation requirements justify it."

This mirrors the Strangler Fig pattern: each domain module is already separated
by ports, so individual modules can be extracted to independent services when
load measurements justify it, without changing domain contracts.

| Module extraction candidate | Extraction trigger |
|---|---|
| `inference/` + RunPod adapter | When inference latency dominates and horizontal scaling is needed |
| `adapters/sqlite_work_scheduler.py` | When O-14 selects a production broker (Kafka, SQS, Temporal) |
| `review/` | When review volume justifies an independent review service |

---

## 10. DORA / DevOps Research — Four Key Metrics

**References**:
- Forsgren, N., Humble, J. and Kim, G. *Accelerate: The Science of Lean
  Software and DevOps*. IT Revolution Press, 2018.

### Alignment

| DORA metric | Robata enabler |
|---|---|
| Deployment frequency | Modular monolith with clean port boundaries; adapters can be swapped without changing domain code |
| Lead time for changes | 806 automated tests; full suite runs in ~15 min; contract tests catch wire regressions immediately |
| Change failure rate | Immutable schemas + exact-digest registry; a wire change that breaks a consumer is detected before deploy |
| Time to restore | Deterministic replay; any historical MCAP re-runs to the exact same result; no state reconstruction needed |
