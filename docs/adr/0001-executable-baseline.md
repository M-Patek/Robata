# ADR 0001: Executable Baseline

- Status: Accepted, amended for Architecture V1.1
- Date: 2026-07-18
- Scope: Phase 0, Phase 1A, and the Phase 1B admission boundary

## Context

Architecture V1 leaves the implementation language and deployment products open. Implementation still needs a reproducible runtime, one wire-contract authority, and stable module boundaries. Architecture V1.1 Section 25 is normative and supersedes the earlier phase numbering: Phase 0 is the security, privacy, and provider-data-governance hard gate; Phase 1A is the executable contract foundation; and Phase 1B is source/time admission on representative real MCAPs.

Local `file.zip` inventory contains 37 MCAP members. Availability is not admission: the archive is ignored by Git, is not an approved production corpus, and supplies no implicit answer to O-03 or O-04. It cannot support Phase 1B promotion until data-governance approval and the source/time decisions are recorded.

## Decision

### Normative phase boundary

- **Phase 0 - security and data governance:** complete the classification and threat model, least-privilege identities/RBAC, secrets, encryption, audit, retention/deletion/legal hold, artifact boundaries, provider data-use/residency terms, shadow approval, and incident/credential-rotation tests. No governed production frame or prompt may leave the approved trust boundary before this gate passes.
- **Phase 1A - executable contract foundation:** implement schema registry/compatibility checks, canonical digest fixtures, integer-nanosecond time and interval types, exact rational-grid vectors, artifact registry behavior, run-independent logical nodes and run-node links, and immutable revision/current-selection primitives.
- **Phase 1B - source/time foundation:** only after 0 and 1A, exercise separate validation-report/READY-manifest publication, exactly-six-camera mapping, raw provenance, separate source/alignment ledgers, and alignment uncertainty on representative approved MCAPs.

Synthetic-data spikes for later phases are allowed in isolation, but they do not satisfy a later phase's exit gate and must not process governed production data.

### Runtime and tooling

- Use Python `>=3.12,<3.14` as the initial application runtime.
- Use `uv` for environment creation, dependency resolution, locking, and project commands.
- Keep importable code under `src/robata` and tests under `tests`.
- Use Pydantic for strict in-process models and parsing at application boundaries.

Python is selected for the initial implementation because the MCAP, media, validation, and model-integration ecosystems support rapid contract testing. Video decoding and other expensive operations remain behind ports so measured bottlenecks can later be replaced by native workers without changing domain contracts.

### Contract authority

Checked-in, immutable, versioned JSON Schema documents are authoritative for wire compatibility. Pydantic models must conform to those schemas but are not a second wire-contract authority. Registry entries include schema ID, semantic version, artifact ID, SHA-256, owner, canonicalization/projection version, compatibility mode, lifecycle, and supported software range. Unknown or conflicting digests fail closed.

Semantic validators are authoritative for invariants that JSON Schema cannot fully express, including:

- exactly the canonical camera slots `cam_01` through `cam_06`;
- aggregate counts matching their child records;
- interval, rational-grid, and recording-duration relationships;
- cross-record mapping uniqueness and READY publication rules;
- digest preimages, canonical ordering, run-independent identity, and lineage consistency;
- immutable revisions and deterministic current-selection projection.

Every accepted payload must pass both wire-schema validation and the applicable semantic validator. Nanosecond values remain canonical base-10 strings on the wire and Python integers internally.

### Architecture style

Begin as a modular monolith using ports and adapters:

- domain modules own invariants and value types;
- application services own use cases and transaction boundaries;
- ports describe MCAP inspection, artifact storage, metadata persistence, clocks, and future queues/providers;
- adapters contain filesystem, MCAP library, database, and provider-specific behavior;
- CLI entry points call application services and do not contain domain policy.

This keeps the first executable path small without coupling domain contracts to a database, broker, object store, decoder, or model provider. Process separation is deferred until measurements or isolation requirements justify it.

### Source admission artifacts

Phase 1B keeps container validation evidence separate from READY publication. `MCAPValidationReport` has `VALID`, `INVALID`, or `INCONCLUSIVE`; infrastructure failure is `INCONCLUSIVE`. `MCAPReadyManifest` exists only after a selected `VALID` report, durable source artifact, and exactly-six-camera mapping pass. Alignment has a separate ledger and does not change source-validity history.

Unsupported source schemas or codecs produce explicit validation evidence and never a READY manifest. Primary package admission later requires both a selected READY manifest and a selected alignment admissible for the consuming policy.

## Promotion Gates

### Phase 0

Phase 0 cannot exit until the Section 25.10 security/privacy/governance controls are approved and tested. Possession of local data does not waive this gate.

### Phase 1A

Phase 1A can exit on executable contract and conformance evidence without real data. A synthetic MCAP may test interfaces or failure handling as an explicitly isolated spike, but it is not Phase 1B evidence.

### Phase 1B

Phase 1B cannot exit until all of the following are available and exercised:

1. Data-governance approval covering the intended use of the representative real MCAP corpus.
2. Representative expected and invalid real MCAP recordings; the 37-member local archive is only an inventory until approved and characterized.
3. O-03 resolution: topics, schemas, camera mapping rules, auxiliary channels, codecs, resolutions, rates, and decoder expectations.
4. O-04 resolution: clock sources, synchronization evidence, resets, drift assumptions, residual thresholds, and missing-frame tolerance.
5. A real-data replay that produces immutable validation reports, selected READY manifests only for admitted sources, separate source/alignment ledger reconciliation, quarantine evidence, and alignment results without inferred-but-unrecorded policy.

Until then, source/time results are development evidence and alignment remains `UNVERIFIED` unless an approved explicit clock policy provides sufficient evidence. No result may be labeled Phase 1B complete or production-admitted.

## Consequences

- Security and data-governance work is a predecessor, not a follow-up hardening task.
- Contract and invariant work can proceed before source-specific decisions or governed real-data access.
- Source-specific readers, decoders, persistence, and providers remain replaceable adapters.
- JSON Schema and Pydantic representations require conformance tests to prevent drift.
- A local filesystem adapter is suitable for development evidence but does not satisfy production durability, concurrency, or disaster-recovery requirements.
- PostgreSQL, broker, object-store, deployment, and provider selections remain open decisions and must not be silently encoded as domain policy.
