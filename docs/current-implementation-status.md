# Current Implementation Status

- Status date: 2026-07-22
- Scope: live repository outside `archive/old_mvp`
- Evidence class: local development unless a row explicitly says otherwise

## Authority Boundary

The product execution specification owns intent and experiment priorities. Architecture V1.1
Section 25 owns normative security, identity, time, status, provider-trust, and dependency
rules. Accepted ADRs and registered schemas refine those rules. `IMPLEMENTATION_PLAN.md`
orders the work but cannot promote a phase without its required evidence.

The archive is intentionally outside this authority chain. Archived reports may explain prior
work, but their commands, paths, test totals, and conclusions do not describe the live tree
unless independently reproduced.

## Next Internal Engineering Iteration

The non-normative
[Streaming Throughput Next-Iteration Guide V1](architecture/streaming-throughput-next-iteration-v1.md)
defines the next local sequence: instrument the current fresh path, register incremental streaming
identities, replace repeated whole-file media work with single-pass bounded ingest, extend the
durable scheduler to a window DAG, batch inference/evidence persistence, commit incremental window
evidence under its own local transaction, and finalize recording-scoped canonical truth at end of
stream. Its capacity values are candidate-local observations, derived projections, external vendor
benchmarks, or unvalidated targets; none is production qualification. Section 25, accepted ADRs,
and registered schemas remain authoritative when implementation begins.

## Live Slices

| Slice | Live state | Promotion boundary |
|---|---|---|
| Contract and schema kernel | Strict values, canonical JSON/SHA-256, six-camera maps, atomic single-schema, independent-schema-bundle, and evolution-bundle registration, immutability checks, logical nodes, immutable revisions, current-selection primitives, and 64 exact-pinned schemas are executable. An independent bundle publishes 1..N new `NONE` schemas atomically; an evolution bundle publishes one new target, one or more incoming upcasters, and their exact code/runtime/golden artifacts only after complete temporary registry and graph validation. Registered payloads include compact/detailed completion, output, work/barrier, event-identity outbox, review, media-quality, supplemental-QA, pre-EOS capture/segment/window, stream inference evidence, stream work, expected-window closure, finalization, and Artifact Registry V3 contracts. Compact V1 bytes remain frozen and exact-readable without an implicit upcast. | The live catalog still declares `upcasters=[]`. The evolution command proves publication mechanics only. Existing completion targets were already published with `compatibility_mode=NONE` and no predecessors, so they cannot be retrofitted without violating catalog immutability; detailed V1 also lacks facts required by later versions. A future governed chain must target new versions. Detailed-result bytes remain embedded in local SQLite rather than a production artifact store. |
| Ingestion and alignment | Deterministic six-stream indexing, immutable validation/READY state transitions, exact rational transforms, reset segments, residuals, and admissibility gates are implemented behind injected ports. | No governed corpus, O-03, O-04, or Phase 0 approval; local results cannot claim Phase 1B. |
| Media inspection and export | Live CLIs expose explicitly authorized inspection and local video export. The canonical raw-MCAP command composes the same inspector, six H.264 decoder probes, registered MP4/sidecar export, canonical frame index, selected PNG publication, and an exact-pinned local media-quality report before entering the shared canonical runner. Report replay is exact-byte and semantic-binding checked. The former fake-model analysis runner remains archived history only. | Source access must be explicitly authorized. The current mapping profile requires an explicit local-development override; its V2 evidence is local only, non-promotional, and sends no provider traffic. Deterministic luma/edge/freeze/cadence/sequence/skew observations are local evidence, not governed semantic blur or occlusion labels. |
| Sampling and package planning | Exact rational-grid selection, deterministic frame tie/dedup behavior, materialized provider-neutral frame-budget package sets, and an exact frozen-adaptive resolver are implemented and locally verified. `AdaptiveSampler.sample()` executes injected detectors and deterministically reduces enabled triggers. Separately, the raw-MCAP canonical path freezes exact neighbor targets from its registered local media-quality report, materializes provider-neutral PNG packages, verifies selected artifact bytes, and persists deterministic supplemental QA evidence on fresh execution and revalidates it on recovery. | The generic `AdaptiveSampler` detector-trigger stream is not yet composed into canonical package planning, and local media observations are not governed adaptive triggers. Low edge energy is explicitly a proxy, not a semantic blur or occlusion decision. O-13 still owns promoted rates, padding, thresholds, adaptive triggers, and budgets. Provider-limit splitting belongs to `InferenceInputPlan`, not package mutation. |
| Primary inference boundary | Provider-neutral request, capability, intent, terminal-attempt, output-validation, selection-decision logical keys, and strict `InferenceInputPlan` contracts are executable. The canonical runner injects `VisionModelAdapter`, the exact raw-byte store, and the strict claim parser independently. A credential-redacting `RunPodVisionAdapter` now covers strict request/response binding, bounded deterministic retries, timeout and protocol failure, exact raw-byte preservation, and mock-transport execution without real network traffic. Exact-pinned schemas and one append-only local SQLite ledger preserve the typed evidence graph across fresh instances. | The local composition root still selects only the offline fixture adapter. The RunPod adapter has no qualified endpoint, real credential path, governed capability evidence, soak result, production attempt/artifact store, or production recovery decision. |
| Canonical offline vertical slice | The fixture and explicitly authorized raw-MCAP commands share one runner through coarse/dense QA, proposal, candidate reduction, per-candidate action evidence, deterministic provisional fusion, separate padded ONSET/OFFSET boundary passes, and final fusion. A versioned final-fusion context binds the exact ordered refined-action closure into the input-plan dependency identity and every adapter request; final reduction accepts explicit zero output or requires exact 1:1 coverage before local admission. Zero proposals/actions terminate as `NO_EVENTS`; indeterminate evidence fails closed; replay and restart-safe inference/barrier recovery avoid redispatch. For raw MCAP, the registered media-quality report and any exact-target supplemental QA artifact closure are validated before the primary command and bound into primary-completion command v2 through ordered role/schema/semantic-digest/exact-byte-digest/byte-count references. Detailed completion V4 retains the complete stage chain, while compact completion V3 records the exact terminal stage. The local aggregate atomically commits identity, ActionEvent genesis, completion, and pending outbox for event-producing and early `NO_EVENTS` outcomes. On recovery the local command uses authoritative primary completion to reconcile `ACTION_PUBLISH` and committed outbox delivery before it revalidates side evidence for the receipt; nonblocking review remains downstream of successful side-evidence validation. | These post-completion integrations remain `LOCAL_CONFORMANCE`: the scheduler, primary ledger, outbox sink, and review queue span separate SQLite authorities and use commit-first reconciliation rather than one production transaction. Local atomic replacement does not prove power-loss durability without a selected filesystem/storage contract and parent-directory `fsync`. Real providers, governed policies, target storage/broker topology, credentials, production isolation, and recovery ownership remain open. Every local result remains `evidence_class=LOCAL_CONFORMANCE` and `production_eligible=false`. |
| Shadow and evaluation | Deterministic random/hard-case routing, explicit budget outcomes, primary isolation, paired comparison, and append-only disagreement evidence are executable locally. | GPT/provider governance, quota, retention, and Phase 8 predecessors remain blocked. |
| QA | The runner executes fixture-backed `QA_COARSE` and, for explicit degraded/unusable coordinates, `QA_DENSE` through the same raw/parsed/enriched/barrier chain. The completion projector produces `QA_COMPLETE` or `QA_INCOMPLETE` from exact six-camera evidence. A separate deterministic supplemental QA consumer verifies every explicitly targeted PNG's media type, byte count, and digest and persists exact-pinned local evidence; it does not invent a semantic blur or occlusion claim. All results remain non-promotable. | Real model quality, adaptive suspicion policy, governed O-10 thresholds, semantic visual-quality classification, and external corpus validation remain unresolved. |
| Event processing | `EventProposalProjector` normalizes authoritative EVENT_PROPOSAL outputs into six-camera facts, `CandidateReducer` deterministically merges compatible intervals, `ActionEvidenceProjector` emits exact six-camera evidence, and `ProvisionalPhysicalActionFuser` produces ordered 0/1/N coarse actions. `BoundaryRefinementProjector` validates role-bound ONSET/OFFSET closure and emits one refined result per action. A versioned final-fusion context then binds all refined actions to provider requests and requires explicit zero output or exact 1:1 final hypotheses before admission. Run/attempt locators do not enter logical identity and all outputs remain non-production. | Governed O-11/O-12 association/ontology/tolerance policies and real-model quality remain absent. |
| Queueing | `queue/barrier.py` provides deterministic aggregation and `adapters/sqlite_barrier.py` persists generic/inference-call barriers plus an exact-pinned persisted-barrier snapshot. `SQLiteWorkScheduler` provides an authoritative durable ledger with dependency progression, atomic claim/CAS, lease epoch and fencing token, retry wait, deadlines, hard expiry, cancellation, skip, invalidation, and restart recovery. The canonical local command uses it for `ACTION_PUBLISH` commit/reconciliation. `OutboxRelay` exact-validates the registered event-identity outbox Wire payload and is composed after primary completion, including recovered completion; local delivery failures remain visible in the receipt and durable retry/DLQ state without replacing primary truth. | The inference stages are not a production work DAG, and scheduler terminalization is not in the primary-completion transaction. The local SQLite sink is not a broker. Production database/broker selection, concurrency/isolation, operator reconciliation and DLQ ownership, metrics, credentials, and O-14 recovery policy remain open. |
| Retrieval | Append-only event revisions/current selections, structured filters, lexical ranking, optional fail-closed reranking, and clip/provenance registration are local. | Production index/storage, clip service, ontology, and retrieval SLO remain open. |
| Benchmarking | Offline QA/event/boundary/calibration metrics, grouped leakage-safe splits, clustered local statistics, and fail-closed promotion gates are executable. Every calculation binds an explicit content-addressed metric-definition policy; unsupported PR-AUC/mAP labels are not emitted. A deterministic synthetic capacity harness reports recording-hours and camera-video-hours, backlog, deadline, utilization, SLO, and same-profile regressions while remaining `SYNTHETIC_LOCAL` and `NOT_MEASURED`. Throughput reports require a complete evidence context before becoming `MEASURED`. | No governed metric policy, ranked scoring contract for AP/AUC, frozen ground truth, O-16 thresholds/power, representative production load, long soak, or measured capacity evidence exists. Evidence binding and synthetic simulation are mechanisms, not external approval. |

## Architecture Section 25 Conformance Matrix

These statuses assess every normative subsection as a whole: `COMPLETE` means all requirements in
that subsection have live implementation and local evidence; `PARTIAL` means only a stated subset
is executable; `BLOCKED` means an authority or policy gate prevents the required implementation or
promotion; and `ABSENT` means there is no runnable implementation. None of these labels is a phase
exit or production-readiness claim.

| Section | Status | Live evidence | Remaining boundary or blocker |
|---|---|---|---|
| 25.1 VLM trust boundary | PARTIAL | Provider outputs are preserved as typed raw bytes, parsed as untrusted claims, enriched against the exact input-plan catalog, and reduced only after validation across QA, proposal, action-evidence, ONSET/OFFSET boundary-refinement, and final local-inference stages. The RunPod adapter applies the same binding and fail-closed parser boundary under mock transport. | Only fixture-backed inference is composed. Real Qwen/RunPod capability, provider governance, credentials, artifact storage, and governed replay evidence are absent. |
| 25.2 provider-neutral packages and durable call barrier | PARTIAL | Temporal packages remain provider-neutral; provider limits and render details live in `InferenceInputPlan`; independently retried call parts and deterministic reduction are executable. The local composition persists generic and inference-call barrier facts and can recover without provider redispatch. The persisted generic-barrier snapshot and lease-bound work-message projection have registered exact schemas. The SQLite scheduler now coordinates `ACTION_PUBLISH` and deterministically reconciles recovered primary completions. | The scheduler does not yet own the inference-stage DAG, and its database is outside the primary-completion transaction. Production broker topology, real provider qualification, concurrency/isolation, credentials, observability, and O-14 recovery ownership remain open. |
| 25.3 exact and adaptive sampling | PARTIAL | Exact rational-grid selection and materialization are tested. `sampling/adaptive.py` executes injected detectors and resolves frozen trigger evidence deterministically. The canonical raw-MCAP path now independently connects its registered local media-quality report to frozen explicit neighbor targets, provider-neutral PNG materialization, exact artifact-byte verification and registered supplemental QA evidence on fresh execution, with exact evidence/artifact revalidation on recovery. Independent Python and Node BigInt implementations verify the checked-in half-even, negative-index, clipping, tolerance, tie-break, decode-failure, and dedupe vectors. | The generic `AdaptiveSampler` detector-trigger stream is not yet composed into canonical package planning or published as governed trigger evidence. Low edge energy remains a local proxy rather than semantic blur/occlusion classification. O-13 owns promoted rates, padding, thresholds, trigger policy, and budgets. |
| 25.4 logical identities and run membership | PARTIAL | Generic logical nodes and the complete local QA/event/boundary/fusion chain use run-independent projections and append-only memberships. Retry/run changes preserve logical keys, and proposal replay tests prove exact manifest bytes remain audit facts rather than semantic identity. The durable scheduler supports explicit `INVALIDATED` terminal state. | Canonical invalidation propagation and governed successor rebuild policy are not yet wired to the scheduler. |
| 25.5 immutable revisions and current selection | PARTIAL | Generic immutable node revisions, append-only selection decisions, current-selection compare-and-swap, rebuild, and local verification are implemented. Deterministic local ActionEvent genesis revisions, selections, current projections, and identity current-revision references are applied with identity/completion/outbox facts in one local aggregate transaction. | The governed production ActionEvent contract and REUSED successor-selection policy remain absent; generic revision `ELIGIBLE` means locally selectable, not production-qualified. |
| 25.6 validation, READY, and separate ledgers | PARTIAL | V2 validation/READY/alignment contracts and local source/alignment ledger reconciliation exist behind injected ports and reject insufficient V1 evidence. The raw canonical bridge derives exact source/schema, six-stream decoder-probe, registered-media, frame-index, and alignment facts from an explicitly mapped real MCAP before resolving V2 admission. | The exercised mapping and development admission/alignment policies are unapproved. Governed O-03/O-04 decisions, production stores, independent promoted ledger reconciliation, and Phase 0 approval are absent. |
| 25.7 exact schema references and evolution | PARTIAL | Atomic single-schema, 1..N independent-schema bundle, and one-target/1..N-incoming-upcaster evolution publication are executable. Publication stages exact schema/code/runtime/golden artifacts as applicable, validates the complete temporary registry and upcaster graph, and makes catalog replacement the commit point with marker-based recovery. Exact digest verification, published-entry immutability, deterministic upcasting, and golden-vector enforcement are tested. The 64-entry catalog includes exact-pinned output, work/barrier, outbox, review, media-quality, supplemental-QA, pre-EOS capture/segment/window, stream inference evidence, stream work, expected-window closure, finalization, and Artifact Registry V3 contracts. | The live catalog still has no domain upcaster (`upcasters=[]`). Supplemental V1 remains byte-frozen and corrected V2 was published separately; both declare `compatibility_mode=NONE`, so no migration edge is implied. Bundle publication is mechanism evidence, not approved business transformation semantics. Existing completion versions likewise cannot be retrofitted without mutating published governance metadata. |
| 25.8 nonblocking human review | PARTIAL | Versioned nonblocking routing covers the five Section 25.8 triggers. The canonical local receipt routes only after primary completion; media-quality flags route the report through `QA_DEGRADATION`. In the verified no-quality-flag real-source flow, fresh work is `ENQUEUED` and exact replay is `ALREADY_ENQUEUED`. An early `NO_EVENTS` completion without an output decision routes the primary-completion subject for review sampling. A routing exception remains observable without replacing primary truth. Immutable review tasks, annotations/adjudications, optional revision/selection references, priority/SLA ordering, lease/fence, expired reclaim, exact replay/conflict, reopen history, and independently visible pending/overdue state are executable in a local SQLite queue. | The selected local routing policy is unapproved and non-promotional. Governed O-10/O-11/O-12 routing and blocking policies, named reviewers and service owner, real capacity, backlog/latency evidence, escalation operations, review-to-authored-revision/selection composition, and SLA ownership remain absent. |
| 25.9 recording-scoped serialized identity | PARTIAL | The one-command local path obtains a recording snapshot, prepares identity side-effect-free, and applies it with run completion, ActionEvent genesis, and outbox under one generation/fence CAS transaction. `EventIdentityOutboxRecord` retains its original domain shape where it is embedded in frozen completion detail; SQLite persistence and publication project it to the exact-pinned `EventIdentityOutboxWireRecord`. Exact replay creates no duplicate business result or outbox. Primary-completion command v2 also binds ordered media-quality and supplemental-QA evidence references when present. The command relays committed and recovered-completion outbox rows with local at-least-once delivery, retry, acknowledgement, exact-byte idempotency, stale-fence rejection, and DLQ state; recovery performs this authoritative reconciliation before validating receipt-side evidence, and delivery failure is observable but does not revoke primary completion. | This is local conformance evidence, not the O-14 production authority. A selected production broker, authenticated delivery, operator-owned reconciliation/DLQ procedures, monitoring, retention, filesystem power-loss guarantees including parent-directory `fsync`, and governed merge/split/REUSED successor behavior remain absent. |
| 25.10 security and phase dependencies | BLOCKED | Later phases run only as local conformance slices with `production_eligible=false`. Coarse/dense QA, proposal, candidate reduction, per-candidate action evidence, provisional fusion, per-action ONSET/OFFSET boundary refinement, and exact 0/1/N final-fusion handling are runnable without provider traffic. | Phase 0 and open O-decisions still block promotion. Real providers, governed policies, and later production stages remain fail-closed rather than fabricated. |
| 25.11 promotion evidence | PARTIAL | Focused local tests cover canonical 0/1/N execution, replay, boundary refinement, call reduction, SQLite inference/barrier/work/outbox/review recovery, scheduler post-commit reconciliation, recovered pending-outbox delivery, sink-corruption visibility without primary rollback, atomic completion, exact Wire validation, Python/Node rational-grid vectors, content-addressed benchmark policy, and synthetic capacity/SLO regression semantics. | Governed source/provider replay, representative production load, production outbox/broker evidence, approved metric/review policies, frozen ground truth, thresholds, security regression, and long soak remain missing. Direct conformance cases for rendering one provider-neutral package into different provider plans without changing package identity, concurrent different-recording identity isolation, and production-like Phase 0 route non-bypass are also still absent. |

### Named-file readiness

| File | Status | Exact executable boundary |
|---|---|---|
| `sampling/adaptive.py` | PARTIAL | The detector coordinator, deterministic trigger ordering/filtering, bounded hysteresis rate reduction, and frozen-artifact target resolver are implemented and tested. The generic detector-trigger stream is not yet composed into canonical package planning; governed thresholds and trigger publication remain absent. |
| `sampling/supplemental.py` | PARTIAL | Frozen explicit media-quality targets are materialized into provider-neutral PNG packages with deterministic nearest/tie/dedupe behavior and exact artifact identities. The local raw-MCAP composition executes and recovers this path; promoted budgets and trigger policy remain O-13 inputs. |
| `qa_pipeline/supplemental.py` | PARTIAL | The deterministic supplemental consumer reads each selected artifact, verifies media type, byte count, and SHA-256, and emits registered local evidence. It is availability/integrity evidence only, not a real-model or semantic blur/occlusion classifier. |
| `qa_pipeline/coarse.py` | PARTIAL | `CoarseQAProjector` validates authoritative `QA_COARSE` enriched coverage and emits a local result consumed by the separate completion gate. It does not itself authorize fusion or execute dense QA; every result is `production_eligible=false`. Model-quality calibration remains absent. |
| `qa_pipeline/completion.py` | PARTIAL | The deterministic three-state completion gate, explicit zero-child dense outcome, exact dense-work manifest, and six-camera all-GOOD aggregate are implemented and tested. It does not execute dense inference, and every result is `production_eligible=false`. |
| `qa_pipeline/dense.py` | PARTIAL | `DenseQAProjector` and the canonical runner execute exact planned dense work through the shared inference evidence chain. Real-model quality and governed dense policy remain external. |
| `event_pipeline/proposer.py` | PARTIAL | `EventProposalProjector` validates plan-bound enriched outputs, preserves attempt provenance separately, and emits stable `CLAIMS` or `NO_EVENTS` identities. The legacy `EventProposer.propose()` remains fail-closed. |
| `event_pipeline/candidate.py` | PARTIAL | `CandidateReducer` deterministically merges connected proposal intervals, binds policy and proposal-result digests, emits source-bound empty reductions, and keeps all candidates non-production. Legacy mutation APIs remain fail-closed; governed merge/split lifecycle is absent. |
| `event_pipeline/evidence.py` | PARTIAL | `ActionEvidenceProjector` validates candidate-scoped ACTION_DENSE package/input-plan lineage, authoritative enriched claims, exact part coverage, time/evidence bounds, and all six camera slots, then emits stable non-production `SUPPORTED`, `NO_ACTION`, or `INDETERMINATE` evidence. The incompatible legacy `ActionEvidenceExtractor.extract_evidence()` entry remains fail-closed. Governed ontology/calibration and real-model quality remain absent. |
| `event_pipeline/provisional_fusion.py` | PARTIAL | `ProvisionalPhysicalActionFuser` validates an exact candidate/ACTION_EVIDENCE closure and deterministically emits ordered 0/1/N coarse physical actions with six explicit camera slots, source closure, ambiguity, versioned semantic identities, and `production_eligible=false`. Compatible candidates may merge and one candidate may split across disconnected or incompatible positive evidence. Missing, duplicate, foreign, or indeterminate evidence fails closed. | Boundary refinement, governed O-11/O-12 association and ontology, stable event identity, revisions, and publication remain downstream and are intentionally not performed here. |
| `event_pipeline/boundary_refinement.py` | PARTIAL | `BoundaryRefinementProjector` validates separate orchestrator-owned ONSET/OFFSET windows, exact package/input-plan/alignment/enriched-output closure, uncertainty, context truncation, and all six camera slots, then deterministically reduces exactly one non-production result per provisional action without an `event_id`. | Real-model quality and governed O-12 boundary tolerance remain downstream. |
| `queue/barrier.py` | PARTIAL | The coordinator and terminal criticality semantics are executable. SQLite persists generic and inference-call facts, emits a registered exact-pinned barrier snapshot, and proves replay/reopen/crash recovery. The durable scheduler supplies deadline/lease/fence/invalidation behavior, emits a registered lease-bound work-message projection, and coordinates local `ACTION_PUBLISH` recovery. It does not yet own the full inference DAG or a production transaction/broker boundary; concurrency qualification and O-14 recovery policy remain external. |

## Explicit Blockers

The following are deliberately skipped rather than replaced with convenient defaults:

- Phase 0 security, privacy, retention, audit, residency, and provider-data-governance approval.
- O-03 source topics/schemas/camera roles/codecs and O-04 clock/reset/alignment policy.
- O-06/O-07 real Qwen/GPT deployment, limits, cost, data handling, and shadow approval.
- O-10/O-11/O-12 QA taxonomy thresholds, action ontology, candidate identity, merge/split,
  boundary tolerance, calibration, and review policy.
- O-13 promoted sampling rates, padding, adaptive triggers, and frame budgets.
- O-14 production database, broker, object store, vector index, isolation, and recovery.
- O-16 governed ground truth, annotator agreement, statistical power, and numeric promotion
  thresholds.
- Section 25.7 governed domain upcaster selection. Atomic publication infrastructure now accepts
  one new target plus 1..N incoming edges with exact code/runtime/golden artifacts and validates
  the complete temporary registry and graph before the catalog commit. The 47-entry catalog
  includes work-message,
  persisted-barrier, event-identity outbox, review-task, review-annotation, and review-reopen-command
  contracts, media-quality-report V1, frozen supplemental-QA-evidence V1, and corrected supplemental-QA-evidence V2 beside the standalone V2 output contracts. It still declares `upcasters=[]`: existing
  completion targets were published with `compatibility_mode=NONE` and empty predecessor sets, so
  retroactive chains would mutate published catalog governance. Detailed V1 also lacks required
  QA/stage evidence. A future governed business chain must target new versions; the publication
  mechanism does not supply or approve the missing transformation semantics.
- Production barrier/work recovery. The local SQLite scheduler now coordinates `ACTION_PUBLISH`
  and reconciles fresh or recovered primary completion, while inference still uses the run-scoped
  barrier/evidence composition. Scheduler state and primary completion are in separate databases.
  A production work DAG, shared transaction/fence policy, concurrency/isolation qualification,
  broker credentials, observability, and O-14 recovery ownership remain absent.
- Production outbox delivery. The canonical local command now relays committed and recovered
  outbox rows to an exact-byte idempotent SQLite sink. Delivery failure enters durable retry/DLQ
  state and remains visible without replacing primary completion. This does not select or qualify
  a production broker, authenticated route, retention policy, monitoring, reconciliation owner,
  or operator redrive procedure.
- Governed raw-MCAP admission and publication under approved O-03/O-04 policy and production
  ledgers. The local command now derives registered V2 validation/READY/alignment evidence from
  exact raw facts, but its explicitly unapproved development policy cannot be promoted. Frozen V1
  evidence remains insufficient and cannot be promoted as V2.

## External Promotion Inputs

The remaining promotion conditions are not all model integration. Internal implementation can
continue behind explicit ports, but the repository cannot manufacture the following approvals,
owners, infrastructure facts, or operational evidence:

| Authority | Still required outside local conformance | Required owner or evidence authority |
|---|---|---|
| Phase 0 and O-15 | Approved classification/threat model, provider data-use and residency terms, RBAC/service identities, secrets, encryption, audit, retention/deletion/legal hold, incident response, and non-bypass tests. | Security, privacy, legal, and platform owners with immutable approval evidence. |
| O-03 and O-04 | Approved source topic/schema/camera-role/codec profile and approved clock/reset/alignment/tolerance policy over a governed representative MCAP corpus. | Source-system plus hardware/data owners; corpus scan and calibration benchmark. |
| O-06 and O-07 | Qualified Qwen/GPT endpoint and exact model/capability pins, provider quotas/limits/cost/data handling, real credentials through the approved secret path, and primary/shadow authorization. | Model platform and model-evaluation/governance owners. |
| O-10, O-11, O-12, O-13, and O-16 | Registered and approved QA severity/acceptance labels, action/hand/object/relation ontology, ambiguity/merge/split/review risk classes, boundary/calibration rules, sampling matrix, frozen ground truth, agreement targets, and numeric promotion thresholds. | Annotation/product/domain, review, data-science, and finance owners named in the applicable policy artifacts. |
| O-14 | Selected production database, broker, object/artifact store, vector index, deployment region, isolation, authenticated identities/credentials, backup/restore, disaster recovery, monitoring, DLQ/redrive, and reconciliation authority. | Platform architecture plus measured integration, recovery, and load-test evidence. |
| O-01, O-02, O-14, O-16, and Section 25.11 | Representative workload and peak-arrival profile, both 500-hour interpretations until resolved, T+1 clock/cohort, live provider and infrastructure pins, registered warm-up/burst/repetition/long-soak policy, failover and primary-isolation runs, headroom, cost, and statistical confidence. | Data operations, product/operations, platform, model platform, and data science. Synthetic reports remain `SYNTHETIC_LOCAL` and `NOT_MEASURED`. |
| Section 25.7 evolution | Approved business transformation semantics and owner for new target versions, with exact code/runtime/golden artifacts; published `NONE` entries cannot be retrofitted. | Contract/schema owner and domain owner. The current empty upcaster graph is mechanism evidence only. |
| Schema/release gate | A clean exact candidate commit, a `SCHEMA_BASELINE_REF` naming a protected prior release commit or tag, event-baseline comparison, required code-owner/release approvals, and archive/manifest publication from that unchanged commit. | Repository/release governance. A local `origin/main` remote-tracking ref does not prove server-side protection or candidate approval. |

## Local Gate Snapshot

The following facts were verified locally on 2026-07-22. They establish local mechanism
conformance only. The full-gate results below describe the pre-commit candidate working tree;
candidate commit identity, clean-clone release evidence, and protected-baseline immutability remain
unverified until those gates run against the unchanged commit:

- The full Python suite completed with 924 passed and 3 skipped in 1547.61 seconds.
- `ruff check .` passed; `ruff format --check .` reported 261 files already formatted.
- Strict Mypy passed across 166 source files.
- Schema registry verification passed for 64 exact-pinned JSON Schema 2020-12 documents and zero
  registered upcaster artifact sets. It includes media-quality report V1, byte-frozen supplemental
  QA evidence V1, and corrected supplemental QA evidence V2.
- Documentation link checking and `git diff --check` passed.
- The checked-in `sample-medium.mcap` spans 40.890455 seconds, so the 180-second first-segment cap
  covered the entire recording.
- The verified local-composition v18 fresh real-source command exited 0 in 853.8 seconds and
  returned `SUCCEEDED`: 16 deterministic fixture-backed inference calls, zero network calls, one
  outbox row delivered on relay attempt 1, review `ENQUEUED`, no local quality flags, and no
  supplemental QA evidence.
- Exact replay exited 0 in 26 seconds with the same command, run, completion, event, revision, and
  outbox identifiers, zero fixture calls, zero relay attempts, and review `ALREADY_ENQUEUED`.
- SQLite corroborated one committed completion, one `SUCCEEDED` processing run, one delivered
  outbox row, one sink row, `ACTION_PUBLISH` `SUCCEEDED`, and one `PENDING` review task. The
  completion evidence contains a `MEDIA_QUALITY_REPORT` reference. Six camera ledgers contain
  7,350 observations; cross-camera skew p50/p95/max is 33,000/84,000/108,000 ns.
- The fresh and replay evidence remains `LOCAL_CONFORMANCE` with `production_eligible=false` and
  does not establish production or model-quality qualification.
- Representative production capacity, long-soak, real-provider, production broker/storage, and
  protected Schema-baseline approval remain outside this local run.

SQLAlchemy/Redis/provider adapters are not silently emulated as production infrastructure.
Missing optional dependencies fail closed. No implementation may turn a local fake score,
empty cohort, synthetic replay, or archived report into `MEASURED` or promotional evidence.

## Current Interpretation

The repository has one connected source-to-completion path for an immutable fixture or explicitly
mapped raw MCAP. The raw path first publishes and validates registered media-quality evidence; a
nonempty explicit neighbor-target plan is materialized as exact PNG artifacts and registered
supplemental QA evidence. On fresh execution, the path continues through processing-run membership,
identity preparation, ActionEvent genesis, atomic completion, and pending outbox. Primary-completion
command v2 binds the ordered side-evidence closure by role, exact Schema reference, semantic digest,
exact-byte digest, and byte count. Same-run and cross-run replay reuse durable logical/inference facts
without provider redispatch. On recovery, the local composition first uses authoritative primary
completion to reconcile durable `ACTION_PUBLISH` work and committed outbox delivery, then revalidates
the referenced report, supplemental evidence, and selected artifacts before constructing a receipt;
a side-artifact mismatch therefore fails closed without erasing primary truth. Nonblocking review
follows successful receipt-side validation. Delivery or review failure remains observable and does
not replace primary truth.

These are connected local components, not a production execution topology. Primary completion,
work scheduling, the sink, and review queue remain separate SQLite authorities. External artifact
storage, broker choice, authenticated identities/credentials, production isolation and recovery,
operator reconciliation, filesystem power-loss semantics including parent-directory `fsync`,
governed provider routes, and O-14 ownership remain open. Work behind an authority gate still
stops at an explicit port, policy input, or fail-closed state.

The locally selected nonblocking review policy and priority/SLA queue are implementation evidence,
not an approved review service. The queue persists exact-pinned task, annotation, and reopen-command
contracts, but supplies no governed labels or thresholds, named reviewer/service ownership, real
capacity, backlog/latency qualification, escalation operations, or production SLA. The RunPod
adapter is likewise mock-transport preparation only; no real endpoint or credential was used.

Injected generic adaptive detectors can be executed and reduced deterministically, but their
detector-trigger stream is not yet composed into canonical package planning. Separately, registered
local media-quality observations now drive exact explicit neighbor targets, PNG materialization,
and deterministic supplemental QA evidence in the raw-MCAP fresh/recovery path. Low edge energy is
a proxy and does not establish semantic blur or occlusion. Governing these observations, thresholds,
rates, and budgets remains blocked by O-13. Compact and detailed primary-completion
records now have exact registered schemas and versioned semantic projections; compact V1 remains
frozen/readable without implicit upcast. Side-effect-free identity and ActionEvent genesis
preparation feeds the local aggregate for CREATED/AMBIGUOUS assignments and exact replay. The
local composition invokes this sequence after the runner; REUSED successor lineage, governed
production ActionEvent contracts, external artifact storage, delivery, and production recovery
remain open.

The removed surface includes `robata.contracts.mainline`, `robata.ports.mainline`, their
smoke-only bundle/report/stage contracts, and the duplicate legacy model-adapter port. There are no
compatibility shims. Shared contracts and frame-materialization boundaries that still serve the
canonical, QA, event, and sampling code live under `robata.contracts.pipeline` and
`robata.ports.frame_materialization`.

The local output proof, output decision, and event-hypothesis payloads are exact-pinned V2
contracts; their registered validator requires the schema quartet out of band so the already
published completion-detail V4 nesting remains byte-for-byte unchanged. V1 payloads fail closed.
The canonical run binding remains `canonical-offline-v5`;
execution-policy semantic projection v3, fusion projector policy v2, local composition v18,
raw-MCAP source-binding policy v5, primary-completion command projection v2, and runtime-policy
projection v8 prevent recovery under earlier incompatible policy namespaces.

The canonical runner executes coarse QA and any exact planned dense QA before deterministic QA
completion. `QA_COMPLETE` enters EVENT_PROPOSAL and candidate reduction, then executes a
candidate-scoped ACTION_DENSE to ACTION_EVIDENCE chain for every candidate. The deterministic
provisional fuser validates that exact closure and emits ordered 0/1/N coarse physical actions.
Every action then executes separate padded ONSET/OFFSET boundary windows through the shared
raw/parsed/enriched/barrier chain and reduces to one exact six-camera refined result. The runner
builds one versioned final-fusion context from the complete ordered refined-action set, binds it to
the input plan and adapter request, and rejects any nonempty result that does not cover that set
exactly once. Zero proposals/actions and explicit empty final fusion remain distinct source-bound
`NO_EVENTS` outcomes; proposal/action/boundary failure and indeterminate evidence stop explicitly;
retry/run identity remains provenance. No QA or downstream local result is production eligible.

The reusable package-set, input-plan, and inference identities now exclude exact artifact locators
and manifests while preserving them in audit validation. Locally persisted derived hashes created
by the former locator-polluted formula are incompatible evidence and must be rebuilt; no registered
wire shape or version changed. The output contracts changed wire version before their V2 schemas
were registered; older local payloads must still be rebuilt rather than relabeled.

No Architecture V1.1 phase is declared complete by this document. Real-model integration is a
later gated adapter task, not a prerequisite for continuing contract, orchestration, replay,
benchmark, and local state-machine development.

## Verification Commands

The live baseline is checked with the commands in the repository README. Results are meaningful
only for the exact worktree and environment in which they were run. Quality, capacity, and SLO
measurements remain `NOT_MEASURED` until governed corpora and registered benchmark inputs exist.
