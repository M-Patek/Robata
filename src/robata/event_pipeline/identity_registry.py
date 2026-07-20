"""Recording-scoped event identity allocation and assignment.

This module is deliberately separate from the legacy fusion engine. Fusion
produces hypotheses; this registry is the only boundary that can allocate a
stable event identity. The in-memory implementation is a reference adapter
for tests and local execution. A durable adapter must preserve the same
compare-and-swap and append-only guarantees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, Protocol, Self
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import (
    NanosecondInterval,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp

if TYPE_CHECKING:
    from robata.admission.context import AdmittedRecordingContextV2
    from robata.inference.enrichment import OrchestratorEnrichedOutput

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

ADMISSION_PROOF_SEMANTIC_PROJECTION_VERSION: Final = "admission-proof-semantic-v2"
OUTPUT_ADMISSION_SEMANTIC_PROJECTION_VERSION: Final = "output-admission-semantic-v2"
EVENT_HYPOTHESIS_SEMANTIC_PROJECTION_VERSION: Final = "event-hypothesis-semantic-v2"
EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE: Final = "event-hypothesis-v2"


class AdmissionEvidenceClass(StrEnum):
    """Strength of evidence behind an admission decision."""

    LOCAL_CONFORMANCE = "LOCAL_CONFORMANCE"
    GOVERNED_BENCHMARK = "GOVERNED_BENCHMARK"
    PRODUCTION_QUALIFIED = "PRODUCTION_QUALIFIED"


def validate_evidence_eligibility(
    evidence_class: AdmissionEvidenceClass,
    production_eligible: bool,
) -> None:
    """Keep evidence strength separate from the admission outcome."""

    if not isinstance(evidence_class, AdmissionEvidenceClass):
        raise TypeError("evidence_class must be an AdmissionEvidenceClass")
    if not isinstance(production_eligible, bool):
        raise TypeError("production_eligible must be a boolean")
    if evidence_class is AdmissionEvidenceClass.PRODUCTION_QUALIFIED:
        raise ProductionQualificationUnavailableError(
            "PRODUCTION_QUALIFIED requires the governed qualification gateway"
        )
    if production_eligible:
        raise ValueError("production_eligible must remain false without production qualification")


class EventIdentityAssignmentDisposition(StrEnum):
    """Registry outcome for one hypothesis under one identity policy."""

    CREATED = "CREATED"
    REUSED = "REUSED"
    AMBIGUOUS = "AMBIGUOUS"


class EventIdentityCandidateRelation(StrEnum):
    """How one scored existing identity relates to a hypothesis."""

    SAME_EVENT = "SAME_EVENT"
    SPLIT_FROM = "SPLIT_FROM"
    MERGED_FROM = "MERGED_FROM"
    SUPERSEDES = "SUPERSEDES"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"


class EventIdentityAssignmentRelation(StrEnum):
    """Summary relations persisted on an assignment."""

    NEW_IDENTITY = "NEW_IDENTITY"
    SAME_EVENT = "SAME_EVENT"
    SPLIT_FROM = "SPLIT_FROM"
    MERGED_FROM = "MERGED_FROM"
    SUPERSEDES = "SUPERSEDES"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"


class EventIdentityRegistryError(RuntimeError):
    """Base class for registry and transaction failures."""


class EventIdentityInputError(EventIdentityRegistryError, ValueError):
    """A hypothesis, policy, or resolution is invalid."""


class ProductionQualificationUnavailableError(EventIdentityInputError):
    """Production qualification cannot be asserted without its governed gateway."""


class EventIdentityConflictError(EventIdentityRegistryError):
    """Append-only state conflicts with an existing record."""


class CrossRecordingEventIdentityError(EventIdentityConflictError):
    """An identity candidate or allocation crosses recording scope."""


class EventIdentityAllocationError(EventIdentityRegistryError):
    """The injected allocator did not return a valid fresh identity."""


class StaleEventRegistryFenceError(EventIdentityRegistryError):
    """A compare-and-swap used an old recording generation or fence."""

    def __init__(self, recording_identity: str, message: str | None = None) -> None:
        self.recording_identity = recording_identity
        super().__init__(message or f"stale event identity registry fence: {recording_identity}")


class EventIdentityPolicyRef(StrictModel):
    """Immutable identity-policy reference used by the idempotency key."""

    version: SchemaVersion
    semantic_sha256: Sha256Digest


class ProductionOutputAdmissionPolicyRef(StrictModel):
    """Pinned policy allowed to admit enriched outputs into the identity domain."""

    version: SchemaVersion
    semantic_sha256: Sha256Digest


class AdmissionProof(StrictModel):
    """Immutable primary-admission proof with an explicit evidence tier."""

    decision: Literal["ADMITTED"]
    evidence_class: AdmissionEvidenceClass
    production_eligible: bool
    recording_identity: Sha256Digest
    admitted_context_semantic_sha256: Sha256Digest
    admission_policy_version: SchemaVersion
    admission_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        validate_evidence_eligibility(self.evidence_class, self.production_eligible)
        return self

    @classmethod
    def from_context(
        cls,
        context: AdmittedRecordingContextV2,
        *,
        evidence_class: AdmissionEvidenceClass = AdmissionEvidenceClass.LOCAL_CONFORMANCE,
    ) -> Self:
        """Derive the proof only from a validated admitted context."""

        from robata.admission.context import AdmittedRecordingContextV2

        if not isinstance(context, AdmittedRecordingContextV2):
            raise EventIdentityInputError(
                "identity assignment requires registered V2 admission evidence"
            )
        if not context.evaluation.admissible:
            raise EventIdentityInputError(
                "an inadmissible context cannot produce an admission proof"
            )
        production_eligible = evidence_class is AdmissionEvidenceClass.PRODUCTION_QUALIFIED
        validate_evidence_eligibility(evidence_class, production_eligible)
        return cls(
            decision="ADMITTED",
            evidence_class=evidence_class,
            production_eligible=production_eligible,
            recording_identity=context.recording_identity,
            admitted_context_semantic_sha256=context.semantic_sha256,
            admission_policy_version=context.policy.version,
            admission_policy_sha256=context.policy.semantic_sha256,
        )

    @classmethod
    def from_local_conformance_context(
        cls,
        context: AdmittedRecordingContextV2,
    ) -> Self:
        """Create the only admission proof issued by the offline canonical path."""

        return cls.from_context(
            context,
            evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        )


class PlatformEnrichedOutputReference(StrictModel):
    """Reference to orchestrator enrichment, never raw provider bytes."""

    authority: Literal["ORCHESTRATOR_ENRICHED"]
    recording_identity: Sha256Digest
    enrichment_logical_key: NodeLogicalKey
    enriched_output_semantic_sha256: Sha256Digest
    enrichment_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if not self.enrichment_logical_key.startswith("orchestrator-enrichment:"):
            raise ValueError("event hypotheses require an orchestrator enrichment logical key")
        return self

    @classmethod
    def from_output(cls, output: OrchestratorEnrichedOutput) -> Self:
        """Create a lineage reference from a validated enriched output."""

        return cls(
            authority="ORCHESTRATOR_ENRICHED",
            recording_identity=output.authority.recording_identity,
            enrichment_logical_key=output.enrichment_logical_key,
            enriched_output_semantic_sha256=output.semantic_sha256,
            enrichment_policy_version=output.enrichment_policy_version,
        )


def platform_enriched_output_logical_projection(
    reference: PlatformEnrichedOutputReference,
) -> dict[str, object]:
    """Project reusable enrichment identity without exact artifact evidence."""

    return {
        "authority": reference.authority,
        "recording_identity": reference.recording_identity,
        "enrichment_logical_key": reference.enrichment_logical_key,
        "enrichment_policy_version": reference.enrichment_policy_version,
    }


class ProductionAdmittedHypothesisFact(StrictModel):
    """Non-circular semantic facts admitted from one enriched fusion claim."""

    fusion_output_ordinal: NonNegativeInt
    effective_interval: NanosecondInterval
    semantic_fingerprint_sha256: Sha256Digest
    fusion_logical_key: NodeLogicalKey

    @model_validator(mode="after")
    def validate_fusion_key(self) -> Self:
        if not self.fusion_logical_key.startswith("fusion:"):
            raise ValueError("admitted hypothesis fact requires a fusion logical key")
        return self


class OutputAdmissionProof(StrictModel):
    """Immutable admission of exact enriched outputs with evidence qualification."""

    schema_version: Literal["2.0"]
    decision: Literal["ADMITTED"]
    evidence_class: AdmissionEvidenceClass
    production_eligible: bool
    recording_identity: Sha256Digest
    source_enrichments: tuple[PlatformEnrichedOutputReference, ...]
    admitted_hypothesis_facts: tuple[ProductionAdmittedHypothesisFact, ...]
    output_admission_policy_version: SchemaVersion
    output_admission_policy_sha256: Sha256Digest
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_proof(self) -> Self:
        validate_evidence_eligibility(self.evidence_class, self.production_eligible)
        if not self.source_enrichments:
            raise ValueError("output admission requires enriched output lineage")
        keys = tuple(item.enrichment_logical_key for item in self.source_enrichments)
        if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
            raise ValueError("output admission lineage must be unique and canonical")
        if any(
            item.recording_identity != self.recording_identity for item in self.source_enrichments
        ):
            raise ValueError("output admission lineage crosses recording scope")
        if not self.admitted_hypothesis_facts:
            raise ValueError("output admission requires admitted hypothesis facts")
        expected_facts = tuple(
            sorted(self.admitted_hypothesis_facts, key=_admitted_hypothesis_fact_sort_key)
        )
        if self.admitted_hypothesis_facts != expected_facts:
            raise ValueError("admitted hypothesis facts must use canonical order")
        ordinals = tuple(item.fusion_output_ordinal for item in self.admitted_hypothesis_facts)
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("admitted hypothesis facts require unique fusion output ordinals")
        expected = semantic_sha256(output_admission_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("output admission semantic_sha256 is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        recording_identity: str,
        source_enrichments: Sequence[PlatformEnrichedOutputReference],
        admitted_hypothesis_facts: Sequence[ProductionAdmittedHypothesisFact],
        policy: ProductionOutputAdmissionPolicyRef,
        evidence_class: AdmissionEvidenceClass = AdmissionEvidenceClass.LOCAL_CONFORMANCE,
    ) -> Self:
        """Create a proof for exact enriched outputs and admitted claim facts."""

        production_eligible = evidence_class is AdmissionEvidenceClass.PRODUCTION_QUALIFIED
        validate_evidence_eligibility(evidence_class, production_eligible)
        refs = tuple(sorted(source_enrichments, key=lambda item: item.enrichment_logical_key))
        facts = tuple(admitted_hypothesis_facts)
        if any(not isinstance(item, ProductionAdmittedHypothesisFact) for item in facts):
            raise TypeError(
                "admitted_hypothesis_facts must contain ProductionAdmittedHypothesisFact values"
            )
        facts = tuple(sorted(facts, key=_admitted_hypothesis_fact_sort_key))
        values: dict[str, object] = {
            "decision": "ADMITTED",
            "evidence_class": evidence_class,
            "production_eligible": production_eligible,
            "recording_identity": recording_identity,
            "source_enrichments": refs,
            "admitted_hypothesis_facts": facts,
            "output_admission_policy_version": policy.version,
            "output_admission_policy_sha256": policy.semantic_sha256,
        }
        digest = semantic_sha256(output_admission_projection_values(values))
        return cls(
            schema_version="2.0",
            decision="ADMITTED",
            evidence_class=evidence_class,
            production_eligible=production_eligible,
            recording_identity=recording_identity,
            source_enrichments=refs,
            admitted_hypothesis_facts=facts,
            output_admission_policy_version=policy.version,
            output_admission_policy_sha256=policy.semantic_sha256,
            semantic_sha256=digest,
        )

    @classmethod
    def create_local_conformance(
        cls,
        *,
        recording_identity: str,
        source_enrichments: Sequence[PlatformEnrichedOutputReference],
        admitted_hypothesis_facts: Sequence[ProductionAdmittedHypothesisFact],
        policy: ProductionOutputAdmissionPolicyRef,
    ) -> Self:
        """Create an output proof that is explicitly ineligible for production."""

        return cls.create(
            recording_identity=recording_identity,
            source_enrichments=source_enrichments,
            admitted_hypothesis_facts=admitted_hypothesis_facts,
            policy=policy,
            evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        )


class PlatformEnrichedEventHypothesis(StrictModel):
    """Immutable, run-independent event hypothesis accepted by the registry."""

    schema_version: Literal["2.0"]
    recording_identity: Sha256Digest
    event_hypothesis_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    effective_interval: NanosecondInterval
    semantic_fingerprint_sha256: Sha256Digest
    fusion_logical_key: NodeLogicalKey
    fusion_output_ordinal: NonNegativeInt
    source_enrichments: tuple[PlatformEnrichedOutputReference, ...]
    production_admission: AdmissionProof
    production_output_admission: OutputAdmissionProof

    @model_validator(mode="after")
    def validate_hypothesis(self) -> Self:
        if not self.event_hypothesis_logical_key.startswith(
            f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:"
        ):
            raise ValueError("event hypothesis logical key has an unexpected namespace")
        if self.production_admission.recording_identity != self.recording_identity:
            raise ValueError("admission proof recording does not match hypothesis")
        if self.production_output_admission.recording_identity != self.recording_identity:
            raise ValueError("output admission proof recording does not match hypothesis")
        if (
            self.production_admission.evidence_class
            is not self.production_output_admission.evidence_class
            or self.production_admission.production_eligible
            is not self.production_output_admission.production_eligible
        ):
            raise ValueError("primary and output admission evidence metadata must match")
        if not self.source_enrichments:
            raise ValueError("an event hypothesis requires enriched output lineage")
        keys = tuple(item.enrichment_logical_key for item in self.source_enrichments)
        if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
            raise ValueError("source enrichment references must be unique and canonical")
        if any(
            item.recording_identity != self.recording_identity for item in self.source_enrichments
        ):
            raise ValueError("enriched output lineage crosses recording scope")
        if self.production_output_admission.source_enrichments != self.source_enrichments:
            raise ValueError("output admission proof does not exactly bind hypothesis lineage")
        admitted_fact = _admitted_hypothesis_fact_from_values(
            fusion_output_ordinal=self.fusion_output_ordinal,
            effective_interval=self.effective_interval,
            semantic_fingerprint_sha256=self.semantic_fingerprint_sha256,
            fusion_logical_key=self.fusion_logical_key,
        )
        if admitted_fact not in self.production_output_admission.admitted_hypothesis_facts:
            raise ValueError("output admission proof does not admit the exact hypothesis facts")
        expected = semantic_sha256(event_hypothesis_semantic_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("event hypothesis semantic_sha256 is inconsistent")
        if self.event_hypothesis_logical_key != (
            f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:{expected}"
        ):
            raise ValueError("event hypothesis logical key is inconsistent")
        return self

    @property
    def effective_start_ns(self) -> int:
        return self.effective_interval.start_ns

    @property
    def effective_end_ns(self) -> int:
        return self.effective_interval.end_ns

    @classmethod
    def create(
        cls,
        *,
        recording_identity: str,
        effective_interval: NanosecondInterval,
        semantic_fingerprint_sha256: str,
        fusion_logical_key: str,
        fusion_output_ordinal: int,
        source_enrichments: Sequence[PlatformEnrichedOutputReference],
        production_admission: AdmissionProof,
        production_output_admission: OutputAdmissionProof,
    ) -> Self:
        refs = tuple(sorted(source_enrichments, key=lambda item: item.enrichment_logical_key))
        values: dict[str, object] = {
            "recording_identity": recording_identity,
            "effective_interval": effective_interval,
            "semantic_fingerprint_sha256": semantic_fingerprint_sha256,
            "fusion_logical_key": fusion_logical_key,
            "fusion_output_ordinal": fusion_output_ordinal,
            "source_enrichments": refs,
            "production_admission": production_admission,
            "production_output_admission": production_output_admission,
        }
        digest = semantic_sha256(event_hypothesis_projection_values(values))
        return cls(
            schema_version="2.0",
            recording_identity=recording_identity,
            event_hypothesis_logical_key=(f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:{digest}"),
            semantic_sha256=digest,
            effective_interval=effective_interval,
            semantic_fingerprint_sha256=semantic_fingerprint_sha256,
            fusion_logical_key=fusion_logical_key,
            fusion_output_ordinal=fusion_output_ordinal,
            source_enrichments=refs,
            production_admission=production_admission,
            production_output_admission=production_output_admission,
        )


class EventIdentityCandidate(StrictModel):
    """One deterministic scored candidate from the registered policy."""

    event_id: OpaqueUuid
    score: UnitInterval
    relation: EventIdentityCandidateRelation
    reason: NonEmptyString

    @model_validator(mode="after")
    def validate_reason(self) -> Self:
        if not self.reason.strip():
            raise ValueError("candidate reason cannot be blank")
        return self


class EventIdentityResolution(StrictModel):
    """Policy result before the registry injects a new event ID."""

    disposition: EventIdentityAssignmentDisposition
    selected_event_id: OpaqueUuid | None
    candidates: tuple[EventIdentityCandidate, ...]
    reason: NonEmptyString

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        if not self.reason.strip():
            raise ValueError("resolution reason cannot be blank")
        ids = tuple(item.event_id for item in self.candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("resolution candidates must contain unique event IDs")
        if self.candidates != tuple(sorted(self.candidates, key=lambda item: item.event_id)):
            raise ValueError("resolution candidates must be in canonical event-ID order")
        if self.disposition is EventIdentityAssignmentDisposition.REUSED:
            if self.selected_event_id is None:
                raise ValueError("REUSED resolution requires selected_event_id")
            selected = [item for item in self.candidates if item.event_id == self.selected_event_id]
            if (
                len(selected) != 1
                or selected[0].relation is not EventIdentityCandidateRelation.SAME_EVENT
            ):
                raise ValueError("REUSED resolution must select a SAME_EVENT candidate")
        elif self.selected_event_id is not None:
            raise ValueError("CREATED or AMBIGUOUS resolution cannot select an existing ID")
        if self.disposition is EventIdentityAssignmentDisposition.AMBIGUOUS and (
            not self.candidates
            or any(
                item.relation is not EventIdentityCandidateRelation.POSSIBLE_MATCH
                for item in self.candidates
            )
        ):
            raise ValueError("AMBIGUOUS resolution requires only POSSIBLE_MATCH candidates")
        if self.disposition is EventIdentityAssignmentDisposition.CREATED and any(
            item.relation
            in {
                EventIdentityCandidateRelation.SAME_EVENT,
                EventIdentityCandidateRelation.POSSIBLE_MATCH,
            }
            for item in self.candidates
        ):
            raise ValueError("CREATED resolution cannot silently contain reuse candidates")
        return self


class StableEventIdentity(StrictModel):
    """Stable event ID allocated by the registry's injected allocator."""

    schema_version: Literal["1.0"]
    recording_identity: Sha256Digest
    event_id: OpaqueUuid
    semantic_fingerprint_sha256: Sha256Digest
    created_generation: PositiveInt
    created_by_hypothesis_logical_key: NodeLogicalKey
    creation_disposition: Literal["CREATED", "AMBIGUOUS"]
    allocator_version: SchemaVersion

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if not self.created_by_hypothesis_logical_key.startswith(
            f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:"
        ):
            raise ValueError("stable identity must retain hypothesis lineage")
        return self


class EventIdentityAssignment(StrictModel):
    """Append-only assignment for one hypothesis and exact policy."""

    schema_version: Literal["1.0"]
    assignment_id: OpaqueUuid
    assignment_logical_key: NodeLogicalKey
    assignment_semantic_sha256: Sha256Digest
    recording_identity: Sha256Digest
    event_hypothesis_logical_key: NodeLogicalKey
    event_hypothesis_semantic_sha256: Sha256Digest
    event_id: OpaqueUuid
    disposition: EventIdentityAssignmentDisposition
    relation: tuple[EventIdentityAssignmentRelation, ...]
    identity_policy_version: SchemaVersion
    identity_policy_sha256: Sha256Digest
    candidates: tuple[EventIdentityCandidate, ...]
    reason: NonEmptyString
    registry_generation: PositiveInt
    decided_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_assignment(self) -> Self:
        if not self.assignment_logical_key.startswith("event-identity-assignment:"):
            raise ValueError("assignment logical key has an unexpected namespace")
        if not self.event_hypothesis_logical_key.startswith(
            f"{EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE}:"
        ):
            raise ValueError("assignment references a stale event-hypothesis namespace")
        if tuple(item.event_id for item in self.candidates) != tuple(
            sorted(item.event_id for item in self.candidates)
        ):
            raise ValueError("assignment candidates must be canonical")
        canonical_relations = tuple(sorted(self.relation, key=lambda item: item.value))
        if len(set(self.relation)) != len(self.relation) or (self.relation != canonical_relations):
            raise ValueError("assignment relations must be unique and canonical")
        expected = semantic_sha256(event_identity_assignment_semantic_projection(self))
        if self.assignment_semantic_sha256 != expected:
            raise ValueError("assignment semantic_sha256 is inconsistent")
        if self.assignment_logical_key != f"event-identity-assignment:{expected}":
            raise ValueError("assignment logical key is inconsistent")
        if self.assignment_id != _stable_uuid("event-identity-assignment", expected):
            raise ValueError("assignment ID is inconsistent")
        return self


class EventCurrentRevisionReference(StrictModel):
    """Replaceable current-revision projection read by matching policies."""

    recording_identity: Sha256Digest
    event_id: OpaqueUuid
    revision_logical_key: NodeLogicalKey
    revision_semantic_sha256: Sha256Digest
    effective_interval: NanosecondInterval


class EventRegistrySnapshot(StrictModel):
    """Stable per-recording read view under one generation and fence."""

    schema_version: Literal["1.0"]
    recording_identity: Sha256Digest
    generation: NonNegativeInt
    fence: PositiveInt
    identities: tuple[StableEventIdentity, ...]
    current_revisions: tuple[EventCurrentRevisionReference, ...]
    assignments: tuple[EventIdentityAssignment, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        identity_ids = tuple(item.event_id for item in self.identities)
        if identity_ids != tuple(sorted(identity_ids)) or (
            len(set(identity_ids)) != len(identity_ids)
        ):
            raise ValueError("snapshot identities must be unique and canonical")
        if any(item.recording_identity != self.recording_identity for item in self.identities):
            raise ValueError("snapshot identity crosses recording scope")
        revision_ids = tuple(item.event_id for item in self.current_revisions)
        if revision_ids != tuple(sorted(revision_ids)):
            raise ValueError("snapshot current revisions must be canonical")
        if any(
            item.recording_identity != self.recording_identity for item in self.current_revisions
        ):
            raise ValueError("snapshot revision crosses recording scope")
        assignment_keys = tuple(_assignment_sort_key(item) for item in self.assignments)
        if assignment_keys != tuple(sorted(assignment_keys)):
            raise ValueError("snapshot assignments must be canonical")
        if any(item.recording_identity != self.recording_identity for item in self.assignments):
            raise ValueError("snapshot assignment crosses recording scope")
        return self


class EventIdentityRelation(StrictModel):
    """Typed split, merge, supersedes, or possible-match audit edge."""

    schema_version: Literal["1.0"]
    relation_logical_key: NodeLogicalKey
    recording_identity: Sha256Digest
    assignment_logical_key: NodeLogicalKey
    from_event_id: OpaqueUuid
    to_event_id: OpaqueUuid
    relation: EventIdentityCandidateRelation
    score: UnitInterval
    reason: NonEmptyString
    identity_policy_version: SchemaVersion
    identity_policy_sha256: Sha256Digest
    registry_generation: PositiveInt

    @model_validator(mode="after")
    def validate_relation(self) -> Self:
        if self.relation is EventIdentityCandidateRelation.SAME_EVENT:
            raise ValueError("SAME_EVENT is represented by assignment, not a correction edge")
        if not self.assignment_logical_key.startswith("event-identity-assignment:"):
            raise ValueError("relation must reference an identity assignment")
        expected = semantic_sha256(event_identity_relation_semantic_projection(self))
        if self.relation_logical_key != f"event-identity-relation:{expected}":
            raise ValueError("relation logical key is inconsistent")
        return self


class EventIdentityOutboxRecord(StrictModel):
    """Transactional successor publication for one committed assignment."""

    schema_version: Literal["1.0"]
    outbox_id: OpaqueUuid
    topic: Literal["event.identity.assignment"]
    recording_identity: Sha256Digest
    key: Sha256Digest
    assignment_logical_key: NodeLogicalKey
    payload_reference: NodeLogicalKey
    registry_generation: PositiveInt

    @model_validator(mode="after")
    def validate_outbox(self) -> Self:
        if self.key != self.recording_identity:
            raise ValueError("outbox key must be recording scoped")
        if self.payload_reference != self.assignment_logical_key:
            raise ValueError("outbox payload must reference the assignment")
        expected = _stable_uuid("event-identity-outbox", self.assignment_logical_key)
        if self.outbox_id != expected:
            raise ValueError("outbox ID is inconsistent")
        return self


class EventIdentityRegistryMutation(StrictModel):
    """Atomic CAS payload for one recording partition."""

    schema_version: Literal["1.0"]
    recording_identity: Sha256Digest
    expected_generation: NonNegativeInt
    fence: PositiveInt
    next_generation: PositiveInt
    identities: tuple[StableEventIdentity, ...]
    assignments: tuple[EventIdentityAssignment, ...]
    relations: tuple[EventIdentityRelation, ...]
    outbox: tuple[EventIdentityOutboxRecord, ...]

    @model_validator(mode="after")
    def validate_mutation(self) -> Self:
        if self.next_generation != self.expected_generation + 1:
            raise ValueError("mutation must advance exactly one registry generation")
        if not self.assignments:
            raise ValueError("a mutation must contain at least one new assignment")
        if len(self.outbox) != len(self.assignments):
            raise ValueError("each assignment requires one transactional outbox row")
        recording_identities = (
            *(item.recording_identity for item in self.identities),
            *(item.recording_identity for item in self.assignments),
            *(item.recording_identity for item in self.relations),
            *(item.recording_identity for item in self.outbox),
        )
        if any(item != self.recording_identity for item in recording_identities):
            raise ValueError("mutation contains a cross-recording row")
        if any(item.created_generation != self.next_generation for item in self.identities):
            raise ValueError("new identities must use the committed generation")
        generations = (
            *(item.registry_generation for item in self.assignments),
            *(item.registry_generation for item in self.relations),
            *(item.registry_generation for item in self.outbox),
        )
        if any(item != self.next_generation for item in generations):
            raise ValueError("mutation rows must use the committed generation")
        return self


class EventIdentityBatchResult(StrictModel):
    """Result containing replayed and newly committed assignment rows."""

    recording_identity: Sha256Digest
    initial_generation: NonNegativeInt
    final_generation: NonNegativeInt
    fence: PositiveInt
    assignments: tuple[EventIdentityAssignment, ...]
    new_identities: tuple[StableEventIdentity, ...]
    relations: tuple[EventIdentityRelation, ...]
    outbox: tuple[EventIdentityOutboxRecord, ...]
    replayed_assignment_logical_keys: tuple[NodeLogicalKey, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.final_generation < self.initial_generation:
            raise ValueError("final generation cannot precede initial generation")
        if not self.assignments:
            raise ValueError("an identity batch result requires assignments")
        if any(item.recording_identity != self.recording_identity for item in self.assignments):
            raise ValueError("batch assignments cross recording scope")
        keys = tuple(item.assignment_logical_key for item in self.assignments)
        if len(set(keys)) != len(keys):
            raise ValueError("batch assignments must be unique")
        replayed = self.replayed_assignment_logical_keys
        if len(set(replayed)) != len(replayed):
            raise ValueError("replayed assignment keys must be unique")
        if not set(replayed).issubset(keys):
            raise ValueError("replayed assignment keys must belong to the batch")
        return self


class EventIdAllocator(Protocol):
    """Registry-owned ID allocation boundary; resolvers and models cannot call it."""

    @property
    def version(self) -> str:
        """Return the immutable allocator implementation version."""

    def allocate(
        self,
        *,
        recording_identity: str,
        hypothesis: PlatformEnrichedEventHypothesis,
        registry_generation: int,
    ) -> str:
        """Allocate an opaque event ID inside one recording transaction."""


class EventIdentityResolver(Protocol):
    """Versioned pure policy for selecting candidates from a stable snapshot."""

    @property
    def policy(self) -> EventIdentityPolicyRef:
        """Return the exact policy reference implemented by this resolver."""

    def resolve(
        self,
        *,
        hypothesis: PlatformEnrichedEventHypothesis,
        snapshot: EventRegistrySnapshot,
    ) -> EventIdentityResolution:
        """Resolve against only the supplied recording-scoped snapshot."""


class EventIdentityRegistryRepository(Protocol):
    """Atomic recording-partition persistence port."""

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        """Read one stable generation and fence."""

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        """Compare-and-swap one complete mutation and return its committed snapshot."""


class RandomEventIdAllocator:
    """Local allocator whose only authority is possession by the registry service."""

    def __init__(self, version: str = "uuid4-v1") -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("allocator version must be a nonempty string")
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def allocate(
        self,
        *,
        recording_identity: str,
        hypothesis: PlatformEnrichedEventHypothesis,
        registry_generation: int,
    ) -> str:
        del recording_identity, hypothesis, registry_generation
        return str(uuid4())


class ExactFingerprintEventIdentityResolver:
    """Conservative reference policy: reuse only one exact platform fingerprint."""

    def __init__(self, policy: EventIdentityPolicyRef) -> None:
        if not isinstance(policy, EventIdentityPolicyRef):
            raise TypeError("policy must be an EventIdentityPolicyRef")
        self._policy = policy

    @property
    def policy(self) -> EventIdentityPolicyRef:
        return self._policy

    def resolve(
        self,
        *,
        hypothesis: PlatformEnrichedEventHypothesis,
        snapshot: EventRegistrySnapshot,
    ) -> EventIdentityResolution:
        if snapshot.recording_identity != hypothesis.recording_identity:
            raise CrossRecordingEventIdentityError(
                "resolver snapshot does not match the hypothesis recording"
            )
        matches = tuple(
            item
            for item in snapshot.identities
            if item.semantic_fingerprint_sha256 == hypothesis.semantic_fingerprint_sha256
        )
        if not matches:
            return EventIdentityResolution(
                disposition=EventIdentityAssignmentDisposition.CREATED,
                selected_event_id=None,
                candidates=(),
                reason="no exact platform semantic fingerprint exists",
            )
        if len(matches) == 1:
            candidate = EventIdentityCandidate(
                event_id=matches[0].event_id,
                score=1.0,
                relation=EventIdentityCandidateRelation.SAME_EVENT,
                reason="exact platform semantic fingerprint match",
            )
            return EventIdentityResolution(
                disposition=EventIdentityAssignmentDisposition.REUSED,
                selected_event_id=matches[0].event_id,
                candidates=(candidate,),
                reason="one exact platform semantic fingerprint exists",
            )
        candidates = tuple(
            EventIdentityCandidate(
                event_id=item.event_id,
                score=1.0,
                relation=EventIdentityCandidateRelation.POSSIBLE_MATCH,
                reason="multiple exact platform semantic fingerprints exist",
            )
            for item in matches
        )
        return EventIdentityResolution(
            disposition=EventIdentityAssignmentDisposition.AMBIGUOUS,
            selected_event_id=None,
            candidates=candidates,
            reason="exact fingerprint is ambiguous across existing identities",
        )


@dataclass(slots=True)
class _RecordingRegistryState:
    recording_identity: str
    generation: int = 0
    fence: int = 1
    identities: dict[str, StableEventIdentity] = field(default_factory=dict)
    current_revisions: dict[str, EventCurrentRevisionReference] = field(default_factory=dict)
    assignments: dict[tuple[str, str, str], EventIdentityAssignment] = field(default_factory=dict)
    relations: dict[str, EventIdentityRelation] = field(default_factory=dict)
    outbox: dict[str, EventIdentityOutboxRecord] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock)


class InMemoryEventIdentityRegistryRepository:
    """Thread-safe reference repository with per-recording CAS partitions."""

    def __init__(self) -> None:
        self._states: dict[str, _RecordingRegistryState] = {}
        self._event_owners: dict[str, str] = {}
        self._states_lock = RLock()

    def snapshot(self, recording_identity: str) -> EventRegistrySnapshot:
        state = self._state(recording_identity)
        with state.lock:
            return self._snapshot_locked(state)

    def commit(self, mutation: EventIdentityRegistryMutation) -> EventRegistrySnapshot:
        if not isinstance(mutation, EventIdentityRegistryMutation):
            raise TypeError("mutation must be an EventIdentityRegistryMutation")
        try:
            mutation = EventIdentityRegistryMutation.model_validate(
                mutation.model_dump(mode="python"), strict=True
            )
        except ValueError as exc:
            raise EventIdentityInputError("registry mutation failed validation") from exc

        state = self._state(mutation.recording_identity)
        with state.lock:
            if state.generation != mutation.expected_generation or state.fence != mutation.fence:
                raise StaleEventRegistryFenceError(mutation.recording_identity)
            self._validate_commit_locked(state, mutation)
            with self._states_lock:
                for identity in mutation.identities:
                    self._event_owners[identity.event_id] = identity.recording_identity
            for identity in mutation.identities:
                state.identities[identity.event_id] = identity
            for assignment in mutation.assignments:
                state.assignments[_assignment_idempotency_key(assignment)] = assignment
            for relation in mutation.relations:
                state.relations[relation.relation_logical_key] = relation
            for outbox in mutation.outbox:
                state.outbox[outbox.outbox_id] = outbox
            state.generation = mutation.next_generation
            state.fence += 1
            return self._snapshot_locked(state)

    def list_relations(self, recording_identity: str) -> tuple[EventIdentityRelation, ...]:
        state = self._state(recording_identity)
        with state.lock:
            return tuple(
                sorted(
                    state.relations.values(),
                    key=lambda item: item.relation_logical_key,
                )
            )

    def list_outbox(self, recording_identity: str) -> tuple[EventIdentityOutboxRecord, ...]:
        state = self._state(recording_identity)
        with state.lock:
            return tuple(sorted(state.outbox.values(), key=lambda item: item.outbox_id))

    def _state(self, recording_identity: str) -> _RecordingRegistryState:
        if not isinstance(recording_identity, str) or not recording_identity:
            raise EventIdentityInputError("recording_identity must be nonempty")
        with self._states_lock:
            state = self._states.get(recording_identity)
            if state is None:
                state = _RecordingRegistryState(recording_identity=recording_identity)
                self._states[recording_identity] = state
            return state

    def _snapshot_locked(self, state: _RecordingRegistryState) -> EventRegistrySnapshot:
        return EventRegistrySnapshot(
            schema_version="1.0",
            recording_identity=state.recording_identity,
            generation=state.generation,
            fence=state.fence,
            identities=tuple(sorted(state.identities.values(), key=lambda item: item.event_id)),
            current_revisions=tuple(
                sorted(
                    state.current_revisions.values(),
                    key=lambda item: item.event_id,
                )
            ),
            assignments=tuple(sorted(state.assignments.values(), key=_assignment_sort_key)),
        )

    def _validate_commit_locked(
        self,
        state: _RecordingRegistryState,
        mutation: EventIdentityRegistryMutation,
    ) -> None:
        new_identity_ids = tuple(item.event_id for item in mutation.identities)
        if len(set(new_identity_ids)) != len(new_identity_ids):
            raise EventIdentityConflictError("mutation repeats a new event ID")
        if any(event_id in state.identities for event_id in new_identity_ids):
            raise EventIdentityConflictError("mutation reallocates an existing event ID")
        with self._states_lock:
            for event_id in new_identity_ids:
                owner = self._event_owners.get(event_id)
                if owner is not None and owner != state.recording_identity:
                    raise CrossRecordingEventIdentityError(
                        "event ID is already owned by another recording"
                    )

        known_event_ids = set(state.identities) | set(new_identity_ids)
        mutation_assignment_keys: set[tuple[str, str, str]] = set()
        mutation_assignment_logical_keys: set[str] = set()
        for assignment in mutation.assignments:
            key = _assignment_idempotency_key(assignment)
            if key in state.assignments or key in mutation_assignment_keys:
                raise EventIdentityConflictError(
                    "hypothesis already has an assignment for this exact policy"
                )
            if assignment.event_id not in known_event_ids:
                raise EventIdentityConflictError(
                    "assignment references an unknown recording-scoped event ID"
                )
            mutation_assignment_keys.add(key)
            mutation_assignment_logical_keys.add(assignment.assignment_logical_key)

        relation_keys = tuple(item.relation_logical_key for item in mutation.relations)
        if len(set(relation_keys)) != len(relation_keys) or any(
            item in state.relations for item in relation_keys
        ):
            raise EventIdentityConflictError("relation logical key conflicts")
        for relation in mutation.relations:
            if relation.assignment_logical_key not in mutation_assignment_logical_keys:
                raise EventIdentityConflictError(
                    "relation does not belong to this mutation's assignment"
                )
            if (
                relation.from_event_id not in known_event_ids
                or relation.to_event_id not in known_event_ids
                or relation.from_event_id == relation.to_event_id
            ):
                raise EventIdentityConflictError(
                    "relation endpoints must be distinct identities in this recording"
                )

        outbox_ids = tuple(item.outbox_id for item in mutation.outbox)
        if len(set(outbox_ids)) != len(outbox_ids) or any(
            item in state.outbox for item in outbox_ids
        ):
            raise EventIdentityConflictError("outbox ID conflicts")
        if {
            item.assignment_logical_key for item in mutation.outbox
        } != mutation_assignment_logical_keys:
            raise EventIdentityConflictError(
                "transactional outbox does not exactly cover assignments"
            )


class EventIdentityRegistryService:
    """Validate authority, resolve canonically, and atomically assign event IDs."""

    def __init__(
        self,
        *,
        repository: EventIdentityRegistryRepository,
        resolver: EventIdentityResolver,
        allocator: EventIdAllocator,
        output_admission_policy: ProductionOutputAdmissionPolicyRef,
        admission_evidence_class: AdmissionEvidenceClass = (
            AdmissionEvidenceClass.LOCAL_CONFORMANCE
        ),
        max_cas_retries: int = 4,
    ) -> None:
        if isinstance(max_cas_retries, bool) or not isinstance(max_cas_retries, int):
            raise TypeError("max_cas_retries must be an integer")
        if max_cas_retries < 0:
            raise ValueError("max_cas_retries cannot be negative")
        policy = resolver.policy
        if not isinstance(policy, EventIdentityPolicyRef):
            raise TypeError("resolver policy must be an EventIdentityPolicyRef")
        if not isinstance(allocator.version, str) or not allocator.version:
            raise TypeError("allocator version must be a nonempty string")
        if not isinstance(output_admission_policy, ProductionOutputAdmissionPolicyRef):
            raise TypeError("output_admission_policy must be a ProductionOutputAdmissionPolicyRef")
        if not isinstance(admission_evidence_class, AdmissionEvidenceClass):
            raise TypeError("admission_evidence_class must be an AdmissionEvidenceClass")
        validate_evidence_eligibility(
            admission_evidence_class,
            admission_evidence_class is AdmissionEvidenceClass.PRODUCTION_QUALIFIED,
        )
        self._repository = repository
        self._resolver = resolver
        self._allocator = allocator
        self._policy = policy
        self._output_admission_policy = output_admission_policy
        self._admission_evidence_class = admission_evidence_class
        self._max_cas_retries = max_cas_retries

    @property
    def policy(self) -> EventIdentityPolicyRef:
        return self._policy

    @property
    def output_admission_policy(self) -> ProductionOutputAdmissionPolicyRef:
        return self._output_admission_policy

    def assign_batch(
        self,
        *,
        admitted_context: AdmittedRecordingContextV2,
        hypotheses: Sequence[PlatformEnrichedEventHypothesis],
        enriched_outputs: Sequence[OrchestratorEnrichedOutput],
        decided_at: str,
    ) -> EventIdentityBatchResult:
        """Assign one canonical batch, retrying stale per-recording generations."""

        context, ordered = _validate_authority_inputs(
            admitted_context=admitted_context,
            hypotheses=hypotheses,
            enriched_outputs=enriched_outputs,
            output_admission_policy=self._output_admission_policy,
            admission_evidence_class=self._admission_evidence_class,
        )
        initial_generation: int | None = None
        last_stale: StaleEventRegistryFenceError | None = None

        for _ in range(self._max_cas_retries + 1):
            snapshot = self._repository.snapshot(context.recording_identity)
            if initial_generation is None:
                initial_generation = snapshot.generation
            (
                result_assignments,
                replayed_keys,
                new_identities,
                new_assignments,
                relations,
                outbox,
            ) = self._prepare_mutation(
                snapshot=snapshot,
                hypotheses=ordered,
                decided_at=decided_at,
            )
            if not new_assignments:
                return EventIdentityBatchResult(
                    recording_identity=context.recording_identity,
                    initial_generation=initial_generation,
                    final_generation=snapshot.generation,
                    fence=snapshot.fence,
                    assignments=result_assignments,
                    new_identities=(),
                    relations=(),
                    outbox=(),
                    replayed_assignment_logical_keys=replayed_keys,
                )

            mutation = EventIdentityRegistryMutation(
                schema_version="1.0",
                recording_identity=context.recording_identity,
                expected_generation=snapshot.generation,
                fence=snapshot.fence,
                next_generation=snapshot.generation + 1,
                identities=new_identities,
                assignments=new_assignments,
                relations=relations,
                outbox=outbox,
            )
            try:
                committed = self._repository.commit(mutation)
            except StaleEventRegistryFenceError as exc:
                last_stale = exc
                continue
            return EventIdentityBatchResult(
                recording_identity=context.recording_identity,
                initial_generation=initial_generation,
                final_generation=committed.generation,
                fence=committed.fence,
                assignments=result_assignments,
                new_identities=new_identities,
                relations=relations,
                outbox=outbox,
                replayed_assignment_logical_keys=replayed_keys,
            )

        assert last_stale is not None
        raise last_stale

    def _prepare_mutation(
        self,
        *,
        snapshot: EventRegistrySnapshot,
        hypotheses: tuple[PlatformEnrichedEventHypothesis, ...],
        decided_at: str,
    ) -> tuple[
        tuple[EventIdentityAssignment, ...],
        tuple[NodeLogicalKey, ...],
        tuple[StableEventIdentity, ...],
        tuple[EventIdentityAssignment, ...],
        tuple[EventIdentityRelation, ...],
        tuple[EventIdentityOutboxRecord, ...],
    ]:
        next_generation = snapshot.generation + 1
        existing_assignments = {
            _assignment_idempotency_key(item): item for item in snapshot.assignments
        }
        working_identities = {item.event_id: item for item in snapshot.identities}
        new_identities: list[StableEventIdentity] = []
        new_assignments: list[EventIdentityAssignment] = []
        relations: list[EventIdentityRelation] = []
        outbox: list[EventIdentityOutboxRecord] = []
        replayed_keys: list[str] = []
        result_by_hypothesis: dict[str, EventIdentityAssignment] = {}

        for hypothesis in hypotheses:
            idempotency_key = _hypothesis_idempotency_key(hypothesis, self._policy)
            existing = existing_assignments.get(idempotency_key)
            if existing is not None:
                if existing.event_hypothesis_semantic_sha256 != hypothesis.semantic_sha256:
                    raise EventIdentityConflictError(
                        "idempotent hypothesis key has different immutable content"
                    )
                result_by_hypothesis[hypothesis.event_hypothesis_logical_key] = existing
                replayed_keys.append(existing.assignment_logical_key)
                continue

            working_snapshot = EventRegistrySnapshot(
                schema_version="1.0",
                recording_identity=snapshot.recording_identity,
                generation=snapshot.generation,
                fence=snapshot.fence,
                identities=tuple(
                    sorted(working_identities.values(), key=lambda item: item.event_id)
                ),
                current_revisions=snapshot.current_revisions,
                assignments=tuple(
                    sorted(
                        (*snapshot.assignments, *new_assignments),
                        key=_assignment_sort_key,
                    )
                ),
            )
            try:
                resolution = self._resolver.resolve(
                    hypothesis=hypothesis,
                    snapshot=working_snapshot,
                )
                resolution = EventIdentityResolution.model_validate(
                    resolution.model_dump(mode="python"), strict=True
                )
            except EventIdentityRegistryError:
                raise
            except (AttributeError, TypeError, ValueError) as exc:
                raise EventIdentityInputError(
                    "event identity resolver returned an invalid resolution"
                ) from exc
            _validate_resolution_scope(resolution, working_snapshot)

            identity: StableEventIdentity | None = None
            if resolution.disposition is EventIdentityAssignmentDisposition.REUSED:
                assert resolution.selected_event_id is not None
                event_id = resolution.selected_event_id
            else:
                try:
                    event_id = self._allocator.allocate(
                        recording_identity=hypothesis.recording_identity,
                        hypothesis=hypothesis,
                        registry_generation=next_generation,
                    )
                except Exception as exc:
                    raise EventIdentityAllocationError("event ID allocator failed") from exc
                if event_id in working_identities:
                    raise EventIdentityAllocationError(
                        "event ID allocator returned an existing identity"
                    )
                try:
                    creation_disposition: Literal["CREATED", "AMBIGUOUS"] = (
                        "AMBIGUOUS"
                        if resolution.disposition is EventIdentityAssignmentDisposition.AMBIGUOUS
                        else "CREATED"
                    )
                    identity = StableEventIdentity(
                        schema_version="1.0",
                        recording_identity=hypothesis.recording_identity,
                        event_id=event_id,
                        semantic_fingerprint_sha256=(hypothesis.semantic_fingerprint_sha256),
                        created_generation=next_generation,
                        created_by_hypothesis_logical_key=(hypothesis.event_hypothesis_logical_key),
                        creation_disposition=creation_disposition,
                        allocator_version=self._allocator.version,
                    )
                except (TypeError, ValueError) as exc:
                    raise EventIdentityAllocationError(
                        "event ID allocator returned an invalid identity"
                    ) from exc
                working_identities[event_id] = identity
                new_identities.append(identity)

            assignment = _build_assignment(
                hypothesis=hypothesis,
                resolution=resolution,
                event_id=event_id,
                policy=self._policy,
                registry_generation=next_generation,
                decided_at=decided_at,
            )
            assignment_relations = _build_relations(assignment, resolution)
            assignment_outbox = EventIdentityOutboxRecord(
                schema_version="1.0",
                outbox_id=_stable_uuid("event-identity-outbox", assignment.assignment_logical_key),
                topic="event.identity.assignment",
                recording_identity=assignment.recording_identity,
                key=assignment.recording_identity,
                assignment_logical_key=assignment.assignment_logical_key,
                payload_reference=assignment.assignment_logical_key,
                registry_generation=assignment.registry_generation,
            )
            new_assignments.append(assignment)
            relations.extend(assignment_relations)
            outbox.append(assignment_outbox)
            existing_assignments[idempotency_key] = assignment
            result_by_hypothesis[hypothesis.event_hypothesis_logical_key] = assignment

        return (
            tuple(result_by_hypothesis[item.event_hypothesis_logical_key] for item in hypotheses),
            tuple(replayed_keys),
            tuple(new_identities),
            tuple(new_assignments),
            tuple(relations),
            tuple(outbox),
        )


def _validate_authority_inputs(
    *,
    admitted_context: AdmittedRecordingContextV2,
    hypotheses: Sequence[PlatformEnrichedEventHypothesis],
    enriched_outputs: Sequence[OrchestratorEnrichedOutput],
    output_admission_policy: ProductionOutputAdmissionPolicyRef,
    admission_evidence_class: AdmissionEvidenceClass,
) -> tuple[AdmittedRecordingContextV2, tuple[PlatformEnrichedEventHypothesis, ...]]:
    from robata.admission.context import AdmittedRecordingContextV2
    from robata.inference.enrichment import OrchestratorEnrichedOutput

    if not isinstance(admitted_context, AdmittedRecordingContextV2):
        raise EventIdentityInputError("event registry requires a validated V2 admission context")
    try:
        context = AdmittedRecordingContextV2.model_validate(
            admitted_context.model_dump(mode="python"), strict=True
        )
    except ValueError as exc:
        raise EventIdentityInputError("V2 admission context failed validation") from exc

    validated_hypotheses: list[PlatformEnrichedEventHypothesis] = []
    for item in hypotheses:
        if not isinstance(item, PlatformEnrichedEventHypothesis):
            raise EventIdentityInputError("hypotheses must be platform-enriched event contracts")
        try:
            validated_hypotheses.append(
                PlatformEnrichedEventHypothesis.model_validate(
                    item.model_dump(mode="python"), strict=True
                )
            )
        except ValueError as exc:
            raise EventIdentityInputError("event hypothesis failed validation") from exc
    if not validated_hypotheses:
        raise EventIdentityInputError("event identity assignment requires hypotheses")
    hypothesis_keys = tuple(item.event_hypothesis_logical_key for item in validated_hypotheses)
    if len(set(hypothesis_keys)) != len(hypothesis_keys):
        raise EventIdentityInputError("event hypothesis batch contains duplicates")
    admitted_facts = tuple(
        sorted(
            (
                _admitted_hypothesis_fact_from_values(
                    fusion_output_ordinal=item.fusion_output_ordinal,
                    effective_interval=item.effective_interval,
                    semantic_fingerprint_sha256=item.semantic_fingerprint_sha256,
                    fusion_logical_key=item.fusion_logical_key,
                )
                for item in validated_hypotheses
            ),
            key=_admitted_hypothesis_fact_sort_key,
        )
    )
    fact_ordinals = tuple(item.fusion_output_ordinal for item in admitted_facts)
    if len(set(fact_ordinals)) != len(fact_ordinals):
        raise EventIdentityInputError(
            "event hypothesis batch contains duplicate fusion output ordinals"
        )

    output_by_key: dict[str, OrchestratorEnrichedOutput] = {}
    for enriched_item in enriched_outputs:
        if not isinstance(enriched_item, OrchestratorEnrichedOutput):
            raise EventIdentityInputError(
                "event registry requires actual orchestrator-enriched outputs"
            )
        try:
            output = OrchestratorEnrichedOutput.model_validate(
                enriched_item.model_dump(mode="python"), strict=True
            )
        except ValueError as exc:
            raise EventIdentityInputError("enriched output failed validation") from exc
        if output.enrichment_logical_key in output_by_key:
            raise EventIdentityInputError("enriched outputs contain duplicate logical keys")
        authority = output.authority
        if (
            authority.recording_identity != context.recording_identity
            or authority.mcap_id != context.ready_manifest.mcap_id
            or authority.camera_mapping_run_id != context.ready_manifest.camera_mapping_run_id
            or authority.alignment_id != context.alignment_manifest.alignment_id
        ):
            raise EventIdentityInputError(
                "enriched output authority does not match V2 admission evidence"
            )
        output_by_key[output.enrichment_logical_key] = output

    expected_proof = AdmissionProof.from_context(
        context,
        evidence_class=admission_evidence_class,
    )
    referenced_keys: set[str] = set()
    for hypothesis in validated_hypotheses:
        if hypothesis.recording_identity != context.recording_identity:
            raise CrossRecordingEventIdentityError(
                "event hypothesis crosses the admitted recording scope"
            )
        if hypothesis.production_admission.evidence_class is not admission_evidence_class:
            raise EventIdentityInputError(
                "event hypothesis evidence class differs from the configured identity boundary"
            )
        if hypothesis.production_admission != expected_proof:
            raise EventIdentityInputError(
                "event hypothesis does not bind the supplied admission context"
            )
        output_proof = hypothesis.production_output_admission
        if output_proof.evidence_class is not admission_evidence_class:
            raise EventIdentityInputError(
                "event hypothesis evidence class differs from the configured identity boundary"
            )
        if (
            output_proof.output_admission_policy_version != output_admission_policy.version
            or output_proof.output_admission_policy_sha256
            != output_admission_policy.semantic_sha256
        ):
            raise EventIdentityInputError(
                "event hypothesis does not bind the configured output admission policy"
            )
        if output_proof.admitted_hypothesis_facts != admitted_facts:
            raise EventIdentityInputError(
                "output admission proof does not exactly cover the hypothesis batch"
            )
        for reference in hypothesis.source_enrichments:
            referenced_output = output_by_key.get(reference.enrichment_logical_key)
            if referenced_output is None or reference != (
                PlatformEnrichedOutputReference.from_output(referenced_output)
            ):
                raise EventIdentityInputError(
                    "event hypothesis contains absent or forged enrichment lineage"
                )
            referenced_keys.add(reference.enrichment_logical_key)
    if referenced_keys != set(output_by_key):
        raise EventIdentityInputError(
            "supplied enriched outputs must exactly match hypothesis lineage"
        )

    ordered = tuple(
        sorted(
            validated_hypotheses,
            key=lambda item: (
                item.effective_start_ns,
                item.effective_end_ns,
                item.event_hypothesis_logical_key,
            ),
        )
    )
    return context, ordered


def _validate_resolution_scope(
    resolution: EventIdentityResolution,
    snapshot: EventRegistrySnapshot,
) -> None:
    known = {item.event_id for item in snapshot.identities}
    candidate_ids = {item.event_id for item in resolution.candidates}
    if not candidate_ids.issubset(known):
        raise CrossRecordingEventIdentityError(
            "identity resolver returned a candidate outside the recording snapshot"
        )
    if resolution.selected_event_id is not None and resolution.selected_event_id not in known:
        raise CrossRecordingEventIdentityError(
            "identity resolver selected an ID outside the recording snapshot"
        )


def _build_assignment(
    *,
    hypothesis: PlatformEnrichedEventHypothesis,
    resolution: EventIdentityResolution,
    event_id: str,
    policy: EventIdentityPolicyRef,
    registry_generation: int,
    decided_at: str,
) -> EventIdentityAssignment:
    relations = _assignment_relations(resolution)
    projection = {
        "recording_identity": hypothesis.recording_identity,
        "event_hypothesis_logical_key": hypothesis.event_hypothesis_logical_key,
        "event_hypothesis_semantic_sha256": hypothesis.semantic_sha256,
        "event_id": event_id,
        "disposition": resolution.disposition.value,
        "relation": [item.value for item in relations],
        "identity_policy_version": policy.version,
        "identity_policy_sha256": policy.semantic_sha256,
        "candidates": [item.model_dump(mode="json") for item in resolution.candidates],
        "reason": resolution.reason,
        "registry_generation": registry_generation,
    }
    digest = semantic_sha256(projection)
    return EventIdentityAssignment(
        schema_version="1.0",
        assignment_id=_stable_uuid("event-identity-assignment", digest),
        assignment_logical_key=f"event-identity-assignment:{digest}",
        assignment_semantic_sha256=digest,
        recording_identity=hypothesis.recording_identity,
        event_hypothesis_logical_key=hypothesis.event_hypothesis_logical_key,
        event_hypothesis_semantic_sha256=hypothesis.semantic_sha256,
        event_id=event_id,
        disposition=resolution.disposition,
        relation=relations,
        identity_policy_version=policy.version,
        identity_policy_sha256=policy.semantic_sha256,
        candidates=resolution.candidates,
        reason=resolution.reason,
        registry_generation=registry_generation,
        decided_at=decided_at,
    )


def _assignment_relations(
    resolution: EventIdentityResolution,
) -> tuple[EventIdentityAssignmentRelation, ...]:
    values: set[EventIdentityAssignmentRelation] = set()
    if resolution.disposition is EventIdentityAssignmentDisposition.REUSED:
        values.add(EventIdentityAssignmentRelation.SAME_EVENT)
    else:
        values.add(EventIdentityAssignmentRelation.NEW_IDENTITY)
    for candidate in resolution.candidates:
        if candidate.relation is EventIdentityCandidateRelation.SAME_EVENT:
            values.add(EventIdentityAssignmentRelation.SAME_EVENT)
        else:
            values.add(EventIdentityAssignmentRelation(candidate.relation.value))
    return tuple(sorted(values, key=lambda item: item.value))


def _build_relations(
    assignment: EventIdentityAssignment,
    resolution: EventIdentityResolution,
) -> tuple[EventIdentityRelation, ...]:
    rows: list[EventIdentityRelation] = []
    for candidate in resolution.candidates:
        if candidate.relation is EventIdentityCandidateRelation.SAME_EVENT:
            continue
        projection = {
            "recording_identity": assignment.recording_identity,
            "assignment_logical_key": assignment.assignment_logical_key,
            "from_event_id": candidate.event_id,
            "to_event_id": assignment.event_id,
            "relation": candidate.relation.value,
            "score": candidate.score,
            "reason": candidate.reason,
            "identity_policy_version": assignment.identity_policy_version,
            "identity_policy_sha256": assignment.identity_policy_sha256,
            "registry_generation": assignment.registry_generation,
        }
        digest = semantic_sha256(projection)
        rows.append(
            EventIdentityRelation(
                schema_version="1.0",
                relation_logical_key=f"event-identity-relation:{digest}",
                recording_identity=assignment.recording_identity,
                assignment_logical_key=assignment.assignment_logical_key,
                from_event_id=candidate.event_id,
                to_event_id=assignment.event_id,
                relation=candidate.relation,
                score=candidate.score,
                reason=candidate.reason,
                identity_policy_version=assignment.identity_policy_version,
                identity_policy_sha256=assignment.identity_policy_sha256,
                registry_generation=assignment.registry_generation,
            )
        )
    return tuple(rows)


def event_hypothesis_projection_values(
    values: dict[str, object],
) -> dict[str, object]:
    """Project immutable hypothesis content without row IDs, clocks, or run IDs."""

    interval = values["effective_interval"]
    source_enrichments = values["source_enrichments"]
    production_admission = values["production_admission"]
    production_output_admission = values["production_output_admission"]
    if not isinstance(interval, NanosecondInterval):
        raise TypeError("effective_interval must be a NanosecondInterval")
    if not isinstance(source_enrichments, tuple) or any(
        not isinstance(item, PlatformEnrichedOutputReference) for item in source_enrichments
    ):
        raise TypeError("source_enrichments must contain enrichment references")
    if not isinstance(production_admission, AdmissionProof):
        raise TypeError("production_admission must be an AdmissionProof")
    if not isinstance(production_output_admission, OutputAdmissionProof):
        raise TypeError("production_output_admission must be an OutputAdmissionProof")
    return {
        "semantic_projection_version": EVENT_HYPOTHESIS_SEMANTIC_PROJECTION_VERSION,
        "recording_identity": values["recording_identity"],
        "effective_interval": interval.model_dump(mode="json"),
        "semantic_fingerprint_sha256": values["semantic_fingerprint_sha256"],
        "fusion_logical_key": values["fusion_logical_key"],
        "fusion_output_ordinal": values["fusion_output_ordinal"],
        "source_enrichments": [
            platform_enriched_output_logical_projection(item) for item in source_enrichments
        ],
        "production_admission": admission_proof_projection(production_admission),
        "production_output_admission": output_admission_projection(production_output_admission),
    }


def event_hypothesis_semantic_projection(
    hypothesis: PlatformEnrichedEventHypothesis,
) -> dict[str, object]:
    return event_hypothesis_projection_values(
        {
            "recording_identity": hypothesis.recording_identity,
            "effective_interval": hypothesis.effective_interval,
            "semantic_fingerprint_sha256": hypothesis.semantic_fingerprint_sha256,
            "fusion_logical_key": hypothesis.fusion_logical_key,
            "fusion_output_ordinal": hypothesis.fusion_output_ordinal,
            "source_enrichments": hypothesis.source_enrichments,
            "production_admission": hypothesis.production_admission,
            "production_output_admission": hypothesis.production_output_admission,
        }
    )


def admission_proof_projection(proof: AdmissionProof) -> dict[str, object]:
    """Project primary admission evidence under the V2 meaning."""

    return {
        "semantic_projection_version": ADMISSION_PROOF_SEMANTIC_PROJECTION_VERSION,
        "decision": proof.decision,
        "evidence_class": proof.evidence_class.value,
        "production_eligible": proof.production_eligible,
        "recording_identity": proof.recording_identity,
        "admitted_context_semantic_sha256": proof.admitted_context_semantic_sha256,
        "admission_policy_version": proof.admission_policy_version,
        "admission_policy_sha256": proof.admission_policy_sha256,
    }


def _admitted_hypothesis_fact_from_values(
    *,
    fusion_output_ordinal: int,
    effective_interval: NanosecondInterval,
    semantic_fingerprint_sha256: str,
    fusion_logical_key: str,
) -> ProductionAdmittedHypothesisFact:
    return ProductionAdmittedHypothesisFact(
        fusion_output_ordinal=fusion_output_ordinal,
        effective_interval=effective_interval,
        semantic_fingerprint_sha256=semantic_fingerprint_sha256,
        fusion_logical_key=fusion_logical_key,
    )


def _admitted_hypothesis_fact_sort_key(
    fact: ProductionAdmittedHypothesisFact,
) -> tuple[int, int, str, int, str]:
    return (
        fact.effective_interval.start_ns,
        fact.effective_interval.end_ns,
        fact.fusion_logical_key,
        fact.fusion_output_ordinal,
        fact.semantic_fingerprint_sha256,
    )


def output_admission_projection_values(
    values: dict[str, object],
) -> dict[str, object]:
    """Project an output-admission decision without row IDs or clocks."""

    source_enrichments = values["source_enrichments"]
    if not isinstance(source_enrichments, tuple) or any(
        not isinstance(item, PlatformEnrichedOutputReference) for item in source_enrichments
    ):
        raise TypeError("source_enrichments must contain enrichment references")
    admitted_facts = values["admitted_hypothesis_facts"]
    if not isinstance(admitted_facts, tuple) or any(
        not isinstance(item, ProductionAdmittedHypothesisFact) for item in admitted_facts
    ):
        raise TypeError("admitted_hypothesis_facts must contain admitted facts")
    evidence_class = values["evidence_class"]
    production_eligible = values["production_eligible"]
    if not isinstance(evidence_class, AdmissionEvidenceClass):
        raise TypeError("evidence_class must be an AdmissionEvidenceClass")
    if not isinstance(production_eligible, bool):
        raise TypeError("production_eligible must be a boolean")
    validate_evidence_eligibility(evidence_class, production_eligible)
    return {
        "semantic_projection_version": OUTPUT_ADMISSION_SEMANTIC_PROJECTION_VERSION,
        "decision": values["decision"],
        "evidence_class": evidence_class.value,
        "production_eligible": production_eligible,
        "recording_identity": values["recording_identity"],
        "source_enrichments": [
            platform_enriched_output_logical_projection(item) for item in source_enrichments
        ],
        "admitted_hypothesis_facts": [item.model_dump(mode="json") for item in admitted_facts],
        "output_admission_policy_version": values["output_admission_policy_version"],
        "output_admission_policy_sha256": values["output_admission_policy_sha256"],
    }


def output_admission_projection(
    proof: OutputAdmissionProof,
) -> dict[str, object]:
    return output_admission_projection_values(
        {
            "decision": proof.decision,
            "evidence_class": proof.evidence_class,
            "production_eligible": proof.production_eligible,
            "recording_identity": proof.recording_identity,
            "source_enrichments": proof.source_enrichments,
            "admitted_hypothesis_facts": proof.admitted_hypothesis_facts,
            "output_admission_policy_version": proof.output_admission_policy_version,
            "output_admission_policy_sha256": proof.output_admission_policy_sha256,
        }
    )


def event_identity_assignment_semantic_projection(
    assignment: EventIdentityAssignment,
) -> dict[str, object]:
    """Project an assignment without its row ID or decision clock."""

    return {
        "recording_identity": assignment.recording_identity,
        "event_hypothesis_logical_key": assignment.event_hypothesis_logical_key,
        "event_hypothesis_semantic_sha256": (assignment.event_hypothesis_semantic_sha256),
        "event_id": assignment.event_id,
        "disposition": assignment.disposition.value,
        "relation": [item.value for item in assignment.relation],
        "identity_policy_version": assignment.identity_policy_version,
        "identity_policy_sha256": assignment.identity_policy_sha256,
        "candidates": [item.model_dump(mode="json") for item in assignment.candidates],
        "reason": assignment.reason,
        "registry_generation": assignment.registry_generation,
    }


def event_identity_relation_semantic_projection(
    relation: EventIdentityRelation,
) -> dict[str, object]:
    return {
        "recording_identity": relation.recording_identity,
        "assignment_logical_key": relation.assignment_logical_key,
        "from_event_id": relation.from_event_id,
        "to_event_id": relation.to_event_id,
        "relation": relation.relation.value,
        "score": relation.score,
        "reason": relation.reason,
        "identity_policy_version": relation.identity_policy_version,
        "identity_policy_sha256": relation.identity_policy_sha256,
        "registry_generation": relation.registry_generation,
    }


def _hypothesis_idempotency_key(
    hypothesis: PlatformEnrichedEventHypothesis,
    policy: EventIdentityPolicyRef,
) -> tuple[str, str, str]:
    return (
        hypothesis.event_hypothesis_logical_key,
        policy.version,
        policy.semantic_sha256,
    )


def _assignment_idempotency_key(
    assignment: EventIdentityAssignment,
) -> tuple[str, str, str]:
    return (
        assignment.event_hypothesis_logical_key,
        assignment.identity_policy_version,
        assignment.identity_policy_sha256,
    )


def _assignment_sort_key(
    assignment: EventIdentityAssignment,
) -> tuple[str, str, str, str]:
    return (
        assignment.event_hypothesis_logical_key,
        assignment.identity_policy_version,
        assignment.identity_policy_sha256,
        assignment.assignment_logical_key,
    )


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


__all__ = [
    "ADMISSION_PROOF_SEMANTIC_PROJECTION_VERSION",
    "EVENT_HYPOTHESIS_LOGICAL_KEY_NAMESPACE",
    "EVENT_HYPOTHESIS_SEMANTIC_PROJECTION_VERSION",
    "OUTPUT_ADMISSION_SEMANTIC_PROJECTION_VERSION",
    "AdmissionEvidenceClass",
    "AdmissionProof",
    "CrossRecordingEventIdentityError",
    "EventCurrentRevisionReference",
    "EventIdAllocator",
    "EventIdentityAllocationError",
    "EventIdentityAssignment",
    "EventIdentityAssignmentDisposition",
    "EventIdentityAssignmentRelation",
    "EventIdentityBatchResult",
    "EventIdentityCandidate",
    "EventIdentityCandidateRelation",
    "EventIdentityConflictError",
    "EventIdentityInputError",
    "EventIdentityOutboxRecord",
    "EventIdentityPolicyRef",
    "EventIdentityRegistryError",
    "EventIdentityRegistryMutation",
    "EventIdentityRegistryRepository",
    "EventIdentityRegistryService",
    "EventIdentityRelation",
    "EventIdentityResolution",
    "EventIdentityResolver",
    "EventRegistrySnapshot",
    "ExactFingerprintEventIdentityResolver",
    "InMemoryEventIdentityRegistryRepository",
    "OutputAdmissionProof",
    "PlatformEnrichedEventHypothesis",
    "PlatformEnrichedOutputReference",
    "ProductionAdmittedHypothesisFact",
    "ProductionOutputAdmissionPolicyRef",
    "ProductionQualificationUnavailableError",
    "RandomEventIdAllocator",
    "StableEventIdentity",
    "StaleEventRegistryFenceError",
    "admission_proof_projection",
    "event_hypothesis_semantic_projection",
    "event_identity_assignment_semantic_projection",
    "event_identity_relation_semantic_projection",
    "output_admission_projection",
    "platform_enriched_output_logical_projection",
    "validate_evidence_eligibility",
]
