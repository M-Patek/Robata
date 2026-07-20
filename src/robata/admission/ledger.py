"""Independent source-admission and alignment ledgers.

Architecture V1.1 Section 25.6 requires two denominators and two current
outcome projections. Records here are append-only; current state is rebuilt
from verified predecessor chains and is never embedded in an old decision.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from enum import StrEnum
from itertools import pairwise
from threading import RLock
from typing import Annotated, Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid, Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class AdmissionLedgerError(ValueError):
    """Raised when append-only admission accounting would become ambiguous."""


class SourceAdmissionOutcome(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    INVALID = "INVALID"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    QUARANTINED = "QUARANTINED"


class AlignmentAdmissionOutcome(StrEnum):
    NOT_SCHEDULED = "NOT_SCHEDULED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SourceAliasObservation(StrictModel):
    """One immutable source notification, separate from unique-content state."""

    schema_version: Literal["1.0"]
    alias_observation_id: OpaqueUuid
    source_uri: NonEmptyString
    source_version: NonEmptyString
    source_content_sha256: Sha256Digest
    recording_identity: Sha256Digest
    semantic_sha256: Sha256Digest
    observed_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_semantic_identity(self) -> Self:
        expected = semantic_sha256(_source_alias_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("source alias semantic_sha256 is inconsistent")
        if self.alias_observation_id != _stable_uuid("source-alias", expected):
            raise ValueError("source alias observation ID is inconsistent")
        return self


class SourceAdmissionDecision(StrictModel):
    """One append-only source-content outcome selection."""

    schema_version: Literal["1.0"]
    decision_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    sequence: NonNegativeInt
    outcome: SourceAdmissionOutcome
    validation_report_id: OpaqueUuid | None
    validation_report_semantic_sha256: Sha256Digest | None
    ready_manifest_id: OpaqueUuid | None
    ready_manifest_semantic_sha256: Sha256Digest | None
    policy_version: SchemaVersion
    policy_sha256: Sha256Digest
    reason_code: NonEmptyString | None
    predecessor_logical_key: NodeLogicalKey | None
    decision_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    decided_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_pair(
            "validation report",
            self.validation_report_id,
            self.validation_report_semantic_sha256,
        )
        _require_pair(
            "READY manifest",
            self.ready_manifest_id,
            self.ready_manifest_semantic_sha256,
        )
        if self.sequence == 0 and self.predecessor_logical_key is not None:
            raise ValueError("source decision genesis cannot have a predecessor")
        if self.sequence > 0 and self.predecessor_logical_key is None:
            raise ValueError("non-genesis source decision requires a predecessor")
        if self.outcome is SourceAdmissionOutcome.PENDING:
            if self.validation_report_id is not None or self.ready_manifest_id is not None:
                raise ValueError("PENDING source decisions cannot select evidence")
        elif self.outcome is SourceAdmissionOutcome.READY:
            if self.validation_report_id is None or self.ready_manifest_id is None:
                raise ValueError("READY requires selected validation and READY artifacts")
        elif self.outcome in {
            SourceAdmissionOutcome.INVALID,
            SourceAdmissionOutcome.FAILED_VALIDATION,
        }:
            if self.validation_report_id is None:
                raise ValueError("validation outcome requires a validation report")
            if self.ready_manifest_id is not None:
                raise ValueError("non-READY outcome cannot select a READY manifest")
        elif self.outcome is SourceAdmissionOutcome.QUARANTINED:
            if self.ready_manifest_id is not None:
                raise ValueError("QUARANTINED cannot select a READY manifest")
            if self.reason_code is None:
                raise ValueError("QUARANTINED requires a reason code")
        expected = semantic_sha256(_source_decision_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("source decision semantic_sha256 is inconsistent")
        logical_key = f"source-admission-decision:{expected}"
        if self.decision_logical_key != logical_key:
            raise ValueError("source decision logical key is inconsistent")
        if self.decision_id != _stable_uuid("source-admission-decision", expected):
            raise ValueError("source decision ID is inconsistent")
        return self


class AlignmentAdmissionDecision(StrictModel):
    """One append-only alignment outcome for a READY-manifest cohort member."""

    schema_version: Literal["1.0"]
    decision_id: OpaqueUuid
    ready_manifest_id: OpaqueUuid
    ready_manifest_semantic_sha256: Sha256Digest
    sequence: NonNegativeInt
    outcome: AlignmentAdmissionOutcome
    alignment_id: OpaqueUuid | None
    alignment_semantic_sha256: Sha256Digest | None
    validation_evidence_sha256: Sha256Digest | None
    policy_version: SchemaVersion
    policy_sha256: Sha256Digest
    reason_code: NonEmptyString | None
    predecessor_logical_key: NodeLogicalKey | None
    decision_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    decided_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_pair("alignment", self.alignment_id, self.alignment_semantic_sha256)
        if self.sequence == 0 and self.predecessor_logical_key is not None:
            raise ValueError("alignment decision genesis cannot have a predecessor")
        if self.sequence > 0 and self.predecessor_logical_key is None:
            raise ValueError("non-genesis alignment decision requires a predecessor")
        if self.outcome in {
            AlignmentAdmissionOutcome.NOT_SCHEDULED,
            AlignmentAdmissionOutcome.QUEUED,
        }:
            if self.alignment_id is not None or self.validation_evidence_sha256 is not None:
                raise ValueError(f"{self.outcome.value} cannot select alignment evidence")
        elif self.outcome is AlignmentAdmissionOutcome.RUNNING:
            if self.alignment_id is None:
                raise ValueError("RUNNING requires an alignment derivation")
            if self.validation_evidence_sha256 is not None:
                raise ValueError("RUNNING cannot carry terminal validation evidence")
        elif self.outcome in {
            AlignmentAdmissionOutcome.VALID,
            AlignmentAdmissionOutcome.DEGRADED,
            AlignmentAdmissionOutcome.INVALID,
        }:
            if self.alignment_id is None or self.validation_evidence_sha256 is None:
                raise ValueError("terminal alignment verdict requires derivation and evidence")
        elif (
            self.outcome
            in {
                AlignmentAdmissionOutcome.FAILED,
                AlignmentAdmissionOutcome.CANCELLED,
            }
            and self.reason_code is None
        ):
            raise ValueError(f"{self.outcome.value} requires a reason code")
        expected = semantic_sha256(_alignment_decision_projection(self))
        if self.semantic_sha256 != expected:
            raise ValueError("alignment decision semantic_sha256 is inconsistent")
        logical_key = f"alignment-admission-decision:{expected}"
        if self.decision_logical_key != logical_key:
            raise ValueError("alignment decision logical key is inconsistent")
        if self.decision_id != _stable_uuid("alignment-admission-decision", expected):
            raise ValueError("alignment decision ID is inconsistent")
        return self


class SourceAdmissionReconciliation(StrictModel):
    discovered_unique_content: NonNegativeInt
    alias_observations: NonNegativeInt
    pending: NonNegativeInt
    ready: NonNegativeInt
    invalid: NonNegativeInt
    failed_validation: NonNegativeInt
    quarantined: NonNegativeInt

    @model_validator(mode="after")
    def reconcile(self) -> Self:
        total = self.pending + self.ready + self.invalid + self.failed_validation + self.quarantined
        if total != self.discovered_unique_content:
            raise ValueError("source outcome counts do not reconcile")
        return self


class AlignmentReconciliation(StrictModel):
    ready_manifest_cohort: NonNegativeInt
    not_scheduled: NonNegativeInt
    queued: NonNegativeInt
    running: NonNegativeInt
    valid: NonNegativeInt
    degraded: NonNegativeInt
    invalid: NonNegativeInt
    failed: NonNegativeInt
    cancelled: NonNegativeInt

    @model_validator(mode="after")
    def reconcile(self) -> Self:
        total = (
            self.not_scheduled
            + self.queued
            + self.running
            + self.valid
            + self.degraded
            + self.invalid
            + self.failed
            + self.cancelled
        )
        if total != self.ready_manifest_cohort:
            raise ValueError("alignment outcome counts do not reconcile")
        return self


class PrimaryAdmissionPolicy(StrictModel):
    version: SchemaVersion
    semantic_sha256: Sha256Digest
    admissible_alignment_outcomes: tuple[AlignmentAdmissionOutcome, ...]

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        outcomes = self.admissible_alignment_outcomes
        if not outcomes or len(set(outcomes)) != len(outcomes):
            raise ValueError("admissible alignment outcomes must be nonempty and unique")
        allowed = {AlignmentAdmissionOutcome.VALID, AlignmentAdmissionOutcome.DEGRADED}
        if not set(outcomes) <= allowed:
            raise ValueError("only VALID or DEGRADED alignment may be admissible")
        expected = semantic_sha256(
            {
                "version": self.version,
                "admissible_alignment_outcomes": [item.value for item in outcomes],
            }
        )
        if self.semantic_sha256 != expected:
            raise ValueError("primary admission policy digest is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        version: str,
        admissible_alignment_outcomes: tuple[AlignmentAdmissionOutcome, ...],
    ) -> PrimaryAdmissionPolicy:
        projection = {
            "version": version,
            "admissible_alignment_outcomes": [item.value for item in admissible_alignment_outcomes],
        }
        return cls(
            version=version,
            semantic_sha256=semantic_sha256(projection),
            admissible_alignment_outcomes=admissible_alignment_outcomes,
        )


class PrimaryAdmissionEvaluation(StrictModel):
    recording_identity: Sha256Digest
    ready_manifest_id: OpaqueUuid | None
    ready_manifest_semantic_sha256: Sha256Digest | None
    source_outcome: SourceAdmissionOutcome | None
    alignment_outcome: AlignmentAdmissionOutcome | None
    alignment_id: OpaqueUuid | None
    alignment_semantic_sha256: Sha256Digest | None
    policy_version: SchemaVersion
    policy_sha256: Sha256Digest
    admissible: bool
    reason_code: NonEmptyString

    @model_validator(mode="after")
    def validate_selected_evidence(self) -> Self:
        _require_pair(
            "evaluated READY manifest",
            self.ready_manifest_id,
            self.ready_manifest_semantic_sha256,
        )
        _require_pair(
            "evaluated alignment",
            self.alignment_id,
            self.alignment_semantic_sha256,
        )
        if self.admissible:
            if (
                self.source_outcome is not SourceAdmissionOutcome.READY
                or self.ready_manifest_id is None
                or self.alignment_outcome
                not in {
                    AlignmentAdmissionOutcome.VALID,
                    AlignmentAdmissionOutcome.DEGRADED,
                }
                or self.alignment_id is None
            ):
                raise ValueError("admissible evaluation requires selected READY and alignment")
            if self.reason_code != "ADMISSIBLE":
                raise ValueError("admissible evaluation requires ADMISSIBLE reason")
        return self


class InMemoryAdmissionLedger:
    """Thread-safe local reference ledger with deterministic projection rebuild."""

    def __init__(self) -> None:
        self._aliases: dict[str, SourceAliasObservation] = {}
        self._alias_bindings: dict[tuple[str, str], tuple[str, str]] = {}
        self._source_decisions: dict[str, SourceAdmissionDecision] = {}
        self._source_history: dict[str, list[str]] = defaultdict(list)
        self._source_current: dict[str, str] = {}
        self._content_recording: dict[str, str] = {}
        self._alignment_decisions: dict[str, AlignmentAdmissionDecision] = {}
        self._alignment_history: dict[str, list[str]] = defaultdict(list)
        self._alignment_current: dict[str, str] = {}
        self._lock = RLock()

    def observe_alias(self, observation: SourceAliasObservation) -> SourceAliasObservation:
        key = (observation.source_uri, observation.source_version)
        binding = (observation.source_content_sha256, observation.recording_identity)
        with self._lock:
            previous = self._alias_bindings.get(key)
            if previous is not None and previous != binding:
                raise AdmissionLedgerError("source alias/version resolved to different content")
            existing = self._aliases.get(observation.alias_observation_id)
            if existing is not None:
                return existing
            self._alias_bindings[key] = binding
            self._aliases[observation.alias_observation_id] = observation
            return observation

    def append_source_decision(self, decision: SourceAdmissionDecision) -> SourceAdmissionDecision:
        with self._lock:
            existing = self._source_decisions.get(decision.decision_logical_key)
            if existing is not None:
                return existing
            bound_recording = self._content_recording.get(decision.source_content_sha256)
            if bound_recording is not None and bound_recording != decision.recording_identity:
                raise AdmissionLedgerError("source content is bound to another recording identity")
            current = self.current_source(decision.recording_identity)
            _verify_chain_append(
                current=current,
                sequence=decision.sequence,
                predecessor=decision.predecessor_logical_key,
                genesis_outcome=decision.outcome,
                required_genesis=SourceAdmissionOutcome.PENDING,
            )
            if (
                current is not None
                and current.source_content_sha256 != decision.source_content_sha256
            ):
                raise AdmissionLedgerError("recording identity cannot change source content")
            self._content_recording[decision.source_content_sha256] = decision.recording_identity
            self._source_decisions[decision.decision_logical_key] = decision
            self._source_history[decision.recording_identity].append(decision.decision_logical_key)
            self._source_current[decision.recording_identity] = decision.decision_logical_key
            return decision

    def append_alignment_decision(
        self, decision: AlignmentAdmissionDecision
    ) -> AlignmentAdmissionDecision:
        with self._lock:
            existing = self._alignment_decisions.get(decision.decision_logical_key)
            if existing is not None:
                return existing
            owning_source = self._source_for_ready_manifest(decision.ready_manifest_id)
            if owning_source is None:
                raise AdmissionLedgerError(
                    "alignment cohort requires a currently selected READY manifest"
                )
            if (
                owning_source.ready_manifest_semantic_sha256
                != decision.ready_manifest_semantic_sha256
            ):
                raise AdmissionLedgerError("alignment cohort READY digest mismatch")
            current = self.current_alignment(decision.ready_manifest_id)
            _verify_chain_append(
                current=current,
                sequence=decision.sequence,
                predecessor=decision.predecessor_logical_key,
                genesis_outcome=decision.outcome,
                required_genesis=AlignmentAdmissionOutcome.NOT_SCHEDULED,
            )
            self._alignment_decisions[decision.decision_logical_key] = decision
            self._alignment_history[decision.ready_manifest_id].append(
                decision.decision_logical_key
            )
            self._alignment_current[decision.ready_manifest_id] = decision.decision_logical_key
            return decision

    def current_source(self, recording_identity: str) -> SourceAdmissionDecision | None:
        key = self._source_current.get(recording_identity)
        return self._source_decisions.get(key) if key is not None else None

    def current_alignment(self, ready_manifest_id: str) -> AlignmentAdmissionDecision | None:
        key = self._alignment_current.get(ready_manifest_id)
        return self._alignment_decisions.get(key) if key is not None else None

    def source_history(self, recording_identity: str) -> tuple[SourceAdmissionDecision, ...]:
        with self._lock:
            return tuple(
                self._source_decisions[key]
                for key in self._source_history.get(recording_identity, ())
            )

    def alignment_history(self, ready_manifest_id: str) -> tuple[AlignmentAdmissionDecision, ...]:
        with self._lock:
            return tuple(
                self._alignment_decisions[key]
                for key in self._alignment_history.get(ready_manifest_id, ())
            )

    def reconcile_sources(self) -> SourceAdmissionReconciliation:
        with self._lock:
            counts = Counter(
                self._source_decisions[key].outcome for key in self._source_current.values()
            )
            return SourceAdmissionReconciliation(
                discovered_unique_content=len(self._source_current),
                alias_observations=len(self._aliases),
                pending=counts[SourceAdmissionOutcome.PENDING],
                ready=counts[SourceAdmissionOutcome.READY],
                invalid=counts[SourceAdmissionOutcome.INVALID],
                failed_validation=counts[SourceAdmissionOutcome.FAILED_VALIDATION],
                quarantined=counts[SourceAdmissionOutcome.QUARANTINED],
            )

    def reconcile_alignments(self) -> AlignmentReconciliation:
        with self._lock:
            counts = Counter(
                self._alignment_decisions[key].outcome for key in self._alignment_current.values()
            )
            return AlignmentReconciliation(
                ready_manifest_cohort=len(self._alignment_current),
                not_scheduled=counts[AlignmentAdmissionOutcome.NOT_SCHEDULED],
                queued=counts[AlignmentAdmissionOutcome.QUEUED],
                running=counts[AlignmentAdmissionOutcome.RUNNING],
                valid=counts[AlignmentAdmissionOutcome.VALID],
                degraded=counts[AlignmentAdmissionOutcome.DEGRADED],
                invalid=counts[AlignmentAdmissionOutcome.INVALID],
                failed=counts[AlignmentAdmissionOutcome.FAILED],
                cancelled=counts[AlignmentAdmissionOutcome.CANCELLED],
            )

    def evaluate_primary(
        self,
        *,
        recording_identity: str,
        policy: PrimaryAdmissionPolicy,
    ) -> PrimaryAdmissionEvaluation:
        with self._lock:
            source = self.current_source(recording_identity)
            if source is None:
                return _primary_evaluation(
                    recording_identity,
                    policy,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "SOURCE_NOT_DISCOVERED",
                )
            if source.outcome is not SourceAdmissionOutcome.READY:
                return _primary_evaluation(
                    recording_identity,
                    policy,
                    source.ready_manifest_id,
                    source.ready_manifest_semantic_sha256,
                    source.outcome,
                    None,
                    None,
                    None,
                    "SOURCE_NOT_READY",
                )
            assert source.ready_manifest_id is not None
            alignment = self.current_alignment(source.ready_manifest_id)
            if alignment is None:
                return _primary_evaluation(
                    recording_identity,
                    policy,
                    source.ready_manifest_id,
                    source.ready_manifest_semantic_sha256,
                    source.outcome,
                    None,
                    None,
                    None,
                    "ALIGNMENT_NOT_IN_COHORT",
                )
            if alignment.outcome not in policy.admissible_alignment_outcomes:
                return _primary_evaluation(
                    recording_identity,
                    policy,
                    source.ready_manifest_id,
                    source.ready_manifest_semantic_sha256,
                    source.outcome,
                    alignment.outcome,
                    alignment.alignment_id,
                    alignment.alignment_semantic_sha256,
                    "ALIGNMENT_NOT_ADMISSIBLE",
                )
            return PrimaryAdmissionEvaluation(
                recording_identity=recording_identity,
                ready_manifest_id=source.ready_manifest_id,
                ready_manifest_semantic_sha256=source.ready_manifest_semantic_sha256,
                source_outcome=source.outcome,
                alignment_outcome=alignment.outcome,
                alignment_id=alignment.alignment_id,
                alignment_semantic_sha256=alignment.alignment_semantic_sha256,
                policy_version=policy.version,
                policy_sha256=policy.semantic_sha256,
                admissible=True,
                reason_code="ADMISSIBLE",
            )

    def rebuild_current_projections(self) -> None:
        with self._lock:
            self._source_current = _rebuild_projection(
                self._source_decisions.values(),
                subject=lambda item: item.recording_identity,
            )
            self._alignment_current = _rebuild_projection(
                self._alignment_decisions.values(),
                subject=lambda item: item.ready_manifest_id,
            )

    def _source_for_ready_manifest(self, ready_manifest_id: str) -> SourceAdmissionDecision | None:
        matches = [
            self._source_decisions[key]
            for key in self._source_current.values()
            if self._source_decisions[key].outcome is SourceAdmissionOutcome.READY
            and self._source_decisions[key].ready_manifest_id == ready_manifest_id
        ]
        if len(matches) > 1:
            raise AdmissionLedgerError("READY manifest is selected by multiple source identities")
        return matches[0] if matches else None


def create_source_alias_observation(
    *,
    source_uri: str,
    source_version: str,
    source_content_sha256: str,
    recording_identity: str,
    observed_at: str,
) -> SourceAliasObservation:
    projection = {
        "source_uri": source_uri,
        "source_version": source_version,
        "source_content_sha256": source_content_sha256,
        "recording_identity": recording_identity,
    }
    digest = semantic_sha256(projection)
    return SourceAliasObservation(
        schema_version="1.0",
        alias_observation_id=_stable_uuid("source-alias", digest),
        source_uri=source_uri,
        source_version=source_version,
        source_content_sha256=source_content_sha256,
        recording_identity=recording_identity,
        semantic_sha256=digest,
        observed_at=observed_at,
    )


def create_source_admission_decision(
    *,
    recording_identity: str,
    source_content_sha256: str,
    sequence: int,
    outcome: SourceAdmissionOutcome,
    policy_version: str,
    policy_sha256: str,
    decided_at: str,
    predecessor_logical_key: str | None = None,
    validation_report_id: str | None = None,
    validation_report_semantic_sha256: str | None = None,
    ready_manifest_id: str | None = None,
    ready_manifest_semantic_sha256: str | None = None,
    reason_code: str | None = None,
) -> SourceAdmissionDecision:
    values = {
        "recording_identity": recording_identity,
        "source_content_sha256": source_content_sha256,
        "sequence": sequence,
        "outcome": outcome,
        "validation_report_id": validation_report_id,
        "validation_report_semantic_sha256": validation_report_semantic_sha256,
        "ready_manifest_id": ready_manifest_id,
        "ready_manifest_semantic_sha256": ready_manifest_semantic_sha256,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "reason_code": reason_code,
        "predecessor_logical_key": predecessor_logical_key,
    }
    digest = semantic_sha256(_source_decision_projection(values))
    return SourceAdmissionDecision(
        schema_version="1.0",
        decision_id=_stable_uuid("source-admission-decision", digest),
        decision_logical_key=f"source-admission-decision:{digest}",
        semantic_sha256=digest,
        decided_at=decided_at,
        recording_identity=recording_identity,
        source_content_sha256=source_content_sha256,
        sequence=sequence,
        outcome=outcome,
        validation_report_id=validation_report_id,
        validation_report_semantic_sha256=validation_report_semantic_sha256,
        ready_manifest_id=ready_manifest_id,
        ready_manifest_semantic_sha256=ready_manifest_semantic_sha256,
        policy_version=policy_version,
        policy_sha256=policy_sha256,
        reason_code=reason_code,
        predecessor_logical_key=predecessor_logical_key,
    )


def create_alignment_admission_decision(
    *,
    ready_manifest_id: str,
    ready_manifest_semantic_sha256: str,
    sequence: int,
    outcome: AlignmentAdmissionOutcome,
    policy_version: str,
    policy_sha256: str,
    decided_at: str,
    predecessor_logical_key: str | None = None,
    alignment_id: str | None = None,
    alignment_semantic_sha256: str | None = None,
    validation_evidence_sha256: str | None = None,
    reason_code: str | None = None,
) -> AlignmentAdmissionDecision:
    values = {
        "ready_manifest_id": ready_manifest_id,
        "ready_manifest_semantic_sha256": ready_manifest_semantic_sha256,
        "sequence": sequence,
        "outcome": outcome,
        "alignment_id": alignment_id,
        "alignment_semantic_sha256": alignment_semantic_sha256,
        "validation_evidence_sha256": validation_evidence_sha256,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "reason_code": reason_code,
        "predecessor_logical_key": predecessor_logical_key,
    }
    digest = semantic_sha256(_alignment_decision_projection(values))
    return AlignmentAdmissionDecision(
        schema_version="1.0",
        decision_id=_stable_uuid("alignment-admission-decision", digest),
        decision_logical_key=f"alignment-admission-decision:{digest}",
        semantic_sha256=digest,
        decided_at=decided_at,
        ready_manifest_id=ready_manifest_id,
        ready_manifest_semantic_sha256=ready_manifest_semantic_sha256,
        sequence=sequence,
        outcome=outcome,
        alignment_id=alignment_id,
        alignment_semantic_sha256=alignment_semantic_sha256,
        validation_evidence_sha256=validation_evidence_sha256,
        policy_version=policy_version,
        policy_sha256=policy_sha256,
        reason_code=reason_code,
        predecessor_logical_key=predecessor_logical_key,
    )


def _source_alias_projection(value: SourceAliasObservation) -> dict[str, object]:
    return {
        "source_uri": value.source_uri,
        "source_version": value.source_version,
        "source_content_sha256": value.source_content_sha256,
        "recording_identity": value.recording_identity,
    }


def _source_decision_projection(value: object) -> dict[str, object]:
    return _projection(
        value,
        (
            "recording_identity",
            "source_content_sha256",
            "sequence",
            "outcome",
            "validation_report_semantic_sha256",
            "ready_manifest_semantic_sha256",
            "policy_version",
            "policy_sha256",
            "reason_code",
            "predecessor_logical_key",
        ),
    )


def _alignment_decision_projection(value: object) -> dict[str, object]:
    return _projection(
        value,
        (
            "ready_manifest_semantic_sha256",
            "sequence",
            "outcome",
            "alignment_semantic_sha256",
            "validation_evidence_sha256",
            "policy_version",
            "policy_sha256",
            "reason_code",
            "predecessor_logical_key",
        ),
    )


def _projection(value: object, names: tuple[str, ...]) -> dict[str, object]:
    source: dict[str, Any] = (
        value if isinstance(value, dict) else {name: getattr(value, name) for name in names}
    )
    return {name: source[name] for name in names}


def _require_pair(name: str, identity: object | None, digest: object | None) -> None:
    if (identity is None) != (digest is None):
        raise ValueError(f"{name} ID and digest must be supplied together")


def _stable_uuid(namespace: str, digest: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{digest}"))


def _verify_chain_append(
    *,
    current: SourceAdmissionDecision | AlignmentAdmissionDecision | None,
    sequence: int,
    predecessor: str | None,
    genesis_outcome: StrEnum,
    required_genesis: StrEnum,
) -> None:
    if current is None:
        if sequence != 0 or predecessor is not None or genesis_outcome is not required_genesis:
            raise AdmissionLedgerError(
                f"first decision must be sequence 0 {required_genesis.value}"
            )
        return
    if sequence != current.sequence + 1:
        raise AdmissionLedgerError("decision sequence is not contiguous")
    if predecessor != current.decision_logical_key:
        raise AdmissionLedgerError("decision predecessor is stale")


def _rebuild_projection(items: object, *, subject: Callable[[Any], str]) -> dict[str, str]:
    records: tuple[Any, ...] = tuple(items)  # type: ignore[arg-type]
    by_subject: dict[str, list[Any]] = defaultdict(list)
    for item in records:
        by_subject[subject(item)].append(item)
    projection: dict[str, str] = {}
    for subject_id, history in by_subject.items():
        ordered = sorted(history, key=lambda item: item.sequence)
        if ordered[0].sequence != 0 or ordered[0].predecessor_logical_key is not None:
            raise AdmissionLedgerError("decision chain has no valid genesis")
        for previous, current in pairwise(ordered):
            if current.sequence != previous.sequence + 1:
                raise AdmissionLedgerError("decision chain sequence is not contiguous")
            if current.predecessor_logical_key != previous.decision_logical_key:
                raise AdmissionLedgerError("decision chain predecessor is invalid")
        projection[subject_id] = ordered[-1].decision_logical_key
    return projection


def _primary_evaluation(
    recording_identity: str,
    policy: PrimaryAdmissionPolicy,
    ready_manifest_id: str | None,
    ready_manifest_semantic_sha256: str | None,
    source_outcome: SourceAdmissionOutcome | None,
    alignment_outcome: AlignmentAdmissionOutcome | None,
    alignment_id: str | None,
    alignment_semantic_sha256: str | None,
    reason_code: str,
) -> PrimaryAdmissionEvaluation:
    return PrimaryAdmissionEvaluation(
        recording_identity=recording_identity,
        ready_manifest_id=ready_manifest_id,
        ready_manifest_semantic_sha256=ready_manifest_semantic_sha256,
        source_outcome=source_outcome,
        alignment_outcome=alignment_outcome,
        alignment_id=alignment_id,
        alignment_semantic_sha256=alignment_semantic_sha256,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admissible=False,
        reason_code=reason_code,
    )


__all__ = [
    "AdmissionLedgerError",
    "AlignmentAdmissionDecision",
    "AlignmentAdmissionOutcome",
    "AlignmentReconciliation",
    "InMemoryAdmissionLedger",
    "PrimaryAdmissionEvaluation",
    "PrimaryAdmissionPolicy",
    "SourceAdmissionDecision",
    "SourceAdmissionOutcome",
    "SourceAdmissionReconciliation",
    "SourceAliasObservation",
    "create_alignment_admission_decision",
    "create_source_admission_decision",
    "create_source_alias_observation",
]
