"""Immutable recovery evidence emitted by the canonical integration boundary.

The module records what a qualification runner observed without deciding whether
that observation promotes a workload.  In particular, a local-conformance
receipt remains local-conformance evidence when it is bound into this record.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel

if TYPE_CHECKING:
    from robata.application.canonical.local_composition import CanonicalLocalRunReceipt

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
CanonicalOutboxDeliveryOutcome = Literal[
    "NOT_APPLICABLE", "DELIVERED", "PENDING", "DEAD_LETTER", "FAILED"
]
CanonicalReviewRoutingDisposition = Literal[
    "ENQUEUED", "ALREADY_ENQUEUED", "NOT_ROUTED", "ROUTING_FAILED"
]

CANONICAL_RECOVERY_QUALIFICATION_EVIDENCE_VERSION: Final[
    Literal["canonical-recovery-qualification-evidence-v1"]
] = "canonical-recovery-qualification-evidence-v1"


class CanonicalRecoveryEvidenceClass(StrEnum):
    """Provenance of the receipts bound into one recovery observation."""

    LOCAL_CONFORMANCE = "LOCAL_CONFORMANCE"
    REPRESENTATIVE_BENCHMARK = "REPRESENTATIVE_BENCHMARK"
    PRODUCTION_QUALIFICATION = "PRODUCTION_QUALIFICATION"


class CanonicalRecoveryScenario(StrEnum):
    """P10 recovery and sustained-operation scenarios."""

    RESTART_REPLAY = "RESTART_REPLAY"
    PROCESS_CRASH = "PROCESS_CRASH"
    LEASE_EXPIRY = "LEASE_EXPIRY"
    DUPLICATE_INJECTION = "DUPLICATE_INJECTION"
    PROVIDER_RETRY = "PROVIDER_RETRY"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    BROKER_FAILURE = "BROKER_FAILURE"
    OBJECT_STORE_FAILURE = "OBJECT_STORE_FAILURE"
    BACKLOG_DRAIN = "BACKLOG_DRAIN"
    SOAK = "SOAK"


_FAILURE_SCENARIOS: Final[frozenset[CanonicalRecoveryScenario]] = frozenset(
    {
        CanonicalRecoveryScenario.RESTART_REPLAY,
        CanonicalRecoveryScenario.PROVIDER_RETRY,
        CanonicalRecoveryScenario.PROVIDER_TIMEOUT,
        CanonicalRecoveryScenario.BROKER_FAILURE,
        CanonicalRecoveryScenario.OBJECT_STORE_FAILURE,
        CanonicalRecoveryScenario.PROCESS_CRASH,
        CanonicalRecoveryScenario.LEASE_EXPIRY,
        CanonicalRecoveryScenario.DUPLICATE_INJECTION,
    }
)


class CanonicalRecoveryReceiptEvidence(StrictModel):
    """Identity-bearing facts from one fresh or replay canonical receipt."""

    run_id: NonEmptyString
    recording_identity: NonEmptyString
    status: Literal["SUCCEEDED", "NO_EVENTS"]
    command_sha256: Sha256Digest
    completion_semantic_sha256: Sha256Digest
    event_ids: tuple[NonEmptyString, ...]
    revision_ids: tuple[NonEmptyString, ...]
    outbox_ids: tuple[NonEmptyString, ...]
    review_task_id: NonEmptyString | None
    outbox_delivery_outcome: CanonicalOutboxDeliveryOutcome
    review_routing_disposition: CanonicalReviewRoutingDisposition
    replayed: bool
    evidence_class: CanonicalRecoveryEvidenceClass
    production_eligible: Literal[False]

    @classmethod
    def from_local_receipt(cls, receipt: CanonicalLocalRunReceipt) -> Self:
        """Project a local receipt without changing its evidence provenance."""

        from robata.application.canonical.local_composition import CanonicalLocalRunReceipt

        if not isinstance(receipt, CanonicalLocalRunReceipt):
            raise TypeError("receipt must be CanonicalLocalRunReceipt")
        return cls(
            run_id=receipt.run_id,
            recording_identity=receipt.recording_identity,
            status=receipt.status,
            command_sha256=receipt.command_sha256,
            completion_semantic_sha256=receipt.completion_semantic_sha256,
            event_ids=receipt.event_ids,
            revision_ids=receipt.revision_ids,
            outbox_ids=receipt.outbox_ids,
            review_task_id=receipt.review_routing.review_task_id,
            outbox_delivery_outcome=receipt.outbox_delivery.outcome.value,
            review_routing_disposition=receipt.review_routing.disposition.value,
            replayed=receipt.replayed,
            evidence_class=CanonicalRecoveryEvidenceClass(receipt.evidence_class),
            production_eligible=receipt.production_eligible,
        )

    @model_validator(mode="after")
    def validate_unique_identity_collections(self) -> Self:
        for field_name, values in (
            ("event_ids", self.event_ids),
            ("revision_ids", self.revision_ids),
            ("outbox_ids", self.outbox_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")
        return self


class CanonicalRecoveryQualificationEvidence(StrictModel):
    """One immutable, report-bindable canonical recovery observation.

    This is deliberately a fact record, not a promotion gate.  A report may
    retain a failed observation (`recovery_completed=False`) so that it cannot
    be confused with a passing recovery scenario.
    """

    model_version: Literal["canonical-recovery-qualification-evidence-v1"] = (
        CANONICAL_RECOVERY_QUALIFICATION_EVIDENCE_VERSION
    )
    workload_fingerprint: Sha256Digest
    run_namespace: NonEmptyString
    qualification_scope_sha256: Sha256Digest | None = None
    scenario: CanonicalRecoveryScenario
    scenario_evidence_sha256: Sha256Digest
    fresh: CanonicalRecoveryReceiptEvidence
    replay: CanonicalRecoveryReceiptEvidence
    fresh_completed: Literal[True] = True
    replay_completed: Literal[True] = True
    failure_observed: bool
    duplicate_injection_count: NonNegativeInt = 0
    recovery_completed: bool
    authoritative_terminal_ids: tuple[NonEmptyString, ...]
    authoritative_terminal_count: NonNegativeInt
    duplicate_terminal_count: NonNegativeInt
    outbox_delivery_ids: tuple[NonEmptyString, ...]
    outbox_delivery_count: NonNegativeInt
    duplicate_outbox_delivery_count: NonNegativeInt
    review_task_ids: tuple[NonEmptyString, ...]
    review_task_count: NonNegativeInt
    duplicate_review_task_count: NonNegativeInt
    backlog_start_count: NonNegativeInt = 0
    backlog_end_count: NonNegativeInt = 0
    run_duration_ns: Nanoseconds
    evidence_class: CanonicalRecoveryEvidenceClass
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_recovery_closure(self) -> Self:
        self._assert_recovery_closure()
        return self

    def _assert_recovery_closure(self) -> None:
        if self.run_duration_ns <= 0:
            raise ValueError("run_duration_ns must be positive")
        if self.scenario is CanonicalRecoveryScenario.BACKLOG_DRAIN:
            if self.backlog_start_count == 0 or self.backlog_end_count != 0:
                raise ValueError("backlog-drain evidence must drain a nonzero backlog")
        elif self.backlog_start_count or self.backlog_end_count:
            raise ValueError("only backlog-drain evidence may carry backlog counts")
        if self.scenario is CanonicalRecoveryScenario.DUPLICATE_INJECTION:
            if self.duplicate_injection_count == 0:
                raise ValueError("duplicate-injection evidence must record an injected duplicate")
        elif self.duplicate_injection_count:
            raise ValueError("only duplicate-injection evidence may carry injected duplicates")
        if self.fresh.replayed or not self.replay.replayed:
            raise ValueError("fresh and replay evidence must bind a fresh receipt then a replay")
        for field_name in (
            "run_id",
            "recording_identity",
            "status",
            "command_sha256",
            "completion_semantic_sha256",
            "event_ids",
            "revision_ids",
            "outbox_ids",
            "review_task_id",
            "evidence_class",
            "production_eligible",
        ):
            if getattr(self.fresh, field_name) != getattr(self.replay, field_name):
                raise ValueError(f"fresh and replay receipts disagree on {field_name}")
        if (
            self.evidence_class is not self.fresh.evidence_class
            or self.evidence_class is not self.replay.evidence_class
        ):
            raise ValueError("recovery evidence cannot upgrade receipt provenance")
        self._validate_observed_collection(
            field_name="authoritative terminals",
            values=self.authoritative_terminal_ids,
            count=self.authoritative_terminal_count,
            duplicate_count=self.duplicate_terminal_count,
        )
        self._validate_observed_collection(
            field_name="outbox deliveries",
            values=self.outbox_delivery_ids,
            count=self.outbox_delivery_count,
            duplicate_count=self.duplicate_outbox_delivery_count,
        )
        self._validate_observed_collection(
            field_name="review tasks",
            values=self.review_task_ids,
            count=self.review_task_count,
            duplicate_count=self.duplicate_review_task_count,
        )
        if self.recovery_completed:
            if self.scenario in _FAILURE_SCENARIOS and not self.failure_observed:
                raise ValueError("a completed failure scenario must record the injected failure")
            if self.authoritative_terminal_count != 1 or self.duplicate_terminal_count:
                raise ValueError(
                    "completed recovery requires one non-duplicate authoritative terminal"
                )
            if self.authoritative_terminal_ids != (self.fresh.completion_semantic_sha256,):
                raise ValueError(
                    "completed recovery terminal does not match the canonical completion"
                )
            if (
                self.outbox_delivery_ids != self.fresh.outbox_ids
                or self.duplicate_outbox_delivery_count
            ):
                raise ValueError("completed recovery outbox delivery does not match the completion")
            expected_review_ids = (
                () if self.fresh.review_task_id is None else (self.fresh.review_task_id,)
            )
            if self.fresh.outbox_ids:
                if (
                    self.fresh.outbox_delivery_outcome != "DELIVERED"
                    or self.replay.outbox_delivery_outcome != "DELIVERED"
                ):
                    raise ValueError("completed recovery requires delivered outbox evidence")
            elif (
                self.fresh.outbox_delivery_outcome != "NOT_APPLICABLE"
                or self.replay.outbox_delivery_outcome != "NOT_APPLICABLE"
            ):
                raise ValueError("empty recovery outbox must be not applicable")
            if self.fresh.review_task_id is not None:
                if (
                    self.fresh.review_routing_disposition != "ENQUEUED"
                    or self.replay.review_routing_disposition != "ALREADY_ENQUEUED"
                ):
                    raise ValueError("completed recovery requires reconciled review routing")
            elif (
                self.fresh.review_routing_disposition != "NOT_ROUTED"
                or self.replay.review_routing_disposition != "NOT_ROUTED"
            ):
                raise ValueError("unrouted recovery must declare no review routing")
            if self.review_task_ids != expected_review_ids or self.duplicate_review_task_count:
                raise ValueError("completed recovery review tasks do not match the completion")

    @staticmethod
    def _validate_observed_collection(
        *,
        field_name: str,
        values: tuple[str, ...],
        count: int,
        duplicate_count: int,
    ) -> None:
        if count != len(values):
            raise ValueError(f"{field_name} count does not match observed IDs")
        if duplicate_count != len(values) - len(set(values)):
            raise ValueError(f"{field_name} duplicate count does not match observed IDs")


def build_canonical_recovery_qualification_evidence(
    *,
    workload_fingerprint: str,
    run_namespace: str,
    scenario: CanonicalRecoveryScenario,
    scenario_evidence_sha256: str,
    fresh_receipt: CanonicalLocalRunReceipt,
    replay_receipt: CanonicalLocalRunReceipt,
    authoritative_terminal_ids: tuple[str, ...],
    outbox_delivery_ids: tuple[str, ...],
    review_task_ids: tuple[str, ...],
    failure_observed: bool,
    recovery_completed: bool,
    run_duration_ns: int,
    qualification_scope_sha256: str | None = None,
    backlog_start_count: int = 0,
    backlog_end_count: int = 0,
    duplicate_injection_count: int = 0,
) -> CanonicalRecoveryQualificationEvidence:
    """Bind local fresh/replay receipts to observed recovery-state identifiers.

    The current local composition emits `LOCAL_CONFORMANCE` receipts.  The
    builder intentionally derives the enclosing evidence class from them, so a
    local fixture cannot be relabeled as representative or production evidence.
    """

    fresh = CanonicalRecoveryReceiptEvidence.from_local_receipt(fresh_receipt)
    replay = CanonicalRecoveryReceiptEvidence.from_local_receipt(replay_receipt)
    return CanonicalRecoveryQualificationEvidence(
        workload_fingerprint=workload_fingerprint,
        run_namespace=run_namespace,
        qualification_scope_sha256=qualification_scope_sha256,
        scenario=scenario,
        scenario_evidence_sha256=scenario_evidence_sha256,
        fresh=fresh,
        replay=replay,
        failure_observed=failure_observed,
        recovery_completed=recovery_completed,
        authoritative_terminal_ids=authoritative_terminal_ids,
        authoritative_terminal_count=len(authoritative_terminal_ids),
        duplicate_terminal_count=(
            len(authoritative_terminal_ids) - len(set(authoritative_terminal_ids))
        ),
        outbox_delivery_ids=outbox_delivery_ids,
        outbox_delivery_count=len(outbox_delivery_ids),
        duplicate_outbox_delivery_count=(len(outbox_delivery_ids) - len(set(outbox_delivery_ids))),
        review_task_ids=review_task_ids,
        review_task_count=len(review_task_ids),
        duplicate_review_task_count=len(review_task_ids) - len(set(review_task_ids)),
        backlog_start_count=backlog_start_count,
        backlog_end_count=backlog_end_count,
        duplicate_injection_count=duplicate_injection_count,
        run_duration_ns=run_duration_ns,
        evidence_class=fresh.evidence_class,
    )


__all__ = [
    "CANONICAL_RECOVERY_QUALIFICATION_EVIDENCE_VERSION",
    "CanonicalRecoveryEvidenceClass",
    "CanonicalRecoveryQualificationEvidence",
    "CanonicalRecoveryReceiptEvidence",
    "CanonicalRecoveryScenario",
    "build_canonical_recovery_qualification_evidence",
]
