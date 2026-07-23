"""Provider-neutral durable stream-work state and capability contracts.

The stream scheduler is deliberately additive to :mod:`robata.queue.models`.
The V1 work ledger carries an ``mcap_id`` and has success-only result
semantics; a pre-EOS stream work item has a typed source subject and must keep
an exact evidence reference for *every* terminal outcome.  These models are
therefore not subclasses of the V1 ledger models and must not be upcast into
them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from robata.contracts.common import SchemaVersion, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.contracts.stream_common import (
    ArtifactEvidenceRef,
    NonEmptyString,
    PreEosCaptureSubjectRef,
    StreamStage,
    StreamSubjectRef,
    TerminalOutcome,
    validate_rfc3339,
)
from robata.contracts.stream_planning import StreamWorkDependency, StreamWorkItemPlan

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class StreamWorkItemState(StrEnum):
    """Durable stream-work lifecycle.

    The terminal vocabulary intentionally follows the stream closure
    contract, rather than the V1 scheduler's ``FAILED_PERMANENT`` result
    shape.  In particular, abstention, no-events, and incomplete input are
    explicit terminal states and cannot disappear from the expected set.
    """

    PLANNED = "PLANNED"
    READY = "READY"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED_POLICY = "SKIPPED_POLICY"
    SKIPPED_NOT_NEEDED = "SKIPPED_NOT_NEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"
    LATE_INPUT = "LATE_INPUT"
    INCOMPLETE = "INCOMPLETE"
    ABSTAINED = "ABSTAINED"
    NO_EVENTS = "NO_EVENTS"
    INVALIDATED = "INVALIDATED"


TERMINAL_STREAM_WORK_STATES = frozenset(
    {
        StreamWorkItemState.SUCCEEDED,
        StreamWorkItemState.SKIPPED_POLICY,
        StreamWorkItemState.SKIPPED_NOT_NEEDED,
        StreamWorkItemState.FAILED,
        StreamWorkItemState.CANCELLED,
        StreamWorkItemState.EXPIRED,
        StreamWorkItemState.QUARANTINED,
        StreamWorkItemState.LATE_INPUT,
        StreamWorkItemState.INCOMPLETE,
        StreamWorkItemState.ABSTAINED,
        StreamWorkItemState.NO_EVENTS,
        StreamWorkItemState.INVALIDATED,
    }
)


class StreamWorkAttemptOutcome(StrEnum):
    """Outcome of one fenced dispatch attempt."""

    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"
    LATE_INPUT = "LATE_INPUT"
    INCOMPLETE = "INCOMPLETE"
    ABSTAINED = "ABSTAINED"
    NO_EVENTS = "NO_EVENTS"
    INVALIDATED = "INVALIDATED"
    SKIPPED = "SKIPPED"


_TERMINAL_EVIDENCE_OUTCOME_BY_STATE: dict[StreamWorkItemState, TerminalOutcome] = {
    StreamWorkItemState.SUCCEEDED: TerminalOutcome.SUCCEEDED,
    StreamWorkItemState.SKIPPED_POLICY: TerminalOutcome.SKIPPED_POLICY,
    StreamWorkItemState.SKIPPED_NOT_NEEDED: TerminalOutcome.SKIPPED_NOT_NEEDED,
    StreamWorkItemState.FAILED: TerminalOutcome.FAILED,
    StreamWorkItemState.CANCELLED: TerminalOutcome.CANCELLED,
    StreamWorkItemState.EXPIRED: TerminalOutcome.EXPIRED,
    StreamWorkItemState.QUARANTINED: TerminalOutcome.QUARANTINED,
    StreamWorkItemState.LATE_INPUT: TerminalOutcome.LATE_INPUT,
    StreamWorkItemState.INCOMPLETE: TerminalOutcome.INCOMPLETE,
    StreamWorkItemState.ABSTAINED: TerminalOutcome.ABSTAINED,
    StreamWorkItemState.NO_EVENTS: TerminalOutcome.NO_EVENTS,
    StreamWorkItemState.INVALIDATED: TerminalOutcome.INVALIDATED,
}


def _parse_timestamp(value: str, field_name: str) -> datetime:
    """Parse and normalize a timestamp after the strict wire shape check."""

    validate_rfc3339(value, field_name)
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    return parsed.astimezone(UTC)


class StreamTerminalEvidence(StrictModel):
    """Exact evidence required to close one stream-work item.

    Unlike V1 result fields, ``evidence_ref`` is mandatory for success,
    failure, cancellation, abstention, no-events, and every other terminal
    outcome.  A terminal row without bytes that prove its outcome is invalid.
    """

    outcome: TerminalOutcome
    evidence_ref: ArtifactEvidenceRef
    terminal_policy_version: SchemaVersion
    completed_at: Rfc3339Timestamp
    reason_code: NonEmptyString | None = None
    reason_detail: NonEmptyString | None = None

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: str) -> str:
        _parse_timestamp(value, "completed_at")
        return value

    @model_validator(mode="after")
    def validate_reason_shape(self) -> Self:
        if self.reason_detail is not None and self.reason_code is None:
            raise ValueError("terminal reason detail requires a reason code")
        return self


class StreamWorkItem(StreamWorkItemPlan):
    """Authoritative mutable state layered over an immutable stream plan."""

    state: StreamWorkItemState
    cancel_requested: bool = False
    lease_epoch: NonNegativeInt = 0
    fencing_token: NonEmptyString | None = None
    leased_by: NonEmptyString | None = None
    lease_expires_at: Rfc3339Timestamp | None = None
    attempt: NonNegativeInt = 0
    retry_not_before_at: Rfc3339Timestamp | None = None
    terminal_evidence: StreamTerminalEvidence | None = None
    updated_at: Rfc3339Timestamp
    row_version: NonNegativeInt = 0

    @property
    def terminal_evidence_ref(self) -> ArtifactEvidenceRef | None:
        """Compatibility view used by closure writers."""

        return None if self.terminal_evidence is None else self.terminal_evidence.evidence_ref

    @property
    def terminal_outcome(self) -> TerminalOutcome | None:
        """Return the closure outcome without exposing V1 result fields."""

        return None if self.terminal_evidence is None else self.terminal_evidence.outcome

    @property
    def completed_at(self) -> Rfc3339Timestamp | None:
        """Compatibility view for closure ledgers that index completion time."""

        return None if self.terminal_evidence is None else self.terminal_evidence.completed_at

    @field_validator("lease_expires_at", "retry_not_before_at", "updated_at")
    @classmethod
    def validate_mutable_timestamp(cls, value: str | None, info: object) -> str | None:
        if value is not None:
            _parse_timestamp(value, getattr(info, "field_name", "timestamp"))
        return value

    @model_validator(mode="after")
    def validate_state_shape(self) -> Self:
        active_lease = self.state in {
            StreamWorkItemState.LEASED,
            StreamWorkItemState.RUNNING,
        }
        lease_values = (self.fencing_token, self.leased_by, self.lease_expires_at)
        if active_lease and any(value is None for value in lease_values):
            raise ValueError("leased or running stream work requires complete lease metadata")
        if not active_lease and any(value is not None for value in lease_values):
            raise ValueError("only leased or running stream work may retain lease metadata")
        if active_lease and (self.lease_epoch == 0 or self.attempt == 0):
            raise ValueError("leased or running stream work requires positive epoch and attempt")

        if self.state is StreamWorkItemState.RETRY_WAIT:
            if self.retry_not_before_at is None:
                raise ValueError("retry-wait stream work requires retry_not_before_at")
        elif self.retry_not_before_at is not None:
            raise ValueError("only retry-wait stream work may have retry_not_before_at")

        terminal = self.state in TERMINAL_STREAM_WORK_STATES
        if terminal != (self.terminal_evidence is not None):
            raise ValueError("every terminal stream work item requires terminal evidence")
        if not terminal and self.terminal_evidence is not None:
            raise ValueError("nonterminal stream work cannot have terminal evidence")
        evidence = self.terminal_evidence
        if terminal:
            assert evidence is not None  # narrowed for type checkers
            expected = _TERMINAL_EVIDENCE_OUTCOME_BY_STATE[self.state]
            if evidence.outcome is not expected:
                raise ValueError("terminal evidence outcome does not match stream work state")

        if _parse_timestamp(self.updated_at, "updated_at") < _parse_timestamp(
            self.created_at, "created_at"
        ):
            raise ValueError("updated_at cannot precede created_at")
        if terminal:
            assert evidence is not None
            if _parse_timestamp(evidence.completed_at, "completed_at") < _parse_timestamp(
                self.created_at, "created_at"
            ):
                raise ValueError("terminal evidence cannot complete before created_at")
            if _parse_timestamp(evidence.completed_at, "completed_at") > _parse_timestamp(
                self.updated_at, "updated_at"
            ):
                raise ValueError("terminal evidence cannot complete after updated_at")
        return self


class StreamWorkLease(StrictModel):
    """Opaque capability required for every stream-work mutation."""

    work_item_id: OpaqueUuid
    worker_id: NonEmptyString
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    lease_expires_at: Rfc3339Timestamp

    @field_validator("lease_expires_at")
    @classmethod
    def validate_expiry(cls, value: str) -> str:
        _parse_timestamp(value, "lease_expires_at")
        return value


class StreamWorkLeaseClaim(StrictModel):
    """Atomic claim result containing a stream row and its lease capability."""

    work_item: StreamWorkItem
    lease: StreamWorkLease

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        item = self.work_item
        lease = self.lease
        if item.state is not StreamWorkItemState.LEASED:
            raise ValueError("a stream claim must contain a leased work item")
        if (
            item.work_item_id != lease.work_item_id
            or item.leased_by != lease.worker_id
            or item.lease_epoch != lease.lease_epoch
            or item.fencing_token != lease.fencing_token
            or item.lease_expires_at != lease.lease_expires_at
        ):
            raise ValueError("stream claim lease does not match its work-item snapshot")
        return self


class StreamWorkAttempt(StrictModel):
    """Append-oriented audit record for one fenced stream dispatch."""

    work_item_id: OpaqueUuid
    attempt_number: PositiveInt
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    worker_id: NonEmptyString
    claimed_at: Rfc3339Timestamp
    started_at: Rfc3339Timestamp | None = None
    completed_at: Rfc3339Timestamp | None = None
    outcome: StreamWorkAttemptOutcome = StreamWorkAttemptOutcome.ACTIVE
    evidence_ref: ArtifactEvidenceRef | None = None
    error_code: NonEmptyString | None = None
    error_detail: NonEmptyString | None = None

    @field_validator("claimed_at", "started_at", "completed_at")
    @classmethod
    def validate_attempt_timestamp(cls, value: str | None, info: object) -> str | None:
        if value is not None:
            _parse_timestamp(value, getattr(info, "field_name", "timestamp"))
        return value

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> Self:
        active = self.outcome is StreamWorkAttemptOutcome.ACTIVE
        if active and (self.completed_at is not None or self.evidence_ref is not None):
            raise ValueError("active stream attempts must be open and evidence-free")
        if not active and (self.completed_at is None or self.evidence_ref is None):
            raise ValueError("completed stream attempts require terminal evidence")
        if self.error_detail is not None and self.error_code is None:
            raise ValueError("attempt error detail requires an error code")
        if (
            self.outcome
            in {
                StreamWorkAttemptOutcome.FAILED,
                StreamWorkAttemptOutcome.FAILED_RETRYABLE,
                StreamWorkAttemptOutcome.ABANDONED,
                StreamWorkAttemptOutcome.CANCELLED,
                StreamWorkAttemptOutcome.EXPIRED,
                StreamWorkAttemptOutcome.QUARANTINED,
                StreamWorkAttemptOutcome.LATE_INPUT,
                StreamWorkAttemptOutcome.INCOMPLETE,
                StreamWorkAttemptOutcome.INVALIDATED,
            }
            and self.error_code is None
        ):
            raise ValueError("failed stream attempts require an error code")
        return self


class SupportedWorkContractPin(StrictModel):
    """Exact contract capability required to claim one stream work family."""

    plan_schema_ref: SchemaRef
    message_schema_ref: SchemaRef
    work_projection_version: SchemaVersion
    work_key_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_distinct_contracts(self) -> Self:
        if self.plan_schema_ref == self.message_schema_ref:
            raise ValueError("plan and message schema pins must be distinct")
        return self

    @property
    def identity(self) -> tuple[object, ...]:
        """Canonical exact pin tuple used for sorting and capability matching."""

        return (
            self.plan_schema_ref.schema_id,
            self.plan_schema_ref.version,
            self.plan_schema_ref.artifact_id,
            self.plan_schema_ref.sha256,
            self.message_schema_ref.schema_id,
            self.message_schema_ref.version,
            self.message_schema_ref.artifact_id,
            self.message_schema_ref.sha256,
            self.work_projection_version,
            self.work_key_policy_version,
        )

    @property
    def plan_schema(self) -> SchemaRef:
        """Short alias for callers that treat the pin as a contract pair."""

        return self.plan_schema_ref

    @property
    def message_schema(self) -> SchemaRef:
        return self.message_schema_ref

    def matches(
        self,
        *,
        plan_schema_ref: SchemaRef,
        message_schema_ref: SchemaRef,
        work_projection_version: str,
        work_key_policy_version: str,
    ) -> bool:
        return (
            self.plan_schema_ref == plan_schema_ref
            and self.message_schema_ref == message_schema_ref
            and self.work_projection_version == work_projection_version
            and self.work_key_policy_version == work_key_policy_version
        )


class WorkerCapabilityClaim(StrictModel):
    """Worker registration with a canonical set of exact stream pins."""

    schema_version: Literal["1.0"] = "1.0"
    worker_id: NonEmptyString
    supported_work_contracts: tuple[SupportedWorkContractPin, ...]
    capability_projection_version: SchemaVersion = "stream-worker-capability-semantic-v1"

    @property
    def supported_contracts(self) -> tuple[SupportedWorkContractPin, ...]:
        """Alias retained for scheduler adapters using shorter terminology."""

        return self.supported_work_contracts

    @model_validator(mode="after")
    def validate_contract_set(self) -> Self:
        if not self.supported_work_contracts:
            raise ValueError("worker capability claim must advertise at least one contract pin")
        identities = tuple(pin.identity for pin in self.supported_work_contracts)
        if len(set(identities)) != len(identities):
            raise ValueError("worker capability contract pins must be unique")
        logical_identities = tuple(
            (
                pin.plan_schema_ref.key,
                pin.message_schema_ref.key,
                pin.work_projection_version,
                pin.work_key_policy_version,
            )
            for pin in self.supported_work_contracts
        )
        if len(set(logical_identities)) != len(logical_identities):
            raise ValueError("worker capability contract versions must not have conflicting pins")
        if identities != tuple(sorted(identities)):
            raise ValueError("worker capability contract pins must be canonically ordered")
        return self

    def supports(
        self,
        *,
        plan_schema_ref: SchemaRef,
        message_schema_ref: SchemaRef,
        work_projection_version: str,
        work_key_policy_version: str,
    ) -> bool:
        return any(
            pin.matches(
                plan_schema_ref=plan_schema_ref,
                message_schema_ref=message_schema_ref,
                work_projection_version=work_projection_version,
                work_key_policy_version=work_key_policy_version,
            )
            for pin in self.supported_work_contracts
        )


def validate_worker_capability_claim(
    claim: WorkerCapabilityClaim,
    registry: SchemaRegistry | None = None,
) -> WorkerCapabilityClaim:
    """Revalidate a claim and, when supplied, resolve both exact schema pins.

    The registry is optional because the stream schemas are published in a
    later atomic bundle.  Omitting it still performs strict model and
    canonical-set validation; a scheduler must provide its current registry
    before leasing work.
    """

    checked = WorkerCapabilityClaim.model_validate(claim.model_dump(mode="python"), strict=True)
    if registry is not None:
        for pin in checked.supported_work_contracts:
            registry.resolve_exact(pin.plan_schema_ref)
            registry.resolve_exact(pin.message_schema_ref)
    return checked


__all__ = [
    "TERMINAL_STREAM_WORK_STATES",
    "PreEosCaptureSubjectRef",
    "StreamStage",
    "StreamSubjectRef",
    "StreamTerminalEvidence",
    "StreamWorkAttempt",
    "StreamWorkAttemptOutcome",
    "StreamWorkDependency",
    "StreamWorkItem",
    "StreamWorkItemPlan",
    "StreamWorkItemState",
    "StreamWorkLease",
    "StreamWorkLeaseClaim",
    "SupportedWorkContractPin",
    "WorkerCapabilityClaim",
    "validate_worker_capability_claim",
]
