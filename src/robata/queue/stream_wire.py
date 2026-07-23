"""Lease-bound Wire projections for the provider-neutral stream scheduler."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.contracts.stream_common import (
    NonEmptyString,
    PreEosCaptureSubjectRef,
    StreamStage,
    StreamSubjectRef,
    validate_rfc3339,
)
from robata.contracts.stream_planning import (
    STREAM_WORK_PLAN_SCHEMA_ID,
    STREAM_WORK_PLAN_SCHEMA_VERSION,
    StreamWorkDependency,
    StreamWorkItemPlan,
)
from robata.queue.stage import DependencyCriticality
from robata.queue.stream_models import (
    StreamWorkItem,
    StreamWorkItemState,
    StreamWorkLeaseClaim,
    SupportedWorkContractPin,
)

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

STREAM_WORK_MESSAGE_SCHEMA_ID = "https://schemas.robata.dev/stream-work-message"
STREAM_WORK_MESSAGE_SCHEMA_VERSION = "1.0.0"
STREAM_WORK_MESSAGE_WIRE_VERSION: Literal["1.0"] = "1.0"
STREAM_WORK_MESSAGE_PROJECTION_VERSION = "stream-work-message-semantic-v1"


def _parse_timestamp(value: str, field_name: str) -> datetime:
    validate_rfc3339(value, field_name)
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized).astimezone(UTC)


class StreamWorkMessageDependency(StrictModel):
    """Canonical dependency projection carried by a stream work message."""

    upstream_work_logical_key: NonEmptyString
    criticality: DependencyCriticality


class StreamWorkMessage(StrictModel):
    """Immutable broker projection issued only from an active stream lease."""

    schema_version: Literal["1.0"]
    schema_ref: SchemaRef
    plan_schema_ref: SchemaRef
    work_item_id: OpaqueUuid
    work_logical_key: NonEmptyString
    stream_run_id: OpaqueUuid
    source_subject: PreEosCaptureSubjectRef
    stage: StreamStage
    subject: StreamSubjectRef
    ordered_dependencies: tuple[StreamWorkMessageDependency, ...]
    input_semantic_sha256: Sha256Digest
    config_semantic_sha256: Sha256Digest
    work_projection_version: SchemaVersion
    work_key_policy_version: SchemaVersion
    priority: NonNegativeInt
    sla_deadline_at: Rfc3339Timestamp | None
    execution_expiry_at: Rfc3339Timestamp | None
    max_attempts: PositiveInt
    trace_id: NonEmptyString | None
    created_at: Rfc3339Timestamp
    cancel_requested: bool
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    leased_by: NonEmptyString
    lease_expires_at: Rfc3339Timestamp
    attempt: PositiveInt

    @property
    def dependencies(self) -> tuple[StreamWorkMessageDependency, ...]:
        """Short compatibility view; the wire name stays explicit and ordered."""

        return self.ordered_dependencies

    @property
    def work_plan_schema_ref(self) -> SchemaRef:
        return self.plan_schema_ref

    @model_validator(mode="after")
    def validate_wire_shape(self) -> Self:
        if self.schema_ref.schema_id != STREAM_WORK_MESSAGE_SCHEMA_ID:
            raise ValueError(
                "schema_ref must identify "
                f"{STREAM_WORK_MESSAGE_SCHEMA_ID}@{STREAM_WORK_MESSAGE_SCHEMA_VERSION}"
            )
        if self.schema_ref.version != STREAM_WORK_MESSAGE_SCHEMA_VERSION:
            raise ValueError(
                "schema_ref must identify "
                f"{STREAM_WORK_MESSAGE_SCHEMA_ID}@{STREAM_WORK_MESSAGE_SCHEMA_VERSION}"
            )
        if self.plan_schema_ref.schema_id != STREAM_WORK_PLAN_SCHEMA_ID:
            raise ValueError(
                "plan_schema_ref must identify "
                f"{STREAM_WORK_PLAN_SCHEMA_ID}@{STREAM_WORK_PLAN_SCHEMA_VERSION}"
            )
        if self.plan_schema_ref.version != STREAM_WORK_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "plan_schema_ref must identify "
                f"{STREAM_WORK_PLAN_SCHEMA_ID}@{STREAM_WORK_PLAN_SCHEMA_VERSION}"
            )
        if tuple(self.ordered_dependencies) != tuple(
            sorted(self.ordered_dependencies, key=lambda item: item.upstream_work_logical_key)
        ):
            raise ValueError("stream work dependencies must be canonically ordered")
        keys = tuple(item.upstream_work_logical_key for item in self.ordered_dependencies)
        if len(set(keys)) != len(keys):
            raise ValueError("stream work dependency logical keys must be unique")
        for field_name, value in (
            ("created_at", self.created_at),
            ("lease_expires_at", self.lease_expires_at),
            ("sla_deadline_at", self.sla_deadline_at),
            ("execution_expiry_at", self.execution_expiry_at),
        ):
            if value is not None:
                _parse_timestamp(value, field_name)
        if self.attempt > self.max_attempts:
            raise ValueError("stream work attempt cannot exceed max_attempts")
        if self.subject.capture_scope_digest != self.source_subject.capture_scope_digest:
            raise ValueError("stream work subject must bind to its source capture scope")
        self.to_plan()
        return self

    def to_plan(self) -> StreamWorkItemPlan:
        """Reconstruct and validate the exact immutable plan projection."""

        return StreamWorkItemPlan(
            schema_ref=self.plan_schema_ref,
            work_item_id=self.work_item_id,
            work_logical_key=self.work_logical_key,
            stream_run_id=self.stream_run_id,
            source_subject=self.source_subject,
            stage=self.stage,
            subject=self.subject,
            ordered_dependencies=tuple(
                StreamWorkDependency(
                    upstream_work_logical_key=dependency.upstream_work_logical_key,
                    criticality=dependency.criticality,
                )
                for dependency in self.ordered_dependencies
            ),
            input_semantic_sha256=self.input_semantic_sha256,
            config_semantic_sha256=self.config_semantic_sha256,
            work_projection_version=self.work_projection_version,
            work_key_policy_version=self.work_key_policy_version,
            priority=self.priority,
            sla_deadline_at=self.sla_deadline_at,
            execution_expiry_at=self.execution_expiry_at,
            max_attempts=self.max_attempts,
            trace_id=self.trace_id,
            created_at=self.created_at,
        )

    @classmethod
    def from_ledger(
        cls,
        item: StreamWorkItem,
        *,
        plan_schema_ref: SchemaRef | None = None,
        schema_ref: SchemaRef,
        dependencies: tuple[StreamWorkDependency, ...] | None = None,
    ) -> StreamWorkMessage:
        """Build a message from a complete authoritative active lease.

        ``dependencies`` is accepted as an explicit argument for scheduler
        adapters that read edges from a separate table.  When supplied it
        must exactly equal the immutable plan projection; it cannot rewrite
        criticality at delivery time.
        """

        if item.state not in {StreamWorkItemState.LEASED, StreamWorkItemState.RUNNING}:
            raise ValueError("stream-work message requires a leased or running work item")
        if item.lease_epoch < 1 or item.attempt < 1:
            raise ValueError("stream-work message requires positive lease epoch and attempt")
        if item.fencing_token is None or item.leased_by is None or item.lease_expires_at is None:
            raise ValueError("stream-work message requires complete active lease metadata")
        effective_plan_schema_ref = item.schema_ref if plan_schema_ref is None else plan_schema_ref
        if effective_plan_schema_ref != item.schema_ref:
            raise ValueError("plan_schema_ref must match the authoritative stream work plan pin")
        expected_dependencies = item.ordered_dependencies
        if dependencies is not None and tuple(dependencies) != tuple(expected_dependencies):
            raise ValueError("stream-work dependencies cannot differ from the authoritative plan")
        projected = tuple(
            StreamWorkMessageDependency(
                upstream_work_logical_key=dependency.upstream_work_logical_key,
                criticality=dependency.criticality,
            )
            for dependency in expected_dependencies
        )
        return cls(
            schema_version=STREAM_WORK_MESSAGE_WIRE_VERSION,
            schema_ref=schema_ref,
            plan_schema_ref=effective_plan_schema_ref,
            work_item_id=item.work_item_id,
            work_logical_key=item.work_logical_key,
            stream_run_id=item.stream_run_id,
            source_subject=item.source_subject,
            stage=item.stage,
            subject=item.subject,
            ordered_dependencies=projected,
            input_semantic_sha256=item.input_semantic_sha256,
            config_semantic_sha256=item.config_semantic_sha256,
            work_projection_version=item.work_projection_version,
            work_key_policy_version=item.work_key_policy_version,
            priority=item.priority,
            sla_deadline_at=item.sla_deadline_at,
            execution_expiry_at=item.execution_expiry_at,
            max_attempts=item.max_attempts,
            trace_id=item.trace_id,
            created_at=item.created_at,
            cancel_requested=item.cancel_requested,
            lease_epoch=item.lease_epoch,
            fencing_token=item.fencing_token,
            leased_by=item.leased_by,
            lease_expires_at=item.lease_expires_at,
            attempt=item.attempt,
        )

    @classmethod
    def from_lease(
        cls,
        item: StreamWorkItem,
        *,
        plan_schema_ref: SchemaRef | None = None,
        schema_ref: SchemaRef,
    ) -> StreamWorkMessage:
        """Readable alias for adapters that call their claim object a lease."""

        return cls.from_ledger(
            item,
            plan_schema_ref=plan_schema_ref,
            schema_ref=schema_ref,
        )

    @classmethod
    def from_claim(
        cls,
        claim: StreamWorkLeaseClaim,
        *,
        schema_ref: SchemaRef,
        plan_schema_ref: SchemaRef | None = None,
    ) -> StreamWorkMessage:
        """Build a message directly from an atomic stream lease claim."""

        return cls.from_ledger(
            claim.work_item,
            plan_schema_ref=plan_schema_ref,
            schema_ref=schema_ref,
        )

    def capability_pin(self) -> SupportedWorkContractPin:
        """Return the exact capability pin required to execute this message."""

        return SupportedWorkContractPin(
            plan_schema_ref=self.plan_schema_ref,
            message_schema_ref=self.schema_ref,
            work_projection_version=self.work_projection_version,
            work_key_policy_version=self.work_key_policy_version,
        )


def validate_registered_stream_work_message(
    message: StreamWorkMessage,
    registry: SchemaRegistry,
) -> StreamWorkMessage:
    """Validate both exact plan/message pins against a supplied catalog."""

    checked = StreamWorkMessage.model_validate(message.model_dump(mode="python"), strict=True)
    registry.resolve_exact(checked.plan_schema_ref)
    registry.resolve_exact(checked.schema_ref)
    plan = checked.to_plan()
    registry.validate_pinned(checked.plan_schema_ref, plan.model_dump(mode="json"))
    registry.validate_pinned(checked.schema_ref, checked.model_dump(mode="json"))
    return checked


__all__ = [
    "STREAM_WORK_MESSAGE_PROJECTION_VERSION",
    "STREAM_WORK_MESSAGE_SCHEMA_ID",
    "STREAM_WORK_MESSAGE_SCHEMA_VERSION",
    "STREAM_WORK_MESSAGE_WIRE_VERSION",
    "STREAM_WORK_PLAN_SCHEMA_ID",
    "STREAM_WORK_PLAN_SCHEMA_VERSION",
    "StreamWorkMessage",
    "StreamWorkMessageDependency",
    "validate_registered_stream_work_message",
]
