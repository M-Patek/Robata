"""Strict in-process processing-run context and ordered node memberships.

Nothing in this module is a registered wire schema.  It composes already constructed
``LogicalNode`` values and never derives node identity from processing-run facts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import TypeAdapter, ValidationError, field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import (
    LogicalNode,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    Rfc3339Timestamp,
    RunNodeDisposition,
    RunNodeRole,
)
from robata.ports.logical_node_registry import (
    ExistingNodeDisposition,
    LogicalNodeRegistry,
    PublishedRunNodeMembership,
)

_UUID_ADAPTER = TypeAdapter(OpaqueUuid)
_ROLE_ADAPTER = TypeAdapter(RunNodeRole)
_TIMESTAMP_ADAPTER = TypeAdapter(Rfc3339Timestamp)


class CanonicalProcessingRunMode(StrEnum):
    FRESH = "FRESH"
    RESUME = "RESUME"


class CanonicalProcessingRunPrimaryStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    NO_EVENTS = "NO_EVENTS"
    ABSTAINED = "ABSTAINED"
    INCOMPLETE = "INCOMPLETE"
    MATERIALIZATION_FAILED = "MATERIALIZATION_FAILED"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    IDENTITY_FAILED = "IDENTITY_FAILED"
    RUN_MEMBERSHIP_FAILED = "RUN_MEMBERSHIP_FAILED"
    CONFIGURATION_FAILED = "CONFIGURATION_FAILED"


class CanonicalProcessingRunShadowStatus(StrEnum):
    NOT_SCHEDULED = "NOT_SCHEDULED"


class CanonicalProcessingRunDeadlineStatus(StrEnum):
    UNRESOLVED = "UNRESOLVED"


class CanonicalRunMembershipErrorCode(StrEnum):
    INVALID_CONTEXT = "INVALID_CONTEXT"
    INVALID_REQUEST = "INVALID_REQUEST"
    RUN_ALREADY_COMPLETED = "RUN_ALREADY_COMPLETED"
    RUN_COMPLETION_CONFLICT = "RUN_COMPLETION_CONFLICT"
    MEMBERSHIP_CONFLICT = "MEMBERSHIP_CONFLICT"
    REGISTRY_RESULT_INVALID = "REGISTRY_RESULT_INVALID"


class CanonicalRunMembershipError(RuntimeError):
    def __init__(self, code: CanonicalRunMembershipErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _timestamp(value: str, field: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an RFC3339 timezone")
    return parsed


class CanonicalProcessingRunContext(StrictModel):
    """An active run whose fresh UUID is supplied by the caller or allocator."""

    run_id: OpaqueUuid
    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    pipeline_version: SchemaVersion
    config_sha256: Sha256Digest
    started_at: Rfc3339Timestamp
    deadline_status: Literal[CanonicalProcessingRunDeadlineStatus.UNRESOLVED] = (
        CanonicalProcessingRunDeadlineStatus.UNRESOLVED
    )
    deadline_at: None = None
    primary_status: Literal[CanonicalProcessingRunPrimaryStatus.RUNNING] = (
        CanonicalProcessingRunPrimaryStatus.RUNNING
    )
    shadow_status: Literal[CanonicalProcessingRunShadowStatus.NOT_SCHEDULED] = (
        CanonicalProcessingRunShadowStatus.NOT_SCHEDULED
    )
    completed_at: None = None
    mode: CanonicalProcessingRunMode = CanonicalProcessingRunMode.FRESH

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: str) -> str:
        _timestamp(value, "started_at")
        return value

    @classmethod
    def fresh(
        cls,
        *,
        run_id: OpaqueUuid,
        recording_identity: Sha256Digest,
        mcap_id: OpaqueUuid,
        pipeline_version: SchemaVersion,
        config_sha256: Sha256Digest,
        started_at: Rfc3339Timestamp,
    ) -> Self:
        return cls(
            run_id=run_id,
            recording_identity=recording_identity,
            mcap_id=mcap_id,
            pipeline_version=pipeline_version,
            config_sha256=config_sha256,
            started_at=started_at,
            mode=CanonicalProcessingRunMode.FRESH,
        )

    @classmethod
    def resume(cls, record: CanonicalProcessingRunRecord) -> Self:
        """Explicitly reuse the identity and immutable facts of an unfinished run."""

        checked = _strict_record(record)
        if checked.primary_status is not CanonicalProcessingRunPrimaryStatus.RUNNING:
            raise ValueError("only a RUNNING processing run can be resumed")
        return cls(
            run_id=checked.run_id,
            recording_identity=checked.recording_identity,
            mcap_id=checked.mcap_id,
            pipeline_version=checked.pipeline_version,
            config_sha256=checked.config_sha256,
            started_at=checked.started_at,
            deadline_status=checked.deadline_status,
            deadline_at=None,
            primary_status=CanonicalProcessingRunPrimaryStatus.RUNNING,
            shadow_status=checked.shadow_status,
            completed_at=None,
            mode=CanonicalProcessingRunMode.RESUME,
        )

    def to_record(self) -> CanonicalProcessingRunRecord:
        return CanonicalProcessingRunRecord.from_context(self)


class CanonicalProcessingRunRecord(StrictModel):
    """Inspectable lifecycle record; deliberately not a registered persistence contract."""

    run_id: OpaqueUuid
    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    pipeline_version: SchemaVersion
    config_sha256: Sha256Digest
    started_at: Rfc3339Timestamp
    completed_at: Rfc3339Timestamp | None
    primary_status: CanonicalProcessingRunPrimaryStatus
    shadow_status: Literal[CanonicalProcessingRunShadowStatus.NOT_SCHEDULED] = (
        CanonicalProcessingRunShadowStatus.NOT_SCHEDULED
    )
    deadline_status: Literal[CanonicalProcessingRunDeadlineStatus.UNRESOLVED] = (
        CanonicalProcessingRunDeadlineStatus.UNRESOLVED
    )
    deadline_at: None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            _timestamp(value, "run timestamp")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.primary_status is CanonicalProcessingRunPrimaryStatus.RUNNING:
            if self.completed_at is not None:
                raise ValueError("RUNNING processing runs cannot have completed_at")
        elif self.completed_at is None:
            raise ValueError("terminal processing runs require completed_at")
        if self.completed_at is not None and _timestamp(
            self.completed_at, "completed_at"
        ) < _timestamp(self.started_at, "started_at"):
            raise ValueError("completed_at cannot precede started_at")
        return self

    @classmethod
    def from_context(cls, context: CanonicalProcessingRunContext) -> Self:
        checked = _strict_context(context)
        return cls(
            run_id=checked.run_id,
            recording_identity=checked.recording_identity,
            mcap_id=checked.mcap_id,
            pipeline_version=checked.pipeline_version,
            config_sha256=checked.config_sha256,
            started_at=checked.started_at,
            completed_at=None,
            primary_status=CanonicalProcessingRunPrimaryStatus.RUNNING,
            shadow_status=checked.shadow_status,
            deadline_status=checked.deadline_status,
            deadline_at=None,
        )

    def complete(
        self,
        primary_status: CanonicalProcessingRunPrimaryStatus,
        completed_at: Rfc3339Timestamp,
    ) -> Self:
        if (
            not isinstance(primary_status, CanonicalProcessingRunPrimaryStatus)
            or primary_status is CanonicalProcessingRunPrimaryStatus.RUNNING
        ):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                "primary_status must be terminal",
            )
        if self.primary_status is not CanonicalProcessingRunPrimaryStatus.RUNNING:
            if self.primary_status is primary_status and self.completed_at == completed_at:
                return self
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.RUN_COMPLETION_CONFLICT,
                "run already has different terminal facts",
            )
        try:
            return type(self).model_validate(
                {
                    **self.model_dump(mode="python"),
                    "primary_status": primary_status,
                    "completed_at": completed_at,
                },
                strict=True,
            )
        except ValidationError as error:
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                f"invalid run completion: {error}",
            ) from error


def _strict_context(value: object) -> CanonicalProcessingRunContext:
    if not isinstance(value, CanonicalProcessingRunContext):
        raise TypeError("context must be a CanonicalProcessingRunContext")
    return CanonicalProcessingRunContext.model_validate(
        value.model_dump(mode="python"), strict=True
    )


def _strict_record(value: object) -> CanonicalProcessingRunRecord:
    if not isinstance(value, CanonicalProcessingRunRecord):
        raise TypeError("record must be a CanonicalProcessingRunRecord")
    return CanonicalProcessingRunRecord.model_validate(value.model_dump(mode="python"), strict=True)


def canonical_first_work_item_id(
    *,
    run_id: OpaqueUuid,
    node: LogicalNode,
    role: RunNodeRole,
) -> OpaqueUuid:
    """Derive an execution-local UUID5 from run, complete node identity, and role."""

    if not isinstance(node, LogicalNode):
        raise TypeError("node must be a LogicalNode")
    checked_node = LogicalNode.model_validate(node.model_dump(mode="python"), strict=True)
    try:
        checked_run_id = _UUID_ADAPTER.validate_python(run_id, strict=True)
        checked_role = _ROLE_ADAPTER.validate_python(role, strict=True)
    except ValidationError as error:
        raise ValueError(f"invalid first-work-item input: {error}") from error
    name = ":".join(
        (
            "robata",
            "canonical-run-membership",
            "first-work-item",
            "v1",
            checked_run_id,
            checked_node.node_type,
            checked_node.node_logical_key,
            checked_role,
        )
    )
    return str(uuid5(NAMESPACE_URL, name))


class _AttachCommand(StrictModel):
    node: LogicalNode
    role: RunNodeRole
    attached_at: Rfc3339Timestamp
    first_work_item_id: OpaqueUuid
    existing_node_disposition: ExistingNodeDisposition

    @field_validator("attached_at")
    @classmethod
    def validate_attached_at(cls, value: str) -> str:
        _timestamp(value, "attached_at")
        return value


class CanonicalRunMembershipJournal:
    """Attach preconstructed nodes and retain first-success call order for one run."""

    def __init__(
        self,
        *,
        context: CanonicalProcessingRunContext,
        registry: LogicalNodeRegistry,
    ) -> None:
        try:
            checked_context = _strict_context(context)
        except (TypeError, ValidationError) as error:
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_CONTEXT,
                f"invalid processing-run context: {error}",
            ) from error
        if not callable(getattr(registry, "attach_run_node", None)):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                "registry must implement attach_run_node",
            )
        self._context = checked_context
        self._registry = registry
        self._record = checked_context.to_record()
        self._commands: dict[tuple[str, str, str], _AttachCommand] = {}
        self._memberships: list[ProcessingRunNodeMembership] = []
        self._lock = RLock()

    @property
    def context(self) -> CanonicalProcessingRunContext:
        return self._context

    @property
    def record(self) -> CanonicalProcessingRunRecord:
        with self._lock:
            return self._record

    @property
    def memberships(self) -> tuple[ProcessingRunNodeMembership, ...]:
        """Memberships in first successful ``attach`` call order."""

        with self._lock:
            return tuple(self._memberships)

    def attach(
        self,
        node: LogicalNode,
        role: RunNodeRole,
        attached_at: Rfc3339Timestamp,
        *,
        existing_node_disposition: ExistingNodeDisposition = RunNodeDisposition.REUSED,
    ) -> ProcessingRunNodeMembership:
        """Attach one node; exact retries reverify storage without duplicating order."""

        with self._lock:
            if self._record.primary_status is not CanonicalProcessingRunPrimaryStatus.RUNNING:
                raise CanonicalRunMembershipError(
                    CanonicalRunMembershipErrorCode.RUN_ALREADY_COMPLETED,
                    "cannot attach after run completion",
                )
            command = self._command(node, role, attached_at, existing_node_disposition)
            key = (command.node.node_type, command.node.node_logical_key, command.role)
            prior = self._commands.get(key)
            if prior is not None and prior != command:
                raise CanonicalRunMembershipError(
                    CanonicalRunMembershipErrorCode.MEMBERSHIP_CONFLICT,
                    "run-node-role identity was retried with different facts",
                )
            published = self._registry.attach_run_node(
                node=command.node,
                run_id=self._context.run_id,
                role=command.role,
                first_work_item_id=command.first_work_item_id,
                attached_at=command.attached_at,
                existing_node_disposition=command.existing_node_disposition,
            )
            membership = self._publication(command, published)
            if prior is not None:
                stored = next(
                    item for item in self._memberships if item.identity == membership.identity
                )
                if stored != membership:
                    raise CanonicalRunMembershipError(
                        CanonicalRunMembershipErrorCode.REGISTRY_RESULT_INVALID,
                        "registry changed an immutable membership on retry",
                    )
                return stored
            self._commands[key] = command
            self._memberships.append(membership)
            return membership

    def complete(
        self,
        primary_status: CanonicalProcessingRunPrimaryStatus,
    ) -> CanonicalProcessingRunRecord:
        """Complete at the latest durable evidence time observed by this journal."""

        with self._lock:
            evidence_times = [
                self._record.started_at,
                *(item.attached_at for item in self._memberships),
            ]
            completed_at = max(evidence_times, key=lambda value: _timestamp(value, "run evidence"))
            completed = self._record.complete(primary_status, completed_at)
            self._record = completed
            return completed

    def _command(
        self,
        node: LogicalNode,
        role: RunNodeRole,
        attached_at: Rfc3339Timestamp,
        disposition: ExistingNodeDisposition,
    ) -> _AttachCommand:
        if not isinstance(node, LogicalNode):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                "node must be a preconstructed LogicalNode",
            )
        if not isinstance(disposition, RunNodeDisposition):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                "existing disposition must be REUSED, INVALIDATED, or OBSERVED",
            )
        try:
            checked_node = LogicalNode.model_validate(node.model_dump(mode="python"), strict=True)
            checked_role = _ROLE_ADAPTER.validate_python(role, strict=True)
            checked_at = _TIMESTAMP_ADAPTER.validate_python(attached_at, strict=True)
            command = _AttachCommand(
                node=checked_node,
                role=checked_role,
                attached_at=checked_at,
                first_work_item_id=canonical_first_work_item_id(
                    run_id=self._context.run_id,
                    node=checked_node,
                    role=checked_role,
                ),
                existing_node_disposition=disposition,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                f"invalid run-node attachment: {error}",
            ) from error
        if _timestamp(command.attached_at, "attached_at") < _timestamp(
            self._context.started_at, "started_at"
        ):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.INVALID_REQUEST,
                "attached_at cannot precede started_at",
            )
        return command

    def _publication(
        self,
        command: _AttachCommand,
        published: PublishedRunNodeMembership,
    ) -> ProcessingRunNodeMembership:
        if not isinstance(published, PublishedRunNodeMembership):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.REGISTRY_RESULT_INVALID,
                "registry returned an invalid publication result",
            )
        try:
            checked_node = LogicalNode.model_validate(
                published.node.model_dump(mode="python"), strict=True
            )
            membership = ProcessingRunNodeMembership.model_validate(
                published.membership.model_dump(mode="python"), strict=True
            )
        except (AttributeError, ValidationError) as error:
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.REGISTRY_RESULT_INVALID,
                f"registry publication failed strict validation: {error}",
            ) from error
        allowed = (
            {RunNodeDisposition.CREATED, RunNodeDisposition.REUSED}
            if command.existing_node_disposition is RunNodeDisposition.REUSED
            else {command.existing_node_disposition}
        )
        expected_identity = (
            self._context.run_id,
            command.node.node_type,
            command.node.node_logical_key,
            command.role,
        )
        if (
            checked_node != command.node
            or membership.identity != expected_identity
            or membership.first_work_item_id != command.first_work_item_id
            or membership.attached_at != command.attached_at
            or membership.disposition not in allowed
        ):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.REGISTRY_RESULT_INVALID,
                "registry publication does not match the attach command",
            )
        if (
            published.node_inserted is True
            and membership.disposition is not RunNodeDisposition.CREATED
        ):
            raise CanonicalRunMembershipError(
                CanonicalRunMembershipErrorCode.REGISTRY_RESULT_INVALID,
                "new node must have a CREATED membership",
            )
        return membership


__all__ = [
    "CanonicalProcessingRunContext",
    "CanonicalProcessingRunDeadlineStatus",
    "CanonicalProcessingRunMode",
    "CanonicalProcessingRunPrimaryStatus",
    "CanonicalProcessingRunRecord",
    "CanonicalProcessingRunShadowStatus",
    "CanonicalRunMembershipError",
    "CanonicalRunMembershipErrorCode",
    "CanonicalRunMembershipJournal",
    "canonical_first_work_item_id",
]
