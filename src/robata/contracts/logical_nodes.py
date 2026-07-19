"""Run-independent logical-node and processing-run membership contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import StringConstraints, field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel

NodeType = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
    ),
]
RunNodeRole = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
    ),
]
KeyNamespace = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?::[a-z0-9][a-z0-9-]*)*$",
    ),
]
NodeLogicalKey = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=66,
        max_length=193,
        pattern=(
            r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
            r"(?::[a-z0-9][a-z0-9-]*)*:[0-9a-f]{64}$"
        ),
    ),
]
OpaqueUuid = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    ),
]
Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
        ),
    ),
]


class RunNodeDisposition(StrEnum):
    """How one processing run became associated with a logical node."""

    CREATED = "CREATED"
    REUSED = "REUSED"
    INVALIDATED = "INVALIDATED"
    OBSERVED = "OBSERVED"


class LogicalNode(StrictModel):
    """Immutable semantic node whose identity is independent of execution runs."""

    schema_version: Literal["1.0"]
    node_type: NodeType
    key_namespace: KeyNamespace
    node_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    identity_policy_version: SchemaVersion

    @model_validator(mode="after")
    def validate_logical_key(self) -> Self:
        expected = f"{self.key_namespace}:{self.semantic_sha256}"
        if self.node_logical_key != expected:
            raise ValueError("node_logical_key must equal key_namespace:semantic_sha256")
        return self

    @property
    def identity(self) -> tuple[str, str]:
        """Return the complete run-independent storage identity."""

        return self.node_type, self.node_logical_key


class ProcessingRunNodeMembership(StrictModel):
    """Immutable audit association between one run and one logical node role."""

    schema_version: Literal["1.0"]
    run_id: OpaqueUuid
    node_type: NodeType
    node_logical_key: NodeLogicalKey
    role: RunNodeRole
    disposition: RunNodeDisposition
    first_work_item_id: OpaqueUuid
    attached_at: Rfc3339Timestamp

    @field_validator("attached_at")
    @classmethod
    def validate_attached_at(cls, value: str) -> str:
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("attached_at must be a valid RFC3339 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("attached_at must include an RFC3339 timezone")
        return value

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Return the exact Architecture V1.1 membership unique key."""

        return self.run_id, self.node_type, self.node_logical_key, self.role


def logical_node_from_semantic_digest(
    *,
    node_type: NodeType,
    key_namespace: KeyNamespace,
    semantic_sha256: Sha256Digest,
    identity_policy_version: SchemaVersion,
) -> LogicalNode:
    """Bind an already typed producer projection digest to a logical-node key."""

    return LogicalNode(
        schema_version="1.0",
        node_type=node_type,
        key_namespace=key_namespace,
        node_logical_key=f"{key_namespace}:{semantic_sha256}",
        semantic_sha256=semantic_sha256,
        identity_policy_version=identity_policy_version,
    )


__all__ = [
    "KeyNamespace",
    "LogicalNode",
    "NodeLogicalKey",
    "NodeType",
    "OpaqueUuid",
    "ProcessingRunNodeMembership",
    "Rfc3339Timestamp",
    "RunNodeDisposition",
    "RunNodeRole",
    "logical_node_from_semantic_digest",
]
