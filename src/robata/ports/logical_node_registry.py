"""Boundary for run-independent logical nodes and immutable run memberships."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from robata.contracts.logical_nodes import (
    LogicalNode,
    NodeLogicalKey,
    NodeType,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    Rfc3339Timestamp,
    RunNodeDisposition,
    RunNodeRole,
)

type ExistingNodeDisposition = Literal[
    RunNodeDisposition.REUSED,
    RunNodeDisposition.INVALIDATED,
    RunNodeDisposition.OBSERVED,
]


class LogicalNodeRegistryErrorCode(StrEnum):
    """Stable machine-readable logical-node registry failures."""

    INVALID_REQUEST = "INVALID_REQUEST"
    NODE_NOT_FOUND = "NODE_NOT_FOUND"
    NODE_CONFLICT = "NODE_CONFLICT"
    MEMBERSHIP_CONFLICT = "MEMBERSHIP_CONFLICT"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    STORAGE_IO_ERROR = "STORAGE_IO_ERROR"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


class LogicalNodeRegistryError(RuntimeError):
    """A logical-node registry failure carrying a stable error code."""

    def __init__(self, code: LogicalNodeRegistryErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PublishedRunNodeMembership:
    """Verified attach result and attributable immutable-row insertions.

    An insertion flag is ``None`` when commit recovery verified the final rows but could
    not prove which concurrent transaction inserted them.
    """

    node: LogicalNode
    membership: ProcessingRunNodeMembership
    node_inserted: bool | None
    membership_inserted: bool | None


@dataclass(frozen=True, slots=True)
class VerifiedLogicalNode:
    """One verified node together with all canonically ordered memberships."""

    node: LogicalNode
    memberships: tuple[ProcessingRunNodeMembership, ...]


class LogicalNodeRegistry(Protocol):
    """Durable attach and verified-query boundary for logical-node membership."""

    def attach_run_node(
        self,
        *,
        node: LogicalNode,
        run_id: OpaqueUuid,
        role: RunNodeRole,
        first_work_item_id: OpaqueUuid,
        attached_at: Rfc3339Timestamp,
        existing_node_disposition: ExistingNodeDisposition = RunNodeDisposition.REUSED,
    ) -> PublishedRunNodeMembership:
        """Atomically create/reuse a node and attach one immutable run membership."""

    def lookup_node(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> LogicalNode | None:
        """Return a verified node, or ``None`` when no such identity exists."""

    def lookup_membership(
        self,
        run_id: OpaqueUuid,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
        role: RunNodeRole,
    ) -> ProcessingRunNodeMembership | None:
        """Return one verified membership by its exact four-part identity."""

    def list_run_memberships(
        self,
        run_id: OpaqueUuid,
    ) -> tuple[ProcessingRunNodeMembership, ...]:
        """List a run's memberships in canonical node/type/role order."""

    def list_node_memberships(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> tuple[ProcessingRunNodeMembership, ...]:
        """List all run memberships for a node in canonical run/role order."""

    def verify_node(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> VerifiedLogicalNode:
        """Verify normalized rows, canonical documents, parentage, and creator."""


__all__ = [
    "ExistingNodeDisposition",
    "LogicalNodeRegistry",
    "LogicalNodeRegistryError",
    "LogicalNodeRegistryErrorCode",
    "PublishedRunNodeMembership",
    "VerifiedLogicalNode",
]
