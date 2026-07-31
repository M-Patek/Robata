"""Focused no-provider tests for PostgreSQL logical and review adapters."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from uuid import UUID

import pytest

from robata.adapters.postgres_authority import PostgresCanonicalAuthority
from robata.adapters.postgres_logical_review import (
    PostgresLogicalNodeRegistry,
    PostgresReviewQueue,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import RunNodeDisposition, logical_node_from_semantic_digest
from robata.ports.logical_node_registry import (
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
)
from robata.review.models import (
    ReviewRequest,
    ReviewRoutingRule,
    ReviewSubject,
    ReviewTrigger,
    create_nonblocking_review_routing_policy,
    create_review_task,
)


class _Cursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> sqlite3.Row | None:
        return self._cursor.fetchone()

    def fetchall(self) -> Sequence[sqlite3.Row]:
        return self._cursor.fetchall()


class _EmptyCursor:
    rowcount = 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> tuple[object, ...]:
        return ()


class _PostgresSqliteHarness:
    """Small DB-API double for adapter transaction and canonical-byte behavior."""

    _tables = frozenset(
        {
            "logical_nodes",
            "processing_run_nodes",
            "immutable_node_revisions",
            "selection_decisions",
            "current_selections",
            "review_tasks",
            "review_annotations",
            "review_reopen_commands",
        }
    )

    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:", isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self.commits = 0
        self._create_schema()

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _Cursor | _EmptyCursor:
        normalized = query.strip()
        if normalized.startswith("BEGIN"):
            self._connection.execute("BEGIN")
            return _EmptyCursor()
        if normalized.startswith("SET LOCAL") or normalized.startswith("SELECT set_config"):
            return _EmptyCursor()
        if "information_schema.tables" in query:
            requested = () if params is None else tuple(params[1])
            rows = [
                {"table_name": name}
                for name in sorted(self._tables.intersection(str(item) for item in requested))
            ]
            return _RowsCursor(rows)
        portable = re.sub(r"\s+FOR UPDATE(?:\s+SKIP LOCKED)?", "", query)
        portable = portable.replace("%s", "?")
        cursor = self._connection.execute(portable, () if params is None else params)
        return _Cursor(cursor)

    def commit(self) -> None:
        self.commits += 1
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        # A production authority opens a fresh connection each operation.  This
        # semantic double retains a single in-memory database across operations.
        return None

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE logical_nodes (
                tenant_id TEXT NOT NULL DEFAULT 'unit-tenant',
                node_type TEXT NOT NULL,
                node_logical_key TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                key_namespace TEXT NOT NULL,
                semantic_sha256 TEXT NOT NULL,
                identity_policy_version TEXT NOT NULL,
                node_json BLOB NOT NULL,
                node_json_sha256 TEXT NOT NULL,
                PRIMARY KEY (tenant_id, node_type, node_logical_key)
            );
            CREATE TABLE processing_run_nodes (
                tenant_id TEXT NOT NULL DEFAULT 'unit-tenant',
                run_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                node_logical_key TEXT NOT NULL,
                role TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                disposition TEXT NOT NULL,
                first_work_item_id TEXT NOT NULL,
                attached_at TEXT NOT NULL,
                membership_json BLOB NOT NULL,
                membership_json_sha256 TEXT NOT NULL,
                PRIMARY KEY (tenant_id, run_id, node_type, node_logical_key, role)
            );
            CREATE UNIQUE INDEX processing_run_nodes_creator_idx
                ON processing_run_nodes (tenant_id, node_type, node_logical_key)
                WHERE disposition = 'CREATED';
            CREATE TABLE immutable_node_revisions (tenant_id TEXT DEFAULT 'unit-tenant');
            CREATE TABLE selection_decisions (tenant_id TEXT DEFAULT 'unit-tenant');
            CREATE TABLE current_selections (tenant_id TEXT DEFAULT 'unit-tenant');
            CREATE TABLE review_tasks (
                tenant_id TEXT NOT NULL DEFAULT 'unit-tenant',
                review_task_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                task_semantic_sha256 TEXT NOT NULL,
                priority INTEGER NOT NULL,
                requested_at_ns INTEGER NOT NULL,
                due_at_ns INTEGER NOT NULL,
                task_json BLOB NOT NULL,
                task_exact_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_fence INTEGER NOT NULL,
                attempt_count INTEGER NOT NULL,
                lease_owner TEXT,
                lease_expires_at_ns INTEGER,
                completed_annotation_id TEXT,
                PRIMARY KEY (tenant_id, review_task_id),
                UNIQUE (tenant_id, request_id),
                UNIQUE (tenant_id, task_semantic_sha256)
            );
            CREATE TABLE review_annotations (
                tenant_id TEXT NOT NULL DEFAULT 'unit-tenant',
                annotation_id TEXT NOT NULL,
                review_task_id TEXT NOT NULL,
                lease_fence INTEGER NOT NULL,
                annotation_semantic_sha256 TEXT NOT NULL,
                annotation_json BLOB NOT NULL,
                annotation_exact_sha256 TEXT NOT NULL,
                PRIMARY KEY (tenant_id, annotation_id),
                UNIQUE (tenant_id, review_task_id, lease_fence)
            );
            CREATE TABLE review_reopen_commands (
                tenant_id TEXT NOT NULL DEFAULT 'unit-tenant',
                reopen_id TEXT NOT NULL,
                review_task_id TEXT NOT NULL,
                expected_annotation_id TEXT NOT NULL,
                command_semantic_sha256 TEXT NOT NULL,
                command_json BLOB NOT NULL,
                command_exact_sha256 TEXT NOT NULL,
                PRIMARY KEY (tenant_id, reopen_id)
            );
            """
        )


class _RowsCursor:
    def __init__(self, rows: Sequence[dict[str, object]]) -> None:
        self._rows = tuple(rows)
        self.rowcount = len(self._rows)

    def fetchone(self) -> dict[str, object] | None:
        return None if not self._rows else self._rows[0]

    def fetchall(self) -> Sequence[dict[str, object]]:
        return self._rows


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _node() -> object:
    return logical_node_from_semantic_digest(
        node_type="CAMERA_VIDEO_EXPORT",
        key_namespace="camera-video-export:v1",
        semantic_sha256=semantic_sha256({"unit": "postgres-logical-review"}),
        identity_policy_version="camera-video-export-v1",
    )


def _task() -> object:
    policy = create_nonblocking_review_routing_policy(
        policy_version="review-routing-unit-v1",
        rules=(
            ReviewRoutingRule(
                trigger=ReviewTrigger.LOW_CONFIDENCE,
                priority=1,
                sla_ns=100,
            ),
        ),
    )
    task = create_review_task(
        ReviewRequest(
            request_id=_uuid(500),
            subject=ReviewSubject(
                subject_type="EVENT_HYPOTHESIS",
                subject_id="event-hypothesis:" + "a" * 64,
                recording_identity="b" * 64,
            ),
            trigger=ReviewTrigger.LOW_CONFIDENCE,
            reason_codes=(ReviewTrigger.LOW_CONFIDENCE.value,),
            requested_at_ns=1_000,
        ),
        policy,
    )
    assert task is not None
    return task


def test_postgres_adapters_use_one_authority_transaction_surface() -> None:
    harness = _PostgresSqliteHarness()
    authority = PostgresCanonicalAuthority(lambda: harness)
    registry = PostgresLogicalNodeRegistry(authority)
    queue = PostgresReviewQueue(authority)
    node = _node()

    registry.verify_startup()
    queue.verify_startup()
    created = registry.attach_run_node(
        node=node,  # type: ignore[arg-type]
        run_id=_uuid(1),
        role="OUTPUT",
        first_work_item_id=_uuid(2),
        attached_at="2026-07-20T12:00:00Z",
        existing_node_disposition=RunNodeDisposition.REUSED,
    )
    replay = registry.attach_run_node(
        node=node,  # type: ignore[arg-type]
        run_id=_uuid(1),
        role="OUTPUT",
        first_work_item_id=_uuid(2),
        attached_at="2026-07-20T12:00:00Z",
        existing_node_disposition=RunNodeDisposition.REUSED,
    )
    assert created.node_inserted is True
    assert created.membership.disposition is RunNodeDisposition.CREATED
    assert replay.membership_inserted is False
    assert registry.verify_node(*created.node.identity).memberships == (created.membership,)

    task = _task()
    first = queue.enqueue(task)  # type: ignore[arg-type]
    assert first.inserted is True
    assert queue.enqueue(task).inserted is False  # type: ignore[arg-type]
    lease = queue.claim_next(worker_id="unit-worker", now_ns=1_001, lease_duration_ns=10)
    assert lease is not None
    assert lease.task == task
    assert harness.commits >= 8


def test_postgres_logical_registry_fails_closed_when_migration_is_incomplete() -> None:
    harness = _PostgresSqliteHarness()
    harness._tables = frozenset({"logical_nodes"})
    registry = PostgresLogicalNodeRegistry(PostgresCanonicalAuthority(lambda: harness))

    with pytest.raises(LogicalNodeRegistryError) as error:
        registry.verify_startup()

    assert error.value.code is LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR
