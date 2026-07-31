"""PostgreSQL/Supabase canonical logical-node and human-review adapters.

The local implementations contain the contract-heavy parts of these stores:
canonical JSON verification, revision-chain validation, current-selection CAS,
and review lease fencing.  This module deliberately reuses that pure behavior
while replacing the local SQLite connection with one transaction supplied by
``PostgresCanonicalAuthority``.  PostgreSQL is the sole persistence authority:
no SQLite path is opened, created, or used as a fallback.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TypeVar, cast

from robata.adapters.local_logical_node_registry import LocalLogicalNodeRegistry
from robata.adapters.postgres_authority import (
    PostgresAuthorityError,
    PostgresCanonicalAuthority,
    PostgresConnection,
    PostgresCursor,
)
from robata.adapters.sqlite_review_queue import SQLiteReviewQueue
from robata.contracts.common import SchemaVersion
from robata.contracts.logical_nodes import (
    KeyNamespace,
    LogicalNode,
    NodeLogicalKey,
    NodeType,
    OpaqueUuid,
    ProcessingRunNodeMembership,
    Rfc3339Timestamp,
    RunNodeDisposition,
    RunNodeRole,
)
from robata.contracts.revisions import (
    CurrentSelection,
    ImmutableNodeRevision,
    SelectionDecision,
)
from robata.contracts.schema_registry import SchemaRegistry, default_schema_registry
from robata.ports.logical_node_registry import (
    ExistingNodeDisposition,
    LogicalNodeRegistryError,
    LogicalNodeRegistryErrorCode,
    PublishedRunNodeMembership,
    VerifiedLogicalNode,
)
from robata.ports.review_queue import (
    ReviewQueueError,
    ReviewQueueErrorCode,
)
from robata.ports.revision_registry import (
    PublishedRevision,
    PublishedSelection,
    RevisionSelectionRegistryError,
    RevisionSelectionRegistryErrorCode,
    VerifiedRevisionSubject,
)
from robata.runtime.observability import RuntimeObserver

_ResultT = TypeVar("_ResultT")
_LOGICAL_TABLES = (
    "logical_nodes",
    "processing_run_nodes",
    "immutable_node_revisions",
    "selection_decisions",
    "current_selections",
)
_REVIEW_TABLES = ("review_tasks", "review_annotations", "review_reopen_commands")


class PostgresLogicalReviewStorageError(RuntimeError):
    """PostgreSQL canonical logical/review storage cannot be used safely."""


class _PostgresCompatibilityOperationalError(sqlite3.OperationalError):
    """SQLite-compatible error surface retaining PostgreSQL's SQLSTATE."""

    sqlstate: str | None = None


class _PostgresCompatibilityIntegrityError(sqlite3.IntegrityError):
    """SQLite-compatible integrity error for shared local-domain code."""

    sqlstate: str | None = None


class _PostgresCursorCompatibility:
    """Expose mapping-shaped Psycopg rows through the local adapter surface."""

    def __init__(self, cursor: PostgresCursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> dict[str, object] | None:
        row = self._cursor.fetchone()
        return None if row is None else _postgres_row(row)

    def fetchall(self) -> tuple[dict[str, object], ...]:
        return tuple(_postgres_row(row) for row in self._cursor.fetchall())


class _PostgresConnectionCompatibility:
    """Translate local DB-API SQL to an authority-owned PostgreSQL transaction."""

    def __init__(self, connection: PostgresConnection) -> None:
        self._connection = connection

    @property
    def in_transaction(self) -> bool:
        # The outer authority owns the transaction.  Shared local code uses this
        # property only for its SQLite rollback bookkeeping.
        return True

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _PostgresCursorCompatibility:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if query.strip().upper().startswith("BEGIN"):
            return _PostgresCursorCompatibility(_EmptyPostgresCursor())
        try:
            cursor = self._connection.execute(
                query.replace("?", "%s"),
                None if params is None else tuple(_postgres_parameter(item) for item in params),
            )
        except Exception as error:
            sqlstate = getattr(error, "sqlstate", None)
            converted_type: type[sqlite3.Error] = (
                _PostgresCompatibilityIntegrityError
                if isinstance(sqlstate, str) and sqlstate.startswith("23")
                else _PostgresCompatibilityOperationalError
            )
            converted = converted_type(str(error))
            if isinstance(
                converted,
                (_PostgresCompatibilityIntegrityError, _PostgresCompatibilityOperationalError),
            ):
                converted.sqlstate = sqlstate if isinstance(sqlstate, str) else None
            raise converted from error
        return _PostgresCursorCompatibility(cursor)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _EmptyPostgresCursor:
    @property
    def rowcount(self) -> int:
        return 0

    def fetchone(self) -> None:
        return None

    def fetchall(self) -> tuple[dict[str, object], ...]:
        return ()


def _postgres_parameter(value: object) -> object:
    return value.tobytes() if isinstance(value, memoryview) else value


def _postgres_row(row: object) -> dict[str, object]:
    try:
        values = dict(cast(Any, row))
    except (TypeError, ValueError) as error:
        raise PostgresLogicalReviewStorageError(
            "PostgreSQL canonical query row must be mapping-shaped"
        ) from error
    return {
        key: value.tobytes() if isinstance(value, memoryview) else value
        for key, value in values.items()
    }


def _retry_after_wrapped_serialization_failure[ResultT](
    operation: Callable[[], ResultT],
) -> ResultT:
    """Retry a pure DB operation whose local implementation wrapped SQLSTATE 40001.

    The inherited local registries deliberately turn DB-API exceptions into their
    stable domain errors before the outer authority sees them.  Retrying here
    preserves the same serializable semantics after that wrapping without
    retrying conflicts, validation failures, or provider I/O.
    """

    for attempt in range(4):
        try:
            return operation()
        except Exception as error:
            if attempt == 3 or not _has_retryable_sqlstate(error):
                raise
    raise AssertionError("unreachable PostgreSQL serialization retry state")


def _has_retryable_sqlstate(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        sqlstate = getattr(current, "sqlstate", None)
        if sqlstate in {"40001", "40P01"}:
            return True
        cause = current.__cause__
        current = cause if isinstance(cause, BaseException) else current.__context__
    return False


class _PostgresLogicalOperationMixin:
    """Run inherited registry methods in exactly one authority transaction."""

    _authority: PostgresCanonicalAuthority
    _active_connection: ContextVar[_PostgresConnectionCompatibility | None]

    def _run_postgres_operation(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        active = self._active_connection.get()
        if active is not None:
            return operation()

        def invoke(connection: PostgresConnection) -> _ResultT:
            token = self._active_connection.set(_PostgresConnectionCompatibility(connection))
            try:
                return operation()
            finally:
                self._active_connection.reset(token)

        return self._authority.run_authority_transaction(
            write=write,
            operation_name=operation_name,
            operation=invoke,
        )

    def _postgres_connection(self) -> _PostgresConnectionCompatibility:
        connection = self._active_connection.get()
        if connection is None:
            raise PostgresLogicalReviewStorageError(
                "PostgreSQL adapter operation was invoked outside its authority transaction"
            )
        return connection


class PostgresLogicalNodeRegistry(_PostgresLogicalOperationMixin, LocalLogicalNodeRegistry):
    """PostgreSQL authority for logical nodes, revisions, and selections.

    The inherited local implementation contributes validation and deterministic
    reconstruction only.  Its SQLite initialization, schema checks, and direct
    transaction lifecycle are bypassed below.
    """

    backend_kind = "POSTGRESQL"

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        *,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority
        self._active_connection = ContextVar("postgres_logical_node_connection", default=None)
        self._runtime_observer = runtime_observer
        # These inherited diagnostic properties are intentionally non-filesystem
        # sentinels.  No code in this adapter opens or writes a SQLite database.
        self._root = Path(f"postgres-{authority.schema}-logical-nodes")
        self._database_path = self._root / "canonical-authority"

    @property
    def authority(self) -> PostgresCanonicalAuthority:
        """Return the sole transaction authority used by this adapter."""

        return self._authority

    def verify_startup(self) -> None:
        """Verify that the reviewed migration created all required authority tables."""

        self._run_logical(
            write=False,
            operation_name="logical_nodes.verify_startup",
            operation=lambda: self._verify_database_schema(self._postgres_connection()),
        )

    def _run_logical(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        try:
            return _retry_after_wrapped_serialization_failure(
                lambda: self._run_postgres_operation(
                    write=write,
                    operation_name=operation_name,
                    operation=operation,
                )
            )
        except (LogicalNodeRegistryError, RevisionSelectionRegistryError):
            raise
        except (PostgresAuthorityError, PostgresLogicalReviewStorageError) as error:
            raise LogicalNodeRegistryError(
                LogicalNodeRegistryErrorCode.STORAGE_IO_ERROR,
                f"PostgreSQL logical-node authority failed: {error}",
            ) from error

    # Each public inherited operation receives an outer PostgreSQL transaction.
    # Nested validation calls (for example attach -> verify_node) reuse it.
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
        return self._run_logical(
            write=True,
            operation_name="logical_nodes.attach_run_node",
            operation=lambda: LocalLogicalNodeRegistry.attach_run_node(
                self,
                node=node,
                run_id=run_id,
                role=role,
                first_work_item_id=first_work_item_id,
                attached_at=attached_at,
                existing_node_disposition=existing_node_disposition,
            ),
        )

    def lookup_node(
        self, node_type: NodeType, node_logical_key: NodeLogicalKey
    ) -> LogicalNode | None:
        return self._run_logical(
            write=False,
            operation_name="logical_nodes.lookup_node",
            operation=lambda: LocalLogicalNodeRegistry.lookup_node(
                self, node_type, node_logical_key
            ),
        )

    def lookup_membership(
        self,
        run_id: OpaqueUuid,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
        role: RunNodeRole,
    ) -> ProcessingRunNodeMembership | None:
        return self._run_logical(
            write=False,
            operation_name="logical_nodes.lookup_membership",
            operation=lambda: LocalLogicalNodeRegistry.lookup_membership(
                self, run_id, node_type, node_logical_key, role
            ),
        )

    def list_run_memberships(self, run_id: OpaqueUuid) -> tuple[ProcessingRunNodeMembership, ...]:
        return self._run_logical(
            write=False,
            operation_name="logical_nodes.list_run_memberships",
            operation=lambda: LocalLogicalNodeRegistry.list_run_memberships(self, run_id),
        )

    def list_node_memberships(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> tuple[ProcessingRunNodeMembership, ...]:
        return self._run_logical(
            write=False,
            operation_name="logical_nodes.list_node_memberships",
            operation=lambda: LocalLogicalNodeRegistry.list_node_memberships(
                self, node_type, node_logical_key
            ),
        )

    def verify_node(
        self,
        node_type: NodeType,
        node_logical_key: NodeLogicalKey,
    ) -> VerifiedLogicalNode:
        return self._run_logical(
            write=False,
            operation_name="logical_nodes.verify_node",
            operation=lambda: LocalLogicalNodeRegistry.verify_node(
                self, node_type, node_logical_key
            ),
        )

    def publish_revision(self, revision: ImmutableNodeRevision) -> PublishedRevision:
        try:
            return _retry_after_wrapped_serialization_failure(
                lambda: self._run_postgres_operation(
                    write=True,
                    operation_name="logical_nodes.publish_revision",
                    operation=lambda: LocalLogicalNodeRegistry.publish_revision(self, revision),
                )
            )
        except RevisionSelectionRegistryError:
            raise
        except (PostgresAuthorityError, LogicalNodeRegistryError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                f"PostgreSQL revision authority failed: {error}",
            ) from error

    def select_revision(
        self,
        *,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        selected_revision_id: OpaqueUuid,
        selection_decision_id: OpaqueUuid,
        selection_key_namespace: KeyNamespace,
        expected_previous_selection_decision_id: OpaqueUuid | None,
        selection_policy_version: SchemaVersion,
        selected_at: Rfc3339Timestamp,
    ) -> PublishedSelection:
        try:
            return _retry_after_wrapped_serialization_failure(
                lambda: self._run_postgres_operation(
                    write=True,
                    operation_name="logical_nodes.select_revision",
                    operation=lambda: LocalLogicalNodeRegistry.select_revision(
                        self,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        selected_revision_id=selected_revision_id,
                        selection_decision_id=selection_decision_id,
                        selection_key_namespace=selection_key_namespace,
                        expected_previous_selection_decision_id=(
                            expected_previous_selection_decision_id
                        ),
                        selection_policy_version=selection_policy_version,
                        selected_at=selected_at,
                    ),
                )
            )
        except RevisionSelectionRegistryError:
            raise
        except (PostgresAuthorityError, LogicalNodeRegistryError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                f"PostgreSQL selection authority failed: {error}",
            ) from error

    def lookup_revision(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        revision_id: OpaqueUuid,
    ) -> ImmutableNodeRevision | None:
        return self._run_revision_read(
            "logical_nodes.lookup_revision",
            lambda: LocalLogicalNodeRegistry.lookup_revision(
                self, subject_type, subject_id, revision_id
            ),
        )

    def lookup_selection_decision(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        selection_decision_id: OpaqueUuid,
    ) -> SelectionDecision | None:
        return self._run_revision_read(
            "logical_nodes.lookup_selection_decision",
            lambda: LocalLogicalNodeRegistry.lookup_selection_decision(
                self, subject_type, subject_id, selection_decision_id
            ),
        )

    def lookup_current_selection(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> CurrentSelection | None:
        return self._run_revision_read(
            "logical_nodes.lookup_current_selection",
            lambda: LocalLogicalNodeRegistry.lookup_current_selection(
                self, subject_type, subject_id
            ),
        )

    def list_revisions(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> tuple[ImmutableNodeRevision, ...]:
        return self._run_revision_read(
            "logical_nodes.list_revisions",
            lambda: LocalLogicalNodeRegistry.list_revisions(self, subject_type, subject_id),
        )

    def list_selection_decisions(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> tuple[SelectionDecision, ...]:
        return self._run_revision_read(
            "logical_nodes.list_selection_decisions",
            lambda: LocalLogicalNodeRegistry.list_selection_decisions(
                self, subject_type, subject_id
            ),
        )

    def verify_subject(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> VerifiedRevisionSubject:
        return self._run_revision_read(
            "logical_nodes.verify_subject",
            lambda: LocalLogicalNodeRegistry.verify_subject(self, subject_type, subject_id),
        )

    def rebuild_current_projection(self) -> tuple[CurrentSelection, ...]:
        try:
            return _retry_after_wrapped_serialization_failure(
                lambda: self._run_postgres_operation(
                    write=True,
                    operation_name="logical_nodes.rebuild_current_projection",
                    operation=lambda: LocalLogicalNodeRegistry.rebuild_current_projection(self),
                )
            )
        except RevisionSelectionRegistryError:
            raise
        except (PostgresAuthorityError, LogicalNodeRegistryError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.TRANSACTION_FAILED,
                f"PostgreSQL current-selection rebuild failed: {error}",
            ) from error

    def _run_revision_read(
        self, operation_name: str, operation: Callable[[], _ResultT]
    ) -> _ResultT:
        try:
            return _retry_after_wrapped_serialization_failure(
                lambda: self._run_postgres_operation(
                    write=False,
                    operation_name=operation_name,
                    operation=operation,
                )
            )
        except RevisionSelectionRegistryError:
            raise
        except (PostgresAuthorityError, LogicalNodeRegistryError) as error:
            raise RevisionSelectionRegistryError(
                RevisionSelectionRegistryErrorCode.STORAGE_IO_ERROR,
                f"PostgreSQL revision authority failed: {error}",
            ) from error

    # LocalLogicalNodeRegistry's public methods call these hooks.  PostgreSQL's
    # outer authority already began the transaction, performs the real commit,
    # and rolls back on any exception.
    def _connect(self) -> Any:
        return self._postgres_connection()

    @contextmanager
    def _observed_transaction_scope(
        self,
        connection: Any,
        *,
        operation: str,
        write: bool,
        rollback_on_success: bool = False,
    ) -> Iterator[None]:
        del connection, operation, write, rollback_on_success
        yield

    def _commit_observed(
        self,
        connection: Any,
        *,
        operation: str,
        write: bool,
        use_commit_hook: bool = True,
    ) -> None:
        del connection, operation, write, use_commit_hook

    def _rollback_observed(
        self,
        connection: Any,
        *,
        operation: str,
        write: bool,
        suppress_errors: bool,
    ) -> None:
        del connection, operation, write, suppress_errors

    def _commit(self, connection: Any) -> None:
        del connection

    def _rollback(self, connection: Any) -> None:
        del connection

    def _verify_database_schema(self, connection: Any, *, expected_version: int = 2) -> None:
        del expected_version
        _verify_required_tables(connection, self._authority.schema, _LOGICAL_TABLES)

    @staticmethod
    def _verify_database_health(connection: Any) -> None:
        connection.execute("SELECT 1").fetchone()


class PostgresReviewQueue(_PostgresLogicalOperationMixin, SQLiteReviewQueue):
    """PostgreSQL authority for fenced human-review work and immutable history."""

    backend_kind = "POSTGRESQL"

    def __init__(
        self,
        authority: PostgresCanonicalAuthority,
        *,
        registry: SchemaRegistry | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority
        self._active_connection = ContextVar("postgres_review_queue_connection", default=None)
        self._registry = registry or default_schema_registry()
        self._runtime_observer = runtime_observer
        self._database_path = Path(f"postgres-{authority.schema}-review-queue")

    @property
    def authority(self) -> PostgresCanonicalAuthority:
        """Return the sole transaction authority used by this queue."""

        return self._authority

    def verify_startup(self) -> None:
        """Verify that the reviewed migration created the queue tables."""

        self._run_review(
            write=False,
            operation_name="review.verify_startup",
            operation=lambda: _verify_required_tables(
                self._postgres_connection(), self._authority.schema, _REVIEW_TABLES
            ),
        )

    def _run_review(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[], _ResultT],
    ) -> _ResultT:
        try:
            return _retry_after_wrapped_serialization_failure(
                lambda: self._run_postgres_operation(
                    write=write,
                    operation_name=operation_name,
                    operation=operation,
                )
            )
        except ReviewQueueError:
            raise
        except (PostgresAuthorityError, PostgresLogicalReviewStorageError) as error:
            raise ReviewQueueError(
                ReviewQueueErrorCode.STORAGE_IO_ERROR,
                f"PostgreSQL review authority failed: {error}",
            ) from error

    def _transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[Any], _ResultT],
    ) -> _ResultT:
        try:
            return self._run_review(
                write=write,
                operation_name=f"review.{operation_name}",
                operation=lambda: operation(self._postgres_connection()),
            )
        except sqlite3.IntegrityError as error:
            raise ReviewQueueError(
                ReviewQueueErrorCode.INTEGRITY_ERROR,
                f"PostgreSQL review queue integrity failure: {error}",
            ) from error
        except sqlite3.Error as error:
            raise ReviewQueueError(
                ReviewQueueErrorCode.STORAGE_IO_ERROR,
                f"PostgreSQL review queue storage failure: {error}",
            ) from error


def _verify_required_tables(
    connection: _PostgresConnectionCompatibility,
    schema: str,
    required_tables: Sequence[str],
) -> None:
    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ANY(?)
        """,
        (schema, list(required_tables)),
    ).fetchall()
    found = {str(row["table_name"]) for row in rows}
    missing = sorted(set(required_tables).difference(found))
    if missing:
        raise PostgresLogicalReviewStorageError(
            "canonical PostgreSQL migration is incomplete: " + ", ".join(missing)
        )


__all__ = [
    "PostgresLogicalNodeRegistry",
    "PostgresLogicalReviewStorageError",
    "PostgresReviewQueue",
]
