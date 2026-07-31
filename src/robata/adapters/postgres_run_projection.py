"""Read-only PostgreSQL projection of committed canonical run bytes.

This adapter keeps the Web/API read side separate from worker mutation authority.
It shares the local projection's strict digest and contract validation, while all
queries run through a tenant-bound repeatable-read PostgreSQL transaction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

from robata.adapters.postgres_authority import (
    PostgresAuthorityError,
    PostgresCanonicalAuthority,
    PostgresConnection,
    PostgresCursor,
)
from robata.web_api.models import RunListResponse, RunSnapshotResponse
from robata.web_api.read_model import (
    LocalStateUnavailable,
    ReadOnlyLocalRunProjection,
)


class PostgresRunProjectionStorageError(LocalStateUnavailable):
    """The canonical PostgreSQL read projection cannot be queried safely."""


class _Cursor:
    def __init__(self, cursor: PostgresCursor) -> None:
        self._cursor = cursor

    def fetchone(self) -> dict[str, object] | None:
        row = self._cursor.fetchone()
        return None if row is None else _row(row)

    def fetchall(self) -> tuple[dict[str, object], ...]:
        return tuple(_row(item) for item in self._cursor.fetchall())


class _Connection:
    """Small SQLite-query-compatible wrapper over one active PostgreSQL read txn."""

    def __init__(self, connection: PostgresConnection) -> None:
        self._connection = connection

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _Cursor:
        try:
            cursor = self._connection.execute(query.replace("?", "%s"), params)
        except Exception as error:
            raise sqlite3.OperationalError(str(error)) from error
        return _Cursor(cursor)


def _row(value: object) -> dict[str, object]:
    try:
        mapping = dict(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise PostgresRunProjectionStorageError(
            "PostgreSQL run projection query row must be mapping-shaped"
        ) from error
    return {
        key: item.tobytes() if isinstance(item, memoryview) else item
        for key, item in mapping.items()
    }


class PostgresCommittedRunProjection(ReadOnlyLocalRunProjection):
    """Read committed primary-completion bytes from canonical PostgreSQL only."""

    backend_kind = "POSTGRESQL"

    def __init__(self, authority: PostgresCanonicalAuthority) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority
        self._active_connection: ContextVar[_Connection | None] = ContextVar(
            "postgres_run_projection_connection",
            default=None,
        )
        # Retain the inherited diagnostic properties without claiming a real
        # filesystem state store.  This adapter never opens the local path.
        self._state_dir = Path(f"postgres-{authority.schema}-run-projection")
        self._database_path = self._state_dir / "canonical-authority"

    @property
    def authority(self) -> PostgresCanonicalAuthority:
        """Return the authority whose tenant/RLS context scopes every read."""

        return self._authority

    def verify_startup(self) -> None:
        """Confirm that the completion migration is visible to this read role."""

        def operation(connection: PostgresConnection) -> None:
            rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (self._authority.schema, "primary_completions"),
            ).fetchall()
            if not rows:
                raise PostgresRunProjectionStorageError(
                    "canonical PostgreSQL migration is incomplete: primary_completions"
                )

        self._run("run_projection.verify_startup", operation)

    def health_check(self) -> None:
        self._run(
            "run_projection.health_check",
            lambda _connection: ReadOnlyLocalRunProjection.health_check(self),
        )

    def list_runs(self) -> RunListResponse:
        return self._run(
            "run_projection.list_runs",
            lambda _connection: ReadOnlyLocalRunProjection.list_runs(self),
        )

    def snapshot(self, run_id: str) -> RunSnapshotResponse:
        return self._run(
            "run_projection.snapshot",
            lambda _connection: ReadOnlyLocalRunProjection.snapshot(self, run_id),
        )

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        connection = self._active_connection.get()
        if connection is None:
            raise PostgresRunProjectionStorageError(
                "PostgreSQL run projection was invoked outside its read transaction"
            )
        try:
            yield connection
        except sqlite3.Error as error:
            raise PostgresRunProjectionStorageError(
                f"cannot read canonical PostgreSQL completion state: {error}"
            ) from error

    def _run[ResultT](
        self,
        operation_name: str,
        operation: Callable[[PostgresConnection], ResultT],
    ) -> ResultT:
        if not isinstance(operation_name, str) or not operation_name:
            raise ValueError("operation_name must be nonempty")

        def invoke(connection: PostgresConnection) -> ResultT:
            token = self._active_connection.set(_Connection(connection))
            try:
                return operation(connection)
            finally:
                self._active_connection.reset(token)

        try:
            return self._authority.run_authority_transaction(
                write=False,
                operation_name=operation_name,
                operation=invoke,
            )
        except LocalStateUnavailable:
            raise
        except PostgresAuthorityError as error:
            raise PostgresRunProjectionStorageError(
                f"PostgreSQL run projection authority failed: {error}"
            ) from error


__all__ = ["PostgresCommittedRunProjection", "PostgresRunProjectionStorageError"]
