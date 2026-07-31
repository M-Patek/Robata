"""Shared PostgreSQL transaction boundary for canonical authority adapters.

The module deliberately contains no connection-string configuration or import-time I/O.
Composition supplies a short-lived DB-API connection factory, so the canonical adapters
can be tested against a double and connected to Psycopg only at process startup.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast, runtime_checkable

Row = Mapping[str, object]
T = TypeVar("T")

_SCHEMA_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")
_ACTIVE_AUTHORITY_TRANSACTION: ContextVar[str | None] = ContextVar(
    "active_postgres_authority_transaction",
    default=None,
)


@runtime_checkable
class PostgresCursor(Protocol):
    """Subset shared by Psycopg cursors and the injectable test double."""

    @property
    def rowcount(self) -> int: ...

    def fetchone(self) -> Row | None: ...

    def fetchall(self) -> Sequence[Row]: ...


@runtime_checkable
class PostgresConnection(Protocol):
    """Minimal DB-API surface required by canonical authority adapters."""

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> PostgresCursor: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


ConnectionFactory = Callable[[], PostgresConnection]


class PostgresAuthorityError(RuntimeError):
    """Base failure for the PostgreSQL canonical authority boundary."""


class PostgresAuthorityStorageError(PostgresAuthorityError):
    """The PostgreSQL authority cannot begin, commit, or verify a transaction."""


class PostgresAuthorityConfigurationError(PostgresAuthorityError):
    """The declared PostgreSQL authority configuration is not safe to use."""


@dataclass(frozen=True, slots=True)
class PostgresAuthorityStartup:
    """Read-only startup facts, intentionally without credentials."""

    backend_kind: str
    schema: str
    required_tables: tuple[str, ...]


class PostgresCanonicalAuthority:
    """One short-lived, non-nestable PostgreSQL transaction authority.

    Every canonical scheduler, stream, delivery, and capture adapter must receive the
    same instance.  A callback receives a connection after ``BEGIN`` and a local,
    validated ``search_path``; provider and object-store calls must stay outside it.
    """

    backend_kind = "POSTGRESQL"

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        schema: str = "robata_canonical",
        tenant_setting: str | None = None,
        tenant_id: str | None = None,
        serialization_retries: int = 3,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        if not isinstance(schema, str) or _SCHEMA_NAME.fullmatch(schema) is None:
            raise ValueError("schema must be a lowercase PostgreSQL identifier")
        if (tenant_setting is None) != (tenant_id is None):
            raise ValueError("tenant_setting and tenant_id must be supplied together")
        if tenant_setting is not None and not tenant_setting.strip():
            raise ValueError("tenant_setting must be nonempty")
        if tenant_id is not None and not tenant_id.strip():
            raise ValueError("tenant_id must be nonempty")
        if isinstance(serialization_retries, bool) or not isinstance(serialization_retries, int):
            raise TypeError("serialization_retries must be an integer")
        if serialization_retries < 0:
            raise ValueError("serialization_retries must be nonnegative")
        self._connection_factory = connection_factory
        self._schema = schema
        self._tenant_setting = tenant_setting
        self._tenant_id = tenant_id
        self._serialization_retries = serialization_retries

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def startup(self) -> PostgresAuthorityStartup:
        return PostgresAuthorityStartup(
            backend_kind=self.backend_kind,
            schema=self._schema,
            required_tables=("work_items", "work_dependencies", "work_attempts"),
        )

    def run_authority_transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[PostgresConnection], T],
    ) -> T:
        """Run one canonical operation atomically, retrying serializable conflicts.

        Callbacks must be deterministic with respect to their supplied input. They can
        be replayed only for PostgreSQL's documented serialization/deadlock failures.
        """

        if not isinstance(write, bool):
            raise TypeError("write must be a boolean")
        checked_name = _nonempty(operation_name, "operation_name")
        if not callable(operation):
            raise TypeError("operation must be callable")
        active = _ACTIVE_AUTHORITY_TRANSACTION.get()
        if active is not None:
            raise PostgresAuthorityConfigurationError(
                "nested PostgreSQL authority transaction is forbidden: "
                f"{checked_name} inside {active}"
            )

        attempt = 0
        while True:
            connection: PostgresConnection | None = None
            token = _ACTIVE_AUTHORITY_TRANSACTION.set(checked_name)
            try:
                connection = self._open()
                connection.execute(
                    "BEGIN ISOLATION LEVEL SERIALIZABLE"
                    if write
                    else "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY"
                )
                connection.execute(f'SET LOCAL search_path TO "{self._schema}", pg_catalog')
                if self._tenant_setting is not None and self._tenant_id is not None:
                    connection.execute(
                        "SELECT set_config(%s, %s, true)",
                        (self._tenant_setting, self._tenant_id),
                    )
                result = operation(connection)
                connection.commit()
                return result
            except BaseException as error:
                if connection is not None:
                    with suppress(Exception):
                        connection.rollback()
                if (
                    _is_retryable_serialization_failure(error)
                    and attempt < self._serialization_retries
                ):
                    attempt += 1
                    continue
                if isinstance(
                    error,
                    (PostgresAuthorityError, TypeError, ValueError, KeyboardInterrupt, SystemExit),
                ):
                    raise
                raise
            finally:
                _ACTIVE_AUTHORITY_TRANSACTION.reset(token)
                if connection is not None:
                    with suppress(Exception):
                        connection.close()

    def verify_startup(self) -> PostgresAuthorityStartup:
        """Confirm that P22 authority tables exist without doing any mutation."""

        required = self.startup.required_tables

        def operation(connection: PostgresConnection) -> None:
            rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = ANY(%s)
                """,
                (self._schema, list(required)),
            ).fetchall()
            found = {str(row["table_name"]) for row in rows}
            missing = sorted(set(required).difference(found))
            if missing:
                raise PostgresAuthorityStorageError(
                    "canonical PostgreSQL migration is incomplete: " + ", ".join(missing)
                )

        self.run_authority_transaction(
            write=False,
            operation_name="verify_startup",
            operation=operation,
        )
        return self.startup

    def _open(self) -> PostgresConnection:
        try:
            connection = self._connection_factory()
        except Exception as error:
            raise PostgresAuthorityStorageError(
                f"cannot open canonical PostgreSQL authority: {error}"
            ) from error
        if not isinstance(connection, PostgresConnection):
            with suppress(Exception):
                connection.close()
            raise PostgresAuthorityStorageError(
                "connection_factory did not return a PostgreSQL DB-API connection"
            )
        return connection


def psycopg_connection_factory(
    dsn: str,
    *,
    connect_timeout_seconds: int = 10,
    application_name: str | None = None,
) -> ConnectionFactory:
    """Return a lazy Psycopg factory without importing Psycopg at module import time."""

    checked_dsn = _nonempty(dsn, "dsn")
    if isinstance(connect_timeout_seconds, bool) or not isinstance(connect_timeout_seconds, int):
        raise TypeError("connect_timeout_seconds must be an integer")
    if connect_timeout_seconds < 1:
        raise ValueError("connect_timeout_seconds must be positive")
    checked_application_name = (
        None if application_name is None else _nonempty(application_name, "application_name")
    )

    def factory() -> PostgresConnection:
        try:
            from psycopg import connect
            from psycopg.rows import dict_row
        except ImportError as error:
            raise PostgresAuthorityConfigurationError(
                "Psycopg is required for PostgreSQL canonical authority; install robata[pgvector]"
            ) from error
        return cast(
            PostgresConnection,
            connect(
                checked_dsn,
                autocommit=True,
                connect_timeout=connect_timeout_seconds,
                row_factory=dict_row,
                application_name=checked_application_name,
            ),
        )

    return factory


def active_postgres_authority_transaction_operation() -> str | None:
    """Return the active PostgreSQL authority operation for the current context."""

    return _ACTIVE_AUTHORITY_TRANSACTION.get()


def require_outside_postgres_authority_transaction(*, activity: str) -> None:
    """Reject provider/media I/O while a PostgreSQL authority transaction is open."""

    checked_activity = _nonempty(activity, "activity")
    operation = active_postgres_authority_transaction_operation()
    if operation is not None:
        raise PostgresAuthorityConfigurationError(
            f"{checked_activity} cannot run inside PostgreSQL authority transaction {operation}"
        )


def postgres_sqlstate(error: BaseException) -> str | None:
    """Return Psycopg/DB-API SQLSTATE without importing an optional driver."""

    sqlstate = getattr(error, "sqlstate", None)
    return sqlstate if isinstance(sqlstate, str) else None


def _is_retryable_serialization_failure(error: BaseException) -> bool:
    return postgres_sqlstate(error) in {"40001", "40P01"}


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


__all__ = [
    "ConnectionFactory",
    "PostgresAuthorityConfigurationError",
    "PostgresAuthorityError",
    "PostgresAuthorityStartup",
    "PostgresAuthorityStorageError",
    "PostgresCanonicalAuthority",
    "PostgresConnection",
    "PostgresCursor",
    "Row",
    "active_postgres_authority_transaction_operation",
    "postgres_sqlstate",
    "psycopg_connection_factory",
    "require_outside_postgres_authority_transaction",
]
