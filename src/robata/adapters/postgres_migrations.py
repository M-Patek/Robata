"""Versioned PostgreSQL migration runner for Robata canonical authority.

Migrations are immutable SQL files.  The runner stores an exact SHA-256 for
every applied file in ``robata_ops.schema_migrations`` and rejects a rewritten
history before it executes further SQL.  It intentionally takes an injected
connection factory so importing the module neither imports Psycopg nor opens a
database connection.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from robata.adapters.postgres_authority import ConnectionFactory, PostgresConnection
from robata.contracts.hashing import exact_bytes_sha256

_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{4})_(?P<name>[a-z0-9_]+)\.sql$")
_MIGRATION_SCHEMA = "robata_ops"
_MIGRATION_TABLE = "schema_migrations"


class PostgresMigrationError(RuntimeError):
    """Base error for immutable canonical PostgreSQL migrations."""


class PostgresMigrationDiscoveryError(PostgresMigrationError):
    """The checked-in migration directory is malformed."""


class PostgresMigrationDriftError(PostgresMigrationError):
    """An already-applied migration has different exact bytes."""


class PostgresMigrationStorageError(PostgresMigrationError):
    """The migration ledger cannot be read or updated safely."""


@dataclass(frozen=True, slots=True)
class PostgresMigration:
    """One immutable SQL migration discovered from the repository."""

    migration_id: str
    path: Path
    sql: str
    exact_bytes_sha256: str


@dataclass(frozen=True, slots=True)
class PostgresMigrationApplication:
    """Outcome of applying a reviewed migration set."""

    applied_ids: tuple[str, ...]
    already_applied_ids: tuple[str, ...]


class PostgresMigrationRunner:
    """Apply and verify immutable canonical PostgreSQL migrations."""

    def __init__(self, connection_factory: ConnectionFactory, migrations_directory: Path) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        if not isinstance(migrations_directory, Path):
            raise TypeError("migrations_directory must be a Path")
        self._connection_factory = connection_factory
        self._migrations_directory = migrations_directory

    @property
    def migrations_directory(self) -> Path:
        return self._migrations_directory

    def migrations(self) -> tuple[PostgresMigration, ...]:
        """Read the immutable ordered migration set without database I/O."""

        return discover_postgres_migrations(self._migrations_directory)

    def apply(self) -> PostgresMigrationApplication:
        """Apply every missing migration in order under one database transaction."""

        migrations = self.migrations()
        connection = self._open()
        try:
            connection.execute("BEGIN")
            _ensure_ledger(connection)
            applied = _load_applied(connection)
            _reject_drift(migrations, applied)
            newly_applied: list[str] = []
            already_applied: list[str] = []
            for migration in migrations:
                if migration.migration_id in applied:
                    already_applied.append(migration.migration_id)
                    continue
                connection.execute(migration.sql)
                connection.execute(
                    """
                    INSERT INTO robata_ops.schema_migrations (
                        migration_id,
                        exact_bytes_sha256
                    ) VALUES (%s, %s)
                    """,
                    (migration.migration_id, migration.exact_bytes_sha256),
                )
                newly_applied.append(migration.migration_id)
            connection.commit()
            return PostgresMigrationApplication(
                applied_ids=tuple(newly_applied),
                already_applied_ids=tuple(already_applied),
            )
        except PostgresMigrationError:
            with suppress(Exception):
                connection.rollback()
            raise
        except Exception as error:
            with suppress(Exception):
                connection.rollback()
            raise PostgresMigrationStorageError(
                "canonical PostgreSQL migration transaction failed"
            ) from error
        finally:
            with suppress(Exception):
                connection.close()

    def verify(self, *, required_migration_ids: Iterable[str] | None = None) -> tuple[str, ...]:
        """Verify exact applied bytes without creating or changing any database object."""

        migrations = self.migrations()
        expected = {item.migration_id: item for item in migrations}
        required = (
            tuple(expected) if required_migration_ids is None else tuple(required_migration_ids)
        )
        if not required:
            raise ValueError("required_migration_ids must not be empty")
        if len(required) != len(set(required)):
            raise ValueError("required_migration_ids must be unique")
        unknown = sorted(set(required).difference(expected))
        if unknown:
            raise PostgresMigrationDiscoveryError(
                "required migration IDs are not checked in: " + ", ".join(unknown)
            )

        connection = self._open()
        try:
            connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            row = connection.execute(
                "SELECT to_regclass(%s) AS migration_table",
                (f"{_MIGRATION_SCHEMA}.{_MIGRATION_TABLE}",),
            ).fetchone()
            if row is None or row.get("migration_table") is None:
                raise PostgresMigrationStorageError(
                    "canonical PostgreSQL migration ledger is absent"
                )
            applied = _load_applied(connection)
            _reject_drift(migrations, applied)
            missing = sorted(set(required).difference(applied))
            if missing:
                raise PostgresMigrationStorageError(
                    "canonical PostgreSQL migrations are missing: " + ", ".join(missing)
                )
            connection.commit()
            return tuple(required)
        except PostgresMigrationError:
            with suppress(Exception):
                connection.rollback()
            raise
        except Exception as error:
            with suppress(Exception):
                connection.rollback()
            raise PostgresMigrationStorageError(
                "cannot verify canonical PostgreSQL migrations"
            ) from error
        finally:
            with suppress(Exception):
                connection.close()

    def _open(self) -> PostgresConnection:
        try:
            connection = self._connection_factory()
        except Exception as error:
            raise PostgresMigrationStorageError(
                f"cannot open PostgreSQL migration connection: {error}"
            ) from error
        if not isinstance(connection, PostgresConnection):
            with suppress(Exception):
                connection.close()
            raise PostgresMigrationStorageError(
                "connection_factory did not return a PostgreSQL DB-API connection"
            )
        return connection


def discover_postgres_migrations(directory: Path) -> tuple[PostgresMigration, ...]:
    """Discover a contiguous, ordered, exact-byte migration set."""

    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    if not directory.is_dir():
        raise PostgresMigrationDiscoveryError(f"migration directory does not exist: {directory}")
    migrations: list[PostgresMigration] = []
    seen_ids: set[str] = set()
    expected_version = 1
    for path in sorted(directory.glob("*.sql")):
        match = _MIGRATION_NAME.fullmatch(path.name)
        if match is None:
            raise PostgresMigrationDiscoveryError(
                f"migration filename must be NNNN_lowercase_name.sql: {path.name}"
            )
        migration_id = match.group("version")
        version = int(migration_id)
        if version != expected_version:
            raise PostgresMigrationDiscoveryError(
                "migration sequence must be contiguous; "
                f"expected {expected_version:04d}, found {migration_id}"
            )
        if migration_id in seen_ids:
            raise PostgresMigrationDiscoveryError(f"duplicate migration ID: {migration_id}")
        raw = path.read_bytes()
        if not raw:
            raise PostgresMigrationDiscoveryError(f"migration is empty: {path.name}")
        try:
            sql = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PostgresMigrationDiscoveryError(
                f"migration must be UTF-8: {path.name}"
            ) from error
        migrations.append(
            PostgresMigration(
                migration_id=migration_id,
                path=path,
                sql=sql,
                exact_bytes_sha256=exact_bytes_sha256(raw),
            )
        )
        seen_ids.add(migration_id)
        expected_version += 1
    if not migrations:
        raise PostgresMigrationDiscoveryError(
            "at least one canonical PostgreSQL migration is required"
        )
    return tuple(migrations)


def _ensure_ledger(connection: PostgresConnection) -> None:
    connection.execute(f"CREATE SCHEMA IF NOT EXISTS {_MIGRATION_SCHEMA}")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_MIGRATION_SCHEMA}.{_MIGRATION_TABLE} (
            migration_id TEXT PRIMARY KEY,
            exact_bytes_sha256 TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _load_applied(connection: PostgresConnection) -> dict[str, str]:
    rows = connection.execute(
        f"""
        SELECT migration_id, exact_bytes_sha256
        FROM {_MIGRATION_SCHEMA}.{_MIGRATION_TABLE}
        ORDER BY migration_id
        """
    ).fetchall()
    applied: dict[str, str] = {}
    for row in rows:
        migration_id = row.get("migration_id")
        checksum = row.get("exact_bytes_sha256")
        if not isinstance(migration_id, str) or not isinstance(checksum, str):
            raise PostgresMigrationStorageError("migration ledger row is malformed")
        if migration_id in applied:
            raise PostgresMigrationStorageError("migration ledger has duplicate IDs")
        applied[migration_id] = checksum
    return applied


def _reject_drift(migrations: Sequence[PostgresMigration], applied: dict[str, str]) -> None:
    expected = {item.migration_id: item.exact_bytes_sha256 for item in migrations}
    unknown = sorted(set(applied).difference(expected))
    if unknown:
        raise PostgresMigrationDriftError(
            "database contains unknown canonical migrations: " + ", ".join(unknown)
        )
    for migration_id, checksum in applied.items():
        if expected[migration_id] != checksum:
            raise PostgresMigrationDriftError(
                f"canonical migration exact bytes changed after apply: {migration_id}"
            )


__all__ = [
    "PostgresMigration",
    "PostgresMigrationApplication",
    "PostgresMigrationDiscoveryError",
    "PostgresMigrationDriftError",
    "PostgresMigrationError",
    "PostgresMigrationRunner",
    "PostgresMigrationStorageError",
    "discover_postgres_migrations",
]
