"""Tests for immutable canonical PostgreSQL migration handling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from robata.adapters.postgres_migrations import (
    PostgresMigrationDiscoveryError,
    PostgresMigrationDriftError,
    PostgresMigrationRunner,
    PostgresMigrationStorageError,
    discover_postgres_migrations,
)


class _Cursor:
    def __init__(
        self,
        *,
        one: dict[str, object] | None = None,
        rows: Sequence[dict[str, object]] = (),
    ) -> None:
        self._one = one
        self._rows = tuple(rows)
        self.rowcount = 1

    def fetchone(self) -> dict[str, object] | None:
        return self._one

    def fetchall(self) -> Sequence[dict[str, object]]:
        return self._rows


class _Connection:
    def __init__(
        self,
        *,
        migration_rows: Sequence[dict[str, object]] = (),
        ledger_present: bool = True,
    ) -> None:
        self.migration_rows = tuple(migration_rows)
        self.ledger_present = ledger_present
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(
        self,
        query: str,
        params: Sequence[object] | None = None,
    ) -> _Cursor:
        self.calls.append((query, None if params is None else tuple(params)))
        if "to_regclass" in query:
            return _Cursor(
                one={"migration_table": "robata_ops.schema_migrations"}
                if self.ledger_present
                else {"migration_table": None}
            )
        if "SELECT migration_id, exact_bytes_sha256" in query:
            return _Cursor(rows=self.migration_rows)
        return _Cursor()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _write_migration(directory: Path, name: str, sql: str = "SELECT 1;\n") -> Path:
    path = directory / name
    path.write_text(sql, encoding="utf-8", newline="\n")
    return path


def test_discovery_requires_contiguous_utf8_numbered_migrations(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0001_base.sql", "SELECT 1;\n")
    _write_migration(tmp_path, "0002_work.sql", "SELECT 2;\n")

    migrations = discover_postgres_migrations(tmp_path)

    assert [item.migration_id for item in migrations] == ["0001", "0002"]
    assert migrations[0].exact_bytes_sha256 != migrations[1].exact_bytes_sha256
    _write_migration(tmp_path, "0004_gap.sql", "SELECT 4;\n")
    with pytest.raises(PostgresMigrationDiscoveryError, match="contiguous"):
        discover_postgres_migrations(tmp_path)


def test_discovery_rejects_unreviewed_filename_or_empty_script(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0001_base.sql", "SELECT 1;\n")
    _write_migration(tmp_path, "notes.sql", "SELECT 2;\n")

    with pytest.raises(PostgresMigrationDiscoveryError, match="filename"):
        discover_postgres_migrations(tmp_path)

    (tmp_path / "notes.sql").unlink()
    _write_migration(tmp_path, "0002_empty.sql", "")
    with pytest.raises(PostgresMigrationDiscoveryError, match="empty"):
        discover_postgres_migrations(tmp_path)


def test_apply_creates_ledger_and_records_exact_bytes(tmp_path: Path) -> None:
    _write_migration(tmp_path, "0001_base.sql", "CREATE SCHEMA robata_canonical;\n")
    _write_migration(tmp_path, "0002_work.sql", "CREATE TABLE work_items ();\n")
    connection = _Connection()
    runner = PostgresMigrationRunner(lambda: connection, tmp_path)

    result = runner.apply()

    assert result.applied_ids == ("0001", "0002")
    assert result.already_applied_ids == ()
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed is True
    inserts = [
        query
        for query, _params in connection.calls
        if "INSERT INTO robata_ops.schema_migrations" in query
    ]
    assert len(inserts) == 2
    executed_sql = [query for query, _params in connection.calls]
    assert "CREATE SCHEMA robata_canonical;\n" in executed_sql
    assert "CREATE TABLE work_items ();\n" in executed_sql


def test_verify_rejects_absent_ledger_and_rewritten_history(tmp_path: Path) -> None:
    path = _write_migration(tmp_path, "0001_base.sql", "SELECT 1;\n")
    expected = discover_postgres_migrations(tmp_path)[0]
    absent = _Connection(ledger_present=False)

    with pytest.raises(PostgresMigrationStorageError, match="ledger is absent"):
        PostgresMigrationRunner(lambda: absent, tmp_path).verify()

    drifted = _Connection(migration_rows=[{"migration_id": "0001", "exact_bytes_sha256": "0" * 64}])
    with pytest.raises(PostgresMigrationDriftError, match="exact bytes changed"):
        PostgresMigrationRunner(lambda: drifted, tmp_path).verify()

    matching = _Connection(
        migration_rows=[{"migration_id": "0001", "exact_bytes_sha256": expected.exact_bytes_sha256}]
    )
    assert PostgresMigrationRunner(lambda: matching, tmp_path).verify() == ("0001",)
    path.write_text("SELECT 2;\n", encoding="utf-8", newline="\n")
    with pytest.raises(PostgresMigrationDriftError, match="exact bytes changed"):
        PostgresMigrationRunner(lambda: matching, tmp_path).verify()
