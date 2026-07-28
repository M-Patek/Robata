"""SQLite persistence for sealed adaptive-sampling decisions.

The store is intentionally narrow.  It owns the exact canonical bytes of a
sealed decision and append-only late-feedback audits; it does not run media
work, reopen a decision, or invoke a provider.  This makes restart/replay a
read of the frozen decision rather than a new adaptive-planning attempt.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.sampling.adaptive_decision import (
    AdaptiveLateFeedbackAudit,
    AdaptiveSamplingDecision,
)

_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 5_000
_MAX_SQLITE_INTEGER = 2**63 - 1

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE adaptive_sampling_decision_store_metadata (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
    )
    """,
    """
    CREATE TABLE adaptive_sampling_decisions (
        decision_scope_sha256 TEXT PRIMARY KEY,
        decision_id TEXT NOT NULL UNIQUE,
        semantic_sha256 TEXT NOT NULL UNIQUE,
        payload_json BLOB NOT NULL,
        exact_bytes_sha256 TEXT NOT NULL,
        sealed_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE adaptive_sampling_late_feedback_audits (
        audit_id TEXT PRIMARY KEY,
        decision_scope_sha256 TEXT NOT NULL
            REFERENCES adaptive_sampling_decisions(decision_scope_sha256)
            ON DELETE RESTRICT,
        arrival_id TEXT NOT NULL,
        semantic_sha256 TEXT NOT NULL UNIQUE,
        payload_json BLOB NOT NULL,
        exact_bytes_sha256 TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        UNIQUE (decision_scope_sha256, arrival_id)
    )
    """,
    """
    CREATE TRIGGER adaptive_sampling_decisions_no_update
    BEFORE UPDATE ON adaptive_sampling_decisions
    BEGIN
        SELECT RAISE(ABORT, 'adaptive sampling decisions are immutable');
    END
    """,
    """
    CREATE TRIGGER adaptive_sampling_decisions_no_delete
    BEFORE DELETE ON adaptive_sampling_decisions
    BEGIN
        SELECT RAISE(ABORT, 'adaptive sampling decisions are immutable');
    END
    """,
    """
    CREATE TRIGGER adaptive_sampling_late_feedback_audits_no_update
    BEFORE UPDATE ON adaptive_sampling_late_feedback_audits
    BEGIN
        SELECT RAISE(ABORT, 'adaptive sampling late-feedback audits are immutable');
    END
    """,
    """
    CREATE TRIGGER adaptive_sampling_late_feedback_audits_no_delete
    BEFORE DELETE ON adaptive_sampling_late_feedback_audits
    BEGIN
        SELECT RAISE(ABORT, 'adaptive sampling late-feedback audits are immutable');
    END
    """,
    """
    CREATE INDEX adaptive_sampling_late_feedback_scope_idx
    ON adaptive_sampling_late_feedback_audits(decision_scope_sha256, recorded_at, audit_id)
    """,
)

T = TypeVar("T")


class AdaptiveDecisionStoreError(RuntimeError):
    """Base error for durable adaptive-decision persistence."""


class AdaptiveDecisionStoreConflict(AdaptiveDecisionStoreError):
    """A sealed decision slot or late-arrival receipt changed its exact bytes."""


class AdaptiveDecisionStoreStorageError(AdaptiveDecisionStoreError):
    """Persisted adaptive-decision state is corrupt or cannot be opened."""


class AdaptiveDecisionNotFoundError(AdaptiveDecisionStoreError):
    """A late-feedback audit named a decision slot that has not been sealed."""


@dataclass(frozen=True)
class StoredAdaptiveSamplingDecision:
    """The authoritative bytes and replay flag returned by :meth:`put_or_get`."""

    decision: AdaptiveSamplingDecision
    exact_bytes: bytes
    exact_bytes_sha256: str
    sealed_at: str
    replayed: bool


@dataclass(frozen=True)
class StoredAdaptiveLateFeedbackAudit:
    """The authoritative bytes and replay flag for one late-feedback receipt."""

    audit: AdaptiveLateFeedbackAudit
    exact_bytes: bytes
    exact_bytes_sha256: str
    recorded_at: str
    replayed: bool


class SQLiteAdaptiveDecisionStore:
    """Append-only local authority for one frozen adaptive decision per scope."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        synchronous: Literal["FULL", "NORMAL"] = "FULL",
    ) -> None:
        self._database_path = Path(database_path)
        if self._database_path.exists() and self._database_path.is_dir():
            raise AdaptiveDecisionStoreStorageError(
                "adaptive-decision store path must identify a file"
            )
        if not self._database_path.parent.exists():
            raise AdaptiveDecisionStoreStorageError(
                "adaptive-decision store parent directory does not exist"
            )
        if synchronous not in {"FULL", "NORMAL"}:
            raise ValueError("synchronous must be FULL or NORMAL")
        self._synchronous = synchronous
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the database that owns the sealed decision bytes."""

        return self._database_path

    def put_or_get(
        self,
        decision: AdaptiveSamplingDecision,
        *,
        sealed_at: str | None = None,
    ) -> StoredAdaptiveSamplingDecision:
        """Seal once or return the exact canonical bytes persisted previously.

        A different decision for the same trigger-independent logical scope is
        a conflict.  In particular, this method never updates policy, targets,
        or trigger provenance after sealing.
        """

        checked = _require_model(decision, AdaptiveSamplingDecision, "decision")
        payload = canonical_json_bytes(checked)
        payload_digest = exact_bytes_sha256(payload)
        requested_sealed_at = _timestamp_or_now(sealed_at, "sealed_at")

        def operation(connection: sqlite3.Connection) -> StoredAdaptiveSamplingDecision:
            row = connection.execute(
                """
                SELECT * FROM adaptive_sampling_decisions
                WHERE decision_scope_sha256 = ?
                """,
                (checked.decision_scope_sha256,),
            ).fetchone()
            if row is not None:
                stored = self._stored_decision_from_row(row, replayed=True)
                if stored.exact_bytes != payload:
                    raise AdaptiveDecisionStoreConflict(
                        "sealed adaptive decision scope received different exact bytes"
                    )
                return stored

            id_row = connection.execute(
                """
                SELECT * FROM adaptive_sampling_decisions WHERE decision_id = ?
                """,
                (checked.decision_id,),
            ).fetchone()
            if id_row is not None:
                stored = self._stored_decision_from_row(id_row, replayed=True)
                if stored.exact_bytes != payload:
                    raise AdaptiveDecisionStoreConflict(
                        "adaptive decision ID collision has different exact bytes"
                    )
                raise AdaptiveDecisionStoreStorageError(
                    "adaptive decision ID is stored under an inconsistent decision scope"
                )

            connection.execute(
                """
                INSERT INTO adaptive_sampling_decisions (
                    decision_scope_sha256, decision_id, semantic_sha256,
                    payload_json, exact_bytes_sha256, sealed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.decision_scope_sha256,
                    checked.decision_id,
                    checked.semantic_sha256,
                    sqlite3.Binary(payload),
                    payload_digest,
                    requested_sealed_at,
                ),
            )
            return StoredAdaptiveSamplingDecision(
                decision=checked,
                exact_bytes=payload,
                exact_bytes_sha256=payload_digest,
                sealed_at=requested_sealed_at,
                replayed=False,
            )

        return self._transaction(write=True, operation_name="put_or_get", operation=operation)

    def get(
        self,
        decision_scope_sha256: str,
    ) -> StoredAdaptiveSamplingDecision | None:
        """Load the authoritative sealed decision for one logical scope."""

        checked_scope = _sha256_text(decision_scope_sha256, "decision_scope_sha256")

        def operation(connection: sqlite3.Connection) -> StoredAdaptiveSamplingDecision | None:
            row = connection.execute(
                """
                SELECT * FROM adaptive_sampling_decisions
                WHERE decision_scope_sha256 = ?
                """,
                (checked_scope,),
            ).fetchone()
            if row is None:
                return None
            return self._stored_decision_from_row(row, replayed=True)

        return self._transaction(write=False, operation_name="get", operation=operation)

    def get_by_decision_id(
        self,
        decision_id: str,
    ) -> StoredAdaptiveSamplingDecision | None:
        """Load a decision by its content-addressed ID."""

        checked_id = _nonempty(decision_id, "decision_id")

        def operation(connection: sqlite3.Connection) -> StoredAdaptiveSamplingDecision | None:
            row = connection.execute(
                "SELECT * FROM adaptive_sampling_decisions WHERE decision_id = ?",
                (checked_id,),
            ).fetchone()
            if row is None:
                return None
            return self._stored_decision_from_row(row, replayed=True)

        return self._transaction(
            write=False, operation_name="get_by_decision_id", operation=operation
        )

    def record_late_feedback(
        self,
        audit: AdaptiveLateFeedbackAudit,
        *,
        recorded_at: str | None = None,
    ) -> StoredAdaptiveLateFeedbackAudit:
        """Append an arrival audit without changing the sealed decision."""

        checked = _require_model(audit, AdaptiveLateFeedbackAudit, "audit")
        payload = canonical_json_bytes(checked)
        payload_digest = exact_bytes_sha256(payload)
        requested_recorded_at = _timestamp_or_now(recorded_at, "recorded_at")

        def operation(connection: sqlite3.Connection) -> StoredAdaptiveLateFeedbackAudit:
            sealed = connection.execute(
                """
                SELECT decision_id FROM adaptive_sampling_decisions
                WHERE decision_scope_sha256 = ?
                """,
                (checked.decision_scope_sha256,),
            ).fetchone()
            if sealed is None:
                raise AdaptiveDecisionNotFoundError(
                    "cannot audit late feedback for an unsealed adaptive decision scope"
                )

            row = connection.execute(
                """
                SELECT * FROM adaptive_sampling_late_feedback_audits
                WHERE decision_scope_sha256 = ? AND arrival_id = ?
                """,
                (checked.decision_scope_sha256, checked.arrival_id),
            ).fetchone()
            if row is not None:
                stored = self._stored_audit_from_row(row, replayed=True)
                if stored.exact_bytes != payload:
                    raise AdaptiveDecisionStoreConflict(
                        "late-feedback arrival receipt received different exact bytes"
                    )
                return stored

            id_row = connection.execute(
                """
                SELECT * FROM adaptive_sampling_late_feedback_audits WHERE audit_id = ?
                """,
                (checked.audit_id,),
            ).fetchone()
            if id_row is not None:
                stored = self._stored_audit_from_row(id_row, replayed=True)
                if stored.exact_bytes != payload:
                    raise AdaptiveDecisionStoreConflict(
                        "late-feedback audit ID collision has different exact bytes"
                    )
                raise AdaptiveDecisionStoreStorageError(
                    "late-feedback audit ID is stored under an inconsistent arrival receipt"
                )

            connection.execute(
                """
                INSERT INTO adaptive_sampling_late_feedback_audits (
                    audit_id, decision_scope_sha256, arrival_id, semantic_sha256,
                    payload_json, exact_bytes_sha256, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.audit_id,
                    checked.decision_scope_sha256,
                    checked.arrival_id,
                    checked.semantic_sha256,
                    sqlite3.Binary(payload),
                    payload_digest,
                    requested_recorded_at,
                ),
            )
            return StoredAdaptiveLateFeedbackAudit(
                audit=checked,
                exact_bytes=payload,
                exact_bytes_sha256=payload_digest,
                recorded_at=requested_recorded_at,
                replayed=False,
            )

        return self._transaction(
            write=True,
            operation_name="record_late_feedback",
            operation=operation,
        )

    def list_late_feedback(
        self,
        decision_scope_sha256: str,
    ) -> tuple[StoredAdaptiveLateFeedbackAudit, ...]:
        """Return all append-only late-feedback receipts in storage order."""

        checked_scope = _sha256_text(decision_scope_sha256, "decision_scope_sha256")

        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[StoredAdaptiveLateFeedbackAudit, ...]:
            rows = connection.execute(
                """
                SELECT * FROM adaptive_sampling_late_feedback_audits
                WHERE decision_scope_sha256 = ?
                ORDER BY recorded_at, audit_id
                """,
                (checked_scope,),
            ).fetchall()
            return tuple(self._stored_audit_from_row(row, replayed=True) for row in rows)

        return self._transaction(
            write=False,
            operation_name="list_late_feedback",
            operation=operation,
        )

    def _initialize_database(self) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            metadata = connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name = 'adaptive_sampling_decision_store_metadata'
                """
            ).fetchone()
            if metadata is None:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO adaptive_sampling_decision_store_metadata (
                        singleton, schema_version
                    )
                    VALUES (1, ?)
                    """,
                    (_SCHEMA_VERSION,),
                )
                return

            row = connection.execute(
                "SELECT singleton, schema_version FROM adaptive_sampling_decision_store_metadata"
            ).fetchall()
            if len(row) != 1 or _row_int(row[0], "singleton") != 1:
                raise AdaptiveDecisionStoreStorageError(
                    "adaptive-decision store metadata singleton is missing or corrupt"
                )
            if _row_int(row[0], "schema_version") != _SCHEMA_VERSION:
                raise AdaptiveDecisionStoreStorageError(
                    "adaptive-decision store belongs to another schema version"
                )
            expected = {
                "adaptive_sampling_decision_store_metadata",
                "adaptive_sampling_decisions",
                "adaptive_sampling_late_feedback_audits",
                "adaptive_sampling_decisions_no_update",
                "adaptive_sampling_decisions_no_delete",
                "adaptive_sampling_late_feedback_audits_no_update",
                "adaptive_sampling_late_feedback_audits_no_delete",
                "adaptive_sampling_late_feedback_scope_idx",
            }
            rows = connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE name LIKE 'adaptive_sampling_%'
                """
            ).fetchall()
            actual = {_row_text(item, "name") for item in rows}
            if actual != expected:
                raise AdaptiveDecisionStoreStorageError(
                    "adaptive-decision store schema inventory changed"
                )

        self._transaction(write=True, operation_name="initialize", operation=operation)

    def _stored_decision_from_row(
        self,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> StoredAdaptiveSamplingDecision:
        payload = _row_bytes(row, "payload_json")
        stored_digest = _sha256_text(_row_text(row, "exact_bytes_sha256"), "exact_bytes_sha256")
        if exact_bytes_sha256(payload) != stored_digest:
            raise AdaptiveDecisionStoreStorageError(
                "adaptive decision exact-byte digest is inconsistent"
            )
        try:
            decision = AdaptiveSamplingDecision.model_validate_json(payload, strict=True)
        except ValidationError as error:
            raise AdaptiveDecisionStoreStorageError(
                "persisted adaptive decision failed strict validation"
            ) from error
        if canonical_json_bytes(decision) != payload:
            raise AdaptiveDecisionStoreStorageError(
                "persisted adaptive decision is not canonical JSON"
            )
        if (
            _row_text(row, "decision_scope_sha256") != decision.decision_scope_sha256
            or _row_text(row, "decision_id") != decision.decision_id
            or _row_text(row, "semantic_sha256") != decision.semantic_sha256
        ):
            raise AdaptiveDecisionStoreStorageError(
                "persisted adaptive decision index columns are inconsistent"
            )
        return StoredAdaptiveSamplingDecision(
            decision=decision,
            exact_bytes=payload,
            exact_bytes_sha256=stored_digest,
            sealed_at=_timestamp_or_error(_row_text(row, "sealed_at"), "sealed_at"),
            replayed=replayed,
        )

    def _stored_audit_from_row(
        self,
        row: sqlite3.Row,
        *,
        replayed: bool,
    ) -> StoredAdaptiveLateFeedbackAudit:
        payload = _row_bytes(row, "payload_json")
        stored_digest = _sha256_text(_row_text(row, "exact_bytes_sha256"), "exact_bytes_sha256")
        if exact_bytes_sha256(payload) != stored_digest:
            raise AdaptiveDecisionStoreStorageError(
                "late-feedback audit exact-byte digest is inconsistent"
            )
        try:
            audit = AdaptiveLateFeedbackAudit.model_validate_json(payload, strict=True)
        except ValidationError as error:
            raise AdaptiveDecisionStoreStorageError(
                "persisted late-feedback audit failed strict validation"
            ) from error
        if canonical_json_bytes(audit) != payload:
            raise AdaptiveDecisionStoreStorageError(
                "persisted late-feedback audit is not canonical JSON"
            )
        if (
            _row_text(row, "audit_id") != audit.audit_id
            or _row_text(row, "decision_scope_sha256") != audit.decision_scope_sha256
            or _row_text(row, "arrival_id") != audit.arrival_id
            or _row_text(row, "semantic_sha256") != audit.semantic_sha256
        ):
            raise AdaptiveDecisionStoreStorageError(
                "persisted late-feedback audit index columns are inconsistent"
            )
        return StoredAdaptiveLateFeedbackAudit(
            audit=audit,
            exact_bytes=payload,
            exact_bytes_sha256=stored_digest,
            recorded_at=_timestamp_or_error(_row_text(row, "recorded_at"), "recorded_at"),
            replayed=replayed,
        )

    def _transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], T],
    ) -> T:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            result = operation(connection)
            connection.execute("COMMIT")
            return result
        except AdaptiveDecisionStoreError:
            if connection is not None:
                self._rollback(connection)
            raise
        except (OSError, sqlite3.Error) as error:
            if connection is not None:
                self._rollback(connection)
            raise AdaptiveDecisionStoreStorageError(
                f"adaptive-decision store {operation_name} failed"
            ) from error
        except Exception:
            if connection is not None:
                self._rollback(connection)
            raise
        finally:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self._database_path,
                isolation_level=None,
                timeout=_BUSY_TIMEOUT_MS / 1_000,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(f"PRAGMA synchronous = {self._synchronous}")
            return connection
        except (OSError, sqlite3.Error) as error:
            with suppress(UnboundLocalError, sqlite3.Error):
                connection.close()
            raise AdaptiveDecisionStoreStorageError(
                "adaptive-decision store cannot open SQLite database"
            ) from error

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        with suppress(sqlite3.Error):
            connection.execute("ROLLBACK")


def _require_model[T](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256_text(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdaptiveDecisionStoreStorageError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _timestamp_or_now(value: str | None, label: str) -> str:
    if value is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return _timestamp_or_error(value, label)


def _timestamp_or_error(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise AdaptiveDecisionStoreStorageError(f"{label} must be an RFC3339 timestamp")
    try:
        # Let Pydantic own the contract's exact lexical shape; this parse additionally
        # rejects impossible dates and timezone-free values before a row is trusted.
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise AdaptiveDecisionStoreStorageError(
            f"{label} must be a valid RFC3339 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdaptiveDecisionStoreStorageError(f"{label} must include a timezone")
    return value


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise AdaptiveDecisionStoreStorageError(
            f"persisted adaptive-decision column {column} must be text"
        )
    return value


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if not isinstance(value, bytes):
        raise AdaptiveDecisionStoreStorageError(
            f"persisted adaptive-decision column {column} must be bytes"
        )
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdaptiveDecisionStoreStorageError(
            f"persisted adaptive-decision column {column} must be a positive integer"
        )
    if value > _MAX_SQLITE_INTEGER:
        raise AdaptiveDecisionStoreStorageError(
            f"persisted adaptive-decision column {column} exceeds SQLite integer range"
        )
    return value


__all__ = [
    "AdaptiveDecisionNotFoundError",
    "AdaptiveDecisionStoreConflict",
    "AdaptiveDecisionStoreError",
    "AdaptiveDecisionStoreStorageError",
    "SQLiteAdaptiveDecisionStore",
    "StoredAdaptiveLateFeedbackAudit",
    "StoredAdaptiveSamplingDecision",
]
