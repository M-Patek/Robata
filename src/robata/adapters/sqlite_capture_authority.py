"""Durable local authority for pre-EOS capture subjects.

Receipt slots are idempotency locators only. They are deliberately absent
from the authority-issued subject and its semantic identity.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import AuthorityBinding, ChannelBinding
from robata.contracts.stream_source import (
    PreEosCaptureSubject,
    create_pre_eos_capture_subject,
)

_EXTENSION_NAME = "local-capture-authority"
_EXTENSION_SCHEMA_VERSION = 1
_EXTENSION_OBJECT_NAMES = frozenset(
    {
        "capture_authority_metadata",
        "capture_authority_receipts",
    }
)
_MAX_SQLITE_INTEGER = 2**63 - 1

_EXTENSION_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE capture_authority_metadata (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        capture_authority_id TEXT NOT NULL,
        capture_authority_epoch INTEGER NOT NULL CHECK (capture_authority_epoch >= 1),
        capture_assignment_policy_version TEXT NOT NULL,
        next_acquisition_sequence INTEGER NOT NULL
            CHECK (next_acquisition_sequence >= 1)
    )
    """,
    """
    CREATE TABLE capture_authority_receipts (
        receipt_slot TEXT PRIMARY KEY,
        acquisition_sequence INTEGER NOT NULL UNIQUE CHECK (acquisition_sequence >= 1),
        request_json BLOB NOT NULL,
        subject_json BLOB NOT NULL,
        capture_scope_digest TEXT NOT NULL UNIQUE
    )
    """,
)


class SQLiteCaptureAuthorityError(RuntimeError):
    """The local capture authority cannot preserve its durable contract."""


class SQLiteCaptureAuthorityConflict(SQLiteCaptureAuthorityError):
    """A receipt slot replay differs from its first authoritative request."""


class SQLiteCaptureAuthorityStorageError(SQLiteCaptureAuthorityError):
    """Persisted capture-authority state is missing, corrupt, or inconsistent."""


class SQLiteLocalCaptureAuthority:
    """Issue durable authority-scoped capture subjects from monotonic receipts."""

    def __init__(
        self,
        authority: SQLiteWorkScheduler,
        *,
        capture_authority_id: str,
        capture_authority_epoch: int,
        capture_assignment_policy_version: str,
    ) -> None:
        if not isinstance(authority, SQLiteWorkScheduler):
            raise TypeError("authority must be SQLiteWorkScheduler")
        self._authority = authority
        self._database_path = authority.database_path
        self._capture_authority_id = _nonempty(
            capture_authority_id,
            "capture_authority_id",
        )
        self._capture_authority_epoch = _positive_int(
            capture_authority_epoch,
            "capture_authority_epoch",
        )
        self._capture_assignment_policy_version = _nonempty(
            capture_assignment_policy_version,
            "capture_assignment_policy_version",
        )
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def issue(
        self,
        receipt_slot: str,
        schema_ref: SchemaRef,
        channel_bindings: Sequence[ChannelBinding],
        mapping_authority: AuthorityBinding,
        clock_authority: AuthorityBinding,
    ) -> PreEosCaptureSubject:
        """Issue once for a slot, or replay its byte-identical persisted subject."""

        checked_slot = _nonempty(receipt_slot, "receipt_slot")
        checked_schema_ref = _require_model(schema_ref, SchemaRef, "schema_ref")
        checked_bindings = tuple(
            _require_model(binding, ChannelBinding, "channel_binding")
            for binding in channel_bindings
        )
        checked_mapping = _require_model(
            mapping_authority,
            AuthorityBinding,
            "mapping_authority",
        )
        checked_clock = _require_model(
            clock_authority,
            AuthorityBinding,
            "clock_authority",
        )
        request_json = canonical_json_bytes(
            {
                "channel_bindings": checked_bindings,
                "clock_authority": checked_clock,
                "mapping_authority": checked_mapping,
                "schema_ref": checked_schema_ref,
            }
        )

        def operation(connection: sqlite3.Connection) -> PreEosCaptureSubject:
            row = connection.execute(
                """
                SELECT acquisition_sequence, request_json, subject_json,
                       capture_scope_digest
                FROM capture_authority_receipts
                WHERE receipt_slot = ?
                """,
                (checked_slot,),
            ).fetchone()
            if row is not None:
                if _row_bytes(row, "request_json") != request_json:
                    raise SQLiteCaptureAuthorityConflict(
                        "capture receipt slot replay changed its authoritative inputs"
                    )
                sequence = _row_positive_int(row, "acquisition_sequence")
                return self._subject_from_receipt(
                    row=row,
                    sequence=sequence,
                    schema_ref=checked_schema_ref,
                    channel_bindings=checked_bindings,
                    mapping_authority=checked_mapping,
                    clock_authority=checked_clock,
                )

            metadata = self._load_metadata(connection)
            sequence = _row_positive_int(metadata, "next_acquisition_sequence")
            if sequence >= _MAX_SQLITE_INTEGER:
                raise SQLiteCaptureAuthorityStorageError(
                    "capture acquisition sequence is exhausted"
                )
            subject = self._create_subject(
                sequence=sequence,
                schema_ref=checked_schema_ref,
                channel_bindings=checked_bindings,
                mapping_authority=checked_mapping,
                clock_authority=checked_clock,
            )
            subject_json = canonical_json_bytes(subject)
            connection.execute(
                """
                INSERT INTO capture_authority_receipts (
                    receipt_slot, acquisition_sequence, request_json,
                    subject_json, capture_scope_digest
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    checked_slot,
                    sequence,
                    sqlite3.Binary(request_json),
                    sqlite3.Binary(subject_json),
                    subject.capture_scope_digest,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE capture_authority_metadata
                SET next_acquisition_sequence = ?
                WHERE singleton = 1 AND next_acquisition_sequence = ?
                """,
                (sequence + 1, sequence),
            )
            if cursor.rowcount != 1:
                raise SQLiteCaptureAuthorityStorageError(
                    "capture acquisition sequence changed inside authority transaction"
                )
            return subject

        return self._run(
            write=True,
            operation_name="issue",
            operation=operation,
        )

    def _create_subject(
        self,
        *,
        sequence: int,
        schema_ref: SchemaRef,
        channel_bindings: tuple[ChannelBinding, ...],
        mapping_authority: AuthorityBinding,
        clock_authority: AuthorityBinding,
    ) -> PreEosCaptureSubject:
        return create_pre_eos_capture_subject(
            schema_ref=schema_ref,
            capture_authority_id=self._capture_authority_id,
            capture_authority_epoch=self._capture_authority_epoch,
            capture_assignment_policy_version=self._capture_assignment_policy_version,
            acquisition_id=f"{self._capture_authority_id}:{sequence}",
            acquisition_epoch=1,
            channel_bindings=channel_bindings,
            mapping_authority=mapping_authority,
            clock_authority=clock_authority,
        )

    def _subject_from_receipt(
        self,
        *,
        row: sqlite3.Row,
        sequence: int,
        schema_ref: SchemaRef,
        channel_bindings: tuple[ChannelBinding, ...],
        mapping_authority: AuthorityBinding,
        clock_authority: AuthorityBinding,
    ) -> PreEosCaptureSubject:
        subject_json = _row_bytes(row, "subject_json")
        expected = self._create_subject(
            sequence=sequence,
            schema_ref=schema_ref,
            channel_bindings=channel_bindings,
            mapping_authority=mapping_authority,
            clock_authority=clock_authority,
        )
        if canonical_json_bytes(expected) != subject_json:
            raise SQLiteCaptureAuthorityStorageError(
                "persisted capture subject differs from its authoritative receipt"
            )
        try:
            subject = PreEosCaptureSubject.model_validate_json(subject_json, strict=True)
        except ValidationError as error:
            raise SQLiteCaptureAuthorityStorageError(
                "persisted capture subject failed strict validation"
            ) from error
        if canonical_json_bytes(subject) != subject_json:
            raise SQLiteCaptureAuthorityStorageError(
                "persisted capture subject is not canonical JSON"
            )
        if _row_text(row, "capture_scope_digest") != subject.capture_scope_digest:
            raise SQLiteCaptureAuthorityStorageError(
                "persisted capture scope digest differs from its subject"
            )
        return subject

    def _initialize_database(self) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_extension_metadata (
                    extension_name TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
                )
                """
            )
            extension = connection.execute(
                """
                SELECT schema_version FROM stream_extension_metadata
                WHERE extension_name = ?
                """,
                (_EXTENSION_NAME,),
            ).fetchone()
            placeholders = ", ".join("?" for _ in _EXTENSION_OBJECT_NAMES)
            rows = connection.execute(
                f"SELECT name FROM sqlite_schema WHERE name IN ({placeholders})",
                tuple(sorted(_EXTENSION_OBJECT_NAMES)),
            ).fetchall()
            existing_objects = {_row_text(row, "name") for row in rows}
            if extension is None:
                if existing_objects:
                    raise SQLiteCaptureAuthorityStorageError(
                        "refusing to adopt unversioned capture-authority tables"
                    )
                for statement in _EXTENSION_SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO stream_extension_metadata (extension_name, schema_version)
                    VALUES (?, ?)
                    """,
                    (_EXTENSION_NAME, _EXTENSION_SCHEMA_VERSION),
                )
                connection.execute(
                    """
                    INSERT INTO capture_authority_metadata (
                        singleton, capture_authority_id, capture_authority_epoch,
                        capture_assignment_policy_version, next_acquisition_sequence
                    ) VALUES (1, ?, ?, ?, 1)
                    """,
                    (
                        self._capture_authority_id,
                        self._capture_authority_epoch,
                        self._capture_assignment_policy_version,
                    ),
                )
                return
            if _row_positive_int(extension, "schema_version") != _EXTENSION_SCHEMA_VERSION:
                raise SQLiteCaptureAuthorityStorageError(
                    "capture-authority extension belongs to another schema version"
                )
            if existing_objects != _EXTENSION_OBJECT_NAMES:
                raise SQLiteCaptureAuthorityStorageError(
                    "capture-authority extension schema inventory changed"
                )
            self._load_metadata(connection)

        self._run(
            write=True,
            operation_name="initialize_extension",
            operation=operation,
        )

    def _load_metadata(self, connection: sqlite3.Connection) -> sqlite3.Row:
        rows = connection.execute("SELECT * FROM capture_authority_metadata").fetchall()
        if len(rows) != 1 or _row_positive_int(rows[0], "singleton") != 1:
            raise SQLiteCaptureAuthorityStorageError(
                "capture-authority metadata singleton is missing or corrupt"
            )
        row = cast(sqlite3.Row, rows[0])
        if (
            _row_text(row, "capture_authority_id") != self._capture_authority_id
            or _row_positive_int(row, "capture_authority_epoch") != self._capture_authority_epoch
            or _row_text(row, "capture_assignment_policy_version")
            != self._capture_assignment_policy_version
        ):
            raise SQLiteCaptureAuthorityConflict(
                "configured capture authority differs from persisted authority metadata"
            )
        _row_positive_int(row, "next_acquisition_sequence")
        return row

    def _run[T](
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], T],
    ) -> T:
        return self._authority.run_authority_transaction(
            write=write,
            operation_name=f"capture_authority.{operation_name}",
            operation=operation,
        )


# Both names are kept public: the longer name emphasizes this is local evidence,
# while the shorter name is convenient at composition sites.
SQLiteCaptureAuthority = SQLiteLocalCaptureAuthority
SQLiteLocalCaptureAuthorityError = SQLiteCaptureAuthorityError
SQLiteLocalCaptureAuthorityConflict = SQLiteCaptureAuthorityConflict
SQLiteLocalCaptureAuthorityStorageError = SQLiteCaptureAuthorityStorageError


def _require_model[T](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    return value


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise SQLiteCaptureAuthorityStorageError(
            f"persisted capture-authority column {column} must be text"
        )
    return value


def _row_positive_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SQLiteCaptureAuthorityStorageError(
            f"persisted capture-authority column {column} must be a positive integer"
        )
    return value


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value: object = row[column]
    if not isinstance(value, bytes):
        raise SQLiteCaptureAuthorityStorageError(
            f"persisted capture-authority column {column} must be bytes"
        )
    return value


__all__ = [
    "SQLiteCaptureAuthority",
    "SQLiteCaptureAuthorityConflict",
    "SQLiteCaptureAuthorityError",
    "SQLiteCaptureAuthorityStorageError",
    "SQLiteLocalCaptureAuthority",
    "SQLiteLocalCaptureAuthorityConflict",
    "SQLiteLocalCaptureAuthorityError",
    "SQLiteLocalCaptureAuthorityStorageError",
]
