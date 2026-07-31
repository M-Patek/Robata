"""PostgreSQL authority for durable pre-EOS capture-subject receipts."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from robata.adapters.postgres_authority import PostgresCanonicalAuthority, PostgresConnection, Row
from robata.adapters.postgres_work_scheduler import PostgresWorkScheduler
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import AuthorityBinding, ChannelBinding
from robata.contracts.stream_source import (
    PreEosCaptureSubject,
    create_pre_eos_capture_subject,
)

_MAX_POSTGRES_BIGINT = 2**63 - 1


class PostgresCaptureAuthorityError(RuntimeError):
    """The PostgreSQL capture authority cannot preserve its durable contract."""


class PostgresCaptureAuthorityConflict(PostgresCaptureAuthorityError):
    """A receipt-slot replay differs from its authoritative first request."""


class PostgresCaptureAuthorityStorageError(PostgresCaptureAuthorityError):
    """Persisted capture state is absent, corrupt, or differs from configuration."""


class PostgresCaptureAuthority:
    """Issue durable capture subjects inside the shared canonical transaction boundary."""

    backend_kind = "POSTGRESQL"

    def __init__(
        self,
        authority: PostgresCanonicalAuthority | PostgresWorkScheduler,
        *,
        capture_authority_id: str,
        capture_authority_epoch: int,
        capture_assignment_policy_version: str,
    ) -> None:
        if isinstance(authority, PostgresWorkScheduler):
            canonical_authority = authority.authority
        elif isinstance(authority, PostgresCanonicalAuthority):
            canonical_authority = authority
        else:
            raise TypeError("authority must be PostgresCanonicalAuthority or PostgresWorkScheduler")
        self._authority = canonical_authority
        self._capture_authority_id = _nonempty(capture_authority_id, "capture_authority_id")
        self._capture_authority_epoch = _positive_int(
            capture_authority_epoch,
            "capture_authority_epoch",
        )
        self._capture_assignment_policy_version = _nonempty(
            capture_assignment_policy_version,
            "capture_assignment_policy_version",
        )

    @property
    def authority(self) -> PostgresCanonicalAuthority:
        return self._authority

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
        checked_mapping = _require_model(mapping_authority, AuthorityBinding, "mapping_authority")
        checked_clock = _require_model(clock_authority, AuthorityBinding, "clock_authority")
        request_json = canonical_json_bytes(
            {
                "channel_bindings": checked_bindings,
                "clock_authority": checked_clock,
                "mapping_authority": checked_mapping,
                "schema_ref": checked_schema_ref,
            }
        )

        def operation(connection: PostgresConnection) -> PreEosCaptureSubject:
            existing = connection.execute(
                """
                SELECT acquisition_sequence, request_json, subject_json, capture_scope_digest
                FROM capture_authority_receipts
                WHERE receipt_slot = %s
                FOR UPDATE
                """,
                (checked_slot,),
            ).fetchone()
            if existing is not None:
                if _row_bytes(existing, "request_json") != request_json:
                    raise PostgresCaptureAuthorityConflict(
                        "capture receipt slot replay changed its authoritative inputs"
                    )
                return self._subject_from_receipt(
                    row=existing,
                    sequence=_row_positive_int(existing, "acquisition_sequence"),
                    schema_ref=checked_schema_ref,
                    channel_bindings=checked_bindings,
                    mapping_authority=checked_mapping,
                    clock_authority=checked_clock,
                )

            self._ensure_metadata(connection)
            metadata = self._load_metadata(connection, for_update=True)
            sequence = _row_positive_int(metadata, "next_acquisition_sequence")
            if sequence >= _MAX_POSTGRES_BIGINT:
                raise PostgresCaptureAuthorityStorageError(
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
                    receipt_slot, acquisition_sequence, request_json, subject_json,
                    capture_scope_digest
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    checked_slot,
                    sequence,
                    request_json,
                    subject_json,
                    subject.capture_scope_digest,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE capture_authority_metadata
                SET next_acquisition_sequence = %s
                WHERE singleton = 1 AND next_acquisition_sequence = %s
                """,
                (sequence + 1, sequence),
            )
            if cursor.rowcount != 1:
                raise PostgresCaptureAuthorityStorageError(
                    "capture acquisition sequence changed inside authority transaction"
                )
            return subject

        return self._authority.run_authority_transaction(
            write=True,
            operation_name="capture_authority.issue",
            operation=operation,
        )

    def _ensure_metadata(self, connection: PostgresConnection) -> None:
        connection.execute(
            """
            INSERT INTO capture_authority_metadata (
                singleton, capture_authority_id, capture_authority_epoch,
                capture_assignment_policy_version, next_acquisition_sequence
            ) VALUES (1, %s, %s, %s, 1)
            ON CONFLICT (tenant_id, singleton) DO NOTHING
            """,
            (
                self._capture_authority_id,
                self._capture_authority_epoch,
                self._capture_assignment_policy_version,
            ),
        )

    def _load_metadata(self, connection: PostgresConnection, *, for_update: bool) -> Row:
        row = connection.execute(
            """
            SELECT * FROM capture_authority_metadata WHERE singleton = 1
            """
            + (" FOR UPDATE" if for_update else ""),
        ).fetchone()
        if row is None or _row_positive_int(row, "singleton") != 1:
            raise PostgresCaptureAuthorityStorageError(
                "capture-authority metadata singleton is missing or corrupt"
            )
        if (
            _row_text(row, "capture_authority_id") != self._capture_authority_id
            or _row_positive_int(row, "capture_authority_epoch") != self._capture_authority_epoch
            or _row_text(row, "capture_assignment_policy_version")
            != self._capture_assignment_policy_version
        ):
            raise PostgresCaptureAuthorityConflict(
                "configured capture authority differs from persisted authority metadata"
            )
        _row_positive_int(row, "next_acquisition_sequence")
        return row

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
        row: Row,
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
            raise PostgresCaptureAuthorityStorageError(
                "persisted capture subject differs from its authoritative receipt"
            )
        try:
            subject = PreEosCaptureSubject.model_validate_json(subject_json, strict=True)
        except ValidationError as error:
            raise PostgresCaptureAuthorityStorageError(
                "persisted capture subject failed strict validation"
            ) from error
        if canonical_json_bytes(subject) != subject_json:
            raise PostgresCaptureAuthorityStorageError(
                "persisted capture subject is not canonical JSON"
            )
        if _row_text(row, "capture_scope_digest") != subject.capture_scope_digest:
            raise PostgresCaptureAuthorityStorageError(
                "persisted capture scope digest differs from its subject"
            )
        return subject


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


def _row_text(row: Row, column: str) -> str:
    value = row[column]
    if not isinstance(value, str):
        raise PostgresCaptureAuthorityStorageError(
            f"persisted capture-authority column {column} must be text"
        )
    return value


def _row_positive_int(row: Row, column: str) -> int:
    value = row[column]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PostgresCaptureAuthorityStorageError(
            f"persisted capture-authority column {column} must be a positive integer"
        )
    return value


def _row_bytes(row: Row, column: str) -> bytes:
    value = row[column]
    if not isinstance(value, bytes):
        raise PostgresCaptureAuthorityStorageError(
            f"persisted capture-authority column {column} must be bytes"
        )
    return value


__all__ = [
    "PostgresCaptureAuthority",
    "PostgresCaptureAuthorityConflict",
    "PostgresCaptureAuthorityError",
    "PostgresCaptureAuthorityStorageError",
]
