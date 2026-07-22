"""SQLite delivery ledger and idempotent sink for primary-completion outbox rows."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Final, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from robata.adapters.sqlite_primary_completion import (
    SQLitePrimaryCompletionRepository,
    _primary_schema_is_current,
)
from robata.application.canonical.primary_completion import PrimaryCompletionError
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    default_schema_registry,
)
from robata.event_pipeline.identity_registry import (
    EventIdentityOutboxWireRecord,
    validate_registered_event_identity_outbox_wire_record,
)
from robata.queue.outbox import (
    Clock,
    OutboxDeliveryClaim,
    OutboxDeliveryError,
    OutboxDeliverySnapshot,
    OutboxDeliveryStatus,
    OutboxFenceError,
    OutboxMessage,
    OutboxRetryPolicy,
)

_PRIMARY_APPLICATION_ID: Final = 0x52504341
_PRIMARY_SCHEMA_VERSION: Final = 2
_SINK_APPLICATION_ID: Final = 0x524F4258  # "ROBX"
_SINK_SCHEMA_VERSION: Final = 1
_BUSY_TIMEOUT_MS: Final = 30_000

_SINK_SCHEMA_STATEMENTS: Final = (
    """
    CREATE TABLE delivered_outbox_messages (
        outbox_id TEXT PRIMARY KEY,
        topic TEXT NOT NULL,
        message_key TEXT NOT NULL,
        payload BLOB NOT NULL,
        payload_sha256 TEXT NOT NULL,
        published_at TEXT NOT NULL
    )
    """,
    """
    CREATE TRIGGER delivered_outbox_messages_no_update
    BEFORE UPDATE ON delivered_outbox_messages
    BEGIN
        SELECT RAISE(ABORT, 'delivered outbox messages are append-only');
    END
    """,
    """
    CREATE TRIGGER delivered_outbox_messages_no_delete
    BEFORE DELETE ON delivered_outbox_messages
    BEGIN
        SELECT RAISE(ABORT, 'delivered outbox messages are append-only');
    END
    """,
)


def _sink_schema_fingerprint(connection: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    return tuple(tuple(row) for row in rows)


@cache
def _expected_sink_schema_fingerprint() -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for statement in _SINK_SCHEMA_STATEMENTS:
            connection.execute(statement)
        return _sink_schema_fingerprint(connection)
    finally:
        connection.close()


def _sink_schema_is_current(connection: sqlite3.Connection) -> bool:
    return _sink_schema_fingerprint(connection) == _expected_sink_schema_fingerprint()


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _require_now(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime):
        raise TypeError("clock must return datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_lease_duration(value: timedelta) -> None:
    if not isinstance(value, timedelta) or value <= timedelta(0):
        raise ValueError("lease_duration must be positive")


def _fencing_token(outbox_id: str, lease_epoch: int) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"robata:primary-outbox-delivery:{outbox_id}:{lease_epoch}",
        )
    )


class SQLitePrimaryOutboxDeliveryStore:
    """Durable lease, retry, DLQ, and acknowledgement state in the primary DB."""

    def __init__(
        self,
        primary_database_path: Path,
        *,
        retry_policy: OutboxRetryPolicy,
        clock: Clock | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None:
        if not isinstance(primary_database_path, Path):
            raise TypeError("primary_database_path must be pathlib.Path")
        if not isinstance(retry_policy, OutboxRetryPolicy):
            raise TypeError("retry_policy must be OutboxRetryPolicy")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._path = primary_database_path.resolve()
        if not self._path.is_file() or self._path.stat().st_size == 0:
            raise OutboxDeliveryError(
                "primary completion database must already exist before starting its relay"
            )
        self._retry_policy = retry_policy
        self._clock = clock if clock is not None else _default_clock
        try:
            self._registry = registry or default_schema_registry()
            SQLitePrimaryCompletionRepository(self._path, registry=self._registry)
        except (PrimaryCompletionError, SchemaRegistryError) as error:
            raise OutboxDeliveryError(
                f"cannot resolve outbox relay schema governance: {error}"
            ) from error
        connection = self._connect()
        connection.close()

    @property
    def path(self) -> Path:
        return self._path

    def claim(
        self,
        *,
        worker_id: str,
        lease_duration: timedelta,
    ) -> OutboxDeliveryClaim | None:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a nonempty string")
        _validate_lease_duration(lease_duration)
        now = _require_now(self._clock)
        now_text = _rfc3339(now)
        expires_at = _rfc3339(now + lease_duration)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._discover(connection, now_text)
            self._expire_leases(connection, now_text)
            row = connection.execute(
                """
                SELECT d.outbox_id, d.attempt_count, d.lease_epoch
                FROM primary_outbox_deliveries AS d
                JOIN primary_outbox AS o ON o.outbox_id = d.outbox_id
                WHERE o.delivered_at IS NULL
                  AND d.status IN ('PENDING', 'RETRY_WAIT')
                  AND d.next_attempt_at <= ?
                ORDER BY o.completion_run_id, o.outbox_ordinal, o.outbox_id
                LIMIT 1
                """,
                (now_text,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            outbox_id = str(row["outbox_id"])
            lease_epoch = int(row["lease_epoch"]) + 1
            attempt_count = int(row["attempt_count"]) + 1
            token = _fencing_token(outbox_id, lease_epoch)
            cursor = connection.execute(
                """
                UPDATE primary_outbox_deliveries
                SET status = 'LEASED',
                    attempt_count = ?,
                    lease_epoch = ?,
                    fencing_token = ?,
                    claimed_by = ?,
                    lease_expires_at = ?,
                    last_error = NULL
                WHERE outbox_id = ?
                  AND status IN ('PENDING', 'RETRY_WAIT')
                  AND lease_epoch = ?
                """,
                (
                    attempt_count,
                    lease_epoch,
                    token,
                    worker_id,
                    expires_at,
                    outbox_id,
                    int(row["lease_epoch"]),
                ),
            )
            if cursor.rowcount != 1:
                raise OutboxDeliveryError("outbox claim lost its atomic state transition")
            claimed = self._joined_row(connection, outbox_id)
            claim = OutboxDeliveryClaim(
                message=self._message_from_row(claimed),
                delivery=self._snapshot_from_row(claimed),
            )
            connection.commit()
            return claim
        except OutboxDeliveryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except (sqlite3.Error, ValidationError, TypeError, ValueError) as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise OutboxDeliveryError(f"cannot claim primary outbox row: {error}") from error
        finally:
            connection.close()

    def acknowledge(self, claim: OutboxDeliveryClaim) -> OutboxDeliverySnapshot:
        checked = self._require_claim(claim)
        now_text = _rfc3339(_require_now(self._clock))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_leases(connection, now_text)
            row = self._joined_row(connection, checked.message.outbox_id)
            if (
                str(row["delivery_status"]) == OutboxDeliveryStatus.DELIVERED
                and str(row["fencing_token"]) == checked.delivery.fencing_token
            ):
                connection.commit()
                return self._snapshot_from_row(row)
            self._require_current_fence(row, checked)
            cursor = connection.execute(
                """
                UPDATE primary_outbox_deliveries
                SET status = 'DELIVERED',
                    lease_expires_at = NULL,
                    next_attempt_at = ?,
                    delivered_at = ?,
                    dead_lettered_at = NULL,
                    last_error = NULL
                WHERE outbox_id = ?
                  AND status = 'LEASED'
                  AND lease_epoch = ?
                  AND fencing_token = ?
                  AND claimed_by = ?
                """,
                (
                    now_text,
                    now_text,
                    checked.message.outbox_id,
                    checked.delivery.lease_epoch,
                    checked.delivery.fencing_token,
                    checked.delivery.claimed_by,
                ),
            )
            if cursor.rowcount != 1:
                raise OutboxFenceError("delivery acknowledgement fence is stale")
            cursor = connection.execute(
                """
                UPDATE primary_outbox
                SET delivered_at = ?
                WHERE outbox_id = ? AND delivered_at IS NULL
                """,
                (now_text, checked.message.outbox_id),
            )
            if cursor.rowcount != 1:
                raise OutboxDeliveryError("primary outbox acknowledgement is inconsistent")
            result = self._snapshot_from_row(
                self._joined_row(connection, checked.message.outbox_id)
            )
            connection.commit()
            return result
        except OutboxDeliveryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except (sqlite3.Error, TypeError, ValueError) as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise OutboxDeliveryError(
                f"cannot acknowledge primary outbox delivery: {error}"
            ) from error
        finally:
            connection.close()

    def record_failure(
        self,
        claim: OutboxDeliveryClaim,
        error: str,
    ) -> OutboxDeliverySnapshot:
        checked = self._require_claim(claim)
        if not isinstance(error, str) or not error:
            raise ValueError("error must be a nonempty string")
        error = error[:1000]
        now = _require_now(self._clock)
        now_text = _rfc3339(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire_leases(connection, now_text)
            row = self._joined_row(connection, checked.message.outbox_id)
            status = OutboxDeliveryStatus(str(row["delivery_status"]))
            if (
                status in (OutboxDeliveryStatus.RETRY_WAIT, OutboxDeliveryStatus.DEAD_LETTER)
                and str(row["fencing_token"]) == checked.delivery.fencing_token
                and str(row["last_error"]) == error
            ):
                connection.commit()
                return self._snapshot_from_row(row)
            self._require_current_fence(row, checked)
            attempt_count = int(row["attempt_count"])
            max_attempts = int(row["max_attempts"])
            if attempt_count >= max_attempts:
                next_status = OutboxDeliveryStatus.DEAD_LETTER
                next_attempt_at = now_text
                dead_lettered_at: str | None = now_text
            else:
                policy = self._policy_from_row(row)
                next_status = OutboxDeliveryStatus.RETRY_WAIT
                next_attempt_at = _rfc3339(now + policy.delay_after(attempt_count))
                dead_lettered_at = None
            cursor = connection.execute(
                """
                UPDATE primary_outbox_deliveries
                SET status = ?,
                    lease_expires_at = NULL,
                    next_attempt_at = ?,
                    last_error = ?,
                    dead_lettered_at = ?
                WHERE outbox_id = ?
                  AND status = 'LEASED'
                  AND lease_epoch = ?
                  AND fencing_token = ?
                  AND claimed_by = ?
                """,
                (
                    next_status,
                    next_attempt_at,
                    error,
                    dead_lettered_at,
                    checked.message.outbox_id,
                    checked.delivery.lease_epoch,
                    checked.delivery.fencing_token,
                    checked.delivery.claimed_by,
                ),
            )
            if cursor.rowcount != 1:
                raise OutboxFenceError("delivery failure fence is stale")
            result = self._snapshot_from_row(
                self._joined_row(connection, checked.message.outbox_id)
            )
            connection.commit()
            return result
        except OutboxDeliveryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except (sqlite3.Error, TypeError, ValueError) as caught:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise OutboxDeliveryError(
                f"cannot record primary outbox delivery failure: {caught}"
            ) from caught
        finally:
            connection.close()

    def get(self, outbox_id: str) -> OutboxDeliverySnapshot | None:
        if not isinstance(outbox_id, str) or not outbox_id:
            raise ValueError("outbox_id must be a nonempty string")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT d.*, d.status AS delivery_status
                FROM primary_outbox_deliveries AS d
                WHERE d.outbox_id = ?
                """,
                (outbox_id,),
            ).fetchone()
            return None if row is None else self._snapshot_from_row(row)
        except (sqlite3.Error, TypeError, ValueError) as error:
            raise OutboxDeliveryError(f"cannot read outbox delivery state: {error}") from error
        finally:
            connection.close()

    def list_dead_letters(self, *, limit: int = 100) -> tuple[OutboxDeliverySnapshot, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT d.*, d.status AS delivery_status
                FROM primary_outbox_deliveries AS d
                JOIN primary_outbox AS o ON o.outbox_id = d.outbox_id
                WHERE d.status = 'DEAD_LETTER'
                ORDER BY o.completion_run_id, o.outbox_ordinal, o.outbox_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(self._snapshot_from_row(row) for row in rows)
        except (sqlite3.Error, TypeError, ValueError) as error:
            raise OutboxDeliveryError(f"cannot list outbox dead letters: {error}") from error
        finally:
            connection.close()

    def _discover(self, connection: sqlite3.Connection, now_text: str) -> None:
        policy = self._retry_policy
        connection.execute(
            """
            INSERT INTO primary_outbox_deliveries (
                outbox_id, status, attempt_count, lease_epoch, fencing_token,
                claimed_by, lease_expires_at, next_attempt_at,
                retry_policy_version, max_attempts, base_delay_seconds,
                max_delay_seconds, last_error, delivered_at, dead_lettered_at
            )
            SELECT
                o.outbox_id,
                CASE WHEN o.delivered_at IS NULL THEN 'PENDING' ELSE 'DELIVERED' END,
                0, 0, NULL, NULL, NULL, COALESCE(o.delivered_at, ?),
                ?, ?, ?, ?, NULL, o.delivered_at, NULL
            FROM primary_outbox AS o
            WHERE NOT EXISTS (
                SELECT 1 FROM primary_outbox_deliveries AS d
                WHERE d.outbox_id = o.outbox_id
            )
            """,
            (
                now_text,
                policy.version,
                policy.max_attempts,
                float(policy.base_delay_seconds),
                float(policy.max_delay_seconds),
            ),
        )

    def _expire_leases(self, connection: sqlite3.Connection, now_text: str) -> None:
        connection.execute(
            """
            UPDATE primary_outbox_deliveries
            SET status = 'DEAD_LETTER',
                lease_expires_at = NULL,
                next_attempt_at = ?,
                last_error = 'delivery lease expired after final attempt',
                dead_lettered_at = ?
            WHERE status = 'LEASED'
              AND lease_expires_at <= ?
              AND attempt_count >= max_attempts
            """,
            (now_text, now_text, now_text),
        )
        connection.execute(
            """
            UPDATE primary_outbox_deliveries
            SET status = 'RETRY_WAIT',
                lease_expires_at = NULL,
                next_attempt_at = ?,
                last_error = 'delivery lease expired before acknowledgement'
            WHERE status = 'LEASED'
              AND lease_expires_at <= ?
              AND attempt_count < max_attempts
            """,
            (now_text, now_text),
        )

    def _require_claim(self, claim: object) -> OutboxDeliveryClaim:
        if not isinstance(claim, OutboxDeliveryClaim):
            raise TypeError("claim must be OutboxDeliveryClaim")
        return claim

    def _require_current_fence(
        self,
        row: sqlite3.Row,
        claim: OutboxDeliveryClaim,
    ) -> None:
        if (
            str(row["delivery_status"]) != OutboxDeliveryStatus.LEASED
            or int(row["lease_epoch"]) != claim.delivery.lease_epoch
            or row["fencing_token"] != claim.delivery.fencing_token
            or row["claimed_by"] != claim.delivery.claimed_by
        ):
            raise OutboxFenceError("outbox delivery claim is stale")

    def _joined_row(self, connection: sqlite3.Connection, outbox_id: str) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT
                o.outbox_id AS source_outbox_id,
                o.completion_run_id,
                o.recording_identity,
                o.outbox_ordinal,
                o.assignment_logical_key,
                o.payload_json,
                o.payload_json_sha256,
                o.delivered_at AS source_delivered_at,
                d.*,
                d.status AS delivery_status
            FROM primary_outbox AS o
            JOIN primary_outbox_deliveries AS d ON d.outbox_id = o.outbox_id
            WHERE o.outbox_id = ?
            """,
            (outbox_id,),
        ).fetchone()
        if row is None:
            raise OutboxDeliveryError("primary outbox delivery row is missing")
        return cast(sqlite3.Row, row)

    def _message_from_row(self, row: sqlite3.Row) -> OutboxMessage:
        payload = bytes(row["payload_json"])
        payload_sha256 = str(row["payload_json_sha256"])
        if exact_bytes_sha256(payload) != payload_sha256:
            raise OutboxDeliveryError("primary outbox payload exact digest is corrupt")
        try:
            record = EventIdentityOutboxWireRecord.model_validate_json(payload, strict=True)
            record = validate_registered_event_identity_outbox_wire_record(record, self._registry)
        except (SchemaRegistryError, ValidationError, TypeError, ValueError) as error:
            raise OutboxDeliveryError("primary outbox payload is not a valid record") from error
        if canonical_json_bytes(record) != payload:
            raise OutboxDeliveryError("primary outbox payload is not canonical JSON")
        if (
            record.outbox_id != str(row["source_outbox_id"])
            or record.recording_identity != str(row["recording_identity"])
            or record.assignment_logical_key != str(row["assignment_logical_key"])
        ):
            raise OutboxDeliveryError("primary outbox columns disagree with exact payload")
        return OutboxMessage(
            outbox_id=record.outbox_id,
            completion_run_id=str(row["completion_run_id"]),
            recording_identity=record.recording_identity,
            outbox_ordinal=int(row["outbox_ordinal"]),
            topic=record.topic,
            key=record.key,
            payload=payload,
            payload_sha256=payload_sha256,
        )

    def _snapshot_from_row(self, row: sqlite3.Row) -> OutboxDeliverySnapshot:
        return OutboxDeliverySnapshot(
            outbox_id=str(row["outbox_id"]),
            status=OutboxDeliveryStatus(str(row["delivery_status"])),
            attempt_count=int(row["attempt_count"]),
            lease_epoch=int(row["lease_epoch"]),
            fencing_token=None if row["fencing_token"] is None else str(row["fencing_token"]),
            claimed_by=None if row["claimed_by"] is None else str(row["claimed_by"]),
            lease_expires_at=None
            if row["lease_expires_at"] is None
            else str(row["lease_expires_at"]),
            next_attempt_at=str(row["next_attempt_at"]),
            retry_policy=self._policy_from_row(row),
            last_error=None if row["last_error"] is None else str(row["last_error"]),
            delivered_at=None if row["delivered_at"] is None else str(row["delivered_at"]),
            dead_lettered_at=None
            if row["dead_lettered_at"] is None
            else str(row["dead_lettered_at"]),
        )

    def _policy_from_row(self, row: sqlite3.Row) -> OutboxRetryPolicy:
        return OutboxRetryPolicy(
            version=str(row["retry_policy_version"]),
            max_attempts=int(row["max_attempts"]),
            base_delay_seconds=float(row["base_delay_seconds"]),
            max_delay_seconds=float(row["max_delay_seconds"]),
        )

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema = OFF")
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if application_id != _PRIMARY_APPLICATION_ID or user_version != _PRIMARY_SCHEMA_VERSION:
                raise OutboxDeliveryError("primary completion database header is incompatible")
            if not _primary_schema_is_current(connection):
                raise OutboxDeliveryError(
                    "primary completion DDL does not match the canonical schema"
                )
            return connection
        except OutboxDeliveryError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise OutboxDeliveryError(
                "cannot open or verify the primary outbox database"
            ) from error


class SQLiteIdempotentOutboxSink:
    """Local broker substitute keyed by outbox ID and exact message bytes."""

    def __init__(self, database_path: Path, *, clock: Clock | None = None) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be pathlib.Path")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._path = database_path.resolve()
        self._clock = clock if clock is not None else _default_clock
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except OutboxDeliveryError:
            raise
        except sqlite3.Error as error:
            raise OutboxDeliveryError("cannot initialize the outbox sink database") from error

    @property
    def path(self) -> Path:
        return self._path

    def publish(self, message: OutboxMessage) -> None:
        if not isinstance(message, OutboxMessage):
            raise TypeError("message must be OutboxMessage")
        published_at = _rfc3339(_require_now(self._clock))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM delivered_outbox_messages WHERE outbox_id = ?",
                (message.outbox_id,),
            ).fetchone()
            if row is not None:
                stored = (
                    str(row["topic"]),
                    str(row["message_key"]),
                    bytes(row["payload"]),
                    str(row["payload_sha256"]),
                )
                candidate = (
                    message.topic,
                    message.key,
                    message.payload,
                    message.payload_sha256,
                )
                if stored != candidate:
                    raise OutboxDeliveryError(
                        "sink already contains different bytes for this outbox ID"
                    )
                connection.commit()
                return
            connection.execute(
                """
                INSERT INTO delivered_outbox_messages (
                    outbox_id, topic, message_key, payload, payload_sha256, published_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.outbox_id,
                    message.topic,
                    message.key,
                    message.payload,
                    message.payload_sha256,
                    published_at,
                ),
            )
            connection.commit()
        except OutboxDeliveryError:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        except sqlite3.Error as error:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise OutboxDeliveryError(f"cannot publish to idempotent sink: {error}") from error
        finally:
            connection.close()

    def count(self) -> int:
        connection = self._connect()
        try:
            row = connection.execute("SELECT COUNT(*) FROM delivered_outbox_messages").fetchone()
            return int(row[0])
        finally:
            connection.close()

    def payload(self, outbox_id: str) -> bytes | None:
        if not isinstance(outbox_id, str) or not outbox_id:
            raise ValueError("outbox_id must be a nonempty string")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload FROM delivered_outbox_messages WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
            return None if row is None else bytes(row["payload"])
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._open_unchecked()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if application_id == _SINK_APPLICATION_ID and user_version == _SINK_SCHEMA_VERSION:
                if not _sink_schema_is_current(connection):
                    raise OutboxDeliveryError("sink DDL does not match the canonical schema")
                return
            if application_id != 0 or user_version != 0:
                raise OutboxDeliveryError("sink database header belongs to another schema")
            existing = connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
            if existing:
                raise OutboxDeliveryError(
                    "refusing to initialize sink over an existing SQLite schema"
                )
            connection.execute("BEGIN IMMEDIATE")
            for statement in _SINK_SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id = {_SINK_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SINK_SCHEMA_VERSION}")
            if not _sink_schema_is_current(connection):
                raise OutboxDeliveryError("new sink DDL is not canonical")
            connection.commit()
        except Exception:
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        finally:
            connection.close()

    def _open_unchecked(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema = OFF")
            return connection
        except sqlite3.Error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._open_unchecked()
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if application_id != _SINK_APPLICATION_ID or user_version != _SINK_SCHEMA_VERSION:
                raise OutboxDeliveryError("sink database header is incompatible")
            if not _sink_schema_is_current(connection):
                raise OutboxDeliveryError("sink DDL does not match the canonical schema")
            return connection
        except OutboxDeliveryError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            raise OutboxDeliveryError("cannot open or verify the outbox sink database") from error


__all__ = [
    "SQLiteIdempotentOutboxSink",
    "SQLitePrimaryOutboxDeliveryStore",
]
