"""Local durable priority/SLA queue for asynchronous human review."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, TypeVar, cast

from pydantic import BaseModel, ValidationError

from robata.contracts.common import INT64_MAX, INT64_MIN
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import (
    SchemaRegistry,
    SchemaRegistryError,
    default_schema_registry,
)
from robata.ports.review_queue import (
    EnqueuedReviewTask,
    ReopenedReviewTask,
    ReviewQueue,
    ReviewQueueError,
    ReviewQueueErrorCode,
    SubmittedReviewAnnotation,
)
from robata.review.models import (
    ReviewAnnotation,
    ReviewLease,
    ReviewReopenCommand,
    ReviewTask,
    ReviewTaskSnapshot,
    ReviewTaskStatus,
    validate_registered_review_annotation,
    validate_registered_review_reopen_command,
    validate_registered_review_task,
)
from robata.runtime.observability import (
    RuntimeAttributeValue,
    RuntimeObserver,
    runtime_increment,
    runtime_span,
)

_APPLICATION_ID: Final = 0x52565257  # "RVRW"
_SCHEMA_VERSION: Final = 2
_BUSY_TIMEOUT_MS: Final = 30_000
_REQUIRED_TABLES: Final = frozenset(
    {"review_tasks", "review_annotations", "review_reopen_commands"}
)

_SCHEMA_SQL: Final = f"""
BEGIN IMMEDIATE;

CREATE TABLE review_tasks (
    review_task_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    task_semantic_sha256 TEXT NOT NULL UNIQUE,
    priority INTEGER NOT NULL CHECK (priority >= 0),
    requested_at_ns INTEGER NOT NULL,
    due_at_ns INTEGER NOT NULL CHECK (due_at_ns > requested_at_ns),
    task_json BLOB NOT NULL,
    task_exact_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'LEASED', 'COMPLETED')),
    lease_fence INTEGER NOT NULL DEFAULT 0 CHECK (lease_fence >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count = lease_fence),
    lease_owner TEXT,
    lease_expires_at_ns INTEGER,
    completed_annotation_id TEXT,
    CHECK (
        (status = 'PENDING' AND lease_owner IS NULL AND lease_expires_at_ns IS NULL
            AND completed_annotation_id IS NULL)
        OR (status = 'LEASED' AND lease_owner IS NOT NULL AND lease_expires_at_ns IS NOT NULL
            AND completed_annotation_id IS NULL AND lease_fence > 0)
        OR (status = 'COMPLETED' AND lease_owner IS NULL AND lease_expires_at_ns IS NULL
            AND completed_annotation_id IS NOT NULL)
    )
) STRICT;

CREATE INDEX review_tasks_schedule
ON review_tasks (priority, due_at_ns, requested_at_ns, review_task_id)
WHERE status != 'COMPLETED';

CREATE TABLE review_annotations (
    annotation_id TEXT PRIMARY KEY,
    review_task_id TEXT NOT NULL,
    lease_fence INTEGER NOT NULL CHECK (lease_fence > 0),
    annotation_semantic_sha256 TEXT NOT NULL UNIQUE,
    annotation_json BLOB NOT NULL,
    annotation_exact_sha256 TEXT NOT NULL,
    UNIQUE (review_task_id, lease_fence),
    FOREIGN KEY (review_task_id) REFERENCES review_tasks (review_task_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT;

CREATE TABLE review_reopen_commands (
    reopen_id TEXT PRIMARY KEY,
    review_task_id TEXT NOT NULL,
    expected_annotation_id TEXT NOT NULL,
    command_semantic_sha256 TEXT NOT NULL,
    command_json BLOB NOT NULL,
    command_exact_sha256 TEXT NOT NULL,
    FOREIGN KEY (review_task_id) REFERENCES review_tasks (review_task_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (expected_annotation_id) REFERENCES review_annotations (annotation_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER review_tasks_immutable_definition
BEFORE UPDATE ON review_tasks
WHEN NEW.review_task_id != OLD.review_task_id
    OR NEW.request_id != OLD.request_id
    OR NEW.task_semantic_sha256 != OLD.task_semantic_sha256
    OR NEW.priority != OLD.priority
    OR NEW.requested_at_ns != OLD.requested_at_ns
    OR NEW.due_at_ns != OLD.due_at_ns
    OR NEW.task_json != OLD.task_json
    OR NEW.task_exact_sha256 != OLD.task_exact_sha256
BEGIN
    SELECT RAISE(ABORT, 'review task definition is immutable');
END;

CREATE TRIGGER review_tasks_no_delete
BEFORE DELETE ON review_tasks
BEGIN
    SELECT RAISE(ABORT, 'review tasks cannot be deleted');
END;

CREATE TRIGGER review_annotations_no_update
BEFORE UPDATE ON review_annotations
BEGIN
    SELECT RAISE(ABORT, 'review annotations are append-only');
END;

CREATE TRIGGER review_annotations_no_delete
BEFORE DELETE ON review_annotations
BEGIN
    SELECT RAISE(ABORT, 'review annotations are append-only');
END;

CREATE TRIGGER review_reopen_commands_no_update
BEFORE UPDATE ON review_reopen_commands
BEGIN
    SELECT RAISE(ABORT, 'review reopen commands are append-only');
END;

CREATE TRIGGER review_reopen_commands_no_delete
BEFORE DELETE ON review_reopen_commands
BEGIN
    SELECT RAISE(ABORT, 'review reopen commands are append-only');
END;

PRAGMA application_id = {_APPLICATION_ID};
PRAGMA user_version = {_SCHEMA_VERSION};
COMMIT;
"""

ResultT = TypeVar("ResultT")


class SQLiteReviewQueue(ReviewQueue):
    """SQLite authority for nonblocking review routing and adjudication history."""

    def __init__(
        self,
        database_path: Path,
        *,
        registry: SchemaRegistry | None = None,
        runtime_observer: RuntimeObserver | None = None,
    ) -> None:
        if not isinstance(database_path, Path):
            raise TypeError("database_path must be a pathlib.Path")
        try:
            database_path.parent.mkdir(parents=True, exist_ok=True)
            self._database_path = database_path.parent.resolve(strict=True) / database_path.name
        except OSError as exc:
            raise ReviewQueueError(
                ReviewQueueErrorCode.STORAGE_IO_ERROR,
                f"cannot prepare review queue database path: {exc}",
            ) from exc
        self._registry = registry or default_schema_registry()
        self._runtime_observer = runtime_observer
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the resolved local database path."""

        return self._database_path

    def enqueue(self, task: ReviewTask) -> EnqueuedReviewTask:
        """Insert a pending task or resolve one exact replay."""

        if not isinstance(task, ReviewTask):
            raise ReviewQueueError(
                ReviewQueueErrorCode.INVALID_REQUEST,
                "task must be a ReviewTask",
            )
        task = _validate_for_write(
            task,
            self._registry,
            validate_registered_review_task,
            "review task",
        )
        payload, payload_sha256 = _encode_model(task)

        def write(connection: sqlite3.Connection) -> EnqueuedReviewTask:
            rows = connection.execute(
                """
                SELECT * FROM review_tasks
                WHERE review_task_id = ? OR request_id = ?
                ORDER BY review_task_id
                """,
                (task.review_task_id, task.request_id),
            ).fetchall()
            if rows:
                if len(rows) != 1:
                    raise _integrity_error("review task identities resolve to multiple rows")
                existing = self._snapshot_from_row(rows[0]).task
                if existing == task:
                    return EnqueuedReviewTask(task=existing, inserted=False)
                raise ReviewQueueError(
                    ReviewQueueErrorCode.TASK_CONFLICT,
                    "review request or task identity already has different immutable content",
                )
            connection.execute(
                """
                INSERT INTO review_tasks (
                    review_task_id, request_id, task_semantic_sha256, priority,
                    requested_at_ns, due_at_ns, task_json, task_exact_sha256,
                    status, lease_fence, attempt_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, 0)
                """,
                (
                    task.review_task_id,
                    task.request_id,
                    task.semantic_sha256,
                    task.priority,
                    task.requested_at_ns,
                    task.due_at_ns,
                    payload,
                    payload_sha256,
                ),
            )
            return EnqueuedReviewTask(task=task, inserted=True)

        return self._transaction(write=True, operation_name="enqueue", operation=write)

    def claim_next(
        self,
        *,
        worker_id: str,
        now_ns: int,
        lease_duration_ns: int,
    ) -> ReviewLease | None:
        """Claim pending work, recovering an expired lease with a new fence."""

        worker = _nonempty_string(worker_id, "worker_id")
        now = _nanoseconds(now_ns, "now_ns")
        duration = _positive_duration(lease_duration_ns)
        expires_at = _checked_add(now, duration, "lease expiry")

        def write(connection: sqlite3.Connection) -> ReviewLease | None:
            row = connection.execute(
                """
                SELECT * FROM review_tasks
                WHERE status = 'PENDING'
                   OR (status = 'LEASED' AND lease_expires_at_ns <= ?)
                ORDER BY priority, due_at_ns, requested_at_ns, review_task_id
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            snapshot = self._snapshot_from_row(row)
            next_fence = snapshot.lease_fence + 1
            if next_fence > INT64_MAX:
                raise _integrity_error("review lease fence exhausted signed int64")
            cursor = connection.execute(
                """
                UPDATE review_tasks
                SET status = 'LEASED', lease_fence = ?, attempt_count = ?,
                    lease_owner = ?, lease_expires_at_ns = ?, completed_annotation_id = NULL
                WHERE review_task_id = ? AND lease_fence = ?
                """,
                (
                    next_fence,
                    next_fence,
                    worker,
                    expires_at,
                    snapshot.task.review_task_id,
                    snapshot.lease_fence,
                ),
            )
            if cursor.rowcount != 1:
                raise _integrity_error("review claim compare-and-swap did not update one row")
            return ReviewLease(
                task=snapshot.task,
                worker_id=worker,
                lease_fence=next_fence,
                lease_expires_at_ns=expires_at,
            )

        return self._transaction(write=True, operation_name="claim_next", operation=write)

    def renew_lease(
        self,
        *,
        review_task_id: str,
        worker_id: str,
        lease_fence: int,
        now_ns: int,
        lease_duration_ns: int,
    ) -> ReviewLease:
        """Renew exactly one still-live fenced lease."""

        task_id = _nonempty_string(review_task_id, "review_task_id")
        worker = _nonempty_string(worker_id, "worker_id")
        fence = _positive_fence(lease_fence)
        now = _nanoseconds(now_ns, "now_ns")
        duration = _positive_duration(lease_duration_ns)
        expires_at = _checked_add(now, duration, "lease expiry")

        def write(connection: sqlite3.Connection) -> ReviewLease:
            snapshot = self._required_snapshot(connection, task_id)
            self._require_matching_lease(snapshot, worker=worker, fence=fence, now_ns=now)
            connection.execute(
                """
                UPDATE review_tasks SET lease_expires_at_ns = ?
                WHERE review_task_id = ?
                """,
                (expires_at, task_id),
            )
            return ReviewLease(
                task=snapshot.task,
                worker_id=worker,
                lease_fence=fence,
                lease_expires_at_ns=expires_at,
            )

        return self._transaction(write=True, operation_name="renew_lease", operation=write)

    def submit_annotation(
        self,
        annotation: ReviewAnnotation,
        *,
        now_ns: int,
    ) -> SubmittedReviewAnnotation:
        """Append one annotation and complete its live fenced lease atomically."""

        if not isinstance(annotation, ReviewAnnotation):
            raise ReviewQueueError(
                ReviewQueueErrorCode.INVALID_REQUEST,
                "annotation must be a ReviewAnnotation",
            )
        now = _nanoseconds(now_ns, "now_ns")
        annotation = _validate_for_write(
            annotation,
            self._registry,
            validate_registered_review_annotation,
            "review annotation",
        )
        payload, payload_sha256 = _encode_model(annotation)

        def write(connection: sqlite3.Connection) -> SubmittedReviewAnnotation:
            existing_row = connection.execute(
                """
                SELECT * FROM review_annotations
                WHERE annotation_id = ?
                   OR (review_task_id = ? AND lease_fence = ?)
                ORDER BY annotation_id
                """,
                (
                    annotation.annotation_id,
                    annotation.review_task_id,
                    annotation.lease_fence,
                ),
            ).fetchall()
            if existing_row:
                if len(existing_row) != 1:
                    raise _integrity_error("annotation identity resolves to multiple rows")
                existing = self._annotation_from_row(existing_row[0])
                if existing == annotation:
                    return SubmittedReviewAnnotation(annotation=existing, inserted=False)
                raise ReviewQueueError(
                    ReviewQueueErrorCode.ANNOTATION_CONFLICT,
                    "review lease attempt already has a different annotation",
                )

            snapshot = self._required_snapshot(connection, annotation.review_task_id)
            if snapshot.task.semantic_sha256 != annotation.review_task_semantic_sha256:
                raise ReviewQueueError(
                    ReviewQueueErrorCode.ANNOTATION_CONFLICT,
                    "annotation references a different immutable review task",
                )
            if snapshot.task.subject != annotation.subject:
                raise ReviewQueueError(
                    ReviewQueueErrorCode.ANNOTATION_CONFLICT,
                    "annotation subject does not match the review task",
                )
            self._require_matching_lease(
                snapshot,
                worker=annotation.lease_owner,
                fence=annotation.lease_fence,
                now_ns=now,
            )
            connection.execute(
                """
                INSERT INTO review_annotations (
                    annotation_id, review_task_id, lease_fence,
                    annotation_semantic_sha256, annotation_json, annotation_exact_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation.annotation_id,
                    annotation.review_task_id,
                    annotation.lease_fence,
                    annotation.semantic_sha256,
                    payload,
                    payload_sha256,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE review_tasks
                SET status = 'COMPLETED', lease_owner = NULL, lease_expires_at_ns = NULL,
                    completed_annotation_id = ?
                WHERE review_task_id = ? AND status = 'LEASED' AND lease_fence = ?
                """,
                (
                    annotation.annotation_id,
                    annotation.review_task_id,
                    annotation.lease_fence,
                ),
            )
            if cursor.rowcount != 1:
                raise _integrity_error("annotation completion did not update one leased task")
            return SubmittedReviewAnnotation(annotation=annotation, inserted=True)

        return self._transaction(
            write=True,
            operation_name="submit_annotation",
            operation=write,
        )

    def reopen(self, command: ReviewReopenCommand) -> ReopenedReviewTask:
        """Return completed work to pending while preserving immutable history."""

        if not isinstance(command, ReviewReopenCommand):
            raise ReviewQueueError(
                ReviewQueueErrorCode.INVALID_REQUEST,
                "command must be a ReviewReopenCommand",
            )
        command = _validate_for_write(
            command,
            self._registry,
            validate_registered_review_reopen_command,
            "review reopen command",
        )
        payload, payload_sha256 = _encode_model(command)

        def write(connection: sqlite3.Connection) -> ReopenedReviewTask:
            existing_row = connection.execute(
                "SELECT * FROM review_reopen_commands WHERE reopen_id = ?",
                (command.reopen_id,),
            ).fetchone()
            if existing_row is not None:
                existing = self._reopen_from_row(existing_row)
                if existing != command:
                    raise ReviewQueueError(
                        ReviewQueueErrorCode.REOPEN_CONFLICT,
                        "reopen ID already has different immutable content",
                    )
                return ReopenedReviewTask(
                    command=existing,
                    snapshot=self._required_snapshot(connection, existing.review_task_id),
                    applied=False,
                )

            snapshot = self._required_snapshot(connection, command.review_task_id)
            if snapshot.status is not ReviewTaskStatus.COMPLETED:
                raise ReviewQueueError(
                    ReviewQueueErrorCode.NOT_CLAIMABLE,
                    "only completed review tasks may be reopened",
                )
            if snapshot.completed_annotation_id != command.expected_annotation_id:
                raise ReviewQueueError(
                    ReviewQueueErrorCode.REOPEN_CONFLICT,
                    "reopen command does not reference the current completed annotation",
                )
            connection.execute(
                """
                INSERT INTO review_reopen_commands (
                    reopen_id, review_task_id, expected_annotation_id,
                    command_semantic_sha256, command_json, command_exact_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    command.reopen_id,
                    command.review_task_id,
                    command.expected_annotation_id,
                    command.semantic_sha256,
                    payload,
                    payload_sha256,
                ),
            )
            connection.execute(
                """
                UPDATE review_tasks
                SET status = 'PENDING', completed_annotation_id = NULL
                WHERE review_task_id = ?
                """,
                (command.review_task_id,),
            )
            return ReopenedReviewTask(
                command=command,
                snapshot=self._required_snapshot(connection, command.review_task_id),
                applied=True,
            )

        return self._transaction(write=True, operation_name="reopen", operation=write)

    def get_task(self, review_task_id: str) -> ReviewTaskSnapshot | None:
        """Read one verified task snapshot."""

        task_id = _nonempty_string(review_task_id, "review_task_id")

        def read(connection: sqlite3.Connection) -> ReviewTaskSnapshot | None:
            row = connection.execute(
                "SELECT * FROM review_tasks WHERE review_task_id = ?",
                (task_id,),
            ).fetchone()
            return None if row is None else self._snapshot_from_row(row)

        return self._transaction(write=False, operation_name="get_task", operation=read)

    def list_open(self) -> tuple[ReviewTaskSnapshot, ...]:
        """List pending and leased work in deterministic scheduling order."""

        def read(connection: sqlite3.Connection) -> tuple[ReviewTaskSnapshot, ...]:
            rows = connection.execute(
                """
                SELECT * FROM review_tasks
                WHERE status != 'COMPLETED'
                ORDER BY priority, due_at_ns, requested_at_ns, review_task_id
                """
            ).fetchall()
            return tuple(self._snapshot_from_row(row) for row in rows)

        return self._transaction(write=False, operation_name="list_open", operation=read)

    def list_overdue(self, *, now_ns: int) -> tuple[ReviewTaskSnapshot, ...]:
        """List incomplete work strictly later than its SLA deadline."""

        now = _nanoseconds(now_ns, "now_ns")

        def read(connection: sqlite3.Connection) -> tuple[ReviewTaskSnapshot, ...]:
            rows = connection.execute(
                """
                SELECT * FROM review_tasks
                WHERE status != 'COMPLETED' AND due_at_ns < ?
                ORDER BY priority, due_at_ns, requested_at_ns, review_task_id
                """,
                (now,),
            ).fetchall()
            return tuple(self._snapshot_from_row(row) for row in rows)

        return self._transaction(write=False, operation_name="list_overdue", operation=read)

    def list_annotations(self, review_task_id: str) -> tuple[ReviewAnnotation, ...]:
        """Read verified append-only annotation history."""

        task_id = _nonempty_string(review_task_id, "review_task_id")

        def read(connection: sqlite3.Connection) -> tuple[ReviewAnnotation, ...]:
            self._required_snapshot(connection, task_id)
            rows = connection.execute(
                """
                SELECT * FROM review_annotations
                WHERE review_task_id = ?
                ORDER BY lease_fence, annotation_id
                """,
                (task_id,),
            ).fetchall()
            return tuple(self._annotation_from_row(row) for row in rows)

        return self._transaction(
            write=False,
            operation_name="list_annotations",
            operation=read,
        )

    def _require_matching_lease(
        self,
        snapshot: ReviewTaskSnapshot,
        *,
        worker: str,
        fence: int,
        now_ns: int,
    ) -> None:
        if (
            snapshot.status is not ReviewTaskStatus.LEASED
            or snapshot.lease_fence != fence
            or snapshot.lease_owner != worker
        ):
            raise ReviewQueueError(
                ReviewQueueErrorCode.STALE_FENCE,
                "review lease owner or fence is stale",
            )
        if snapshot.lease_expires_at_ns is None:
            raise _integrity_error("leased review task has no expiry")
        if snapshot.lease_expires_at_ns <= now_ns:
            raise ReviewQueueError(
                ReviewQueueErrorCode.LEASE_EXPIRED,
                "review lease has expired",
            )

    def _required_snapshot(
        self,
        connection: sqlite3.Connection,
        review_task_id: str,
    ) -> ReviewTaskSnapshot:
        row = connection.execute(
            "SELECT * FROM review_tasks WHERE review_task_id = ?",
            (review_task_id,),
        ).fetchone()
        if row is None:
            raise ReviewQueueError(
                ReviewQueueErrorCode.TASK_NOT_FOUND,
                f"review task not found: {review_task_id}",
            )
        return self._snapshot_from_row(row)

    def _snapshot_from_row(self, row: sqlite3.Row) -> ReviewTaskSnapshot:
        task = _decode_model(
            ReviewTask,
            _row_bytes(row, "task_json"),
            _row_text(row, "task_exact_sha256"),
        )
        task = _validate_persisted(
            task,
            self._registry,
            validate_registered_review_task,
            "review task",
        )
        indexed = (
            (_row_text(row, "review_task_id"), task.review_task_id, "review_task_id"),
            (_row_text(row, "request_id"), task.request_id, "request_id"),
            (
                _row_text(row, "task_semantic_sha256"),
                task.semantic_sha256,
                "task_semantic_sha256",
            ),
            (_row_int(row, "priority"), task.priority, "priority"),
            (_row_int(row, "requested_at_ns"), task.requested_at_ns, "requested_at_ns"),
            (_row_int(row, "due_at_ns"), task.due_at_ns, "due_at_ns"),
        )
        for stored, decoded, label in indexed:
            if stored != decoded:
                raise _integrity_error(f"review task indexed {label} disagrees with payload")
        try:
            return ReviewTaskSnapshot(
                task=task,
                status=ReviewTaskStatus(_row_text(row, "status")),
                lease_fence=_row_int(row, "lease_fence"),
                attempt_count=_row_int(row, "attempt_count"),
                lease_owner=_row_optional_text(row, "lease_owner"),
                lease_expires_at_ns=_row_optional_int(row, "lease_expires_at_ns"),
                completed_annotation_id=_row_optional_text(row, "completed_annotation_id"),
            )
        except (ValueError, ValidationError) as exc:
            raise _integrity_error(f"invalid persisted review task state: {exc}") from exc

    def _annotation_from_row(self, row: sqlite3.Row) -> ReviewAnnotation:
        annotation = _decode_model(
            ReviewAnnotation,
            _row_bytes(row, "annotation_json"),
            _row_text(row, "annotation_exact_sha256"),
        )
        annotation = _validate_persisted(
            annotation,
            self._registry,
            validate_registered_review_annotation,
            "review annotation",
        )
        indexed = (
            (_row_text(row, "annotation_id"), annotation.annotation_id, "annotation_id"),
            (
                _row_text(row, "review_task_id"),
                annotation.review_task_id,
                "review_task_id",
            ),
            (_row_int(row, "lease_fence"), annotation.lease_fence, "lease_fence"),
            (
                _row_text(row, "annotation_semantic_sha256"),
                annotation.semantic_sha256,
                "annotation_semantic_sha256",
            ),
        )
        for stored, decoded, label in indexed:
            if stored != decoded:
                raise _integrity_error(f"annotation indexed {label} disagrees with payload")
        return annotation

    def _reopen_from_row(self, row: sqlite3.Row) -> ReviewReopenCommand:
        command = _decode_model(
            ReviewReopenCommand,
            _row_bytes(row, "command_json"),
            _row_text(row, "command_exact_sha256"),
        )
        command = _validate_persisted(
            command,
            self._registry,
            validate_registered_review_reopen_command,
            "review reopen command",
        )
        indexed = (
            (_row_text(row, "reopen_id"), command.reopen_id, "reopen_id"),
            (
                _row_text(row, "review_task_id"),
                command.review_task_id,
                "review_task_id",
            ),
            (
                _row_text(row, "expected_annotation_id"),
                command.expected_annotation_id,
                "expected_annotation_id",
            ),
            (
                _row_text(row, "command_semantic_sha256"),
                command.semantic_sha256,
                "command_semantic_sha256",
            ),
        )
        for stored, decoded, label in indexed:
            if stored != decoded:
                raise _integrity_error(f"reopen indexed {label} disagrees with payload")
        return command

    def _initialize_database(self) -> None:
        connection = self._open()
        try:
            tables = _user_tables(connection)
            if not tables:
                self._initialize_schema(connection)
            self._verify_database(connection)
        except ReviewQueueError:
            raise
        except sqlite3.Error as exc:
            raise ReviewQueueError(
                ReviewQueueErrorCode.STORAGE_IO_ERROR,
                f"cannot initialize review queue database: {exc}",
            ) from exc
        finally:
            connection.close()

    def _open(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self._database_path,
                isolation_level=None,
                timeout=_BUSY_TIMEOUT_MS / 1000,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            return connection
        except sqlite3.Error as exc:
            raise ReviewQueueError(
                ReviewQueueErrorCode.STORAGE_IO_ERROR,
                f"cannot open review queue database: {exc}",
            ) from exc

    def _verify_database(self, connection: sqlite3.Connection) -> None:
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if application_id != _APPLICATION_ID or user_version != _SCHEMA_VERSION:
            raise _integrity_error(
                "review queue database header does not match the supported schema"
            )
        if _user_tables(connection) != _REQUIRED_TABLES:
            raise _integrity_error("review queue database tables do not match the supported schema")

    def _transaction(
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[sqlite3.Connection], ResultT],
    ) -> ResultT:
        connection = self._open()
        try:
            with self._observed_transaction_scope(
                connection,
                write=write,
                operation_name=operation_name,
            ):
                self._verify_database(connection)
                return operation(connection)
        except ReviewQueueError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ReviewQueueError(
                ReviewQueueErrorCode.INTEGRITY_ERROR,
                f"review queue integrity failure: {exc}",
            ) from exc
        except sqlite3.Error as exc:
            raise ReviewQueueError(
                ReviewQueueErrorCode.STORAGE_IO_ERROR,
                f"review queue storage failure: {exc}",
            ) from exc
        finally:
            connection.close()

    def _initialize_schema(self, connection: sqlite3.Connection) -> None:
        attributes: dict[str, RuntimeAttributeValue] = {
            "operation": "initialize_schema",
            "write": True,
        }
        with runtime_span(
            self._runtime_observer,
            "sqlite.review_queue.transaction",
            attributes,
        ):
            try:
                connection.executescript(_SCHEMA_SQL)
            except BaseException:
                if connection.in_transaction:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.review_queue.transactions",
                        attributes=attributes,
                    )
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.rollbacks",
                            attributes=attributes,
                        )
                raise
            runtime_increment(
                self._runtime_observer,
                "sqlite.review_queue.transactions",
                attributes=attributes,
            )
            runtime_increment(
                self._runtime_observer,
                "sqlite.review_queue.commits",
                attributes=attributes,
            )

    @contextmanager
    def _observed_transaction_scope(
        self,
        connection: sqlite3.Connection,
        *,
        write: bool,
        operation_name: str,
    ) -> Iterator[None]:
        attributes: dict[str, RuntimeAttributeValue] = {
            "operation": operation_name,
            "write": write,
        }
        with runtime_span(
            self._runtime_observer,
            "sqlite.review_queue.transaction",
            attributes,
        ):
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            runtime_increment(
                self._runtime_observer,
                "sqlite.review_queue.transactions",
                attributes=attributes,
            )
            try:
                yield
            except BaseException:
                if connection.in_transaction:
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.rollbacks",
                            attributes=attributes,
                        )
                else:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.review_queue.transaction_outcomes_unknown",
                        attributes=attributes,
                    )
                raise
            try:
                connection.commit()
            except BaseException:
                runtime_increment(
                    self._runtime_observer,
                    "sqlite.review_queue.commit_failures",
                    attributes=attributes,
                )
                if connection.in_transaction:
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.rollback_failures",
                            attributes=attributes,
                        )
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.transaction_outcomes_unknown",
                            attributes=attributes,
                        )
                    else:
                        runtime_increment(
                            self._runtime_observer,
                            "sqlite.review_queue.rollbacks",
                            attributes=attributes,
                        )
                else:
                    runtime_increment(
                        self._runtime_observer,
                        "sqlite.review_queue.transaction_outcomes_unknown",
                        attributes=attributes,
                    )
                raise
            runtime_increment(
                self._runtime_observer,
                "sqlite.review_queue.commits",
                attributes=attributes,
            )


def _encode_model(model: BaseModel) -> tuple[bytes, str]:
    payload = canonical_json_bytes(model)
    return payload, exact_bytes_sha256(payload)


def _decode_model[ModelT: BaseModel](
    model_type: type[ModelT],
    payload: bytes,
    expected_sha256: str,
) -> ModelT:
    if exact_bytes_sha256(payload) != expected_sha256:
        raise _integrity_error("persisted review payload exact digest mismatch")
    try:
        return model_type.model_validate_json(payload)
    except (ValueError, ValidationError) as exc:
        raise _integrity_error(f"invalid persisted review payload: {exc}") from exc


def _validate_for_write[ModelT: BaseModel](
    model: ModelT,
    registry: SchemaRegistry,
    validator: Callable[[ModelT, SchemaRegistry | None], ModelT],
    label: str,
) -> ModelT:
    try:
        return validator(model, registry)
    except (SchemaRegistryError, TypeError, ValueError) as exc:
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            f"{label} registered schema validation failed: {exc}",
        ) from exc


def _validate_persisted[ModelT: BaseModel](
    model: ModelT,
    registry: SchemaRegistry,
    validator: Callable[[ModelT, SchemaRegistry | None], ModelT],
    label: str,
) -> ModelT:
    try:
        return validator(model, registry)
    except (SchemaRegistryError, TypeError, ValueError) as exc:
        raise _integrity_error(
            f"persisted {label} registered schema validation failed: {exc}"
        ) from exc


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            f"{label} must be a nonempty string",
        )
    return value


def _nanoseconds(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            f"{label} must be an integer",
        )
    if value < INT64_MIN or value > INT64_MAX:
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            f"{label} must fit signed int64 nanoseconds",
        )
    return value


def _positive_duration(value: object) -> int:
    duration = _nanoseconds(value, "lease_duration_ns")
    if duration <= 0:
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            "lease_duration_ns must be positive",
        )
    return duration


def _positive_fence(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            "lease_fence must be a positive integer",
        )
    return value


def _checked_add(left: int, right: int, label: str) -> int:
    result = left + right
    if result < INT64_MIN or result > INT64_MAX:
        raise ReviewQueueError(
            ReviewQueueErrorCode.INVALID_REQUEST,
            f"{label} exceeds signed int64 nanoseconds",
        )
    return result


def _integrity_error(message: str) -> ReviewQueueError:
    return ReviewQueueError(ReviewQueueErrorCode.INTEGRITY_ERROR, message)


def _user_tables(connection: sqlite3.Connection) -> frozenset[str]:
    rows = connection.execute(
        """
        SELECT name FROM sqlite_schema
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()
    return frozenset(str(row[0]) for row in rows)


def _row_text(row: sqlite3.Row, column: str) -> str:
    value = row[column]
    if not isinstance(value, str):
        raise _integrity_error(f"review database column {column} must be text")
    return value


def _row_optional_text(row: sqlite3.Row, column: str) -> str | None:
    value = row[column]
    if value is None:
        return None
    if not isinstance(value, str):
        raise _integrity_error(f"review database column {column} must be text or null")
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _integrity_error(f"review database column {column} must be an integer")
    return cast(int, value)


def _row_optional_int(row: sqlite3.Row, column: str) -> int | None:
    value = row[column]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _integrity_error(f"review database column {column} must be an integer or null")
    return cast(int, value)


def _row_bytes(row: sqlite3.Row, column: str) -> bytes:
    value = row[column]
    if not isinstance(value, bytes):
        raise _integrity_error(f"review database column {column} must be bytes")
    return value


__all__ = ["SQLiteReviewQueue"]
