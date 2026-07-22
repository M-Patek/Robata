"""SQLite implementation of the authoritative durable-work ledger.

The scheduler owns lifecycle, dependency, retry, lease, and fencing truth.
Broker messages are delivery projections, never the state authority.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from robata.queue.models import (
    SUCCESSFUL_DEPENDENCY_STATES,
    TERMINAL_WORK_STATES,
    WorkAttempt,
    WorkAttemptOutcome,
    WorkDependency,
    WorkItem,
    WorkItemPlan,
    WorkItemState,
    WorkItemSubjectType,
    WorkLease,
    WorkLeaseClaim,
)
from robata.queue.stage import DependencyCriticality, Stage

_APPLICATION_ID = 0x5242574B
_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 5_000

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE work_items (
        schema_version TEXT NOT NULL,
        work_item_id TEXT PRIMARY KEY,
        work_logical_key TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL,
        mcap_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        subject_type TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        input_digest TEXT NOT NULL,
        config_digest TEXT NOT NULL,
        priority INTEGER NOT NULL CHECK (priority >= 0),
        sla_deadline_at TEXT,
        execution_expiry_at TEXT,
        max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
        trace_id TEXT,
        created_at TEXT NOT NULL,
        state TEXT NOT NULL,
        cancel_requested INTEGER NOT NULL CHECK (cancel_requested IN (0, 1)),
        lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 0),
        fencing_token TEXT,
        leased_by TEXT,
        lease_expires_at TEXT,
        attempt INTEGER NOT NULL CHECK (attempt >= 0),
        retry_not_before_at TEXT,
        terminal_reason_code TEXT,
        terminal_reason_detail TEXT,
        result_reference TEXT,
        result_sha256 TEXT,
        completed_at TEXT,
        updated_at TEXT NOT NULL,
        row_version INTEGER NOT NULL CHECK (row_version >= 0)
    )
    """,
    """
    CREATE TABLE work_dependencies (
        dependency_id TEXT PRIMARY KEY,
        downstream_work_item_id TEXT NOT NULL
            REFERENCES work_items(work_item_id) ON DELETE RESTRICT,
        upstream_work_item_id TEXT NOT NULL
            REFERENCES work_items(work_item_id) ON DELETE RESTRICT,
        criticality TEXT NOT NULL,
        UNIQUE (downstream_work_item_id, upstream_work_item_id),
        CHECK (downstream_work_item_id <> upstream_work_item_id)
    )
    """,
    """
    CREATE INDEX work_dependencies_upstream
    ON work_dependencies(upstream_work_item_id, downstream_work_item_id)
    """,
    """
    CREATE TABLE work_attempts (
        work_item_id TEXT NOT NULL
            REFERENCES work_items(work_item_id) ON DELETE RESTRICT,
        attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
        lease_epoch INTEGER NOT NULL CHECK (lease_epoch >= 1),
        fencing_token TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        claimed_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        outcome TEXT NOT NULL,
        error_code TEXT,
        error_detail TEXT,
        PRIMARY KEY (work_item_id, attempt_number),
        UNIQUE (work_item_id, lease_epoch),
        UNIQUE (fencing_token)
    )
    """,
)


class WorkSchedulerError(RuntimeError):
    """Base error for durable scheduling operations."""


class WorkStorageError(WorkSchedulerError):
    """The SQLite ledger cannot be opened or is internally inconsistent."""


class WorkConflictError(WorkSchedulerError):
    """An idempotent plan conflicts with persisted identity or policy."""


class WorkNotFoundError(WorkSchedulerError):
    """The requested work item is absent from the authoritative ledger."""


class WorkFenceError(WorkSchedulerError):
    """A lease capability is stale, expired, or no longer active."""


class WorkStateError(WorkSchedulerError):
    """The requested transition is illegal for the current state."""


class SQLiteWorkScheduler:
    """Local durable scheduler with atomic SQLite claim and fenced mutation."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        if self._database_path.exists() and self._database_path.is_dir():
            raise WorkStorageError("work-ledger path must identify a file")
        if not self._database_path.parent.exists():
            raise WorkStorageError("work-ledger parent directory does not exist")
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def plan(
        self,
        plan: WorkItemPlan,
        dependencies: Sequence[WorkDependency] = (),
    ) -> WorkItem:
        """Persist one immutable plan and atomically derive initial readiness."""

        checked_plan = _require_model(plan, WorkItemPlan, "plan")
        checked_dependencies = tuple(
            _require_model(value, WorkDependency, "dependency") for value in dependencies
        )
        for dependency in checked_dependencies:
            if dependency.downstream_work_item_id != checked_plan.work_item_id:
                raise WorkConflictError("dependency downstream id does not match the plan")
        if len({value.dependency_id for value in checked_dependencies}) != len(
            checked_dependencies
        ):
            raise WorkConflictError("dependency ids must be unique")
        if len({value.upstream_work_item_id for value in checked_dependencies}) != len(
            checked_dependencies
        ):
            raise WorkConflictError("upstream dependencies must be unique")

        def operation(connection: sqlite3.Connection) -> WorkItem:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE work_item_id = ? OR work_logical_key = ?
                ORDER BY work_item_id
                """,
                (checked_plan.work_item_id, checked_plan.work_logical_key),
            ).fetchall()
            expected_dependencies = tuple(
                sorted(checked_dependencies, key=lambda value: value.dependency_id)
            )
            if rows:
                if len(rows) != 1:
                    raise WorkConflictError("work id and logical key identify different rows")
                existing = self._item_from_row(rows[0])
                if (
                    existing.work_item_id != checked_plan.work_item_id
                    or existing.work_logical_key != checked_plan.work_logical_key
                    or not _plan_matches(existing, checked_plan)
                    or self._dependencies(connection, existing.work_item_id)
                    != expected_dependencies
                ):
                    raise WorkConflictError("work-plan replay conflicts with persisted policy")
                return existing

            for dependency in checked_dependencies:
                upstream = connection.execute(
                    "SELECT run_id FROM work_items WHERE work_item_id = ?",
                    (dependency.upstream_work_item_id,),
                ).fetchone()
                if upstream is None:
                    raise WorkConflictError("every upstream dependency must already be planned")
                if _row_text(upstream, "run_id") != checked_plan.run_id:
                    raise WorkConflictError("dependencies cannot cross processing runs")

            created_at = _normalize_timestamp(checked_plan.created_at)
            connection.execute(
                """
                INSERT INTO work_items (
                    schema_version, work_item_id, work_logical_key, run_id, mcap_id,
                    stage, subject_type, subject_id, input_digest, config_digest,
                    priority, sla_deadline_at, execution_expiry_at, max_attempts,
                    trace_id, created_at, state, cancel_requested, lease_epoch,
                    fencing_token, leased_by, lease_expires_at, attempt,
                    retry_not_before_at, terminal_reason_code, terminal_reason_detail,
                    result_reference, result_sha256, completed_at, updated_at, row_version
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    0, 0, NULL, NULL, NULL, 0, NULL, NULL, NULL,
                    NULL, NULL, NULL, ?, 0
                )
                """,
                (
                    checked_plan.schema_version,
                    checked_plan.work_item_id,
                    checked_plan.work_logical_key,
                    checked_plan.run_id,
                    checked_plan.mcap_id,
                    checked_plan.stage.value,
                    checked_plan.subject_type.value,
                    checked_plan.subject_id,
                    checked_plan.input_digest,
                    checked_plan.config_digest,
                    checked_plan.priority,
                    _normalize_optional_timestamp(checked_plan.sla_deadline_at),
                    _normalize_optional_timestamp(checked_plan.execution_expiry_at),
                    checked_plan.max_attempts,
                    checked_plan.trace_id,
                    created_at,
                    WorkItemState.PLANNED.value,
                    created_at,
                ),
            )
            for dependency in checked_dependencies:
                connection.execute(
                    """
                    INSERT INTO work_dependencies (
                        dependency_id, downstream_work_item_id,
                        upstream_work_item_id, criticality
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        dependency.dependency_id,
                        dependency.downstream_work_item_id,
                        dependency.upstream_work_item_id,
                        dependency.criticality.value,
                    ),
                )
            self._refresh_planned(connection, _parse_timestamp(created_at))
            return self._load_item(connection, checked_plan.work_item_id)

        return self._transaction(write=True, operation=operation)

    def get(self, work_item_id: str) -> WorkItem:
        """Read one snapshot without implicitly advancing wall-clock time."""

        return self._transaction(
            write=False,
            operation=lambda connection: self._load_item(connection, work_item_id),
        )

    def dependencies(self, work_item_id: str) -> tuple[WorkDependency, ...]:
        """Return upstream edges in deterministic dependency-id order."""

        def operation(connection: sqlite3.Connection) -> tuple[WorkDependency, ...]:
            self._load_item(connection, work_item_id)
            return self._dependencies(connection, work_item_id)

        return self._transaction(write=False, operation=operation)

    def list_attempts(self, work_item_id: str) -> tuple[WorkAttempt, ...]:
        """Return append-oriented execution history for one work item."""

        def operation(connection: sqlite3.Connection) -> tuple[WorkAttempt, ...]:
            self._load_item(connection, work_item_id)
            rows = connection.execute(
                """
                SELECT * FROM work_attempts
                WHERE work_item_id = ? ORDER BY attempt_number
                """,
                (work_item_id,),
            ).fetchall()
            return tuple(self._attempt_from_row(row) for row in rows)

        return self._transaction(write=False, operation=operation)

    def reconcile(self, *, now: datetime | None = None) -> int:
        """Advance hard expiry, abandoned leases, retries, and dependencies."""

        checked_now = _checked_now(now)
        return self._transaction(
            write=True,
            operation=lambda connection: self._maintain(connection, checked_now),
        )

    def claim(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> WorkLeaseClaim | None:
        """Atomically claim the highest-priority READY item or one exact item."""

        checked_worker = _nonempty(worker_id, "worker_id")
        duration = _positive_int(lease_duration_seconds, "lease_duration_seconds")
        checked_work_item_id = (
            None if work_item_id is None else _nonempty(work_item_id, "work_item_id")
        )
        checked_now = _checked_now(now)

        def operation(connection: sqlite3.Connection) -> WorkLeaseClaim | None:
            self._maintain(connection, checked_now)
            row = connection.execute(
                """
                SELECT * FROM work_items
                WHERE state = ? AND (? IS NULL OR work_item_id = ?)
                ORDER BY
                    priority DESC,
                    CASE WHEN sla_deadline_at IS NULL THEN 1 ELSE 0 END,
                    sla_deadline_at,
                    created_at,
                    work_item_id
                LIMIT 1
                """,
                (
                    WorkItemState.READY.value,
                    checked_work_item_id,
                    checked_work_item_id,
                ),
            ).fetchone()
            if row is None:
                return None
            work_item_id = _row_text(row, "work_item_id")
            row_version = _row_int(row, "row_version")
            lease_epoch = _row_int(row, "lease_epoch") + 1
            attempt = _row_int(row, "attempt") + 1
            fencing_token = str(uuid4())
            timestamp = _format_timestamp(checked_now)
            lease_expires_at = _format_timestamp(checked_now + timedelta(seconds=duration))
            cursor = connection.execute(
                """
                UPDATE work_items
                SET state = ?, lease_epoch = ?, fencing_token = ?, leased_by = ?,
                    lease_expires_at = ?, attempt = ?, updated_at = ?,
                    row_version = row_version + 1
                WHERE work_item_id = ? AND state = ? AND row_version = ?
                """,
                (
                    WorkItemState.LEASED.value,
                    lease_epoch,
                    fencing_token,
                    checked_worker,
                    lease_expires_at,
                    attempt,
                    timestamp,
                    work_item_id,
                    WorkItemState.READY.value,
                    row_version,
                ),
            )
            if cursor.rowcount != 1:
                raise WorkStorageError("atomic work claim lost its compare-and-swap")
            connection.execute(
                """
                INSERT INTO work_attempts (
                    work_item_id, attempt_number, lease_epoch, fencing_token,
                    worker_id, claimed_at, started_at, completed_at, outcome,
                    error_code, error_detail
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL)
                """,
                (
                    work_item_id,
                    attempt,
                    lease_epoch,
                    fencing_token,
                    checked_worker,
                    timestamp,
                    WorkAttemptOutcome.ACTIVE.value,
                ),
            )
            item = self._load_item(connection, work_item_id)
            lease = WorkLease(
                work_item_id=work_item_id,
                worker_id=checked_worker,
                lease_epoch=lease_epoch,
                fencing_token=fencing_token,
                lease_expires_at=lease_expires_at,
            )
            return WorkLeaseClaim(work_item=item, lease=lease)

        return self._transaction(write=True, operation=operation)

    def start(self, lease: WorkLease, *, now: datetime | None = None) -> WorkItem:
        """Move an exact live claim from LEASED to RUNNING."""

        checked_lease = _require_model(lease, WorkLease, "lease")
        checked_now = _checked_now(now)
        self.reconcile(now=checked_now)

        def operation(connection: sqlite3.Connection) -> WorkItem:
            item = self._require_lease(connection, checked_lease, checked_now)
            if item.state is WorkItemState.RUNNING:
                return item
            if item.state is not WorkItemState.LEASED:
                raise WorkStateError("only leased work can start")
            timestamp = _format_timestamp(checked_now)
            cursor = connection.execute(
                """
                UPDATE work_items
                SET state = ?, updated_at = ?, row_version = row_version + 1
                WHERE work_item_id = ? AND state = ? AND lease_epoch = ?
                    AND fencing_token = ?
                """,
                (
                    WorkItemState.RUNNING.value,
                    timestamp,
                    item.work_item_id,
                    WorkItemState.LEASED.value,
                    checked_lease.lease_epoch,
                    checked_lease.fencing_token,
                ),
            )
            if cursor.rowcount != 1:
                raise WorkFenceError("work lease changed while starting")
            attempt_cursor = connection.execute(
                """
                UPDATE work_attempts
                SET started_at = COALESCE(started_at, ?)
                WHERE work_item_id = ? AND attempt_number = ? AND outcome = ?
                """,
                (
                    timestamp,
                    item.work_item_id,
                    item.attempt,
                    WorkAttemptOutcome.ACTIVE.value,
                ),
            )
            if attempt_cursor.rowcount != 1:
                raise WorkStorageError("active attempt is missing while starting")
            return self._load_item(connection, item.work_item_id)

        return self._transaction(write=True, operation=operation)

    def heartbeat(
        self,
        lease: WorkLease,
        lease_duration_seconds: int,
        *,
        now: datetime | None = None,
    ) -> WorkLease:
        """Renew an active lease and return its updated immutable capability."""

        checked_lease = _require_model(lease, WorkLease, "lease")
        duration = _positive_int(lease_duration_seconds, "lease_duration_seconds")
        checked_now = _checked_now(now)
        self.reconcile(now=checked_now)

        def operation(connection: sqlite3.Connection) -> WorkLease:
            item = self._require_lease(connection, checked_lease, checked_now)
            expires_at = _format_timestamp(checked_now + timedelta(seconds=duration))
            cursor = connection.execute(
                """
                UPDATE work_items
                SET lease_expires_at = ?, updated_at = ?, row_version = row_version + 1
                WHERE work_item_id = ? AND lease_epoch = ? AND fencing_token = ?
                    AND state IN (?, ?)
                """,
                (
                    expires_at,
                    _format_timestamp(checked_now),
                    item.work_item_id,
                    checked_lease.lease_epoch,
                    checked_lease.fencing_token,
                    WorkItemState.LEASED.value,
                    WorkItemState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise WorkFenceError("work lease changed while renewing")
            return WorkLease(
                work_item_id=item.work_item_id,
                worker_id=checked_lease.worker_id,
                lease_epoch=checked_lease.lease_epoch,
                fencing_token=checked_lease.fencing_token,
                lease_expires_at=expires_at,
            )

        return self._transaction(write=True, operation=operation)

    def succeed(
        self,
        lease: WorkLease,
        *,
        result_reference: str | None = None,
        result_sha256: str | None = None,
        now: datetime | None = None,
    ) -> WorkItem:
        """Commit success under the exact running epoch and fencing token."""

        checked_lease = _require_model(lease, WorkLease, "lease")
        if (result_reference is None) != (result_sha256 is None):
            raise ValueError("result reference and digest must be supplied together")
        if result_reference is not None:
            _nonempty(result_reference, "result_reference")
            _sha256(result_sha256)
        checked_now = _checked_now(now)
        self.reconcile(now=checked_now)

        def operation(connection: sqlite3.Connection) -> WorkItem:
            item = self._require_lease(connection, checked_lease, checked_now)
            if item.state is not WorkItemState.RUNNING:
                raise WorkStateError("only running work can succeed")
            self._finish_attempt(
                connection,
                item,
                WorkAttemptOutcome.SUCCEEDED,
                checked_now,
            )
            self._set_terminal(
                connection,
                item,
                WorkItemState.SUCCEEDED,
                checked_now,
                reason_code=None,
                reason_detail=None,
                result_reference=result_reference,
                result_sha256=result_sha256,
            )
            self._maintain(connection, checked_now)
            return self._load_item(connection, item.work_item_id)

        return self._transaction(write=True, operation=operation)

    def fail(
        self,
        lease: WorkLease,
        *,
        error_code: str,
        retryable: bool,
        error_detail: str | None = None,
        retry_delay_seconds: int = 0,
        now: datetime | None = None,
    ) -> WorkItem:
        """Record failure and either enter RETRY_WAIT or fail permanently."""

        checked_lease = _require_model(lease, WorkLease, "lease")
        checked_code = _nonempty(error_code, "error_code")
        checked_detail = None if error_detail is None else _nonempty(error_detail, "error_detail")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean")
        delay = _nonnegative_int(retry_delay_seconds, "retry_delay_seconds")
        checked_now = _checked_now(now)
        self.reconcile(now=checked_now)

        def operation(connection: sqlite3.Connection) -> WorkItem:
            item = self._require_lease(connection, checked_lease, checked_now)
            retry_at = checked_now + timedelta(seconds=delay)
            before_hard_expiry = item.execution_expiry_at is None or retry_at < _parse_timestamp(
                item.execution_expiry_at
            )
            will_retry = retryable and item.attempt < item.max_attempts and before_hard_expiry
            self._finish_attempt(
                connection,
                item,
                (
                    WorkAttemptOutcome.FAILED_RETRYABLE
                    if will_retry
                    else WorkAttemptOutcome.FAILED_PERMANENT
                ),
                checked_now,
                error_code=checked_code,
                error_detail=checked_detail,
            )
            if will_retry:
                cursor = connection.execute(
                    """
                    UPDATE work_items
                    SET state = ?, fencing_token = NULL, leased_by = NULL,
                        lease_expires_at = NULL, retry_not_before_at = ?,
                        updated_at = ?, row_version = row_version + 1
                    WHERE work_item_id = ? AND lease_epoch = ? AND fencing_token = ?
                        AND state IN (?, ?)
                    """,
                    (
                        WorkItemState.RETRY_WAIT.value,
                        _format_timestamp(retry_at),
                        _format_timestamp(checked_now),
                        item.work_item_id,
                        checked_lease.lease_epoch,
                        checked_lease.fencing_token,
                        WorkItemState.LEASED.value,
                        WorkItemState.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise WorkFenceError("work lease changed while recording retry")
            else:
                self._set_terminal(
                    connection,
                    item,
                    WorkItemState.FAILED_PERMANENT,
                    checked_now,
                    reason_code=checked_code,
                    reason_detail=checked_detail,
                )
                self._maintain(connection, checked_now)
            return self._load_item(connection, item.work_item_id)

        return self._transaction(write=True, operation=operation)

    def cancel(
        self,
        work_item_id: str,
        *,
        reason_code: str = "CANCELLED_BY_REQUEST",
        reason_detail: str | None = None,
        now: datetime | None = None,
    ) -> WorkItem:
        """Cancel nonterminal work and revoke any active lease."""

        return self._direct_terminal(
            work_item_id,
            state=WorkItemState.CANCELLED,
            attempt_outcome=WorkAttemptOutcome.CANCELLED,
            reason_code=reason_code,
            reason_detail=reason_detail,
            cancel_requested=True,
            now=now,
        )

    def skip(
        self,
        work_item_id: str,
        *,
        state: WorkItemState,
        reason_code: str,
        reason_detail: str | None = None,
        now: datetime | None = None,
    ) -> WorkItem:
        """Apply one explicit policy/not-needed terminal outcome."""

        if state not in {
            WorkItemState.SKIPPED_POLICY,
            WorkItemState.SKIPPED_NOT_NEEDED,
        }:
            raise ValueError("skip state must be SKIPPED_POLICY or SKIPPED_NOT_NEEDED")
        return self._direct_terminal(
            work_item_id,
            state=state,
            attempt_outcome=WorkAttemptOutcome.SKIPPED,
            reason_code=reason_code,
            reason_detail=reason_detail,
            now=now,
        )

    def invalidate(
        self,
        work_item_id: str,
        *,
        reason_code: str,
        reason_detail: str | None = None,
        now: datetime | None = None,
    ) -> WorkItem:
        """Invalidate nonterminal work and fence an active worker."""

        return self._direct_terminal(
            work_item_id,
            state=WorkItemState.INVALIDATED,
            attempt_outcome=WorkAttemptOutcome.INVALIDATED,
            reason_code=reason_code,
            reason_detail=reason_detail,
            now=now,
        )

    def _direct_terminal(
        self,
        work_item_id: str,
        *,
        state: WorkItemState,
        attempt_outcome: WorkAttemptOutcome,
        reason_code: str,
        reason_detail: str | None,
        cancel_requested: bool = False,
        now: datetime | None,
    ) -> WorkItem:
        checked_code = _nonempty(reason_code, "reason_code")
        checked_detail = (
            None if reason_detail is None else _nonempty(reason_detail, "reason_detail")
        )
        checked_now = _checked_now(now)
        self.reconcile(now=checked_now)

        def operation(connection: sqlite3.Connection) -> WorkItem:
            item = self._load_item(connection, work_item_id)
            if item.state is state:
                return item
            if item.state in TERMINAL_WORK_STATES:
                raise WorkStateError(
                    f"cannot move terminal {item.state.value} work to {state.value}"
                )
            if item.state in {WorkItemState.LEASED, WorkItemState.RUNNING}:
                self._finish_attempt(
                    connection,
                    item,
                    attempt_outcome,
                    checked_now,
                    error_code=checked_code,
                    error_detail=checked_detail,
                )
            self._set_terminal(
                connection,
                item,
                state,
                checked_now,
                reason_code=checked_code,
                reason_detail=checked_detail,
                cancel_requested=cancel_requested,
            )
            self._maintain(connection, checked_now)
            return self._load_item(connection, work_item_id)

        return self._transaction(write=True, operation=operation)

    def _maintain(self, connection: sqlite3.Connection, now: datetime) -> int:
        changed = self._expire_hard_deadlines(connection, now)
        changed += self._recover_expired_leases(connection, now)
        timestamp = _format_timestamp(now)
        cursor = connection.execute(
            """
            UPDATE work_items
            SET state = ?, retry_not_before_at = NULL, updated_at = ?,
                row_version = row_version + 1
            WHERE state = ? AND retry_not_before_at <= ?
            """,
            (
                WorkItemState.READY.value,
                timestamp,
                WorkItemState.RETRY_WAIT.value,
                timestamp,
            ),
        )
        changed += cursor.rowcount
        changed += self._refresh_planned(connection, now)
        return changed

    def _expire_hard_deadlines(
        self,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> int:
        rows = connection.execute(
            """
            SELECT * FROM work_items
            WHERE completed_at IS NULL AND execution_expiry_at IS NOT NULL
            ORDER BY work_item_id
            """
        ).fetchall()
        changed = 0
        for row in rows:
            item = self._item_from_row(row)
            expiry = item.execution_expiry_at
            if expiry is None or _parse_timestamp(expiry) > now:
                continue
            if item.state in {WorkItemState.LEASED, WorkItemState.RUNNING}:
                self._finish_attempt(
                    connection,
                    item,
                    WorkAttemptOutcome.EXPIRED,
                    now,
                    error_code="EXECUTION_EXPIRED",
                )
            self._set_terminal(
                connection,
                item,
                WorkItemState.EXPIRED,
                now,
                reason_code="EXECUTION_EXPIRED",
                reason_detail=None,
            )
            changed += 1
        return changed

    def _recover_expired_leases(
        self,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> int:
        rows = connection.execute(
            """
            SELECT * FROM work_items
            WHERE state IN (?, ?) ORDER BY work_item_id
            """,
            (WorkItemState.LEASED.value, WorkItemState.RUNNING.value),
        ).fetchall()
        changed = 0
        for row in rows:
            item = self._item_from_row(row)
            lease_expiry = item.lease_expires_at
            if lease_expiry is None or _parse_timestamp(lease_expiry) > now:
                continue
            self._finish_attempt(
                connection,
                item,
                WorkAttemptOutcome.ABANDONED,
                now,
                error_code="LEASE_EXPIRED",
            )
            if item.attempt < item.max_attempts:
                cursor = connection.execute(
                    """
                    UPDATE work_items
                    SET state = ?, fencing_token = NULL, leased_by = NULL,
                        lease_expires_at = NULL, updated_at = ?,
                        row_version = row_version + 1
                    WHERE work_item_id = ? AND lease_epoch = ? AND fencing_token = ?
                    """,
                    (
                        WorkItemState.READY.value,
                        _format_timestamp(now),
                        item.work_item_id,
                        item.lease_epoch,
                        item.fencing_token,
                    ),
                )
                if cursor.rowcount != 1:
                    raise WorkFenceError("expired lease changed during recovery")
            else:
                self._set_terminal(
                    connection,
                    item,
                    WorkItemState.FAILED_PERMANENT,
                    now,
                    reason_code="LEASE_ATTEMPTS_EXHAUSTED",
                    reason_detail=None,
                )
            changed += 1
        return changed

    def _refresh_planned(
        self,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> int:
        changed = 0
        while True:
            pass_changed = 0
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE state = ? ORDER BY created_at, work_item_id
                """,
                (WorkItemState.PLANNED.value,),
            ).fetchall()
            for row in rows:
                item = self._item_from_row(row)
                dependency_rows = connection.execute(
                    """
                    SELECT d.criticality, upstream.state
                    FROM work_dependencies AS d
                    JOIN work_items AS upstream
                      ON upstream.work_item_id = d.upstream_work_item_id
                    WHERE d.downstream_work_item_id = ?
                    ORDER BY d.dependency_id
                    """,
                    (item.work_item_id,),
                ).fetchall()
                if any(
                    WorkItemState(_row_text(value, "state")) not in TERMINAL_WORK_STATES
                    for value in dependency_rows
                ):
                    continue
                required_failure = any(
                    DependencyCriticality(_row_text(value, "criticality"))
                    is DependencyCriticality.REQUIRED
                    and WorkItemState(_row_text(value, "state")) not in SUCCESSFUL_DEPENDENCY_STATES
                    for value in dependency_rows
                )
                if required_failure:
                    self._set_terminal(
                        connection,
                        item,
                        WorkItemState.FAILED_PERMANENT,
                        now,
                        reason_code="REQUIRED_DEPENDENCY_FAILED",
                        reason_detail=None,
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE work_items
                        SET state = ?, updated_at = ?, row_version = row_version + 1
                        WHERE work_item_id = ? AND state = ?
                        """,
                        (
                            WorkItemState.READY.value,
                            _format_timestamp(now),
                            item.work_item_id,
                            WorkItemState.PLANNED.value,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise WorkStorageError(
                            "planned readiness transition lost its compare-and-swap"
                        )
                pass_changed += 1
            changed += pass_changed
            if pass_changed == 0:
                return changed

    def _require_lease(
        self,
        connection: sqlite3.Connection,
        lease: WorkLease,
        now: datetime,
    ) -> WorkItem:
        item = self._load_item(connection, lease.work_item_id)
        if (
            item.state not in {WorkItemState.LEASED, WorkItemState.RUNNING}
            or item.lease_epoch != lease.lease_epoch
            or item.fencing_token != lease.fencing_token
            or item.leased_by != lease.worker_id
            or item.lease_expires_at is None
            or _parse_timestamp(item.lease_expires_at) <= now
        ):
            raise WorkFenceError("work lease is stale, expired, or inactive")
        return item

    def _finish_attempt(
        self,
        connection: sqlite3.Connection,
        item: WorkItem,
        outcome: WorkAttemptOutcome,
        now: datetime,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE work_attempts
            SET completed_at = ?, outcome = ?, error_code = ?, error_detail = ?
            WHERE work_item_id = ? AND attempt_number = ? AND lease_epoch = ?
                AND fencing_token = ? AND outcome = ?
            """,
            (
                _format_timestamp(now),
                outcome.value,
                error_code,
                error_detail,
                item.work_item_id,
                item.attempt,
                item.lease_epoch,
                item.fencing_token,
                WorkAttemptOutcome.ACTIVE.value,
            ),
        )
        if cursor.rowcount != 1:
            raise WorkStorageError("active attempt does not match its work-item lease")

    def _set_terminal(
        self,
        connection: sqlite3.Connection,
        item: WorkItem,
        state: WorkItemState,
        now: datetime,
        *,
        reason_code: str | None,
        reason_detail: str | None,
        result_reference: str | None = None,
        result_sha256: str | None = None,
        cancel_requested: bool = False,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE work_items
            SET state = ?, cancel_requested = ?,
                fencing_token = NULL, leased_by = NULL, lease_expires_at = NULL,
                retry_not_before_at = NULL, terminal_reason_code = ?,
                terminal_reason_detail = ?, result_reference = ?, result_sha256 = ?,
                completed_at = ?, updated_at = ?, row_version = row_version + 1
            WHERE work_item_id = ? AND completed_at IS NULL
            """,
            (
                state.value,
                int(cancel_requested),
                reason_code,
                reason_detail,
                result_reference,
                result_sha256,
                _format_timestamp(now),
                _format_timestamp(now),
                item.work_item_id,
            ),
        )
        if cursor.rowcount != 1:
            raise WorkStateError("work became terminal before the requested transition")

    def _dependencies(
        self,
        connection: sqlite3.Connection,
        work_item_id: str,
    ) -> tuple[WorkDependency, ...]:
        rows = connection.execute(
            """
            SELECT * FROM work_dependencies
            WHERE downstream_work_item_id = ? ORDER BY dependency_id
            """,
            (work_item_id,),
        ).fetchall()
        return tuple(
            WorkDependency(
                dependency_id=_row_text(row, "dependency_id"),
                downstream_work_item_id=_row_text(row, "downstream_work_item_id"),
                upstream_work_item_id=_row_text(row, "upstream_work_item_id"),
                criticality=DependencyCriticality(_row_text(row, "criticality")),
            )
            for row in rows
        )

    def _load_item(
        self,
        connection: sqlite3.Connection,
        work_item_id: str,
    ) -> WorkItem:
        row = connection.execute(
            "SELECT * FROM work_items WHERE work_item_id = ?",
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise WorkNotFoundError(f"work item is not registered: {work_item_id}")
        return self._item_from_row(row)

    def _item_from_row(self, row: sqlite3.Row) -> WorkItem:
        schema_version = _row_text(row, "schema_version")
        if schema_version != "1.0":
            raise WorkStorageError("persisted work schema_version is unsupported")
        return WorkItem(
            schema_version="1.0",
            work_item_id=_row_text(row, "work_item_id"),
            work_logical_key=_row_text(row, "work_logical_key"),
            run_id=_row_text(row, "run_id"),
            mcap_id=_row_text(row, "mcap_id"),
            stage=Stage(_row_text(row, "stage")),
            subject_type=WorkItemSubjectType(_row_text(row, "subject_type")),
            subject_id=_row_text(row, "subject_id"),
            input_digest=_row_text(row, "input_digest"),
            config_digest=_row_text(row, "config_digest"),
            priority=_row_int(row, "priority"),
            sla_deadline_at=_row_optional_text(row, "sla_deadline_at"),
            execution_expiry_at=_row_optional_text(row, "execution_expiry_at"),
            max_attempts=_row_int(row, "max_attempts"),
            trace_id=_row_optional_text(row, "trace_id"),
            created_at=_row_text(row, "created_at"),
            state=WorkItemState(_row_text(row, "state")),
            cancel_requested=bool(_row_int(row, "cancel_requested")),
            lease_epoch=_row_int(row, "lease_epoch"),
            fencing_token=_row_optional_text(row, "fencing_token"),
            leased_by=_row_optional_text(row, "leased_by"),
            lease_expires_at=_row_optional_text(row, "lease_expires_at"),
            attempt=_row_int(row, "attempt"),
            retry_not_before_at=_row_optional_text(row, "retry_not_before_at"),
            terminal_reason_code=_row_optional_text(row, "terminal_reason_code"),
            terminal_reason_detail=_row_optional_text(row, "terminal_reason_detail"),
            result_reference=_row_optional_text(row, "result_reference"),
            result_sha256=_row_optional_text(row, "result_sha256"),
            completed_at=_row_optional_text(row, "completed_at"),
            updated_at=_row_text(row, "updated_at"),
            row_version=_row_int(row, "row_version"),
        )

    def _attempt_from_row(self, row: sqlite3.Row) -> WorkAttempt:
        return WorkAttempt(
            work_item_id=_row_text(row, "work_item_id"),
            attempt_number=_row_int(row, "attempt_number"),
            lease_epoch=_row_int(row, "lease_epoch"),
            fencing_token=_row_text(row, "fencing_token"),
            worker_id=_row_text(row, "worker_id"),
            claimed_at=_row_text(row, "claimed_at"),
            started_at=_row_optional_text(row, "started_at"),
            completed_at=_row_optional_text(row, "completed_at"),
            outcome=WorkAttemptOutcome(_row_text(row, "outcome")),
            error_code=_row_optional_text(row, "error_code"),
            error_detail=_row_optional_text(row, "error_detail"),
        )

    def _initialize_database(self) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = self._open(check_header=False)
            journal = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if journal is None or str(journal[0]).lower() != "wal":
                raise WorkStorageError("SQLite WAL mode could not be enabled")
            connection.execute("BEGIN IMMEDIATE")
            user_version = _pragma_int(connection, "user_version")
            application_id = _pragma_int(connection, "application_id")
            inventory = connection.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'
                )
                """
            ).fetchone()
            has_schema = inventory is not None and int(inventory[0]) == 1
            if user_version == 0:
                if application_id != 0 or has_schema:
                    raise WorkStorageError("refusing to adopt a nonempty unversioned work database")
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif user_version != _SCHEMA_VERSION or application_id != _APPLICATION_ID:
                raise WorkStorageError("work database belongs to another application or version")
            connection.commit()
        except WorkSchedulerError:
            if connection is not None:
                _rollback_quietly(connection)
            raise
        except sqlite3.Error as error:
            if connection is not None:
                _rollback_quietly(connection)
            raise WorkStorageError(f"cannot initialize SQLite work scheduler: {error}") from error
        finally:
            if connection is not None:
                connection.close()

    def _open(self, *, check_header: bool = True) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._database_path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            if check_header:
                self._verify_header(connection)
            return connection
        except (sqlite3.Error, WorkStorageError) as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            if isinstance(error, WorkStorageError):
                raise
            raise WorkStorageError(f"cannot open SQLite work scheduler: {error}") from error

    def _verify_header(self, connection: sqlite3.Connection) -> None:
        if (
            _pragma_int(connection, "application_id") != _APPLICATION_ID
            or _pragma_int(connection, "user_version") != _SCHEMA_VERSION
        ):
            raise WorkStorageError("work database header changed")

    def _transaction[T](
        self,
        *,
        write: bool,
        operation: Callable[[sqlite3.Connection], T],
    ) -> T:
        connection = self._open()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            result = operation(connection)
            connection.commit()
            return result
        except (WorkSchedulerError, TypeError, ValueError):
            _rollback_quietly(connection)
            raise
        except sqlite3.IntegrityError as error:
            _rollback_quietly(connection)
            raise WorkConflictError(f"SQLite rejected durable work: {error}") from error
        except sqlite3.Error as error:
            _rollback_quietly(connection)
            raise WorkStorageError(f"work scheduler transaction failed: {error}") from error
        finally:
            connection.close()


def _require_model[T](value: object, model_type: type[T], label: str) -> T:
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    return value


def _plan_matches(item: WorkItem, plan: WorkItemPlan) -> bool:
    direct_fields = (
        "schema_version",
        "work_item_id",
        "work_logical_key",
        "run_id",
        "mcap_id",
        "stage",
        "subject_type",
        "subject_id",
        "input_digest",
        "config_digest",
        "priority",
        "max_attempts",
        "trace_id",
    )
    if any(getattr(item, field) != getattr(plan, field) for field in direct_fields):
        return False
    for field in ("created_at", "sla_deadline_at", "execution_expiry_at"):
        item_value = getattr(item, field)
        plan_value = getattr(plan, field)
        if (item_value is None) != (plan_value is None):
            return False
        if (
            item_value is not None
            and plan_value is not None
            and _parse_timestamp(item_value) != _parse_timestamp(plan_value)
        ):
            return False
    return True


def _checked_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if not isinstance(value, datetime):
        raise TypeError("now must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(UTC)


def _parse_timestamp(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _normalize_timestamp(value: str) -> str:
    return _format_timestamp(_parse_timestamp(value))


def _normalize_optional_timestamp(value: str | None) -> str | None:
    return None if value is None else _normalize_timestamp(value)


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _nonnegative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return value


def _sha256(value: str | None) -> str:
    if (
        value is None
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("result_sha256 must be a lowercase SHA-256 digest")
    return value


def _row_text(row: sqlite3.Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise WorkStorageError(f"persisted {column} must be text")
    return value


def _row_optional_text(row: sqlite3.Row, column: str) -> str | None:
    value: object = row[column]
    if value is not None and not isinstance(value, str):
        raise WorkStorageError(f"persisted {column} must be text or null")
    return value


def _row_int(row: sqlite3.Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkStorageError(f"persisted {column} must be an integer")
    return value


def _pragma_int(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(f"PRAGMA {name}").fetchone()
    value: object = None if row is None else row[0]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkStorageError(f"SQLite PRAGMA {name} returned no integer")
    return value


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        connection.rollback()


__all__ = [
    "SQLiteWorkScheduler",
    "WorkConflictError",
    "WorkFenceError",
    "WorkNotFoundError",
    "WorkSchedulerError",
    "WorkStateError",
    "WorkStorageError",
]
