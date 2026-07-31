"""PostgreSQL implementation of the canonical durable-work ledger.

The scheduler preserves the immutable plan, dependency, lease, retry, fencing, and
recovery semantics of the local conformance scheduler. PostgreSQL is the authority;
broker delivery remains a projection. Migrations, rather than adapter construction,
create its tables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from robata.adapters.postgres_authority import (
    PostgresCanonicalAuthority,
    PostgresConnection,
    Row,
    postgres_sqlstate,
)
from robata.adapters.sqlite_work_scheduler import (
    WorkConflictError,
    WorkFenceError,
    WorkNotFoundError,
    WorkSchedulerError,
    WorkStateError,
    WorkStorageError,
)
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


class PostgresWorkScheduler:
    """Durable PostgreSQL scheduler with serializable, fenced state transitions."""

    backend_kind = "POSTGRESQL"

    def __init__(self, authority: PostgresCanonicalAuthority) -> None:
        if not isinstance(authority, PostgresCanonicalAuthority):
            raise TypeError("authority must be PostgresCanonicalAuthority")
        self._authority = authority

    @property
    def authority(self) -> PostgresCanonicalAuthority:
        """The shared transaction authority for stream/capture/delivery extensions."""

        return self._authority

    @property
    def schema(self) -> str:
        return self._authority.schema

    def verify_startup(self) -> object:
        """Verify P22's migration surface before a production composition starts."""

        return self._authority.verify_startup()

    def run_authority_transaction[T](
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[PostgresConnection], T],
    ) -> T:
        return self._transaction(write=write, operation_name=operation_name, operation=operation)

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
        self._validate_plan_dependencies(checked_plan, checked_dependencies)

        def operation(connection: PostgresConnection) -> WorkItem:
            self._lock_plan_replay_identities(connection, (checked_plan,))
            return self._plan_in_transaction(
                connection,
                checked_plan,
                checked_dependencies,
                refresh_readiness=True,
            )

        return self._transaction(write=True, operation_name="plan", operation=operation)

    def plan_many(
        self,
        plans: Sequence[tuple[WorkItemPlan, Sequence[WorkDependency]]],
    ) -> tuple[WorkItem, ...]:
        """Persist one topologically ordered work batch in one authority transaction.

        The operation retains the exact-replay checks of :meth:`plan`. A caller can
        retry a partially published stream batch after a crash without turning a
        duplicate delivery into another execution item. Dependencies may refer to
        an earlier member of ``plans`` or an item already in the authority ledger.
        """

        if isinstance(plans, (str, bytes)) or not isinstance(plans, Sequence):
            raise TypeError("plans must be a sequence of plan/dependency pairs")
        checked: tuple[tuple[WorkItemPlan, tuple[WorkDependency, ...]], ...] = tuple(
            (
                _require_model(plan, WorkItemPlan, "plan"),
                tuple(
                    _require_model(value, WorkDependency, "dependency") for value in dependencies
                ),
            )
            for plan, dependencies in plans
        )
        if not checked:
            return ()
        for checked_plan, checked_dependencies in checked:
            self._validate_plan_dependencies(checked_plan, checked_dependencies)

        def operation(connection: PostgresConnection) -> tuple[WorkItem, ...]:
            self._lock_plan_replay_identities(
                connection,
                tuple(plan for plan, _dependencies in checked),
            )
            for checked_plan, checked_dependencies in checked:
                self._plan_in_transaction(
                    connection,
                    checked_plan,
                    checked_dependencies,
                    refresh_readiness=False,
                )
            # Readiness is an execution projection, not an independent durable fact.
            # Derive it only for this bounded batch instead of scanning every pending
            # item already owned by older windows.
            self._refresh_planned_for_work_ids(
                connection,
                max(_parse_timestamp(plan.created_at) for plan, _dependencies in checked),
                tuple(plan.work_item_id for plan, _dependencies in checked),
            )
            return tuple(
                self._load_item(connection, checked_plan.work_item_id)
                for checked_plan, _dependencies in checked
            )

        return self._transaction(write=True, operation_name="plan_many", operation=operation)

    def _validate_plan_dependencies(
        self,
        plan: WorkItemPlan,
        dependencies: Sequence[WorkDependency],
    ) -> None:
        for dependency in dependencies:
            if dependency.downstream_work_item_id != plan.work_item_id:
                raise WorkConflictError("dependency downstream id does not match the plan")
        if len({value.dependency_id for value in dependencies}) != len(dependencies):
            raise WorkConflictError("dependency ids must be unique")
        if len({value.upstream_work_item_id for value in dependencies}) != len(dependencies):
            raise WorkConflictError("upstream dependencies must be unique")

    def _lock_plan_replay_identities(
        self,
        connection: PostgresConnection,
        plans: Sequence[WorkItemPlan],
    ) -> None:
        """Serialize exact-plan replays before PostgreSQL unique-key arbitration.

        The keys include both immutable identifiers and are acquired in sorted order so
        overlapping ``plan_many`` batches cannot deadlock each other. The local tenant
        setting scopes the lock while preserving safe, conservative behavior for a
        deliberately unbound conformance connection.
        """

        replay_keys = sorted(
            {f"work_item:{plan.work_item_id}" for plan in plans}
            | {f"work_logical:{plan.work_logical_key}" for plan in plans}
        )
        for replay_key in replay_keys:
            connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtextextended(
                        COALESCE(current_setting('robata.tenant_id', true), '') || ':' || %s,
                        0
                    )
                )
                """,
                (replay_key,),
            )

    def _plan_in_transaction(
        self,
        connection: PostgresConnection,
        plan: WorkItemPlan,
        dependencies: Sequence[WorkDependency],
        *,
        refresh_readiness: bool,
    ) -> WorkItem:
        """Apply one validated immutable plan inside an authority transaction."""

        rows = connection.execute(
            """
            SELECT * FROM work_items
            WHERE work_item_id = %s OR work_logical_key = %s
            ORDER BY work_item_id
            """,
            (plan.work_item_id, plan.work_logical_key),
        ).fetchall()
        expected_dependencies = tuple(sorted(dependencies, key=lambda value: value.dependency_id))
        if rows:
            if len(rows) != 1:
                raise WorkConflictError("work id and logical key identify different rows")
            existing = self._item_from_row(rows[0])
            if (
                existing.work_item_id != plan.work_item_id
                or existing.work_logical_key != plan.work_logical_key
                or not _plan_matches(existing, plan)
                or self._dependencies(connection, existing.work_item_id) != expected_dependencies
            ):
                raise WorkConflictError("work-plan replay conflicts with persisted policy")
            return existing

        for dependency in dependencies:
            upstream = connection.execute(
                "SELECT run_id FROM work_items WHERE work_item_id = %s",
                (dependency.upstream_work_item_id,),
            ).fetchone()
            if upstream is None:
                raise WorkConflictError("every upstream dependency must already be planned")
            if _row_text(upstream, "run_id") != plan.run_id:
                raise WorkConflictError("dependencies cannot cross processing runs")

        created_at = _normalize_timestamp(plan.created_at)
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
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                0, 0, NULL, NULL, NULL, 0, NULL, NULL, NULL,
                NULL, NULL, NULL, %s, 0
            )
            """,
            (
                plan.schema_version,
                plan.work_item_id,
                plan.work_logical_key,
                plan.run_id,
                plan.mcap_id,
                plan.stage.value,
                plan.subject_type.value,
                plan.subject_id,
                plan.input_digest,
                plan.config_digest,
                plan.priority,
                _normalize_optional_timestamp(plan.sla_deadline_at),
                _normalize_optional_timestamp(plan.execution_expiry_at),
                plan.max_attempts,
                plan.trace_id,
                created_at,
                WorkItemState.PLANNED.value,
                created_at,
            ),
        )
        for dependency in dependencies:
            connection.execute(
                """
                INSERT INTO work_dependencies (
                    dependency_id, downstream_work_item_id,
                    upstream_work_item_id, criticality
                ) VALUES (%s, %s, %s, %s)
                """,
                (
                    dependency.dependency_id,
                    dependency.downstream_work_item_id,
                    dependency.upstream_work_item_id,
                    dependency.criticality.value,
                ),
            )
        if refresh_readiness:
            self._refresh_planned_for_work_ids(
                connection,
                _parse_timestamp(created_at),
                (plan.work_item_id,),
            )
        return self._load_item(connection, plan.work_item_id)

    def get(self, work_item_id: str) -> WorkItem:
        """Read one snapshot without implicitly advancing wall-clock time."""

        return self._transaction(
            write=False,
            operation_name="get",
            operation=lambda connection: self._load_item(connection, work_item_id),
        )

    def items_for_run(self, run_id: str) -> tuple[WorkItem, ...]:
        """Read one run's work snapshots in a single authority transaction."""

        checked_run_id = _nonempty(run_id, "run_id")

        def operation(connection: PostgresConnection) -> tuple[WorkItem, ...]:
            rows = connection.execute(
                "SELECT * FROM work_items WHERE run_id = %s ORDER BY work_item_id",
                (checked_run_id,),
            ).fetchall()
            return tuple(self._item_from_row(row) for row in rows)

        return self._transaction(
            write=False,
            operation_name="items_for_run",
            operation=operation,
        )

    def ready_for_run(self, run_id: str, *, limit: int) -> tuple[WorkItem, ...]:
        """Return a bounded priority-ordered READY slice for one run.

        Queue recovery uses this scoped projection instead of loading every
        durable row in a long recording merely to refill a finite ingress
        buffer.  The matching additive index also preserves the scheduler's
        normal dispatch order.
        """

        checked_run_id = _nonempty(run_id, "run_id")
        checked_limit = _positive_int(limit, "limit")

        def operation(connection: PostgresConnection) -> tuple[WorkItem, ...]:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE run_id = %s AND state = %s
                ORDER BY
                    priority DESC,
                    CASE WHEN sla_deadline_at IS NULL THEN 1 ELSE 0 END,
                    sla_deadline_at,
                    created_at,
                    work_item_id
                LIMIT %s
                """,
                (checked_run_id, WorkItemState.READY.value, checked_limit),
            ).fetchall()
            return tuple(self._item_from_row(row) for row in rows)

        return self._transaction(
            write=False,
            operation_name="ready_for_run",
            operation=operation,
        )

    def dependencies(self, work_item_id: str) -> tuple[WorkDependency, ...]:
        """Return upstream edges in deterministic dependency-id order."""

        def operation(connection: PostgresConnection) -> tuple[WorkDependency, ...]:
            self._load_item(connection, work_item_id)
            return self._dependencies(connection, work_item_id)

        return self._transaction(write=False, operation_name="dependencies", operation=operation)

    def list_attempts(self, work_item_id: str) -> tuple[WorkAttempt, ...]:
        """Return append-oriented execution history for one work item."""

        def operation(connection: PostgresConnection) -> tuple[WorkAttempt, ...]:
            self._load_item(connection, work_item_id)
            rows = connection.execute(
                """
                SELECT * FROM work_attempts
                WHERE work_item_id = %s ORDER BY attempt_number
                """,
                (work_item_id,),
            ).fetchall()
            return tuple(self._attempt_from_row(row) for row in rows)

        return self._transaction(write=False, operation_name="list_attempts", operation=operation)

    def reconcile(self, *, now: datetime | None = None) -> int:
        """Advance hard expiry, abandoned leases, retries, and dependencies."""

        checked_now = _checked_now(now)
        return self._transaction(
            write=True,
            operation_name="reconcile",
            operation=lambda connection: self._maintain(
                connection,
                checked_now,
                refresh_all_planned=True,
            ),
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

        return self._claim(
            worker_id,
            lease_duration_seconds,
            work_item_id=work_item_id,
            now=now,
            start_immediately=False,
            operation_name="claim",
        )

    def claim_and_start(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None = None,
        now: datetime | None = None,
    ) -> WorkLeaseClaim | None:
        """Claim and start one READY item in one fenced authority transaction.

        This is appropriate for a same-process executor that begins work immediately.
        Distributed workers can retain the separate :meth:`claim` / :meth:`start`
        handshake while preserving the same lease and attempt semantics.
        """

        return self._claim(
            worker_id,
            lease_duration_seconds,
            work_item_id=work_item_id,
            now=now,
            start_immediately=True,
            operation_name="claim_and_start",
        )

    def _claim(
        self,
        worker_id: str,
        lease_duration_seconds: int,
        *,
        work_item_id: str | None,
        now: datetime | None,
        start_immediately: bool,
        operation_name: str,
    ) -> WorkLeaseClaim | None:
        checked_worker = _nonempty(worker_id, "worker_id")
        duration = _positive_int(lease_duration_seconds, "lease_duration_seconds")
        checked_work_item_id = (
            None if work_item_id is None else _nonempty(work_item_id, "work_item_id")
        )
        checked_now = _checked_now(now)

        return self._transaction(
            write=True,
            operation_name=operation_name,
            operation=lambda connection: self._claim_in_transaction(
                connection,
                worker_id=checked_worker,
                lease_duration_seconds=duration,
                work_item_id=checked_work_item_id,
                now=checked_now,
                start_immediately=start_immediately,
            ),
        )

    def _claim_in_transaction(
        self,
        connection: PostgresConnection,
        *,
        worker_id: str,
        lease_duration_seconds: int,
        work_item_id: str | None,
        now: datetime,
        start_immediately: bool,
    ) -> WorkLeaseClaim | None:
        self._maintain(connection, now)
        if work_item_id is None:
            row = connection.execute(
                """
                SELECT * FROM work_items
                WHERE state = %s
                ORDER BY
                    priority DESC,
                    CASE WHEN sla_deadline_at IS NULL THEN 1 ELSE 0 END,
                    sla_deadline_at,
                    created_at,
                    work_item_id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (WorkItemState.READY.value,),
            ).fetchone()
        else:
            # An exact graph-owned claim is a primary-key lookup, not a ready-queue scan.
            row = connection.execute(
                """
                SELECT * FROM work_items
                WHERE work_item_id = %s AND state = %s
                FOR UPDATE SKIP LOCKED
                """,
                (work_item_id, WorkItemState.READY.value),
            ).fetchone()
        if row is None:
            return None
        claimed_work_item_id = _row_text(row, "work_item_id")
        row_version = _row_int(row, "row_version")
        lease_epoch = _row_int(row, "lease_epoch") + 1
        attempt = _row_int(row, "attempt") + 1
        fencing_token = str(uuid4())
        timestamp = _format_timestamp(now)
        lease_expires_at = _format_timestamp(now + timedelta(seconds=lease_duration_seconds))
        next_state = WorkItemState.RUNNING if start_immediately else WorkItemState.LEASED
        cursor = connection.execute(
            """
            UPDATE work_items
            SET state = %s, lease_epoch = %s, fencing_token = %s, leased_by = %s,
                lease_expires_at = %s, attempt = %s, updated_at = %s,
                row_version = row_version + 1
            WHERE work_item_id = %s AND state = %s AND row_version = %s
            """,
            (
                next_state.value,
                lease_epoch,
                fencing_token,
                worker_id,
                lease_expires_at,
                attempt,
                timestamp,
                claimed_work_item_id,
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
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, %s, NULL, NULL)
            """,
            (
                claimed_work_item_id,
                attempt,
                lease_epoch,
                fencing_token,
                worker_id,
                timestamp,
                timestamp if start_immediately else None,
                WorkAttemptOutcome.ACTIVE.value,
            ),
        )
        item = self._load_item(connection, claimed_work_item_id)
        lease = WorkLease(
            work_item_id=claimed_work_item_id,
            worker_id=worker_id,
            lease_epoch=lease_epoch,
            fencing_token=fencing_token,
            lease_expires_at=lease_expires_at,
        )
        return WorkLeaseClaim(work_item=item, lease=lease)

    def start(self, lease: WorkLease, *, now: datetime | None = None) -> WorkItem:
        """Move an exact live claim from LEASED to RUNNING."""

        checked_lease = _require_model(lease, WorkLease, "lease")
        checked_now = _checked_now(now)

        def operation(connection: PostgresConnection) -> WorkItem:
            item = self._require_lease(connection, checked_lease, checked_now)
            if item.state is WorkItemState.RUNNING:
                return item
            if item.state is not WorkItemState.LEASED:
                raise WorkStateError("only leased work can start")
            timestamp = _format_timestamp(checked_now)
            cursor = connection.execute(
                """
                UPDATE work_items
                SET state = %s, updated_at = %s, row_version = row_version + 1
                WHERE work_item_id = %s AND state = %s AND lease_epoch = %s
                    AND fencing_token = %s
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
                SET started_at = COALESCE(started_at, %s)
                WHERE work_item_id = %s AND attempt_number = %s AND outcome = %s
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

        try:
            return self._transaction(write=True, operation_name="start", operation=operation)
        except WorkFenceError:
            self._reconcile_exact_due_state(checked_lease.work_item_id, checked_now)
            raise

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

        def operation(connection: PostgresConnection) -> WorkLease:
            item = self._require_lease(connection, checked_lease, checked_now)
            expires_at = _format_timestamp(checked_now + timedelta(seconds=duration))
            cursor = connection.execute(
                """
                UPDATE work_items
                SET lease_expires_at = %s, updated_at = %s, row_version = row_version + 1
                WHERE work_item_id = %s AND lease_epoch = %s AND fencing_token = %s
                    AND state IN (%s, %s)
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

        try:
            return self._transaction(write=True, operation_name="heartbeat", operation=operation)
        except WorkFenceError:
            self._reconcile_exact_due_state(checked_lease.work_item_id, checked_now)
            raise

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

        def operation(connection: PostgresConnection) -> WorkItem:
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
            self._refresh_downstream_for_upstreams(
                connection,
                checked_now,
                (item.work_item_id,),
            )
            self._maintain(connection, checked_now)
            return self._load_item(connection, item.work_item_id)

        try:
            return self._transaction(write=True, operation_name="succeed", operation=operation)
        except WorkFenceError:
            self._reconcile_exact_due_state(checked_lease.work_item_id, checked_now)
            raise

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

        def operation(connection: PostgresConnection) -> WorkItem:
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
                    SET state = %s, fencing_token = NULL, leased_by = NULL,
                        lease_expires_at = NULL, retry_not_before_at = %s,
                        updated_at = %s, row_version = row_version + 1
                    WHERE work_item_id = %s AND lease_epoch = %s AND fencing_token = %s
                        AND state IN (%s, %s)
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
                self._refresh_downstream_for_upstreams(
                    connection,
                    checked_now,
                    (item.work_item_id,),
                )
                self._maintain(connection, checked_now)
            return self._load_item(connection, item.work_item_id)

        try:
            result = self._transaction(write=True, operation_name="fail", operation=operation)
        except WorkFenceError:
            self._reconcile_exact_due_state(checked_lease.work_item_id, checked_now)
            raise
        return result

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

        def operation(connection: PostgresConnection) -> WorkItem:
            self._expire_hard_deadlines(connection, checked_now, work_item_id=work_item_id)
            self._recover_expired_leases(connection, checked_now, work_item_id=work_item_id)
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
            self._refresh_downstream_for_upstreams(
                connection,
                checked_now,
                (item.work_item_id,),
            )
            self._maintain(connection, checked_now)
            return self._load_item(connection, work_item_id)

        return self._transaction(
            write=True,
            operation_name=f"terminal_{state.value.lower()}",
            operation=operation,
        )

    def _maintain(
        self,
        connection: PostgresConnection,
        now: datetime,
        *,
        refresh_all_planned: bool = False,
    ) -> int:
        """Advance indexed due state; reserve full planned scans for reconciliation."""

        changed = self._expire_hard_deadlines(connection, now)
        changed += self._recover_expired_leases(connection, now)
        timestamp = _format_timestamp(now)
        cursor = connection.execute(
            """
            UPDATE work_items
            SET state = %s, retry_not_before_at = NULL, updated_at = %s,
                row_version = row_version + 1
            WHERE state = %s AND retry_not_before_at <= %s
            """,
            (
                WorkItemState.READY.value,
                timestamp,
                WorkItemState.RETRY_WAIT.value,
                timestamp,
            ),
        )
        changed += cursor.rowcount
        # A restart/reconciliation must repair any historical partial projection. On
        # ordinary claim/terminal traffic, direct predecessor transitions below keep
        # readiness current without repeatedly walking every older PLANNED row.
        if refresh_all_planned:
            changed += self._refresh_planned(connection, now)
        return changed

    def _reconcile_exact_due_state(self, work_item_id: str, now: datetime) -> int:
        def operation(connection: PostgresConnection) -> int:
            changed = self._expire_hard_deadlines(
                connection,
                now,
                work_item_id=work_item_id,
            )
            changed += self._recover_expired_leases(
                connection,
                now,
                work_item_id=work_item_id,
            )
            if changed:
                item = self._load_item(connection, work_item_id)
                if item.state in TERMINAL_WORK_STATES:
                    changed += self._refresh_downstream_for_upstreams(
                        connection,
                        now,
                        (work_item_id,),
                    )
            return changed

        return self._transaction(
            write=True,
            operation_name="reconcile_exact_due_state",
            operation=operation,
        )

    def _expire_hard_deadlines(
        self,
        connection: PostgresConnection,
        now: datetime,
        *,
        work_item_id: str | None = None,
    ) -> int:
        timestamp = _format_timestamp(now)
        if work_item_id is None:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE completed_at IS NULL
                  AND execution_expiry_at IS NOT NULL
                  AND execution_expiry_at <= %s
                ORDER BY execution_expiry_at, work_item_id
                """,
                (timestamp,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE work_item_id = %s
                  AND completed_at IS NULL
                  AND execution_expiry_at IS NOT NULL
                  AND execution_expiry_at <= %s
                """,
                (work_item_id, timestamp),
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
            self._refresh_downstream_for_upstreams(connection, now, (item.work_item_id,))
            changed += 1
        return changed

    def _recover_expired_leases(
        self,
        connection: PostgresConnection,
        now: datetime,
        *,
        work_item_id: str | None = None,
    ) -> int:
        timestamp = _format_timestamp(now)
        if work_item_id is None:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE lease_expires_at IS NOT NULL
                  AND lease_expires_at <= %s
                  AND state IN (%s, %s)
                ORDER BY state, lease_expires_at, work_item_id
                """,
                (
                    timestamp,
                    WorkItemState.LEASED.value,
                    WorkItemState.RUNNING.value,
                ),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT * FROM work_items
                WHERE work_item_id = %s
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= %s
                  AND state IN (%s, %s)
                """,
                (
                    work_item_id,
                    timestamp,
                    WorkItemState.LEASED.value,
                    WorkItemState.RUNNING.value,
                ),
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
                    SET state = %s, fencing_token = NULL, leased_by = NULL,
                        lease_expires_at = NULL, updated_at = %s,
                        row_version = row_version + 1
                    WHERE work_item_id = %s AND lease_epoch = %s AND fencing_token = %s
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
                self._refresh_downstream_for_upstreams(connection, now, (item.work_item_id,))
            changed += 1
        return changed

    def _refresh_downstream_for_upstreams(
        self,
        connection: PostgresConnection,
        now: datetime,
        upstream_work_item_ids: Sequence[str],
    ) -> int:
        """Re-evaluate only direct downstream work made eligible by a transition."""

        candidates: list[str] = []
        for work_item_id in upstream_work_item_ids:
            rows = connection.execute(
                """
                SELECT downstream_work_item_id
                FROM work_dependencies
                WHERE upstream_work_item_id = %s
                ORDER BY downstream_work_item_id
                """,
                (work_item_id,),
            ).fetchall()
            candidates.extend(_row_text(row, "downstream_work_item_id") for row in rows)
        return self._refresh_planned_for_work_ids(connection, now, candidates)

    def _refresh_planned_for_work_ids(
        self,
        connection: PostgresConnection,
        now: datetime,
        work_item_ids: Sequence[str],
    ) -> int:
        """Advance a bounded set of PLANNED rows and any failure cascade it creates."""

        pending = list(dict.fromkeys(work_item_ids))
        queued = set(pending)
        changed = 0
        while pending:
            work_item_id = pending.pop(0)
            queued.discard(work_item_id)
            transition = self._advance_planned_item(connection, work_item_id, now)
            if transition is None:
                continue
            changed += 1
            if transition not in TERMINAL_WORK_STATES:
                continue
            rows = connection.execute(
                """
                SELECT downstream_work_item_id
                FROM work_dependencies
                WHERE upstream_work_item_id = %s
                ORDER BY downstream_work_item_id
                """,
                (work_item_id,),
            ).fetchall()
            for row in rows:
                downstream_id = _row_text(row, "downstream_work_item_id")
                if downstream_id not in queued:
                    pending.append(downstream_id)
                    queued.add(downstream_id)
        return changed

    def _refresh_planned(self, connection: PostgresConnection, now: datetime) -> int:
        """Repair every PLANNED row during explicit reconciliation only."""

        changed = 0
        while True:
            rows = connection.execute(
                """
                SELECT work_item_id FROM work_items
                WHERE state = %s ORDER BY created_at, work_item_id
                """,
                (WorkItemState.PLANNED.value,),
            ).fetchall()
            pass_changed = sum(
                self._advance_planned_item(
                    connection,
                    _row_text(row, "work_item_id"),
                    now,
                )
                is not None
                for row in rows
            )
            changed += pass_changed
            if pass_changed == 0:
                return changed

    def _advance_planned_item(
        self,
        connection: PostgresConnection,
        work_item_id: str,
        now: datetime,
    ) -> WorkItemState | None:
        item = self._load_item(connection, work_item_id)
        if item.state is not WorkItemState.PLANNED:
            return None
        dependency_rows = connection.execute(
            """
            SELECT d.criticality, upstream.state
            FROM work_dependencies AS d
            JOIN work_items AS upstream
              ON upstream.work_item_id = d.upstream_work_item_id
            WHERE d.downstream_work_item_id = %s
            ORDER BY d.dependency_id
            """,
            (item.work_item_id,),
        ).fetchall()
        if any(
            WorkItemState(_row_text(value, "state")) not in TERMINAL_WORK_STATES
            for value in dependency_rows
        ):
            return None
        required_failure = any(
            DependencyCriticality(_row_text(value, "criticality")) is DependencyCriticality.REQUIRED
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
            return WorkItemState.FAILED_PERMANENT
        cursor = connection.execute(
            """
            UPDATE work_items
            SET state = %s, updated_at = %s, row_version = row_version + 1
            WHERE work_item_id = %s AND state = %s
            """,
            (
                WorkItemState.READY.value,
                _format_timestamp(now),
                item.work_item_id,
                WorkItemState.PLANNED.value,
            ),
        )
        if cursor.rowcount != 1:
            raise WorkStorageError("planned readiness transition lost its compare-and-swap")
        return WorkItemState.READY

    def _require_lease(
        self,
        connection: PostgresConnection,
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
            # The expiry is part of the opaque capability returned by claim. Bind
            # it to the persisted row as well as checking the authority clock; a
            # caller must not be able to forge a different expiry around the same
            # epoch/token pair.
            or _parse_timestamp(item.lease_expires_at) != _parse_timestamp(lease.lease_expires_at)
            or _parse_timestamp(item.lease_expires_at) <= now
            or (
                item.execution_expiry_at is not None
                and _parse_timestamp(item.execution_expiry_at) <= now
            )
        ):
            raise WorkFenceError("work lease is stale, expired, or inactive")
        return item

    def _finish_attempt(
        self,
        connection: PostgresConnection,
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
            SET completed_at = %s, outcome = %s, error_code = %s, error_detail = %s
            WHERE work_item_id = %s AND attempt_number = %s AND lease_epoch = %s
                AND fencing_token = %s AND outcome = %s
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
        connection: PostgresConnection,
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
            SET state = %s, cancel_requested = %s,
                fencing_token = NULL, leased_by = NULL, lease_expires_at = NULL,
                retry_not_before_at = NULL, terminal_reason_code = %s,
                terminal_reason_detail = %s, result_reference = %s, result_sha256 = %s,
                completed_at = %s, updated_at = %s, row_version = row_version + 1
            WHERE work_item_id = %s AND completed_at IS NULL
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
        connection: PostgresConnection,
        work_item_id: str,
    ) -> tuple[WorkDependency, ...]:
        rows = connection.execute(
            """
            SELECT * FROM work_dependencies
            WHERE downstream_work_item_id = %s ORDER BY dependency_id
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
        connection: PostgresConnection,
        work_item_id: str,
    ) -> WorkItem:
        row = connection.execute(
            "SELECT * FROM work_items WHERE work_item_id = %s",
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise WorkNotFoundError(f"work item is not registered: {work_item_id}")
        return self._item_from_row(row)

    def _item_from_row(self, row: Row) -> WorkItem:
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

    def _attempt_from_row(self, row: Row) -> WorkAttempt:
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

    def _transaction[T](
        self,
        *,
        write: bool,
        operation_name: str,
        operation: Callable[[PostgresConnection], T],
    ) -> T:
        try:
            return self._authority.run_authority_transaction(
                write=write,
                operation_name=f"work_scheduler.{_nonempty(operation_name, 'operation_name')}",
                operation=operation,
            )
        except (WorkSchedulerError, TypeError, ValueError):
            raise
        except Exception as error:
            if postgres_sqlstate(error) in {"23505", "23503", "23514"}:
                raise WorkConflictError(f"PostgreSQL rejected durable work: {error}") from error
            raise WorkStorageError(
                f"PostgreSQL work scheduler transaction failed: {error}"
            ) from error


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


def _row_text(row: Row, column: str) -> str:
    value: object = row[column]
    if not isinstance(value, str):
        raise WorkStorageError(f"persisted {column} must be text")
    return value


def _row_optional_text(row: Row, column: str) -> str | None:
    value: object = row[column]
    if value is not None and not isinstance(value, str):
        raise WorkStorageError(f"persisted {column} must be text or null")
    return value


def _row_int(row: Row, column: str) -> int:
    value: object = row[column]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkStorageError(f"persisted {column} must be an integer")
    return value


PostgresWorkConflictError = WorkConflictError
PostgresWorkFenceError = WorkFenceError
PostgresWorkNotFoundError = WorkNotFoundError
PostgresWorkSchedulerError = WorkSchedulerError
PostgresWorkStateError = WorkStateError
PostgresWorkStorageError = WorkStorageError

__all__ = [
    "PostgresWorkConflictError",
    "PostgresWorkFenceError",
    "PostgresWorkNotFoundError",
    "PostgresWorkScheduler",
    "PostgresWorkSchedulerError",
    "PostgresWorkStateError",
    "PostgresWorkStorageError",
]
