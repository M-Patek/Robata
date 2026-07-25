from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sqlite3 import Row, connect
from threading import Barrier as ThreadBarrier

import pytest

from robata.adapters.sqlite_work_scheduler import (
    SQLiteWorkScheduler,
    WorkConflictError,
    WorkFenceError,
    WorkNotFoundError,
)
from robata.queue.models import (
    WorkAttemptOutcome,
    WorkDependency,
    WorkItem,
    WorkItemPlan,
    WorkItemState,
    WorkItemSubjectType,
)
from robata.queue.stage import DependencyCriticality, Stage
from robata.runtime.observability import RuntimeProfileRecorder

_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_RUN_ID = "00000000-0000-4000-8000-000000000001"
_MCAP_ID = "00000000-0000-4000-8000-000000000002"


def _uuid(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012d}"


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _plan(
    value: int,
    *,
    max_attempts: int = 3,
    sla_deadline_at: datetime | None = None,
    execution_expiry_at: datetime | None = None,
    priority: int = 0,
) -> WorkItemPlan:
    return WorkItemPlan(
        work_item_id=_uuid(100 + value),
        work_logical_key=f"work:{value}",
        run_id=_RUN_ID,
        mcap_id=_MCAP_ID,
        stage=Stage.QA_COARSE_PLAN,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=_uuid(200 + value),
        input_digest="a" * 64,
        config_digest="b" * 64,
        priority=priority,
        sla_deadline_at=(None if sla_deadline_at is None else _timestamp(sla_deadline_at)),
        execution_expiry_at=(
            None if execution_expiry_at is None else _timestamp(execution_expiry_at)
        ),
        max_attempts=max_attempts,
        created_at=_timestamp(_BASE),
    )


def _scheduler(tmp_path: Path) -> SQLiteWorkScheduler:
    return SQLiteWorkScheduler(tmp_path / "work.sqlite3")


def test_observes_exact_transaction_boundaries_and_actual_rollback(
    tmp_path: Path,
) -> None:
    recorder = RuntimeProfileRecorder()
    scheduler = SQLiteWorkScheduler(
        tmp_path / "work.sqlite3",
        runtime_observer=recorder,
    )

    with pytest.raises(WorkNotFoundError, match="not registered"):
        scheduler.get(_uuid(999))

    snapshot = recorder.snapshot()
    transactions = tuple(
        counter
        for counter in snapshot.counters
        if counter.name == "sqlite.work_scheduler.transactions"
    )
    commits = sum(
        counter.value
        for counter in snapshot.counters
        if counter.name == "sqlite.work_scheduler.commits"
    )
    rollbacks = sum(
        counter.value
        for counter in snapshot.counters
        if counter.name == "sqlite.work_scheduler.rollbacks"
    )
    writes = {
        attribute.value
        for counter in transactions
        for attribute in counter.attributes
        if attribute.name == "write"
    }

    assert sum(counter.value for counter in transactions) == 2
    assert commits == 1
    assert rollbacks == 1
    assert writes == {False, True}
    assert sum(span.name == "sqlite.work_scheduler.transaction" for span in snapshot.spans) == 2


def test_dependency_stays_planned_until_required_upstream_succeeds(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    upstream = _plan(1)
    downstream = _plan(2)

    assert scheduler.plan(upstream).state is WorkItemState.READY
    dependency = WorkDependency(
        dependency_id=_uuid(301),
        downstream_work_item_id=downstream.work_item_id,
        upstream_work_item_id=upstream.work_item_id,
        criticality=DependencyCriticality.REQUIRED,
    )
    assert scheduler.plan(downstream, (dependency,)).state is WorkItemState.PLANNED

    claim = scheduler.claim("worker-a", 30, now=_BASE)
    assert claim is not None
    assert claim.work_item.work_item_id == upstream.work_item_id
    scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))
    scheduler.succeed(
        claim.lease,
        result_reference="artifact://upstream",
        result_sha256="c" * 64,
        now=_BASE + timedelta(seconds=2),
    )

    assert scheduler.get(downstream.work_item_id).state is WorkItemState.READY


def test_plan_many_is_exact_replay_safe_and_uses_targeted_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3", runtime_observer=recorder)
    upstream = _plan(70)
    downstream = _plan(71)
    dependency = WorkDependency(
        dependency_id=_uuid(370),
        downstream_work_item_id=downstream.work_item_id,
        upstream_work_item_id=upstream.work_item_id,
        criticality=DependencyCriticality.REQUIRED,
    )

    def reject_global_planned_scan(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("normal batch planning must not scan every planned work row")

    monkeypatch.setattr(scheduler, "_refresh_planned", reject_global_planned_scan)
    batch = ((upstream, ()), (downstream, (dependency,)))
    planned = scheduler.plan_many(batch)
    replayed = scheduler.plan_many(batch)

    assert tuple(item.state for item in planned) == (
        WorkItemState.READY,
        WorkItemState.PLANNED,
    )
    assert tuple(item.work_item_id for item in replayed) == (
        upstream.work_item_id,
        downstream.work_item_id,
    )

    claim = scheduler.claim_and_start(
        "same-process-worker",
        30,
        work_item_id=upstream.work_item_id,
        now=_BASE,
    )
    assert claim is not None
    assert claim.work_item.state is WorkItemState.RUNNING
    assert scheduler.list_attempts(upstream.work_item_id)[0].started_at == _timestamp(_BASE)
    scheduler.succeed(claim.lease, now=_BASE + timedelta(seconds=1))
    assert scheduler.get(downstream.work_item_id).state is WorkItemState.READY

    snapshot = recorder.snapshot()

    def transactions_for(operation: str) -> int:
        return sum(
            counter.value
            for counter in snapshot.counters
            if counter.name == "sqlite.work_scheduler.transactions"
            and any(
                attribute.name == "operation" and attribute.value == operation
                for attribute in counter.attributes
            )
        )

    assert transactions_for("plan_many") == 2
    assert transactions_for("claim_and_start") == 1
    assert transactions_for("start") == 0


def test_plan_many_rolls_back_all_members_after_late_dependency_conflict(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    first = _plan(72)
    invalid = _plan(73)
    missing_upstream = _uuid(9_999)
    dependency = WorkDependency(
        dependency_id=_uuid(371),
        downstream_work_item_id=invalid.work_item_id,
        upstream_work_item_id=missing_upstream,
        criticality=DependencyCriticality.REQUIRED,
    )

    with pytest.raises(WorkConflictError, match="upstream dependency"):
        scheduler.plan_many(((first, ()), (invalid, (dependency,))))

    assert scheduler.items_for_run(_RUN_ID) == ()


def test_start_and_succeed_avoid_pretransition_reconcile_and_progress_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(tmp_path)
    upstream = _plan(20)
    downstream = _plan(21)
    scheduler.plan(upstream)
    scheduler.plan(
        downstream,
        (
            WorkDependency(
                dependency_id=_uuid(320),
                downstream_work_item_id=downstream.work_item_id,
                upstream_work_item_id=upstream.work_item_id,
            ),
        ),
    )

    claim = scheduler.claim(
        "worker-a",
        30,
        work_item_id=upstream.work_item_id,
        now=_BASE,
    )
    assert claim is not None

    def reject_pretransition_reconcile(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("lease transitions must not run a pretransition reconcile")

    monkeypatch.setattr(scheduler, "reconcile", reject_pretransition_reconcile)

    scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))
    scheduler.succeed(claim.lease, now=_BASE + timedelta(seconds=2))

    assert scheduler.get(downstream.work_item_id).state is WorkItemState.READY


def test_required_dependency_failure_cascades_without_dispatch(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    upstream = _plan(3)
    downstream = _plan(4)
    scheduler.plan(upstream)
    scheduler.plan(
        downstream,
        (
            WorkDependency(
                dependency_id=_uuid(302),
                downstream_work_item_id=downstream.work_item_id,
                upstream_work_item_id=upstream.work_item_id,
            ),
        ),
    )

    claim = scheduler.claim("worker-a", 30, now=_BASE)
    assert claim is not None
    scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))
    scheduler.fail(
        claim.lease,
        error_code="INPUT_CORRUPT",
        retryable=False,
        now=_BASE + timedelta(seconds=2),
    )

    failed = scheduler.get(downstream.work_item_id)
    assert failed.state is WorkItemState.FAILED_PERMANENT
    assert failed.terminal_reason_code == "REQUIRED_DEPENDENCY_FAILED"


def test_reopen_recovers_expired_lease_and_rejects_stale_fence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "work.sqlite3"
    scheduler = SQLiteWorkScheduler(database)
    plan = _plan(5)
    scheduler.plan(plan)
    abandoned = scheduler.claim("worker-old", 10, now=_BASE)
    assert abandoned is not None
    scheduler.start(abandoned.lease, now=_BASE + timedelta(seconds=1))

    reopened = SQLiteWorkScheduler(database)
    recovered = reopened.claim(
        "worker-new",
        10,
        now=_BASE + timedelta(seconds=11),
    )
    assert recovered is not None
    assert recovered.work_item.work_item_id == plan.work_item_id
    assert recovered.lease.lease_epoch == abandoned.lease.lease_epoch + 1
    assert recovered.lease.fencing_token != abandoned.lease.fencing_token

    with pytest.raises(WorkFenceError, match="stale, expired, or inactive"):
        reopened.succeed(
            abandoned.lease,
            now=_BASE + timedelta(seconds=12),
        )

    attempts = reopened.list_attempts(plan.work_item_id)
    assert [value.outcome for value in attempts] == [
        WorkAttemptOutcome.ABANDONED,
        WorkAttemptOutcome.ACTIVE,
    ]
    reopened.start(recovered.lease, now=_BASE + timedelta(seconds=12))
    assert (
        reopened.succeed(
            recovered.lease,
            now=_BASE + timedelta(seconds=13),
        ).state
        is WorkItemState.SUCCEEDED
    )


def test_stale_commit_persists_lease_recovery_without_global_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(tmp_path)
    plan = _plan(12)
    scheduler.plan(plan)
    claim = scheduler.claim("worker-old", 2, now=_BASE)
    assert claim is not None
    scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))

    def reject_pretransition_reconcile(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("lease transitions must not run a pretransition reconcile")

    monkeypatch.setattr(scheduler, "reconcile", reject_pretransition_reconcile)

    with pytest.raises(WorkFenceError):
        scheduler.succeed(
            claim.lease,
            now=_BASE + timedelta(seconds=2),
        )

    recovered = scheduler.get(plan.work_item_id)
    assert recovered.state is WorkItemState.READY
    assert scheduler.list_attempts(plan.work_item_id)[0].outcome is (WorkAttemptOutcome.ABANDONED)


def test_exact_succeed_persists_hard_expiry_without_global_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(tmp_path)
    plan = _plan(23, execution_expiry_at=_BASE + timedelta(seconds=2))
    scheduler.plan(plan)
    claim = scheduler.claim("worker", 30, now=_BASE)
    assert claim is not None
    scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))

    def reject_pretransition_reconcile(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("lease transitions must not run a pretransition reconcile")

    monkeypatch.setattr(scheduler, "reconcile", reject_pretransition_reconcile)

    with pytest.raises(WorkFenceError):
        scheduler.succeed(claim.lease, now=_BASE + timedelta(seconds=2))

    assert scheduler.get(plan.work_item_id).state is WorkItemState.EXPIRED
    assert scheduler.list_attempts(plan.work_item_id)[0].outcome is WorkAttemptOutcome.EXPIRED


def test_retry_wait_exhaustion_and_soft_sla_do_not_fake_expiry(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    plan = _plan(
        6,
        max_attempts=2,
        sla_deadline_at=_BASE - timedelta(seconds=1),
        execution_expiry_at=_BASE + timedelta(minutes=1),
    )
    scheduler.plan(plan)

    first = scheduler.claim("worker", 20, now=_BASE)
    assert first is not None
    scheduler.start(first.lease, now=_BASE + timedelta(seconds=1))
    waiting = scheduler.fail(
        first.lease,
        error_code="TRANSIENT",
        retryable=True,
        retry_delay_seconds=5,
        now=_BASE + timedelta(seconds=2),
    )
    assert waiting.state is WorkItemState.RETRY_WAIT
    assert (
        scheduler.claim(
            "worker",
            20,
            now=_BASE + timedelta(seconds=6),
        )
        is None
    )

    second = scheduler.claim(
        "worker",
        20,
        now=_BASE + timedelta(seconds=7),
    )
    assert second is not None
    scheduler.start(second.lease, now=_BASE + timedelta(seconds=8))
    failed = scheduler.fail(
        second.lease,
        error_code="STILL_TRANSIENT",
        retryable=True,
        now=_BASE + timedelta(seconds=9),
    )
    assert failed.state is WorkItemState.FAILED_PERMANENT
    assert failed.attempt == 2


def test_hard_expiry_fences_running_attempt(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    plan = _plan(7, execution_expiry_at=_BASE + timedelta(seconds=2))
    scheduler.plan(plan)
    claim = scheduler.claim("worker", 30, now=_BASE)
    assert claim is not None
    scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))

    assert scheduler.reconcile(now=_BASE + timedelta(seconds=2)) == 1
    expired = scheduler.get(plan.work_item_id)
    assert expired.state is WorkItemState.EXPIRED
    assert scheduler.list_attempts(plan.work_item_id)[0].outcome is (WorkAttemptOutcome.EXPIRED)
    with pytest.raises(WorkFenceError):
        scheduler.succeed(claim.lease, now=_BASE + timedelta(seconds=3))


def test_reconcile_materializes_only_due_deadlines_and_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(tmp_path)
    due_deadline = _plan(43, execution_expiry_at=_BASE + timedelta(seconds=2))
    future_deadlines = tuple(
        _plan(value, execution_expiry_at=_BASE + timedelta(hours=1)) for value in range(44, 52)
    )
    due_lease = _plan(52)
    future_leases = tuple(_plan(value) for value in range(53, 61))
    for plan in (due_deadline, *future_deadlines, due_lease, *future_leases):
        scheduler.plan(plan)

    due_claim = scheduler.claim(
        "due-worker",
        2,
        work_item_id=due_lease.work_item_id,
        now=_BASE,
    )
    assert due_claim is not None
    scheduler.start(due_claim.lease, now=_BASE + timedelta(seconds=1))
    for index, plan in enumerate(future_leases):
        claim = scheduler.claim(
            f"future-worker-{index}",
            3600,
            work_item_id=plan.work_item_id,
            now=_BASE,
        )
        assert claim is not None
        scheduler.start(claim.lease, now=_BASE + timedelta(seconds=1))

    observed: list[str] = []
    original_item_from_row = scheduler._item_from_row

    def observe_item(row: Row) -> WorkItem:
        item = original_item_from_row(row)
        observed.append(item.work_item_id)
        return item

    monkeypatch.setattr(scheduler, "_item_from_row", observe_item)

    assert scheduler.reconcile(now=_BASE + timedelta(seconds=2)) == 2
    assert len(observed) == 2
    assert set(observed) == {due_deadline.work_item_id, due_lease.work_item_id}


def test_due_maintenance_queries_use_range_indexes(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    with connect(scheduler.database_path) as connection:
        expiry_plan = tuple(
            row[3]
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM work_items
                WHERE completed_at IS NULL
                  AND execution_expiry_at IS NOT NULL
                  AND execution_expiry_at <= ?
                ORDER BY execution_expiry_at, work_item_id
                """,
                (_timestamp(_BASE),),
            )
        )
        lease_plan = tuple(
            row[3]
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM work_items
                WHERE lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?
                  AND state IN (?, ?)
                ORDER BY state, lease_expires_at, work_item_id
                """,
                (
                    _timestamp(_BASE),
                    WorkItemState.LEASED.value,
                    WorkItemState.RUNNING.value,
                ),
            )
        )

    assert any("work_items_due_expiry" in detail for detail in expiry_plan)
    assert any("work_items_due_lease" in detail for detail in lease_plan)
    assert all("SCAN work_items" not in detail for detail in (*expiry_plan, *lease_plan))
    assert all("USE TEMP B-TREE" not in detail for detail in (*expiry_plan, *lease_plan))


def test_ready_dispatch_uses_ordered_index_without_a_queue_scan(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    with connect(scheduler.database_path) as connection:
        query_plan = tuple(
            row[3]
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM work_items
                WHERE state = ?
                ORDER BY
                    priority DESC,
                    CASE WHEN sla_deadline_at IS NULL THEN 1 ELSE 0 END,
                    sla_deadline_at,
                    created_at,
                    work_item_id
                LIMIT 1
                """,
                (WorkItemState.READY.value,),
            )
        )

    assert any("work_items_dispatch_order" in detail for detail in query_plan)
    assert all("SCAN work_items" not in detail for detail in query_plan)
    assert all("USE TEMP B-TREE" not in detail for detail in query_plan)


def test_cancel_skip_and_invalidate_are_explicit_terminal_states(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path)
    cancelled_plan = _plan(8)
    skipped_plan = _plan(9)
    invalidated_plan = _plan(10)
    scheduler.plan(cancelled_plan)
    scheduler.plan(skipped_plan)
    scheduler.plan(invalidated_plan)

    cancelled = scheduler.cancel(
        cancelled_plan.work_item_id,
        now=_BASE + timedelta(seconds=1),
    )
    skipped = scheduler.skip(
        skipped_plan.work_item_id,
        state=WorkItemState.SKIPPED_NOT_NEEDED,
        reason_code="EMPTY_FANOUT",
        now=_BASE + timedelta(seconds=1),
    )
    invalidated = scheduler.invalidate(
        invalidated_plan.work_item_id,
        reason_code="POLICY_CHANGED",
        now=_BASE + timedelta(seconds=1),
    )

    assert cancelled.state is WorkItemState.CANCELLED
    assert cancelled.cancel_requested is True
    assert skipped.state is WorkItemState.SKIPPED_NOT_NEEDED
    assert invalidated.state is WorkItemState.INVALIDATED


def test_atomic_claim_has_one_winner(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    scheduler.plan(_plan(11))
    gate = ThreadBarrier(2)

    def claim(worker: str) -> object:
        gate.wait()
        return scheduler.claim(worker, 30, now=_BASE)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(claim, ("worker-a", "worker-b")))

    assert sum(value is not None for value in results) == 1


def test_exact_claim_does_not_take_another_ready_item(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    preferred = _plan(13, priority=100)
    requested = _plan(14, priority=0)
    scheduler.plan(preferred)
    scheduler.plan(requested)

    exact = scheduler.claim(
        "worker-exact",
        30,
        work_item_id=requested.work_item_id,
        now=_BASE,
    )

    assert exact is not None
    assert exact.work_item.work_item_id == requested.work_item_id
    assert scheduler.get(preferred.work_item_id).state is WorkItemState.READY

    next_claim = scheduler.claim("worker-next", 30, now=_BASE)
    assert next_claim is not None
    assert next_claim.work_item.work_item_id == preferred.work_item_id
