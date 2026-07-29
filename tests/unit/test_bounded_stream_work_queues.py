from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import pytest

from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.queue import (
    BoundedStreamWorkQueues,
    BoundedStreamWorkQueuesConfig,
    StreamQueueAdmissionStatus,
    StreamQueueRetryableError,
)
from robata.queue.models import WorkAttemptOutcome, WorkItemPlan, WorkItemState, WorkItemSubjectType
from robata.queue.stage import Stage

_RUN_ID = "00000000-0000-4000-8000-000000000701"
_MCAP_ID = "00000000-0000-4000-8000-000000000702"


def _uuid(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012d}"


def _plan(
    value: int,
    *,
    created_at: datetime | None = None,
    run_id: str = _RUN_ID,
) -> WorkItemPlan:
    timestamp = (created_at or (datetime.now(UTC) - timedelta(seconds=10))).isoformat()
    return WorkItemPlan(
        work_item_id=_uuid(800 + value),
        work_logical_key=f"runtime-work:{value}",
        run_id=run_id,
        mcap_id=_MCAP_ID,
        stage=Stage.QWEN_QA_COARSE,
        subject_type=WorkItemSubjectType.MCAP,
        subject_id=_uuid(900 + value),
        input_digest="a" * 64,
        config_digest="b" * 64,
        max_attempts=2,
        created_at=timestamp,
    )


def _queues(
    scheduler: SQLiteWorkScheduler,
    *,
    provider: object,
    publisher: object,
    ingress_capacity: int = 2,
    provider_capacity: int = 2,
    publish_capacity: int = 2,
    lease_duration_seconds: int = 30,
    recovery_poll_seconds: float = 60.0,
) -> BoundedStreamWorkQueues:
    return BoundedStreamWorkQueues(
        scheduler=scheduler,
        config=BoundedStreamWorkQueuesConfig(
            run_id=_RUN_ID,
            ingress_capacity=ingress_capacity,
            provider_capacity=provider_capacity,
            publish_capacity=publish_capacity,
            lease_duration_seconds=lease_duration_seconds,
            recovery_poll_seconds=recovery_poll_seconds,
        ),
        provider_executor=provider,  # type: ignore[arg-type]
        publisher=publisher,  # type: ignore[arg-type]
    )


def _wait_for(predicate: object, *, timeout: float = 3.0) -> None:
    assert callable(predicate)
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        sleep(0.01)
    assert predicate()  # type: ignore[operator]


def test_downstream_provider_pressure_rejects_ingress_and_recovers_after_burst(
    tmp_path: Path,
) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plans = tuple(_plan(value) for value in range(1, 4))
    for plan in plans:
        scheduler.plan(plan)

    provider_started = Event()
    release_provider = Event()
    published: list[str] = []

    def provider(claim: object) -> str:
        provider_started.set()
        assert release_provider.wait(3)
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def publisher(claim: object, result: object) -> None:
        assert result == claim.work_item.work_item_id  # type: ignore[union-attr]
        published.append(result)  # type: ignore[arg-type]

    with _queues(
        scheduler,
        provider=provider,
        publisher=publisher,
        ingress_capacity=1,
        provider_capacity=1,
        publish_capacity=1,
    ) as queues:
        assert queues.admit(plans[0].work_item_id).admitted
        assert provider_started.wait(3)
        assert queues.admit(plans[1].work_item_id).admitted
        _wait_for(lambda: queues.snapshot.provider.queued == 1)

        throttled = queues.admit(plans[2].work_item_id)
        assert throttled.status is StreamQueueAdmissionStatus.PROVIDER_FULL

        release_provider.set()
        assert queues.drain(timeout=3)

        snapshot = queues.snapshot
        assert snapshot.ingress.maximum_queued <= snapshot.ingress.capacity
        assert snapshot.provider.maximum_queued <= snapshot.provider.capacity
        assert snapshot.publish.maximum_queued <= snapshot.publish.capacity

    assert set(published) == {plan.work_item_id for plan in plans}
    assert all(scheduler.get(plan.work_item_id).state is WorkItemState.SUCCEEDED for plan in plans)


def test_retryable_provider_failure_replays_from_durable_state_and_drains(
    tmp_path: Path,
) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plan = _plan(10)
    scheduler.plan(plan)
    provider_attempts = 0
    published: list[str] = []

    def provider(claim: object) -> str:
        nonlocal provider_attempts
        provider_attempts += 1
        if provider_attempts == 1:
            raise StreamQueueRetryableError("TEMPORARY_PROVIDER", retry_delay_seconds=0)
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def publisher(claim: object, result: object) -> None:
        assert result == claim.work_item.work_item_id  # type: ignore[union-attr]
        published.append(result)  # type: ignore[arg-type]

    with _queues(scheduler, provider=provider, publisher=publisher) as queues:
        assert queues.admit(plan.work_item_id).admitted
        assert queues.drain(timeout=3)
        assert queues.snapshot.provider.failed == 1

    attempts = scheduler.list_attempts(plan.work_item_id)
    assert [attempt.outcome for attempt in attempts] == [
        WorkAttemptOutcome.FAILED_RETRYABLE,
        WorkAttemptOutcome.SUCCEEDED,
    ]
    assert scheduler.get(plan.work_item_id).state is WorkItemState.SUCCEEDED
    assert published == [plan.work_item_id]


def test_cancellation_fences_active_provider_result_before_publish(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plan = _plan(20)
    scheduler.plan(plan)
    provider_started = Event()
    release_provider = Event()
    published: list[str] = []

    def provider(claim: object) -> str:
        provider_started.set()
        assert release_provider.wait(3)
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def publisher(claim: object, result: object) -> None:
        published.append(result)  # type: ignore[arg-type]

    with _queues(scheduler, provider=provider, publisher=publisher) as queues:
        assert queues.admit(plan.work_item_id).admitted
        assert provider_started.wait(3)
        cancelled = queues.cancel(plan.work_item_id, reason_code="TEST_CANCEL")
        assert cancelled.state is WorkItemState.CANCELLED
        release_provider.set()
        assert queues.drain(timeout=3)

    assert scheduler.get(plan.work_item_id).state is WorkItemState.CANCELLED
    assert published == []


def test_restart_recovery_releases_expired_lease_and_replays_work(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    crashed_at = datetime.now(UTC) - timedelta(seconds=5)
    plan = _plan(30, created_at=crashed_at - timedelta(seconds=5))
    scheduler.plan(plan)
    abandoned = scheduler.claim_and_start(
        "crashed-worker",
        1,
        work_item_id=plan.work_item_id,
        now=crashed_at,
    )
    assert abandoned is not None

    published: list[str] = []

    def provider(claim: object) -> str:
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def publisher(_claim: object, result: object) -> None:
        published.append(result)  # type: ignore[arg-type]

    with _queues(scheduler, provider=provider, publisher=publisher) as queues:
        assert queues.recover() == 1
        assert queues.drain(timeout=3)

    attempts = scheduler.list_attempts(plan.work_item_id)
    assert [attempt.outcome for attempt in attempts] == [
        WorkAttemptOutcome.ABANDONED,
        WorkAttemptOutcome.SUCCEEDED,
    ]
    assert scheduler.get(plan.work_item_id).state is WorkItemState.SUCCEEDED
    assert published == [plan.work_item_id]


def test_admission_rejects_work_from_a_different_recording_run(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    foreign_plan = _plan(40, run_id=_uuid(711))
    scheduler.plan(foreign_plan)
    calls: list[str] = []

    def provider(claim: object) -> str:
        calls.append(claim.work_item.work_item_id)  # type: ignore[union-attr]
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    with _queues(scheduler, provider=provider, publisher=lambda _claim, _result: None) as queues:
        rejected = queues.admit(foreign_plan.work_item_id)
        assert rejected.status is StreamQueueAdmissionStatus.WRONG_RUN
        assert queues.snapshot.ingress.rejected == 1

    assert scheduler.get(foreign_plan.work_item_id).state is WorkItemState.READY
    assert calls == []


def test_cancellation_rejects_work_from_a_different_recording_run(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    foreign_plan = _plan(41, run_id=_uuid(712))
    scheduler.plan(foreign_plan)

    with (
        _queues(
            scheduler,
            provider=lambda _claim: None,
            publisher=lambda _claim, _result: None,
        ) as queues,
        pytest.raises(ValueError, match="does not belong to this recording-affine runtime"),
    ):
        queues.cancel(foreign_plan.work_item_id)

    assert scheduler.get(foreign_plan.work_item_id).state is WorkItemState.READY


def test_heartbeat_prevents_recovery_from_reexecuting_a_slow_provider(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plan = _plan(50)
    scheduler.plan(plan)
    provider_started = Event()
    release_provider = Event()
    provider_calls = 0

    def provider(claim: object) -> str:
        nonlocal provider_calls
        provider_calls += 1
        provider_started.set()
        assert release_provider.wait(timeout=4)
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    with _queues(
        scheduler,
        provider=provider,
        publisher=lambda _claim, _result: None,
        lease_duration_seconds=1,
        recovery_poll_seconds=0.05,
    ) as queues:
        assert queues.admit(plan.work_item_id).admitted
        assert provider_started.wait(timeout=2)
        sleep(1.3)
        assert provider_calls == 1
        assert scheduler.get(plan.work_item_id).state is WorkItemState.RUNNING
        release_provider.set()
        assert queues.drain(timeout=4)

    assert provider_calls == 1
    assert scheduler.get(plan.work_item_id).state is WorkItemState.SUCCEEDED


def test_heartbeat_keeps_a_slow_publisher_fenced_until_success(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plan = _plan(60)
    scheduler.plan(plan)
    publisher_started = Event()
    release_publisher = Event()
    provider_calls = 0

    def provider(claim: object) -> str:
        nonlocal provider_calls
        provider_calls += 1
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def publisher(_claim: object, _result: object) -> None:
        publisher_started.set()
        assert release_publisher.wait(timeout=4)

    with _queues(
        scheduler,
        provider=provider,
        publisher=publisher,
        lease_duration_seconds=1,
        recovery_poll_seconds=0.05,
    ) as queues:
        assert queues.admit(plan.work_item_id).admitted
        assert publisher_started.wait(timeout=2)
        sleep(1.3)
        assert provider_calls == 1
        assert scheduler.get(plan.work_item_id).state is WorkItemState.RUNNING
        release_publisher.set()
        assert queues.drain(timeout=4)

    assert provider_calls == 1
    assert scheduler.get(plan.work_item_id).state is WorkItemState.SUCCEEDED


def test_non_graceful_stop_releases_queued_publish_lease_for_restart(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    first_plan = _plan(70)
    queued_plan = _plan(71)
    scheduler.plan(first_plan)
    scheduler.plan(queued_plan)
    publisher_started = Event()
    release_publisher = Event()

    def provider(claim: object) -> str:
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def blocking_publisher(_claim: object, result: object) -> None:
        if result == first_plan.work_item_id:
            publisher_started.set()
            assert release_publisher.wait(timeout=4)

    queues = _queues(
        scheduler,
        provider=provider,
        publisher=blocking_publisher,
        lease_duration_seconds=1,
        recovery_poll_seconds=60.0,
    )
    try:
        assert queues.admit(first_plan.work_item_id).admitted
        assert publisher_started.wait(timeout=2)
        assert queues.admit(queued_plan.work_item_id).admitted
        _wait_for(lambda: queues.snapshot.publish.queued == 1)
        assert scheduler.get(queued_plan.work_item_id).state is WorkItemState.RUNNING

        queues.close(wait=False)
        release_publisher.set()
        queues.close(wait=True)

        sleep(1.3)
        scheduler.reconcile()
        assert scheduler.get(queued_plan.work_item_id).state is WorkItemState.READY

        replayed: list[str] = []
        with _queues(
            scheduler,
            provider=provider,
            publisher=lambda _claim, result: replayed.append(result),
            lease_duration_seconds=1,
            recovery_poll_seconds=60.0,
        ) as restarted:
            assert restarted.recover() == 1
            assert restarted.drain(timeout=3)

        assert scheduler.get(queued_plan.work_item_id).state is WorkItemState.SUCCEEDED
        assert replayed == [queued_plan.work_item_id]
    finally:
        release_publisher.set()
        queues.close(wait=True)


def test_optional_work_is_shed_without_durable_mutation_or_drain_credit(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plans = tuple(_plan(value) for value in (80, 81, 82))
    for plan in plans:
        scheduler.plan(plan)
    provider_started = Event()
    release_provider = Event()

    def provider(claim: object) -> str:
        provider_started.set()
        assert release_provider.wait(3)
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    with _queues(
        scheduler,
        provider=provider,
        publisher=lambda _claim, _result: None,
        ingress_capacity=1,
        provider_capacity=1,
        publish_capacity=1,
    ) as queues:
        assert queues.admit(plans[0].work_item_id).admitted
        assert provider_started.wait(3)
        assert queues.admit(plans[1].work_item_id).admitted
        _wait_for(lambda: queues.snapshot.provider.queued == 1)
        shed = queues.admit_optional(plans[2].work_item_id)
        assert shed.status is StreamQueueAdmissionStatus.SHED_OPTIONAL
        assert scheduler.get(plans[2].work_item_id).state is WorkItemState.READY
        assert queues.snapshot.optional_work_offered == 1
        assert queues.snapshot.optional_work_shed == 1
        assert queues.snapshot.backlog >= 1
        assert queues.snapshot.optional_work_shedding_actions[0] == "STOP_OPTIONAL_DEEP"
        assert queues.recover() == 0
        assert queues.snapshot.optional_work_shed == 1
        release_provider.set()
        queues.cancel(plans[2].work_item_id, reason_code="TEST_SHED_CANCEL")
        assert queues.recover() == 0
        assert queues.snapshot.optional_work_shed == 1
        assert queues.drain(timeout=3)
        assert queues.snapshot.backlog_end == 0


def test_optional_watermark_and_recovery_counters_are_visible(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plan = _plan(83)
    scheduler.plan(plan)
    with _queues(
        scheduler,
        provider=lambda claim: claim.work_item.work_item_id,  # type: ignore[union-attr]
        publisher=lambda _claim, _result: None,
    ) as queues:
        assert queues.recover() == 1
        assert queues.snapshot.recovery_count >= 1
        assert queues.snapshot.recovery_admitted >= 1
        assert queues.drain(timeout=3)


def test_cancelling_close_discards_queued_publish_notifications(tmp_path: Path) -> None:
    scheduler = SQLiteWorkScheduler(tmp_path / "work.sqlite3")
    plans = (_plan(84), _plan(85))
    for plan in plans:
        scheduler.plan(plan)
    publisher_started = Event()
    release_publisher = Event()

    def provider(claim: object) -> str:
        return claim.work_item.work_item_id  # type: ignore[union-attr]

    def publisher(claim: object, _result: object) -> None:
        if claim.work_item.work_item_id == plans[0].work_item_id:  # type: ignore[union-attr]
            publisher_started.set()
            assert release_publisher.wait(3)

    queues = _queues(
        scheduler,
        provider=provider,
        publisher=publisher,
        ingress_capacity=1,
        provider_capacity=1,
        publish_capacity=1,
        recovery_poll_seconds=60.0,
    )
    try:
        assert queues.admit(plans[0].work_item_id).admitted
        assert publisher_started.wait(3)
        assert queues.admit(plans[1].work_item_id).admitted
        _wait_for(lambda: queues.snapshot.publish.queued == 1)

        queues.close(wait=False, cancel_pending=True)
        snapshot = queues.snapshot
        assert snapshot.ingress.queued == 0
        assert snapshot.provider.queued == 0
        assert snapshot.publish.queued == 0
        assert snapshot.scheduled_work_count == 0
        assert snapshot.backlog_end == 0
        assert snapshot.publish.cancelled >= 1
        assert scheduler.get(plans[0].work_item_id).state is WorkItemState.CANCELLED
        assert scheduler.get(plans[1].work_item_id).state is WorkItemState.CANCELLED

        release_publisher.set()
        queues.close(wait=True)
    finally:
        release_publisher.set()
        queues.close(wait=True, cancel_pending=True)
