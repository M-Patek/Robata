from __future__ import annotations

import asyncio
import time
from pathlib import Path
from threading import Event, RLock, Thread, Timer
from types import SimpleNamespace

import pytest

from robata.application.canonical.local_composition import (
    CanonicalLocalCompositionError,
    CanonicalLocalCompositionErrorCode,
)
from robata.application.canonical.parallel_service import (
    CanonicalLocalFixtureJob,
    CanonicalLocalProviderQueue,
    CanonicalLocalRecordingService,
)


def test_shared_provider_queue_bounds_actual_async_operations() -> None:
    active = 0
    max_active = 0
    lock = RLock()

    async def operation(value: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.02)
            return value
        finally:
            with lock:
                active -= 1

    async def run() -> tuple[tuple[int, ...], object]:
        with CanonicalLocalProviderQueue(max_concurrency=1, queue_capacity=1) as provider:
            results = await asyncio.gather(
                *(provider.dispatch(lambda value=value: operation(value)) for value in range(4))
            )
            return tuple(results), provider.snapshot

    results, snapshot = asyncio.run(run())

    assert results == (0, 1, 2, 3)
    assert max_active == 1
    assert snapshot.admitted == 4
    assert snapshot.completed == 4
    assert snapshot.failed == 0
    assert snapshot.cancelled == 0
    assert snapshot.queue_depth == 0
    assert snapshot.max_queue_depth <= snapshot.queue_capacity
    assert snapshot.backpressure_waits > 0


def test_cancelling_active_async_dispatch_does_not_break_provider_drain() -> None:
    started = Event()
    release = Event()

    async def operation() -> str:
        started.set()
        await asyncio.to_thread(release.wait, 3)
        return "done"

    async def run() -> object:
        provider = CanonicalLocalProviderQueue(max_concurrency=1, queue_capacity=1)
        task = asyncio.create_task(provider.dispatch(operation))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            provider.drain()
            return provider.snapshot
        finally:
            release.set()
            provider.close(wait=True, cancel_pending=True)

    snapshot = asyncio.run(run())
    assert snapshot.active == 0
    assert snapshot.queue_depth == 0
    assert snapshot.completed == 1
    assert snapshot.failed == 0
    assert snapshot.cancelled == 0


def test_provider_queue_wait_false_close_returns_with_full_queue() -> None:
    started = Event()
    release = Event()
    close_returned = Event()

    def callback() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "done"

    provider = CanonicalLocalProviderQueue(max_concurrency=1, queue_capacity=1)
    closer: Thread | None = None
    try:
        first = provider.submit(callback)
        assert started.wait(timeout=1)
        pending = provider.submit(callback)
        assert provider.snapshot.queue_depth == 1

        def request_nonblocking_close() -> None:
            provider.close(wait=False, cancel_pending=False)
            close_returned.set()

        closer = Thread(target=request_nonblocking_close, daemon=True)
        closer.start()
        assert close_returned.wait(timeout=1)
        assert provider.closed is True
        assert not pending.cancelled()

        release.set()
        assert first.result(timeout=3) == "done"
        assert pending.result(timeout=3) == "done"
        closer.join(timeout=3)
        assert not closer.is_alive()
        provider.close(wait=True)

        snapshot = provider.snapshot
        assert snapshot.active == 0
        assert snapshot.queue_depth == 0
        assert snapshot.completed == 2
        assert snapshot.cancelled == 0
    finally:
        release.set()
        if closer is not None:
            closer.join(timeout=5)
        provider.close(wait=True, cancel_pending=True)


def test_recording_service_bounds_ingress_claims_state_and_cancels_pending() -> None:
    started = Event()
    release = Event()
    fixture_root = Path(__file__).parents[1] / "fixtures"
    source_one = fixture_root / "canonical" / "source-recording.json"
    source_two = fixture_root / "schema_upcasting" / "schema-catalog.json"
    source_three = fixture_root / "schema_upcasting" / "golden" / "synthetic-v1-to-v2.input.json"

    def runner(_job: CanonicalLocalFixtureJob) -> CanonicalLocalRunReceiptLike:
        started.set()
        assert release.wait(timeout=3)
        return CanonicalLocalRunReceiptLike(replayed=False)

    service = CanonicalLocalRecordingService(
        recording_worker_count=1,
        ingress_queue_capacity=1,
        fixture_runner=runner,  # type: ignore[arg-type]
    )
    try:
        first = service.submit_fixture(CanonicalLocalFixtureJob(source_one, Path("state-one")))
        assert started.wait(timeout=1)

        with pytest.raises(CanonicalLocalCompositionError) as duplicate:
            service.submit_fixture(CanonicalLocalFixtureJob(source_one, Path("state-one")))
        assert duplicate.value.code is CanonicalLocalCompositionErrorCode.BACKPRESSURE

        with pytest.raises(CanonicalLocalCompositionError) as same_recording:
            service.submit_fixture(
                CanonicalLocalFixtureJob(
                    source_one,
                    Path("state-one-copy"),
                    run_key="independent-reprocess-key",
                )
            )
        assert same_recording.value.code is CanonicalLocalCompositionErrorCode.BACKPRESSURE

        pending = service.submit_fixture(CanonicalLocalFixtureJob(source_two, Path("state-two")))
        with pytest.raises(CanonicalLocalCompositionError) as saturated:
            service.submit_fixture(CanonicalLocalFixtureJob(source_three, Path("state-three")))
        assert saturated.value.code is CanonicalLocalCompositionErrorCode.BACKPRESSURE

        service.close(wait=False, cancel_pending=True)
        assert pending.cancelled()
        release.set()
        assert first.result(timeout=3).replayed is False
        service.close(wait=True)

        snapshot = service.snapshot
        assert snapshot.closed is True
        assert snapshot.active == 0
        assert snapshot.ingress_depth == 0
        assert snapshot.claimed_state_dir_count == 0
        assert snapshot.unique_state_dir_count == 2
        assert snapshot.state_dir_claim_conflicts == 1
        assert snapshot.claimed_recording_key_count == 0
        assert snapshot.unique_recording_key_count == 2
        assert snapshot.recording_key_claim_conflicts == 1
        assert snapshot.completed == 1
        assert snapshot.cancelled == 1
        assert snapshot.rejected == 3
        assert snapshot.fresh_receipts == 1
        assert snapshot.replayed_receipts == 0
        assert snapshot.completion_outbox_inline is True
        assert snapshot.provider_queue.closed is True
    finally:
        release.set()
        service.close(wait=True, cancel_pending=True)


def test_recording_service_wait_false_close_returns_with_full_ingress() -> None:
    started = Event()
    release = Event()
    close_returned = Event()
    fixture_root = Path(__file__).parents[1] / "fixtures"
    source_one = fixture_root / "canonical" / "source-recording.json"
    source_two = fixture_root / "schema_upcasting" / "schema-catalog.json"

    def runner(_job: CanonicalLocalFixtureJob) -> CanonicalLocalRunReceiptLike:
        started.set()
        assert release.wait(timeout=5)
        return CanonicalLocalRunReceiptLike(replayed=False)

    service = CanonicalLocalRecordingService(
        recording_worker_count=1,
        ingress_queue_capacity=1,
        fixture_runner=runner,  # type: ignore[arg-type]
    )
    closer: Thread | None = None
    try:
        first = service.submit_fixture(
            CanonicalLocalFixtureJob(source_one, Path("close-state-one"))
        )
        assert started.wait(timeout=1)
        pending = service.submit_fixture(
            CanonicalLocalFixtureJob(source_two, Path("close-state-two"))
        )
        assert service.ingress_depth == 1

        def request_nonblocking_close() -> None:
            service.close(wait=False, cancel_pending=False)
            close_returned.set()

        closer = Thread(target=request_nonblocking_close, daemon=True)
        closer.start()
        # The old poison-pill shutdown blocked here until the active recording
        # released its full ingress queue. Non-waiting close must return first.
        assert close_returned.wait(timeout=1)
        assert service.snapshot.closed is True
        assert not pending.cancelled()

        release.set()
        assert first.result(timeout=3).replayed is False
        assert pending.result(timeout=3).replayed is False
        closer.join(timeout=3)
        assert not closer.is_alive()
        service.close(wait=True)

        snapshot = service.snapshot
        assert snapshot.completed == 2
        assert snapshot.cancelled == 0
        assert snapshot.ingress_depth == 0
        assert snapshot.claimed_state_dir_count == 0
        assert snapshot.provider_queue.closed is True
    finally:
        release.set()
        if closer is not None:
            closer.join(timeout=5)
        service.close(wait=True, cancel_pending=True)


def test_async_recording_batch_does_not_block_when_ingress_is_full() -> None:
    started = Event()
    release = Event()
    fixture_root = Path(__file__).parents[1] / "fixtures"
    source_one = fixture_root / "canonical" / "source-recording.json"
    source_two = fixture_root / "schema_upcasting" / "schema-catalog.json"
    source_three = fixture_root / "schema_upcasting" / "golden" / "synthetic-v1-to-v2.input.json"

    def runner(_job: CanonicalLocalFixtureJob) -> CanonicalLocalRunReceiptLike:
        started.set()
        assert release.wait(timeout=3)
        return CanonicalLocalRunReceiptLike(replayed=False)

    service = CanonicalLocalRecordingService(
        recording_worker_count=1,
        ingress_queue_capacity=1,
        fixture_runner=runner,  # type: ignore[arg-type]
    )
    release_timer: Timer | None = None
    try:
        first = service.submit_fixture(
            CanonicalLocalFixtureJob(source_one, Path("async-state-one"))
        )
        assert started.wait(timeout=1)
        pending = service.submit_fixture(
            CanonicalLocalFixtureJob(source_two, Path("async-state-two"))
        )

        async def exercise() -> float:
            heartbeat_at = 0.0

            async def heartbeat() -> None:
                nonlocal heartbeat_at
                await asyncio.sleep(0.02)
                heartbeat_at = time.monotonic()

            heartbeat_task = asyncio.create_task(heartbeat())
            await service.arun_fixtures(
                [CanonicalLocalFixtureJob(source_three, Path("async-state-three"))]
            )
            await heartbeat_task
            return heartbeat_at

        started_at = time.monotonic()
        release_timer = Timer(0.10, release.set)
        release_timer.start()
        heartbeat_at = asyncio.run(exercise())
        assert heartbeat_at - started_at < 0.09
        assert first.result(timeout=3).replayed is False
        assert pending.result(timeout=3).replayed is False
    finally:
        if release_timer is not None:
            release_timer.cancel()
        release.set()
        service.close(wait=True, cancel_pending=True)


def test_cancelling_active_async_recording_keeps_worker_usable() -> None:
    started = Event()
    release = Event()
    fixture_root = Path(__file__).parents[1] / "fixtures"
    source_one = fixture_root / "canonical" / "source-recording.json"
    source_two = fixture_root / "schema_upcasting" / "schema-catalog.json"

    def runner(_job: CanonicalLocalFixtureJob) -> CanonicalLocalRunReceiptLike:
        started.set()
        assert release.wait(timeout=3)
        return CanonicalLocalRunReceiptLike(replayed=False)

    service = CanonicalLocalRecordingService(
        recording_worker_count=1,
        ingress_queue_capacity=1,
        fixture_runner=runner,  # type: ignore[arg-type]
    )
    try:

        async def cancel_active() -> None:
            task = asyncio.create_task(
                service.arun_fixture(CanonicalLocalFixtureJob(source_one, Path("cancel-state")))
            )
            assert await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_active())
        release.set()
        second = service.submit_fixture(
            CanonicalLocalFixtureJob(source_two, Path("after-cancel-state"))
        )
        assert second.result(timeout=3).replayed is False
        snapshot = service.snapshot
        assert snapshot.active == 0
        assert snapshot.completed == 2
        assert snapshot.failed == 0
    finally:
        release.set()
        service.close(wait=True, cancel_pending=True)


class CanonicalLocalRunReceiptLike(SimpleNamespace):
    replayed: bool
