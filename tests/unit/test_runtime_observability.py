from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from robata.runtime.observability import (
    NOOP_RUNTIME_OBSERVER,
    ProcessResourceSample,
    RuntimeProfileRecorder,
    RuntimeProfileSnapshot,
    RuntimeResourceMeasurement,
    RuntimeResourceStatus,
    RuntimeSpanStatus,
    runtime_increment,
    runtime_span,
)


def _available(value: int) -> RuntimeResourceMeasurement:
    return RuntimeResourceMeasurement(
        status=RuntimeResourceStatus.AVAILABLE,
        value=value,
    )


def _resources(*, rss: int, read: int, write: int) -> ProcessResourceSample:
    return ProcessResourceSample(
        rss_bytes=_available(rss),
        read_bytes=_available(read),
        write_bytes=_available(write),
    )


def _iterator_clock(values: tuple[int, ...]) -> Callable[[], int]:
    values_iterator = iter(values)
    return lambda: next(values_iterator)


def test_snapshot_has_deterministic_serialization_shape_and_nesting() -> None:
    wall_clock = _iterator_clock((100, 110, 120, 130, 140, 150))
    cpu_clock = _iterator_clock((1_000, 1_040))
    resource_samples = iter(
        (
            _resources(rss=1_000, read=10, write=20),
            _resources(rss=1_200, read=17, write=31),
        )
    )
    recorder = RuntimeProfileRecorder(
        clock_ns=wall_clock,
        process_cpu_clock_ns=cpu_clock,
        resource_sampler=lambda: next(resource_samples),
    )

    with runtime_span(recorder, "outer", {"z": 2, "a": "first"}):
        runtime_increment(recorder, "work.items", 2, {"kind": "frame"})
        with runtime_span(recorder, "inner"):
            runtime_increment(recorder, "work.items", 3, {"kind": "frame"})

    payload = recorder.snapshot().model_dump(mode="json")

    assert payload == {
        "version": "runtime-profile-v1",
        "elapsed_ns": 50,
        "process_cpu_ns": 40,
        "resources": {
            "rss_bytes": {"status": "AVAILABLE", "value": 1_200, "error_type": None},
            "read_bytes_delta": {"status": "AVAILABLE", "value": 7, "error_type": None},
            "write_bytes_delta": {"status": "AVAILABLE", "value": 11, "error_type": None},
        },
        "spans": [
            {
                "sequence": 1,
                "parent_sequence": None,
                "name": "outer",
                "attributes": [
                    {"name": "a", "value": "first"},
                    {"name": "z", "value": 2},
                ],
                "status": "OK",
                "error_type": None,
                "started_offset_ns": 10,
                "ended_offset_ns": 40,
                "elapsed_ns": 30,
            },
            {
                "sequence": 2,
                "parent_sequence": 1,
                "name": "inner",
                "attributes": [],
                "status": "OK",
                "error_type": None,
                "started_offset_ns": 20,
                "ended_offset_ns": 30,
                "elapsed_ns": 10,
            },
        ],
        "counters": [
            {
                "name": "work.items",
                "attributes": [{"name": "kind", "value": "frame"}],
                "value": 5,
            }
        ],
    }


def test_runtime_span_records_error_type_only_and_rethrows() -> None:
    recorder = RuntimeProfileRecorder()

    with (
        pytest.raises(RuntimeError, match="sensitive detail"),
        runtime_span(
            recorder,
            "failure",
        ),
    ):
        raise RuntimeError("sensitive detail")

    span = recorder.snapshot().spans[0]
    assert span.status is RuntimeSpanStatus.ERROR
    assert span.error_type == "RuntimeError"
    assert "sensitive detail" not in str(span.model_dump(mode="json"))


def test_runtime_span_records_cancellation_and_rethrows() -> None:
    recorder = RuntimeProfileRecorder()

    with pytest.raises(asyncio.CancelledError), runtime_span(recorder, "cancelled"):
        raise asyncio.CancelledError

    span = recorder.snapshot().spans[0]
    assert span.status is RuntimeSpanStatus.CANCELLED
    assert span.error_type == "CancelledError"


def test_async_tasks_inherit_parent_without_becoming_each_others_parents() -> None:
    recorder = RuntimeProfileRecorder()

    async def child(ordinal: int) -> None:
        with runtime_span(recorder, "child", {"ordinal": ordinal}):
            await asyncio.sleep(0)

    async def workload() -> None:
        with runtime_span(recorder, "parent"):
            await asyncio.gather(*(child(ordinal) for ordinal in range(8)))

    asyncio.run(workload())
    spans = recorder.snapshot().spans

    assert spans[0].name == "parent"
    assert all(span.parent_sequence == spans[0].sequence for span in spans[1:])
    assert [span.sequence for span in spans] == list(range(1, 10))


def test_threaded_sequences_and_counter_updates_are_lossless() -> None:
    recorder = RuntimeProfileRecorder()
    worker_count = 12
    increments_per_worker = 100

    def worker(worker_ordinal: int) -> None:
        with runtime_span(recorder, "worker", {"ordinal": worker_ordinal}):
            for _ in range(increments_per_worker):
                runtime_increment(recorder, "items")

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        tuple(pool.map(worker, range(worker_count)))

    snapshot = recorder.snapshot()
    assert tuple(span.sequence for span in snapshot.spans) == tuple(range(1, worker_count + 1))
    assert snapshot.counters[0].value == worker_count * increments_per_worker


def test_unavailable_resources_are_explicit_and_never_encoded_as_zero() -> None:
    unsupported = RuntimeResourceMeasurement(status=RuntimeResourceStatus.UNSUPPORTED)

    def sampler() -> ProcessResourceSample:
        return ProcessResourceSample(
            rss_bytes=unsupported,
            read_bytes=unsupported,
            write_bytes=unsupported,
        )

    recorder = RuntimeProfileRecorder(resource_sampler=sampler)

    resources = recorder.snapshot().resources

    assert resources.rss_bytes.status is RuntimeResourceStatus.UNSUPPORTED
    assert resources.rss_bytes.value is None
    assert resources.read_bytes_delta.status is RuntimeResourceStatus.UNSUPPORTED
    assert resources.read_bytes_delta.value is None
    with pytest.raises(ValidationError, match="UNSUPPORTED resources forbid"):
        RuntimeResourceMeasurement(
            status=RuntimeResourceStatus.UNSUPPORTED,
            value=0,
        )


def test_resource_sampler_failure_is_an_error_type_not_a_fake_value() -> None:
    def broken_sampler() -> ProcessResourceSample:
        raise PermissionError("must not escape into the report")

    recorder = RuntimeProfileRecorder(resource_sampler=broken_sampler)
    resources = recorder.snapshot().resources

    assert resources.rss_bytes.status is RuntimeResourceStatus.ERROR
    assert resources.rss_bytes.value is None
    assert resources.rss_bytes.error_type == "PermissionError"
    assert "must not escape" not in str(resources.model_dump(mode="json"))


def test_snapshot_is_idempotent_and_fail_open_helpers_ignore_frozen_recorder() -> None:
    wall_clock = _iterator_clock((100, 125))
    cpu_clock = _iterator_clock((200, 210))
    unsupported = RuntimeResourceMeasurement(status=RuntimeResourceStatus.UNSUPPORTED)
    sample = ProcessResourceSample(
        rss_bytes=unsupported,
        read_bytes=unsupported,
        write_bytes=unsupported,
    )
    recorder = RuntimeProfileRecorder(
        clock_ns=wall_clock,
        process_cpu_clock_ns=cpu_clock,
        resource_sampler=lambda: sample,
    )

    first = recorder.snapshot()
    runtime_increment(recorder, "ignored")
    reached = False
    with runtime_span(recorder, "also-ignored"):
        reached = True
    second = recorder.snapshot()

    assert reached
    assert first is second
    assert first.elapsed_ns == 25
    assert first.process_cpu_ns == 10
    assert first.spans == ()
    assert first.counters == ()


def test_helpers_are_no_ops_for_none_and_the_noop_observer() -> None:
    for observer in (None, NOOP_RUNTIME_OBSERVER):
        runtime_increment(observer, "counter")
        with runtime_span(observer, "span"):
            pass


def test_snapshot_models_are_strict_frozen_and_closed() -> None:
    snapshot = RuntimeProfileRecorder().snapshot()

    with pytest.raises(ValidationError):
        RuntimeProfileSnapshot.model_validate(
            {**snapshot.model_dump(mode="python"), "elapsed_ns": "1"},
            strict=True,
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RuntimeProfileSnapshot.model_validate(
            {**snapshot.model_dump(mode="python"), "unexpected": True},
            strict=True,
        )
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.elapsed_ns = 1
