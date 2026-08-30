from __future__ import annotations

from threading import Event
from time import sleep

import pytest

from robata.benchmark.wemm_pipeline_benchmark import (
    PhaseRecorder,
    PipelinePhase,
    WemmPipelineBenchmarkError,
    phase_totals_by_name,
    run_bounded_pipeline,
)


def test_phase_recorder_closes_intervals_when_callback_raises() -> None:
    recorder = PhaseRecorder()

    with pytest.raises(RuntimeError, match="phase failure"), recorder.phase(PipelinePhase.PIL):
        raise RuntimeError("phase failure")

    samples = recorder.snapshot()
    assert len(samples) == 1
    assert samples[0].name == "pil"
    assert samples[0].completed_ns >= samples[0].started_ns
    assert samples[0].duration_ns >= 0


def test_bounded_pipeline_overlaps_media_prepare_with_model_consume() -> None:
    first_consume_entered = Event()
    first_model_entered = Event()
    second_prepare_finished = Event()
    release_first_consume = Event()

    def prepare(item: int, recorder: PhaseRecorder) -> int:
        with recorder.phase(PipelinePhase.MEDIA_DECODE):
            if item == 0:
                sleep(0.005)
            if item == 1:
                if not first_model_entered.wait(timeout=2.0):
                    raise AssertionError("first model phase was not entered")
                second_prepare_finished.set()
        with recorder.phase(PipelinePhase.PIL):
            sleep(0.002)
        return item

    def consume(item: int, recorder: PhaseRecorder) -> int:
        with recorder.phase(PipelinePhase.PROCESSOR):
            if item == 0:
                first_consume_entered.set()
        with recorder.phase(PipelinePhase.TENSOR_TRANSFER):
            sleep(0.002)
        with recorder.phase(PipelinePhase.MODEL):
            if item == 0:
                first_model_entered.set()
                if not release_first_consume.wait(timeout=2.0):
                    raise AssertionError("test consumer was not released")
            sleep(0.005)
        with recorder.phase(PipelinePhase.RANK):
            return item

    # Keep the first model call open.  The producer must still prepare item 1
    # and place it in the one-slot queue before the first consumer is released.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_bounded_pipeline,
            range(3),
            prepare=prepare,
            consume=consume,
            key=lambda item, _ordinal: f"window-{item}",
            queue_capacity=1,
        )
        assert first_consume_entered.wait(timeout=2.0)
        assert second_prepare_finished.wait(timeout=2.0)
        release_first_consume.set()
        run = future.result(timeout=5.0)

    assert run.succeeded
    assert run.outputs == (0, 1, 2)
    report = run.report
    assert report.offered_item_count == 3
    assert report.produced_item_count == 3
    assert report.consumed_item_count == 3
    assert report.queue_capacity == 1
    assert report.overlap_ns >= 0
    phases = phase_totals_by_name(report)
    assert set(phases) == {
        "media_decode",
        "pil",
        "processor",
        "tensor_transfer",
        "model",
        "rank",
    }
    assert phases["media_decode"].count == 3
    assert all(item.succeeded for item in report.items)
    # The producer's second media phase begins while the first model phase is
    # still held open.  This is the key producer/consumer overlap invariant.
    first_consumer_model = next(
        sample for sample in report.items[0].consumer_phases if sample.name == "model"
    )
    second_producer_media = next(
        sample for sample in report.items[1].producer_phases if sample.name == "media_decode"
    )
    assert second_producer_media.started_ns < first_consumer_model.completed_ns
    assert first_consumer_model.started_ns < second_producer_media.completed_ns


def test_bounded_pipeline_preserves_order_and_reports_backpressure() -> None:
    def prepare(item: int, recorder: PhaseRecorder) -> int:
        with recorder.phase(PipelinePhase.MEDIA_DECODE):
            return item

    def consume(item: int, recorder: PhaseRecorder) -> int:
        with recorder.phase(PipelinePhase.MODEL):
            sleep(0.01)
            return item

    run = run_bounded_pipeline(
        range(5),
        prepare=prepare,
        consume=consume,
        queue_capacity=1,
    )

    assert run.outputs == (0, 1, 2, 3, 4)
    assert run.report.producer_backpressure_ns >= 0
    assert run.report.consumer_queue_wait_ns >= 0
    assert run.report.serial_estimate_ns >= run.report.wall_ns or run.report.overlap_ns == 0
    assert [item.ordinal for item in run.report.items] == list(range(5))


def test_worker_failure_does_not_deadlock_and_can_be_raised() -> None:
    def prepare(item: int, recorder: PhaseRecorder) -> int:
        with recorder.phase(PipelinePhase.MEDIA_DECODE):
            if item == 1:
                raise RuntimeError("decode failed")
            return item

    def consume(item: int, recorder: PhaseRecorder) -> int:
        with recorder.phase(PipelinePhase.MODEL):
            return item

    run = run_bounded_pipeline(
        range(20),
        prepare=prepare,
        consume=consume,
        queue_capacity=1,
    )
    assert not run.succeeded
    assert run.report.status == "FAILED"
    assert run.report.error_type == "RuntimeError"
    assert run.report.error_detail == "decode failed"
    assert run.report.consumed_item_count <= run.report.produced_item_count

    with pytest.raises(WemmPipelineBenchmarkError, match="decode failed"):
        run_bounded_pipeline(
            range(3),
            prepare=prepare,
            consume=consume,
            queue_capacity=1,
            raise_on_error=True,
        )


def test_invalid_queue_capacity_is_rejected_before_threads_start() -> None:
    with pytest.raises(WemmPipelineBenchmarkError, match="queue_capacity"):
        run_bounded_pipeline(
            (),
            prepare=lambda item, _recorder: item,
            consume=lambda item, _recorder: item,
            queue_capacity=0,
        )
