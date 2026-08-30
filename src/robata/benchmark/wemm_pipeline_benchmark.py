"""Small, benchmark-only producer/consumer timing harness for the WeMM path.

The production WeMM runner intentionally remains serial and owns the durable
source/result envelope.  This module is an opt-in diagnostic seam for answering
one narrower question: can a bounded media producer (MCAP/H.264/Pillow) feed a
single model consumer without retaining a whole recording or allowing an
unbounded queue?  It contains no model imports, media decoding, identity/hash
logic, or published-contract fields.

``run_bounded_pipeline`` starts exactly one producer and one consumer.  The
producer pulls/decodes one bounded item, places it in a bounded queue (capacity
one by default), and the consumer performs processor/tensor/model/ranking work
on that item.  Callbacks receive a :class:`PhaseRecorder`; they may surround
their own work with ``with recorder.phase("media_decode")`` (or any other
stage name).  The harness also records coarse producer/consumer and queue
intervals, so a report remains useful when an optional dependency is absent.

The report is intentionally diagnostic rather than authoritative.  Timings are
process-local monotonic observations and are never suitable for a wire schema or
capacity/quality claim without a workload-bound qualification artifact.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Full, Queue
from typing import Any, Literal

WEMM_PIPELINE_BENCHMARK_VERSION = "robata-wemm-pipeline-benchmark-v1"
DEFAULT_QUEUE_CAPACITY = 1
_QUEUE_POLL_SECONDS = 0.02


class WemmPipelineBenchmarkError(ValueError):
    """Raised for invalid harness configuration or strict worker failures."""


class PipelinePhase(StrEnum):
    """Common labels used by the MCAP/H.264/PIL to model path.

    Callers may use arbitrary non-empty strings as well; these names merely
    prevent spelling drift in small local benchmark callbacks.
    """

    MEDIA_DECODE = "media_decode"
    PIL = "pil"
    PROCESSOR = "processor"
    TENSOR_TRANSFER = "tensor_transfer"
    MODEL = "model"
    RANK = "rank"
    IO = "io"
    GC = "gc"


@dataclass(frozen=True, slots=True)
class PhaseSample:
    """One monotonic interval recorded by a callback."""

    name: str
    started_ns: int
    completed_ns: int
    thread_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise WemmPipelineBenchmarkError("phase name must be non-empty")
        if self.started_ns < 0 or self.completed_ns < self.started_ns:
            raise WemmPipelineBenchmarkError("phase interval is invalid")
        if not isinstance(self.thread_name, str) or not self.thread_name:
            raise WemmPipelineBenchmarkError("phase thread_name must be non-empty")

    @property
    def duration_ns(self) -> int:
        """Elapsed monotonic nanoseconds for this phase."""

        return self.completed_ns - self.started_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_ns": self.started_ns,
            "completed_ns": self.completed_ns,
            "duration_ns": self.duration_ns,
            "thread_name": self.thread_name,
        }


class PhaseRecorder:
    """Mutable per-item phase recorder supplied to producer/consumer callbacks.

    The recorder is intentionally local to one item.  A callback can retain no
    recorder state across items, which keeps phase attribution unambiguous when
    producer and consumer execute concurrently.
    """

    def __init__(self, *, clock: Callable[[], int] = time.perf_counter_ns) -> None:
        self._clock = clock
        self._samples: list[PhaseSample] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[PhaseRecorder]:
        """Record one phase even when the callback raises."""

        if not isinstance(name, str) or not name.strip():
            raise WemmPipelineBenchmarkError("phase name must be non-empty")
        started = self._clock()
        try:
            yield self
        finally:
            completed = self._clock()
            self._samples.append(
                PhaseSample(
                    name=name.strip(),
                    started_ns=started,
                    completed_ns=max(started, completed),
                    thread_name=threading.current_thread().name,
                )
            )

    def record(self, name: str, started_ns: int, completed_ns: int) -> None:
        """Record an externally measured interval."""

        if not isinstance(name, str) or not name.strip():
            raise WemmPipelineBenchmarkError("phase name must be non-empty")
        self._samples.append(
            PhaseSample(
                name=name.strip(),
                started_ns=started_ns,
                completed_ns=completed_ns,
                thread_name=threading.current_thread().name,
            )
        )

    def snapshot(self) -> tuple[PhaseSample, ...]:
        """Return phase samples in callback order."""

        return tuple(self._samples)


@dataclass(frozen=True, slots=True)
class PipelineItemTiming:
    """Timing and phase observations for one producer/consumer item."""

    ordinal: int
    key: str
    producer_started_ns: int
    produced_ns: int
    dequeued_ns: int | None
    consumed_ns: int | None
    producer_phases: tuple[PhaseSample, ...]
    consumer_phases: tuple[PhaseSample, ...]
    producer_backpressure_ns: int
    consumer_queue_wait_ns: int
    succeeded: bool
    error_type: str | None = None
    error_detail: str | None = None

    @property
    def producer_active_ns(self) -> int:
        return max(0, self.produced_ns - self.producer_started_ns)

    @property
    def consumer_active_ns(self) -> int:
        if self.dequeued_ns is None or self.consumed_ns is None:
            return 0
        return max(0, self.consumed_ns - self.dequeued_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "key": self.key,
            "producer_started_ns": self.producer_started_ns,
            "produced_ns": self.produced_ns,
            "dequeued_ns": self.dequeued_ns,
            "consumed_ns": self.consumed_ns,
            "producer_active_ns": self.producer_active_ns,
            "consumer_active_ns": self.consumer_active_ns,
            "producer_backpressure_ns": self.producer_backpressure_ns,
            "consumer_queue_wait_ns": self.consumer_queue_wait_ns,
            "succeeded": self.succeeded,
            "error_type": self.error_type,
            "error_detail": self.error_detail,
            "producer_phases": [sample.to_dict() for sample in self.producer_phases],
            "consumer_phases": [sample.to_dict() for sample in self.consumer_phases],
        }


@dataclass(frozen=True, slots=True)
class PhaseAggregate:
    """Aggregate duration/count for one phase name."""

    name: str
    count: int
    total_ns: int
    max_ns: int

    @property
    def mean_ns(self) -> float:
        return self.total_ns / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_ns": self.total_ns,
            "mean_ns": self.mean_ns,
            "max_ns": self.max_ns,
        }


@dataclass(frozen=True, slots=True)
class WemmPipelineTimingReport:
    """Diagnostic result from one bounded pipeline run."""

    benchmark_version: str
    status: Literal["SUCCEEDED", "FAILED"]
    queue_capacity: int
    offered_item_count: int
    produced_item_count: int
    consumed_item_count: int
    wall_ns: int
    producer_active_ns: int
    consumer_active_ns: int
    producer_backpressure_ns: int
    consumer_queue_wait_ns: int
    overlap_ns: int
    serial_estimate_ns: int
    phase_totals: tuple[PhaseAggregate, ...]
    items: tuple[PipelineItemTiming, ...]
    error_type: str | None = None
    error_detail: str | None = None

    @property
    def wall_seconds(self) -> float:
        return self.wall_ns / 1_000_000_000

    @property
    def overlap_seconds(self) -> float:
        return self.overlap_ns / 1_000_000_000

    @property
    def estimated_speedup(self) -> float:
        if self.wall_ns <= 0:
            return 0.0
        return self.serial_estimate_ns / self.wall_ns

    @property
    def producer_utilization(self) -> float:
        return self.producer_active_ns / self.wall_ns if self.wall_ns else 0.0

    @property
    def consumer_utilization(self) -> float:
        return self.consumer_active_ns / self.wall_ns if self.wall_ns else 0.0

    @property
    def dominant_phase(self) -> str | None:
        if not self.phase_totals:
            return None
        return max(self.phase_totals, key=lambda aggregate: aggregate.total_ns).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "status": self.status,
            "queue_capacity": self.queue_capacity,
            "offered_item_count": self.offered_item_count,
            "produced_item_count": self.produced_item_count,
            "consumed_item_count": self.consumed_item_count,
            "wall_ns": self.wall_ns,
            "wall_seconds": self.wall_seconds,
            "producer_active_ns": self.producer_active_ns,
            "consumer_active_ns": self.consumer_active_ns,
            "producer_backpressure_ns": self.producer_backpressure_ns,
            "consumer_queue_wait_ns": self.consumer_queue_wait_ns,
            "overlap_ns": self.overlap_ns,
            "overlap_seconds": self.overlap_seconds,
            "serial_estimate_ns": self.serial_estimate_ns,
            "estimated_speedup": self.estimated_speedup,
            "producer_utilization": self.producer_utilization,
            "consumer_utilization": self.consumer_utilization,
            "dominant_phase": self.dominant_phase,
            "phase_totals": [aggregate.to_dict() for aggregate in self.phase_totals],
            "items": [item.to_dict() for item in self.items],
            "error_type": self.error_type,
            "error_detail": self.error_detail,
        }


@dataclass(frozen=True, slots=True)
class WemmPipelineRun[V]:
    """Consumed outputs together with their timing report."""

    outputs: tuple[V, ...]
    report: WemmPipelineTimingReport

    @property
    def succeeded(self) -> bool:
        return self.report.status == "SUCCEEDED"


@dataclass(slots=True)
class _PreparedItem[U]:
    ordinal: int
    key: str
    payload: U
    recorder: PhaseRecorder
    producer_started_ns: int
    produced_ns: int
    backpressure_ns: int


def run_bounded_pipeline[T, U, V](
    items: Iterable[T],
    *,
    prepare: Callable[[T, PhaseRecorder], U],
    consume: Callable[[U, PhaseRecorder], V],
    key: Callable[[T, int], str] | None = None,
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    clock: Callable[[], int] = time.perf_counter_ns,
    raise_on_error: bool = False,
) -> WemmPipelineRun[V]:
    """Run one bounded producer and one single-flight consumer.

    ``prepare`` is the media-side callback (MCAP/H.264 decode, PIL conversion,
    or a pre-materialization step).  ``consume`` is the model-side callback
    (processor, tensor transfer, GPU embedding, and ranking).  The callbacks
    are never invoked concurrently with another call to themselves: there is
    one producer and one consumer.  They *can* overlap with each other, which is
    the behavior this harness is intended to measure.

    The queue is bounded and uses short timed puts.  If either callback fails,
    the other worker drains/discards pending items and exits, preventing the
    common producer-blocked-on-full-queue deadlock.  By default the failure is
    retained in the report; ``raise_on_error=True`` raises a
    :class:`WemmPipelineBenchmarkError` after the workers have joined.
    """

    if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
        raise WemmPipelineBenchmarkError("queue_capacity must be a positive integer")
    if queue_capacity <= 0:
        raise WemmPipelineBenchmarkError("queue_capacity must be a positive integer")
    if not callable(prepare) or not callable(consume):
        raise WemmPipelineBenchmarkError("prepare and consume must be callable")
    if not callable(clock):
        raise WemmPipelineBenchmarkError("clock must be callable")
    if key is not None and not callable(key):
        raise WemmPipelineBenchmarkError("key must be callable or None")

    queue: Queue[_PreparedItem[U] | object] = Queue(maxsize=queue_capacity)
    stop = threading.Event()
    sentinel = object()
    lock = threading.Lock()
    prepared_timings: dict[int, PipelineItemTiming] = {}
    consumed_outputs: dict[int, V] = {}
    worker_error: list[BaseException] = []
    offered_count = 0
    produced_count = 0
    consumed_count = 0
    source_exhausted = False

    def _set_error(error: BaseException) -> None:
        with lock:
            if not worker_error:
                worker_error.append(error)
        stop.set()

    def _put_until_accepted(
        value: _PreparedItem[U] | object,
        *,
        force: bool = False,
    ) -> int:
        blocked_ns = 0
        while True:
            started = clock()
            try:
                queue.put(value, timeout=_QUEUE_POLL_SECONDS)
            except Full:
                blocked_ns += max(0, clock() - started)
                if stop.is_set() and not force:
                    return blocked_ns
                continue
            return blocked_ns

    def _producer() -> None:
        nonlocal offered_count, produced_count, source_exhausted
        try:
            for ordinal, item in enumerate(items):
                offered_count = ordinal + 1
                if stop.is_set():
                    break
                item_key = str(key(item, ordinal) if key is not None else ordinal)
                if not item_key:
                    raise WemmPipelineBenchmarkError(
                        f"pipeline item {ordinal} key must be non-empty"
                    )
                recorder = PhaseRecorder(clock=clock)
                producer_started = clock()
                try:
                    payload = prepare(item, recorder)
                except BaseException as error:
                    # Keep a failed item visible even if the callback recorded
                    # its own partial phase interval.
                    completed = clock()
                    prepared_timings[ordinal] = PipelineItemTiming(
                        ordinal=ordinal,
                        key=item_key,
                        producer_started_ns=producer_started,
                        produced_ns=completed,
                        dequeued_ns=None,
                        consumed_ns=None,
                        producer_phases=recorder.snapshot(),
                        consumer_phases=(),
                        producer_backpressure_ns=0,
                        consumer_queue_wait_ns=0,
                        succeeded=False,
                        error_type=type(error).__name__,
                        error_detail=str(error),
                    )
                    raise
                produced = clock()
                prepared = _PreparedItem(
                    ordinal=ordinal,
                    key=item_key,
                    payload=payload,
                    recorder=recorder,
                    producer_started_ns=producer_started,
                    produced_ns=produced,
                    backpressure_ns=0,
                )
                blocked = _put_until_accepted(prepared)
                prepared.backpressure_ns = blocked
                produced_count += 1
            source_exhausted = True
        except BaseException as error:
            _set_error(error)
        finally:
            # The consumer keeps draining after an error so this put cannot
            # strand a producer on a full queue.
            _put_until_accepted(sentinel, force=True)

    def _consumer() -> None:
        nonlocal consumed_count
        while True:
            try:
                value = queue.get(timeout=_QUEUE_POLL_SECONDS)
            except Empty:
                if source_exhausted or stop.is_set():
                    # A producer always publishes a sentinel in ``finally``;
                    # this branch only avoids spinning while that put is in
                    # progress.
                    continue
                continue
            if value is sentinel:
                queue.task_done()
                return
            prepared = value
            assert isinstance(prepared, _PreparedItem)
            dequeued_ns = clock()
            queue_wait = max(0, dequeued_ns - prepared.produced_ns)
            if stop.is_set() and worker_error:
                # Preserve ordering metadata for discarded work, but do not
                # invoke the consumer after a sibling callback failed.
                prepared_timings[prepared.ordinal] = PipelineItemTiming(
                    ordinal=prepared.ordinal,
                    key=prepared.key,
                    producer_started_ns=prepared.producer_started_ns,
                    produced_ns=prepared.produced_ns,
                    dequeued_ns=dequeued_ns,
                    consumed_ns=None,
                    producer_phases=prepared.recorder.snapshot(),
                    consumer_phases=(),
                    producer_backpressure_ns=prepared.backpressure_ns,
                    consumer_queue_wait_ns=queue_wait,
                    succeeded=False,
                    error_type="PIPELINE_CANCELLED",
                    error_detail="sibling producer or consumer failed",
                )
                queue.task_done()
                continue
            consumer_recorder = PhaseRecorder(clock=clock)
            try:
                output = consume(prepared.payload, consumer_recorder)
            except BaseException as error:
                completed = clock()
                prepared_timings[prepared.ordinal] = PipelineItemTiming(
                    ordinal=prepared.ordinal,
                    key=prepared.key,
                    producer_started_ns=prepared.producer_started_ns,
                    produced_ns=prepared.produced_ns,
                    dequeued_ns=dequeued_ns,
                    consumed_ns=completed,
                    producer_phases=prepared.recorder.snapshot(),
                    consumer_phases=consumer_recorder.snapshot(),
                    producer_backpressure_ns=prepared.backpressure_ns,
                    consumer_queue_wait_ns=queue_wait,
                    succeeded=False,
                    error_type=type(error).__name__,
                    error_detail=str(error),
                )
                queue.task_done()
                _set_error(error)
                # Continue draining until sentinel to release a blocked
                # producer and preserve a complete diagnostic report.
                continue
            consumed_at = clock()
            prepared_timings[prepared.ordinal] = PipelineItemTiming(
                ordinal=prepared.ordinal,
                key=prepared.key,
                producer_started_ns=prepared.producer_started_ns,
                produced_ns=prepared.produced_ns,
                dequeued_ns=dequeued_ns,
                consumed_ns=consumed_at,
                producer_phases=prepared.recorder.snapshot(),
                consumer_phases=consumer_recorder.snapshot(),
                producer_backpressure_ns=prepared.backpressure_ns,
                consumer_queue_wait_ns=queue_wait,
                succeeded=True,
            )
            consumed_outputs[prepared.ordinal] = output
            consumed_count += 1
            queue.task_done()

    started_ns = clock()
    producer = threading.Thread(target=_producer, name="wemm-media-producer", daemon=True)
    consumer = threading.Thread(target=_consumer, name="wemm-model-consumer", daemon=True)
    consumer.start()
    producer.start()
    producer.join()
    consumer.join()
    completed_ns = clock()

    ordered_items = tuple(prepared_timings[index] for index in sorted(prepared_timings))
    producer_active_ns = sum(item.producer_active_ns for item in ordered_items)
    consumer_active_ns = sum(item.consumer_active_ns for item in ordered_items)
    producer_backpressure_ns = sum(item.producer_backpressure_ns for item in ordered_items)
    consumer_queue_wait_ns = sum(item.consumer_queue_wait_ns for item in ordered_items)
    wall_ns = max(0, completed_ns - started_ns)
    # The sum of active producer/consumer intervals is a deliberately simple
    # serial estimate.  Any excess over wall time is attributable to overlap;
    # this is a diagnostic lower-bound estimate, not a capacity model.
    serial_estimate_ns = producer_active_ns + consumer_active_ns
    overlap_ns = max(0, serial_estimate_ns - wall_ns)

    phase_rows: dict[str, list[int]] = defaultdict(list)
    for item in ordered_items:
        for sample in (*item.producer_phases, *item.consumer_phases):
            phase_rows[sample.name].append(sample.duration_ns)
    phase_totals = tuple(
        PhaseAggregate(
            name=name,
            count=len(durations),
            total_ns=sum(durations),
            max_ns=max(durations),
        )
        for name, durations in sorted(phase_rows.items())
    )

    error = worker_error[0] if worker_error else None
    report = WemmPipelineTimingReport(
        benchmark_version=WEMM_PIPELINE_BENCHMARK_VERSION,
        status="FAILED" if error is not None else "SUCCEEDED",
        queue_capacity=queue_capacity,
        offered_item_count=offered_count,
        produced_item_count=produced_count,
        consumed_item_count=consumed_count,
        wall_ns=wall_ns,
        producer_active_ns=producer_active_ns,
        consumer_active_ns=consumer_active_ns,
        producer_backpressure_ns=producer_backpressure_ns,
        consumer_queue_wait_ns=consumer_queue_wait_ns,
        overlap_ns=overlap_ns,
        serial_estimate_ns=serial_estimate_ns,
        phase_totals=phase_totals,
        items=ordered_items,
        error_type=None if error is None else type(error).__name__,
        error_detail=None if error is None else str(error),
    )
    outputs = tuple(consumed_outputs[index] for index in sorted(consumed_outputs))
    run = WemmPipelineRun(outputs=outputs, report=report)
    if raise_on_error and error is not None:
        raise WemmPipelineBenchmarkError(
            f"bounded WeMM pipeline failed: {type(error).__name__}: {error}"
        ) from error
    return run


def phase_totals_by_name(report: WemmPipelineTimingReport) -> Mapping[str, PhaseAggregate]:
    """Return phase aggregates as a read-only name mapping."""

    return {aggregate.name: aggregate for aggregate in report.phase_totals}


__all__ = [
    "DEFAULT_QUEUE_CAPACITY",
    "WEMM_PIPELINE_BENCHMARK_VERSION",
    "PhaseAggregate",
    "PhaseRecorder",
    "PhaseSample",
    "PipelineItemTiming",
    "PipelinePhase",
    "WemmPipelineBenchmarkError",
    "WemmPipelineRun",
    "WemmPipelineTimingReport",
    "phase_totals_by_name",
    "run_bounded_pipeline",
]
