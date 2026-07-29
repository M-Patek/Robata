"""Small, dependency-free benchmark accounting primitives for throughput Track T1/T2.

The helpers intentionally do not declare a capacity claim. A measured summary must carry the
complete content-addressed BenchmarkEvidenceContext for its approved frozen inputs. Even a
bound summary remains evidence for a later promotion decision, not self-issued certification.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from robata.benchmark.evidence import BenchmarkEvidenceContext
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityInput,
    MeasuredCapacityReport,
    ProviderMode,
    build_measured_capacity_report,
    required_worker_count_for_rtf,
)

_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_MILLISECONDS_PER_HOUR = 3_600_000


@dataclass(frozen=True, slots=True)
class ThroughputSample:
    """One timed workload sample with both required throughput units."""

    elapsed_ms: int
    recording_duration_ns: int
    camera_count: int = 6

    def __post_init__(self) -> None:
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, int):
            raise TypeError("elapsed_ms must be an integer")
        if self.elapsed_ms <= 0:
            raise ValueError("elapsed_ms must be positive")
        if isinstance(self.recording_duration_ns, bool) or not isinstance(
            self.recording_duration_ns, int
        ):
            raise TypeError("recording_duration_ns must be an integer")
        if self.recording_duration_ns <= 0:
            raise ValueError("recording_duration_ns must be positive")
        if isinstance(self.camera_count, bool) or not isinstance(self.camera_count, int):
            raise TypeError("camera_count must be an integer")
        if self.camera_count <= 0:
            raise ValueError("camera_count must be positive")

    @property
    def recording_hours_per_wall_hour(self) -> float:
        """Return recording-hours processed per wall-clock hour."""

        return self.recording_duration_ns / self.elapsed_ms / 1_000_000

    @property
    def camera_video_hours_per_wall_hour(self) -> float:
        """Return camera-video-hours processed per wall-clock hour."""

        return self.recording_hours_per_wall_hour * self.camera_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "recording_duration_ns": str(self.recording_duration_ns),
            "camera_count": self.camera_count,
            "recording_hours_per_wall_hour": self.recording_hours_per_wall_hour,
            "camera_video_hours_per_wall_hour": self.camera_video_hours_per_wall_hour,
        }


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """A throughput sample with portable CPU and traced-allocation observations."""

    throughput: ThroughputSample
    cpu_time_ms: float
    peak_tracemalloc_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.throughput, ThroughputSample):
            raise TypeError("throughput must be a ThroughputSample")
        if isinstance(self.cpu_time_ms, bool) or not isinstance(self.cpu_time_ms, (int, float)):
            raise TypeError("cpu_time_ms must be numeric")
        if not math.isfinite(self.cpu_time_ms) or self.cpu_time_ms < 0:
            raise ValueError("cpu_time_ms must be finite and nonnegative")
        if isinstance(self.peak_tracemalloc_bytes, bool) or not isinstance(
            self.peak_tracemalloc_bytes, int
        ):
            raise TypeError("peak_tracemalloc_bytes must be an integer")
        if self.peak_tracemalloc_bytes < 0:
            raise ValueError("peak_tracemalloc_bytes must be nonnegative")

    def as_dict(self) -> dict[str, Any]:
        payload = self.throughput.as_dict()
        payload.update(
            {
                "cpu_time_ms": self.cpu_time_ms,
                "peak_tracemalloc_bytes": self.peak_tracemalloc_bytes,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Deterministic summary bound to governed evidence or explicitly local."""

    workload_id: str
    samples: tuple[ThroughputSample, ...]
    evidence_context: BenchmarkEvidenceContext | None = None

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id must be nonempty")
        if not self.samples:
            raise ValueError("samples must be nonempty")
        if self.evidence_context is not None and not isinstance(
            self.evidence_context, BenchmarkEvidenceContext
        ):
            raise TypeError("evidence_context must be a BenchmarkEvidenceContext")

    @property
    def measurement_status(self) -> str:
        return "MEASURED" if self.evidence_context is not None else "NOT_MEASURED"

    @property
    def mean_elapsed_ms(self) -> float:
        return sum(sample.elapsed_ms for sample in self.samples) / len(self.samples)

    @property
    def p50_elapsed_ms(self) -> int:
        return _nearest_rank(
            tuple(sample.elapsed_ms for sample in self.samples),
            0.50,
        )

    @property
    def p95_elapsed_ms(self) -> int:
        return _nearest_rank(
            tuple(sample.elapsed_ms for sample in self.samples),
            0.95,
        )

    @property
    def mean_recording_hours_per_wall_hour(self) -> float:
        return sum(sample.recording_hours_per_wall_hour for sample in self.samples) / len(
            self.samples
        )

    @property
    def mean_camera_video_hours_per_wall_hour(self) -> float:
        return sum(sample.camera_video_hours_per_wall_hour for sample in self.samples) / len(
            self.samples
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "evidence_context": (
                self.evidence_context.model_dump(mode="json")
                if self.evidence_context is not None
                else None
            ),
            "measurement_status": self.measurement_status,
            "sample_count": len(self.samples),
            "elapsed_ms": {
                "mean": self.mean_elapsed_ms,
                "p50": self.p50_elapsed_ms,
                "p95": self.p95_elapsed_ms,
            },
            "throughput": {
                "recording_hours_per_wall_hour": self.mean_recording_hours_per_wall_hour,
                "camera_video_hours_per_wall_hour": self.mean_camera_video_hours_per_wall_hour,
            },
            "samples": [sample.as_dict() for sample in self.samples],
        }


def measure_callable(
    workload: Callable[[], object],
    *,
    recording_duration_ns: int,
    camera_count: int = 6,
    clock: Callable[[], float] = time.perf_counter,
) -> ThroughputSample:
    """Measure one callable without interpreting the result as a capacity claim."""

    started = clock()
    workload()
    elapsed_seconds = clock() - started
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("benchmark clock must advance by a positive finite duration")
    elapsed_ms = max(1, round(elapsed_seconds * 1_000))
    return ThroughputSample(
        elapsed_ms=elapsed_ms,
        recording_duration_ns=recording_duration_ns,
        camera_count=camera_count,
    )


def measure_callable_with_resources(
    workload: Callable[[], object],
    *,
    recording_duration_ns: int,
    camera_count: int = 6,
    clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
) -> ResourceSample:
    """Measure wall throughput plus portable CPU/traced-allocation observations."""

    tracing_was_active = tracemalloc.is_tracing()
    if not tracing_was_active:
        tracemalloc.start()
    try:
        started_cpu = cpu_clock()
        sample = measure_callable(
            workload,
            recording_duration_ns=recording_duration_ns,
            camera_count=camera_count,
            clock=clock,
        )
        cpu_elapsed_ms = max(0.0, (cpu_clock() - started_cpu) * 1_000)
        _current, peak_bytes = tracemalloc.get_traced_memory()
        return ResourceSample(
            throughput=sample,
            cpu_time_ms=cpu_elapsed_ms,
            peak_tracemalloc_bytes=peak_bytes,
        )
    finally:
        # Never leak tracing state when the workload or either clock raises.  If the caller
        # had tracing enabled already, ownership remains with the caller.
        if not tracing_was_active:
            tracemalloc.stop()


def run_repeated(
    workload: Callable[[], object],
    *,
    workload_id: str,
    recording_duration_ns: int,
    iterations: int = 1,
    warmups: int = 0,
    camera_count: int = 6,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkSummary:
    """Run a callable repeatedly and return an explicitly local benchmark summary.

    Warmups are intentionally excluded from the emitted samples.  The helper is suitable
    for local engineering evidence only. A measured report must be rebuilt with a complete
    BenchmarkEvidenceContext; the summary does not itself grant promotion.
    """

    if not callable(workload):
        raise TypeError("workload must be callable")
    for field, value in (("iterations", iterations), ("warmups", warmups)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be an integer")
        if value < 0 or (field == "iterations" and value == 0):
            raise ValueError(
                f"{field} must be positive"
                if field == "iterations"
                else f"{field} must be nonnegative"
            )

    for _ in range(warmups):
        workload()
    samples = tuple(
        measure_callable(
            workload,
            recording_duration_ns=recording_duration_ns,
            camera_count=camera_count,
            clock=clock,
        )
        for _ in range(iterations)
    )
    return summarize_samples(workload_id, samples)


def summarize_samples(
    workload_id: str,
    samples: Iterable[ThroughputSample],
    *,
    evidence_context: BenchmarkEvidenceContext | None = None,
) -> BenchmarkSummary:
    """Build a summary while preserving sample order and evidence identity."""

    return BenchmarkSummary(
        workload_id=workload_id,
        samples=tuple(samples),
        evidence_context=evidence_context,
    )


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    if not values:
        raise ValueError("values must be nonempty")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True, slots=True)
class RecordingWorkerQueueObservation:
    """Queue facts captured from one finite recording-worker batch.

    The observation deliberately accepts an over-capacity high-water mark so a
    qualification report can *fail* its bounded-queue gate instead of discarding
    the evidence as invalid input. ``backpressure_event_count`` records actual
    producer delay/rejection events when the queue implementation exposes them.
    """

    name: str
    capacity: int
    high_watermark: int
    end_depth: int
    backpressure_event_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("queue name must be nonempty")
        for field_name in (
            "capacity",
            "high_watermark",
            "end_depth",
            "backpressure_event_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.end_depth > self.high_watermark:
            raise ValueError("end_depth cannot exceed high_watermark")

    @property
    def bounded(self) -> bool:
        return self.high_watermark <= self.capacity and self.end_depth <= self.capacity

    @property
    def drained(self) -> bool:
        return self.end_depth == 0

    @property
    def burst_observed(self) -> bool:
        """Whether this queue held at least one item during the finite batch."""

        return self.high_watermark > 0

    @property
    def reached_capacity(self) -> bool:
        """Whether occupancy reached the configured backpressure threshold."""

        return self.high_watermark >= self.capacity

    @property
    def backpressure_observed(self) -> bool:
        """Whether a producer actually reported a backpressure event."""

        return self.backpressure_event_count > 0

    @property
    def drained_after_burst(self) -> bool:
        return self.burst_observed and self.drained

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "high_watermark": self.high_watermark,
            "end_depth": self.end_depth,
            "backpressure_event_count": self.backpressure_event_count,
            "bounded": self.bounded,
            "drained": self.drained,
            "burst_observed": self.burst_observed,
            "reached_capacity": self.reached_capacity,
            "backpressure_observed": self.backpressure_observed,
            "drained_after_burst": self.drained_after_burst,
        }


@dataclass(frozen=True, slots=True)
class RecordingWorkerConcurrency:
    """Explicit stage concurrency used by one recording-worker observation.

    A worker matrix is useful only when the shared limits are visible. Keeping
    media, provider, completion, and outbox pools separate prevents a report
    from attributing a provider bottleneck to recording workers. ``None`` on
    :class:`MeasuredRecordingWorkerRun` means the caller did not instrument
    this boundary; it is never represented as zero.
    """

    media_worker_count: int
    provider_worker_count: int
    completion_worker_count: int
    outbox_worker_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "media_worker_count",
            "provider_worker_count",
            "completion_worker_count",
            "outbox_worker_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "media_worker_count": self.media_worker_count,
            "provider_worker_count": self.provider_worker_count,
            "completion_worker_count": self.completion_worker_count,
            "outbox_worker_count": self.outbox_worker_count,
        }


@dataclass(frozen=True, slots=True)
class RecordingWorkerBatchFacts:
    """Queue, completion, cancellation, and state-affinity facts from a real batch."""

    successful_recording_count: int
    failed_recording_count: int
    cancelled_recording_count: int
    replay_verified_recording_count: int
    distinct_state_root_count: int
    state_affinity_violation_count: int
    queues: tuple[RecordingWorkerQueueObservation, ...]
    named_shared_resource_limit: str | None = None
    admission_rejection_count: int = 0
    lease_recovery_count: int = 0
    lease_recovery_succeeded_count: int = 0
    optional_work_offered_count: int = 0
    optional_work_shed_count: int = 0
    optional_work_shedding_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "successful_recording_count",
            "failed_recording_count",
            "cancelled_recording_count",
            "replay_verified_recording_count",
            "distinct_state_root_count",
            "state_affinity_violation_count",
            "admission_rejection_count",
            "lease_recovery_count",
            "lease_recovery_succeeded_count",
            "optional_work_offered_count",
            "optional_work_shed_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if not isinstance(self.queues, tuple):
            raise TypeError("queues must be a tuple")
        if any(not isinstance(queue, RecordingWorkerQueueObservation) for queue in self.queues):
            raise TypeError("queues must contain RecordingWorkerQueueObservation values")
        names = tuple(queue.name for queue in self.queues)
        if names != ("ingress", "provider", "publish"):
            raise ValueError("queues must be ingress, provider, and publish in order")
        if self.named_shared_resource_limit is not None and (
            not isinstance(self.named_shared_resource_limit, str)
            or not self.named_shared_resource_limit.strip()
        ):
            raise ValueError("named_shared_resource_limit must be nonempty or None")
        if not isinstance(self.optional_work_shedding_actions, tuple):
            raise TypeError("optional_work_shedding_actions must be a tuple")
        if any(
            not isinstance(action, str) or not action.strip()
            for action in self.optional_work_shedding_actions
        ):
            raise ValueError("optional_work_shedding_actions must contain nonempty strings")
        if len(set(self.optional_work_shedding_actions)) != len(
            self.optional_work_shedding_actions
        ):
            raise ValueError("optional_work_shedding_actions must be unique")
        if self.lease_recovery_succeeded_count > self.lease_recovery_count:
            raise ValueError("lease recovery successes cannot exceed recovery attempts")
        if self.optional_work_shed_count > self.optional_work_offered_count:
            raise ValueError("optional work shed cannot exceed optional work offered")
        if self.optional_work_shed_count and not self.optional_work_shedding_actions:
            raise ValueError("optional work shedding requires at least one named action")
        if not self.optional_work_shed_count and self.optional_work_shedding_actions:
            raise ValueError("optional shedding actions require shed work")

    @property
    def queues_bounded(self) -> bool:
        return all(queue.bounded for queue in self.queues)

    @property
    def backlog_drained(self) -> bool:
        return all(queue.drained for queue in self.queues)

    @property
    def queue_burst_observed(self) -> bool:
        """All three queue stages carried work during the burst observation."""

        return all(queue.burst_observed for queue in self.queues)

    @property
    def ingress_backpressure_observed(self) -> bool:
        """Ingress reached its bound or reported an actual producer backpressure event."""

        ingress = self.queues[0]
        return ingress.reached_capacity or ingress.backpressure_observed

    @property
    def burst_backpressure_drained(self) -> bool:
        """The exercised ingress bound drained after all queue stages saw the burst."""

        return (
            self.queue_burst_observed
            and self.ingress_backpressure_observed
            and self.backlog_drained
        )

    @property
    def replay_verified_cancelled_recording_count(self) -> int:
        """Cancelled recordings whose restart/replay terminal was actually verified."""

        return min(
            self.cancelled_recording_count,
            self.replay_verified_recording_count,
        )

    @property
    def cancellation_restart_replayable(self) -> bool:
        return (
            self.cancelled_recording_count == 0
            or self.replay_verified_cancelled_recording_count == self.cancelled_recording_count
        )

    @property
    def lease_recovery_reconciled(self) -> bool:
        """Whether every observed lease recovery attempt reached a terminal proof."""

        return self.lease_recovery_succeeded_count == self.lease_recovery_count

    @property
    def optional_work_admitted_count(self) -> int:
        return self.optional_work_offered_count - self.optional_work_shed_count

    @property
    def optional_work_shed_fraction(self) -> float | None:
        if self.optional_work_offered_count == 0:
            return None
        return self.optional_work_shed_count / self.optional_work_offered_count

    @property
    def backlog_peak(self) -> int:
        """Aggregate queue high-water mark for this finite batch."""

        return sum(queue.high_watermark for queue in self.queues)

    @property
    def backlog_end(self) -> int:
        """Aggregate residual queue depth at the observation cutoff."""

        return sum(queue.end_depth for queue in self.queues)

    @property
    def backpressure_event_count(self) -> int:
        return sum(queue.backpressure_event_count for queue in self.queues)

    def as_dict(self) -> dict[str, object]:
        return {
            "successful_recording_count": self.successful_recording_count,
            "failed_recording_count": self.failed_recording_count,
            "cancelled_recording_count": self.cancelled_recording_count,
            "replay_verified_recording_count": self.replay_verified_recording_count,
            "distinct_state_root_count": self.distinct_state_root_count,
            "state_affinity_violation_count": self.state_affinity_violation_count,
            "admission_rejection_count": self.admission_rejection_count,
            "lease_recovery_count": self.lease_recovery_count,
            "lease_recovery_succeeded_count": self.lease_recovery_succeeded_count,
            "lease_recovery_reconciled": self.lease_recovery_reconciled,
            "optional_work_offered_count": self.optional_work_offered_count,
            "optional_work_admitted_count": self.optional_work_admitted_count,
            "optional_work_shed_count": self.optional_work_shed_count,
            "optional_work_shed_fraction": self.optional_work_shed_fraction,
            "optional_work_shedding_actions": list(self.optional_work_shedding_actions),
            "backlog_peak": self.backlog_peak,
            "backlog_end": self.backlog_end,
            "backpressure_event_count": self.backpressure_event_count,
            "named_shared_resource_limit": self.named_shared_resource_limit,
            "queues_bounded": self.queues_bounded,
            "backlog_drained": self.backlog_drained,
            "queues": [queue.as_dict() for queue in self.queues],
        }


@dataclass(frozen=True, slots=True)
class MeasuredRecordingWorkerRun:
    """One timed local 1/2/4/N recording-worker batch observation.

    ``recording_duration_ns`` is the duration of each recording. The capacity
    report below multiplies it by ``recording_count`` and remains visibly local
    conformance evidence rather than a production-capacity assertion.
    """

    workload_id: str
    worker_count: int
    recording_count: int
    recording_duration_ns: int
    elapsed_ns: int
    facts: RecordingWorkerBatchFacts
    camera_count: int = 6
    provider_mode: ProviderMode = ProviderMode.LOCAL_OFFLINE_FIXTURE
    execution_mode: str = "FRESH"
    concurrency: RecordingWorkerConcurrency | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.workload_id, str) or not self.workload_id:
            raise ValueError("workload_id must be nonempty")
        for field_name in (
            "worker_count",
            "recording_count",
            "recording_duration_ns",
            "elapsed_ns",
            "camera_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not isinstance(self.facts, RecordingWorkerBatchFacts):
            raise TypeError("facts must be RecordingWorkerBatchFacts")
        if not isinstance(self.provider_mode, ProviderMode):
            raise TypeError("provider_mode must be ProviderMode")
        if self.execution_mode not in {"FRESH", "REPLAY", "UNKNOWN"}:
            raise ValueError("execution_mode must be FRESH, REPLAY, or UNKNOWN")
        if self.concurrency is not None and not isinstance(
            self.concurrency,
            RecordingWorkerConcurrency,
        ):
            raise TypeError("concurrency must be RecordingWorkerConcurrency or None")
        if (
            self.facts.successful_recording_count
            + self.facts.failed_recording_count
            + self.facts.cancelled_recording_count
            != self.recording_count
        ):
            raise ValueError("recording terminal counts must reconcile")
        if self.facts.successful_recording_count > self.facts.distinct_state_root_count:
            raise ValueError("successful recordings cannot exceed distinct state roots")

    @property
    def capacity(self) -> MeasuredCapacityReport:
        return build_measured_capacity_report(
            MeasuredCapacityInput(
                workload_fingerprint=self.workload_id,
                evidence_class=CapacityEvidenceClass.LOCAL_CONFORMANCE,
                provider_mode=self.provider_mode,
                execution_mode=self.execution_mode,
                recording_count=self.recording_count,
                recording_worker_count=self.worker_count,
                camera_count=self.camera_count,
                recording_duration_ns=self.recording_duration_ns,
                wall_time_ns=self.elapsed_ns,
            )
        )

    @property
    def recording_rtf(self) -> float:
        result = self.capacity.recording_hours_per_wall_hour
        if result is None:
            raise AssertionError("a timed local batch must expose recording RTF")
        return result

    @property
    def terminal_or_replay_completed_recording_count(self) -> int:
        """Fresh terminals plus cancellations proved safe by an independent replay."""

        return (
            self.facts.successful_recording_count
            + self.facts.replay_verified_cancelled_recording_count
        )

    @property
    def complete_without_state_leakage(self) -> bool:
        """All recordings completed freshly or by a verified replay on distinct state roots."""

        return (
            self.terminal_or_replay_completed_recording_count == self.recording_count
            and self.facts.failed_recording_count == 0
            and self.facts.cancellation_restart_replayable
            and self.facts.distinct_state_root_count == self.recording_count
            and self.facts.state_affinity_violation_count == 0
        )

    @property
    def sustainable(self) -> bool:
        return (
            self.complete_without_state_leakage
            and self.facts.queues_bounded
            and self.facts.burst_backpressure_drained
            and self.facts.cancellation_restart_replayable
            and self.facts.lease_recovery_reconciled
        )

    @property
    def backlog_peak(self) -> int:
        return self.facts.backlog_peak

    @property
    def backlog_end(self) -> int:
        return self.facts.backlog_end

    @property
    def optional_work_shed_fraction(self) -> float | None:
        return self.facts.optional_work_shed_fraction

    def as_dict(self, *, throughput_ratio: float) -> dict[str, object]:
        return {
            "worker_count": self.worker_count,
            "recording_count": self.recording_count,
            "recording_duration_ns": str(self.recording_duration_ns),
            "elapsed_ns": str(self.elapsed_ns),
            "recording_rtf": self.recording_rtf,
            "camera_video_rtf": self.capacity.camera_hours_per_wall_hour,
            "throughput_ratio": throughput_ratio,
            "complete_without_state_leakage": self.complete_without_state_leakage,
            "successful_recording_count": self.facts.successful_recording_count,
            "failed_recording_count": self.facts.failed_recording_count,
            "cancelled_recording_count": self.facts.cancelled_recording_count,
            "replay_verified_recording_count": self.facts.replay_verified_recording_count,
            "replay_verified_cancelled_recording_count": (
                self.facts.replay_verified_cancelled_recording_count
            ),
            "terminal_or_replay_completed_recording_count": (
                self.terminal_or_replay_completed_recording_count
            ),
            "distinct_state_root_count": self.facts.distinct_state_root_count,
            "state_affinity_violation_count": self.facts.state_affinity_violation_count,
            "queues_bounded": self.facts.queues_bounded,
            "queue_burst_observed": self.facts.queue_burst_observed,
            "ingress_backpressure_observed": self.facts.ingress_backpressure_observed,
            "backlog_drained": self.facts.backlog_drained,
            "burst_backpressure_drained": self.facts.burst_backpressure_drained,
            "cancellation_restart_replayable": self.facts.cancellation_restart_replayable,
            "backlog_peak": self.backlog_peak,
            "backlog_end": self.backlog_end,
            "backpressure_event_count": self.facts.backpressure_event_count,
            "admission_rejection_count": self.facts.admission_rejection_count,
            "lease_recovery_count": self.facts.lease_recovery_count,
            "lease_recovery_succeeded_count": self.facts.lease_recovery_succeeded_count,
            "lease_recovery_reconciled": self.facts.lease_recovery_reconciled,
            "optional_work_offered_count": self.facts.optional_work_offered_count,
            "optional_work_admitted_count": self.facts.optional_work_admitted_count,
            "optional_work_shed_count": self.facts.optional_work_shed_count,
            "optional_work_shed_fraction": self.optional_work_shed_fraction,
            "optional_work_shedding_actions": list(self.facts.optional_work_shedding_actions),
            "named_shared_resource_limit": self.facts.named_shared_resource_limit,
            "concurrency": (None if self.concurrency is None else self.concurrency.as_dict()),
            "queues": [queue.as_dict() for queue in self.facts.queues],
        }


@dataclass(frozen=True, slots=True)
class MeasuredRecordingWorkerCapacityProjection:
    """A lower-bound CPU/NVMe worker projection from one sustainable local sample."""

    basis_worker_count: int
    basis_recording_rtf: float
    per_worker_recording_rtf: float
    target_recording_rtf: float
    required_cpu_worker_count: int
    required_nvme_worker_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "basis_worker_count",
            "required_cpu_worker_count",
            "required_nvme_worker_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "basis_recording_rtf",
            "per_worker_recording_rtf",
            "target_recording_rtf",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive")

    @property
    def evidence_class(self) -> str:
        return "LOCAL_CONFORMANCE"

    @property
    def measurement_status(self) -> str:
        return "MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False

    def as_dict(self) -> dict[str, object]:
        return {
            "basis_worker_count": self.basis_worker_count,
            "basis_recording_rtf": self.basis_recording_rtf,
            "per_worker_recording_rtf": self.per_worker_recording_rtf,
            "target_recording_rtf": self.target_recording_rtf,
            "required_cpu_worker_count": self.required_cpu_worker_count,
            "required_nvme_worker_count": self.required_nvme_worker_count,
            "evidence_class": self.evidence_class,
            "measurement_status": self.measurement_status,
            "production_eligible": self.production_eligible,
        }


@dataclass(frozen=True, slots=True)
class MeasuredRecordingWorkerScalingReport:
    """Measured 1/2/4/N local worker scaling, queue, and saturation report."""

    workload_id: str
    runs: tuple[MeasuredRecordingWorkerRun, ...]
    target_recording_rtf: float
    capacity_projection: MeasuredRecordingWorkerCapacityProjection | None

    def __post_init__(self) -> None:
        if not isinstance(self.workload_id, str) or not self.workload_id:
            raise ValueError("workload_id must be nonempty")
        if not isinstance(self.runs, tuple) or not self.runs:
            raise ValueError("runs must be a nonempty tuple")
        workers = tuple(run.worker_count for run in self.runs)
        if workers != tuple(sorted(set(workers))):
            raise ValueError("runs must have sorted, unique worker counts")
        if not {1, 2, 4}.issubset(workers):
            raise ValueError("runs must include 1, 2, and 4 workers")
        first = self.runs[0]
        if first.worker_count != 1:
            raise ValueError("runs must start with a one-worker baseline")
        if any(run.workload_id != self.workload_id for run in self.runs):
            raise ValueError("runs must share workload_id")
        if any(
            (
                run.recording_count,
                run.recording_duration_ns,
                run.camera_count,
                run.provider_mode,
                run.execution_mode,
            )
            != (
                first.recording_count,
                first.recording_duration_ns,
                first.camera_count,
                first.provider_mode,
                first.execution_mode,
            )
            for run in self.runs
        ):
            raise ValueError("runs must use the same workload and provider mode")
        if (
            isinstance(self.target_recording_rtf, bool)
            or not isinstance(self.target_recording_rtf, (int, float))
            or not math.isfinite(self.target_recording_rtf)
            or self.target_recording_rtf <= 0
        ):
            raise ValueError("target_recording_rtf must be finite and positive")
        if self.capacity_projection is not None and not isinstance(
            self.capacity_projection,
            MeasuredRecordingWorkerCapacityProjection,
        ):
            raise TypeError("capacity_projection must be MeasuredRecordingWorkerCapacityProjection")

    @property
    def worker_counts(self) -> tuple[int, ...]:
        return tuple(run.worker_count for run in self.runs)

    @property
    def baseline(self) -> MeasuredRecordingWorkerRun:
        return self.runs[0]

    @property
    def four_worker_run(self) -> MeasuredRecordingWorkerRun:
        return next(run for run in self.runs if run.worker_count == 4)

    @property
    def four_worker_speedup(self) -> float:
        return self.four_worker_run.recording_rtf / self.baseline.recording_rtf

    @property
    def four_worker_meets_2_5x(self) -> bool:
        return self.four_worker_speedup >= 2.5

    @property
    def four_worker_named_shared_resource_limit(self) -> str | None:
        return self.four_worker_run.facts.named_shared_resource_limit

    @property
    def four_worker_outcome_explained(self) -> bool:
        return self.four_worker_run.sustainable and (
            self.four_worker_meets_2_5x or self.four_worker_named_shared_resource_limit is not None
        )

    @property
    def named_shared_resource_limits(self) -> tuple[tuple[int, str], ...]:
        """All explicitly observed shared limits, including scale-out points above four."""

        return tuple(
            (run.worker_count, run.facts.named_shared_resource_limit)
            for run in self.runs
            if run.facts.named_shared_resource_limit is not None
        )

    @property
    def saturation_worker_count(self) -> int | None:
        """First 4+ worker point that identifies a shared saturation resource."""

        return next(
            (
                worker_count
                for worker_count, _limit in self.named_shared_resource_limits
                if worker_count >= 4
            ),
            None,
        )

    @property
    def saturation_named_shared_resource_limit(self) -> str | None:
        saturation_worker_count = self.saturation_worker_count
        if saturation_worker_count is None:
            return None
        return next(
            limit
            for worker_count, limit in self.named_shared_resource_limits
            if worker_count == saturation_worker_count
        )

    @property
    def saturation_outcome_explained(self) -> bool | None:
        """Whether every run at/after a named saturation point remains sustainable.

        ``None`` means the matrix did not identify a named saturation resource;
        it does not imply that the workload is unsaturated.
        """

        saturation_worker_count = self.saturation_worker_count
        if saturation_worker_count is None:
            return None
        return all(
            run.sustainable for run in self.runs if run.worker_count >= saturation_worker_count
        )

    @property
    def scale_out_worker_counts(self) -> tuple[int, ...]:
        """Measured worker counts beyond the required 1/2/4 matrix."""

        return tuple(run.worker_count for run in self.runs if run.worker_count > 4)

    @property
    def unexplained_non_scaling_worker_counts(self) -> tuple[int, ...]:
        """N>4 points that stopped improving without a previously named limit."""

        unresolved: list[int] = []
        previous: MeasuredRecordingWorkerRun | None = None
        active_named_limit: str | None = None
        for run in self.runs:
            if run.facts.named_shared_resource_limit is not None:
                active_named_limit = run.facts.named_shared_resource_limit
            if (
                previous is not None
                and run.worker_count > 4
                and run.recording_rtf <= previous.recording_rtf
                and active_named_limit is None
            ):
                unresolved.append(run.worker_count)
            previous = run
        return tuple(unresolved)

    @property
    def scale_out_outcome_explained(self) -> bool:
        """Higher-than-four worker points either improve or carry a named limit."""

        return not self.unexplained_non_scaling_worker_counts and all(
            run.sustainable for run in self.runs if run.worker_count > 4
        )

    @property
    def n_worker_outcome_explained(self) -> bool:
        """The required four-worker gate plus every measured N>4 scale-out point."""

        return self.four_worker_outcome_explained and self.scale_out_outcome_explained

    @property
    def queues_bounded(self) -> bool:
        return all(run.facts.queues_bounded for run in self.runs)

    @property
    def queue_burst_observed(self) -> bool:
        return all(run.facts.queue_burst_observed for run in self.runs)

    @property
    def ingress_backpressure_observed(self) -> bool:
        return all(run.facts.ingress_backpressure_observed for run in self.runs)

    @property
    def backlog_drains_after_burst(self) -> bool:
        return all(run.facts.burst_backpressure_drained for run in self.runs)

    @property
    def cancellation_restart_replayable(self) -> bool:
        return all(run.facts.cancellation_restart_replayable for run in self.runs)

    @property
    def lease_recovery_reconciled(self) -> bool:
        return all(run.facts.lease_recovery_reconciled for run in self.runs)

    @property
    def backlog_peak(self) -> int:
        """Largest aggregate queue backlog observed at any worker point."""

        return max(run.backlog_peak for run in self.runs)

    @property
    def backlog_end(self) -> int:
        """Largest residual aggregate queue depth across worker points."""

        return max(run.backlog_end for run in self.runs)

    @property
    def backpressure_event_count(self) -> int:
        return sum(run.facts.backpressure_event_count for run in self.runs)

    @property
    def admission_rejection_count(self) -> int:
        return sum(run.facts.admission_rejection_count for run in self.runs)

    @property
    def lease_recovery_count(self) -> int:
        return sum(run.facts.lease_recovery_count for run in self.runs)

    @property
    def lease_recovery_succeeded_count(self) -> int:
        return sum(run.facts.lease_recovery_succeeded_count for run in self.runs)

    @property
    def optional_work_offered_count(self) -> int:
        return sum(run.facts.optional_work_offered_count for run in self.runs)

    @property
    def optional_work_shed_count(self) -> int:
        return sum(run.facts.optional_work_shed_count for run in self.runs)

    @property
    def optional_work_shed_fraction(self) -> float | None:
        offered = self.optional_work_offered_count
        if offered == 0:
            return None
        return self.optional_work_shed_count / offered

    @property
    def optional_work_shedding_actions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {action for run in self.runs for action in run.facts.optional_work_shedding_actions}
            )
        )

    @property
    def stage_concurrency_observed(self) -> bool:
        return all(run.concurrency is not None for run in self.runs)

    @property
    def evidence_class(self) -> str:
        return "LOCAL_CONFORMANCE"

    @property
    def measurement_status(self) -> str:
        return "MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False

    def as_dict(self) -> dict[str, object]:
        baseline_rtf = self.baseline.recording_rtf
        return {
            "workload_id": self.workload_id,
            "worker_counts": list(self.worker_counts),
            "target_recording_rtf": self.target_recording_rtf,
            "four_worker_speedup": self.four_worker_speedup,
            "four_worker_meets_2_5x": self.four_worker_meets_2_5x,
            "four_worker_named_shared_resource_limit": self.four_worker_named_shared_resource_limit,
            "four_worker_outcome_explained": self.four_worker_outcome_explained,
            "scale_out_worker_counts": list(self.scale_out_worker_counts),
            "named_shared_resource_limits": [
                {"worker_count": worker_count, "limit": limit}
                for worker_count, limit in self.named_shared_resource_limits
            ],
            "saturation_worker_count": self.saturation_worker_count,
            "saturation_named_shared_resource_limit": self.saturation_named_shared_resource_limit,
            "saturation_outcome_explained": self.saturation_outcome_explained,
            "unexplained_non_scaling_worker_counts": list(
                self.unexplained_non_scaling_worker_counts
            ),
            "scale_out_outcome_explained": self.scale_out_outcome_explained,
            "n_worker_outcome_explained": self.n_worker_outcome_explained,
            "queues_bounded": self.queues_bounded,
            "queue_burst_observed": self.queue_burst_observed,
            "ingress_backpressure_observed": self.ingress_backpressure_observed,
            "backlog_drains_after_burst": self.backlog_drains_after_burst,
            "cancellation_restart_replayable": self.cancellation_restart_replayable,
            "lease_recovery_reconciled": self.lease_recovery_reconciled,
            "backlog_peak": self.backlog_peak,
            "backlog_end": self.backlog_end,
            "backpressure_event_count": self.backpressure_event_count,
            "admission_rejection_count": self.admission_rejection_count,
            "lease_recovery_count": self.lease_recovery_count,
            "lease_recovery_succeeded_count": self.lease_recovery_succeeded_count,
            "optional_work_offered_count": self.optional_work_offered_count,
            "optional_work_shed_count": self.optional_work_shed_count,
            "optional_work_shed_fraction": self.optional_work_shed_fraction,
            "optional_work_shedding_actions": list(self.optional_work_shedding_actions),
            "stage_concurrency_observed": self.stage_concurrency_observed,
            "capacity_projection": (
                None if self.capacity_projection is None else self.capacity_projection.as_dict()
            ),
            "runs": [
                run.as_dict(throughput_ratio=run.recording_rtf / baseline_rtf) for run in self.runs
            ],
            "evidence_class": self.evidence_class,
            "measurement_status": self.measurement_status,
            "production_eligible": self.production_eligible,
        }


def measure_recording_worker_batch(
    workload: Callable[[], RecordingWorkerBatchFacts],
    *,
    workload_id: str,
    worker_count: int,
    recording_count: int,
    recording_duration_ns: int,
    camera_count: int = 6,
    provider_mode: ProviderMode = ProviderMode.LOCAL_OFFLINE_FIXTURE,
    execution_mode: str = "FRESH",
    concurrency: RecordingWorkerConcurrency | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> MeasuredRecordingWorkerRun:
    """Time one real finite worker batch and preserve its topology/recovery facts."""

    if not callable(workload):
        raise TypeError("workload must be callable")
    started = clock_ns()
    facts = workload()
    elapsed_ns = clock_ns() - started
    if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int) or elapsed_ns <= 0:
        raise ValueError("benchmark clock must advance by a positive integer number of nanoseconds")
    return MeasuredRecordingWorkerRun(
        workload_id=workload_id,
        worker_count=worker_count,
        recording_count=recording_count,
        recording_duration_ns=recording_duration_ns,
        elapsed_ns=elapsed_ns,
        facts=facts,
        camera_count=camera_count,
        provider_mode=provider_mode,
        execution_mode=execution_mode,
        concurrency=concurrency,
    )


def build_measured_recording_worker_scaling_report(
    runs: Iterable[MeasuredRecordingWorkerRun],
    *,
    target_recording_rtf: float = 25.0,
) -> MeasuredRecordingWorkerScalingReport:
    """Build a sizing report from like-for-like measured recording-worker runs."""

    raw_runs = tuple(runs)
    if not raw_runs:
        raise ValueError("runs must be nonempty")
    if any(not isinstance(run, MeasuredRecordingWorkerRun) for run in raw_runs):
        raise TypeError("runs must contain MeasuredRecordingWorkerRun values")
    ordered = tuple(sorted(raw_runs, key=lambda run: run.worker_count))
    sustainable = tuple(run for run in ordered if run.sustainable)
    projection = None
    if sustainable:
        basis = max(sustainable, key=lambda run: run.recording_rtf / run.worker_count)
        per_worker_rtf = basis.recording_rtf / basis.worker_count
        required_workers = required_worker_count_for_rtf(
            per_worker_rtf,
            target_recording_rtf=target_recording_rtf,
        )
        projection = MeasuredRecordingWorkerCapacityProjection(
            basis_worker_count=basis.worker_count,
            basis_recording_rtf=basis.recording_rtf,
            per_worker_recording_rtf=per_worker_rtf,
            target_recording_rtf=target_recording_rtf,
            required_cpu_worker_count=required_workers,
            required_nvme_worker_count=required_workers,
        )
    return MeasuredRecordingWorkerScalingReport(
        workload_id=ordered[0].workload_id,
        runs=ordered,
        target_recording_rtf=target_recording_rtf,
        capacity_projection=projection,
    )


def run_measured_recording_worker_matrix(
    workload_factory: Callable[[int], Callable[[], RecordingWorkerBatchFacts]],
    *,
    workload_id: str,
    recording_count: int,
    recording_duration_ns: int,
    worker_counts: tuple[int, ...] = (1, 2, 4),
    camera_count: int = 6,
    provider_mode: ProviderMode = ProviderMode.LOCAL_OFFLINE_FIXTURE,
    execution_mode: str = "FRESH",
    concurrency_factory: Callable[[int], RecordingWorkerConcurrency] | None = None,
    target_recording_rtf: float = 25.0,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> MeasuredRecordingWorkerScalingReport:
    """Execute a caller-provided real 1/2/4/N worker matrix.

    The factory owns process/thread creation and recording-affine state roots;
    it must return observed queue and replay facts after each finite batch.
    """

    if not callable(workload_factory):
        raise TypeError("workload_factory must be callable")
    if concurrency_factory is not None and not callable(concurrency_factory):
        raise TypeError("concurrency_factory must be callable or None")
    if not isinstance(worker_counts, tuple) or not worker_counts:
        raise ValueError("worker_counts must be a nonempty tuple")
    if worker_counts != tuple(sorted(set(worker_counts))):
        raise ValueError("worker_counts must be sorted and unique")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in worker_counts
    ):
        raise ValueError("worker_counts must contain positive integers")
    return build_measured_recording_worker_scaling_report(
        (
            measure_recording_worker_batch(
                workload_factory(worker_count),
                workload_id=workload_id,
                worker_count=worker_count,
                recording_count=recording_count,
                recording_duration_ns=recording_duration_ns,
                camera_count=camera_count,
                provider_mode=provider_mode,
                execution_mode=execution_mode,
                concurrency=(
                    None if concurrency_factory is None else concurrency_factory(worker_count)
                ),
                clock_ns=clock_ns,
            )
            for worker_count in worker_counts
        ),
        target_recording_rtf=target_recording_rtf,
    )


__all__ = [
    "BenchmarkSummary",
    "MeasuredRecordingWorkerCapacityProjection",
    "MeasuredRecordingWorkerRun",
    "MeasuredRecordingWorkerScalingReport",
    "RecordingWorkerBatchFacts",
    "RecordingWorkerConcurrency",
    "RecordingWorkerQueueObservation",
    "ResourceSample",
    "ThroughputSample",
    "build_measured_recording_worker_scaling_report",
    "measure_callable",
    "measure_callable_with_resources",
    "measure_recording_worker_batch",
    "run_measured_recording_worker_matrix",
    "run_repeated",
    "summarize_samples",
]
