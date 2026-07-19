"""Small, dependency-free benchmark accounting primitives for throughput Track T1/T2.

The helpers intentionally do not declare a capacity claim. Callers must provide a governed
corpus identifier and explicitly opt into ``certifying=True`` only after the normative gates
and workload approval are complete.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

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
class BenchmarkSummary:
    """Deterministic summary of local benchmark samples.

    ``certifying`` defaults to false because local synthetic/fake-model runs are evidence for
    engineering only. A certifying summary requires an explicit governed corpus identifier.
    """

    workload_id: str
    samples: tuple[ThroughputSample, ...]
    certifying: bool = False
    corpus_id: str | None = None

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id must be nonempty")
        if not self.samples:
            raise ValueError("samples must be nonempty")
        if self.certifying and not self.corpus_id:
            raise ValueError("certifying summaries require corpus_id")

    @property
    def measurement_status(self) -> str:
        return "CERTIFYING" if self.certifying else "NOT_MEASURED"

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
            "corpus_id": self.corpus_id,
            "measurement_status": self.measurement_status,
            "certifying": self.certifying,
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
    """Run a callable repeatedly and return a non-certifying benchmark summary.

    Warmups are intentionally excluded from the emitted samples.  The helper is suitable
    for local engineering evidence only; callers must still supply a governed corpus and
    explicitly build a certifying summary before making a capacity claim.
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
    certifying: bool = False,
    corpus_id: str | None = None,
) -> BenchmarkSummary:
    """Build a summary while preserving sample order in the emitted report."""

    return BenchmarkSummary(
        workload_id=workload_id,
        samples=tuple(samples),
        certifying=certifying,
        corpus_id=corpus_id,
    )


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    if not values:
        raise ValueError("values must be nonempty")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


__all__ = [
    "BenchmarkSummary",
    "ThroughputSample",
    "measure_callable",
    "run_repeated",
    "summarize_samples",
]
