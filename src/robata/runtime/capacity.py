"""Deterministic synthetic capacity harness with non-promotional evidence semantics."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.hashing import semantic_sha256

_NANOSECONDS_PER_HOUR = 3_600_000_000_000
_MILLISECONDS_PER_HOUR = 3_600_000


class SyntheticOutcome(StrEnum):
    """Mutually exclusive outcome observed at the synthetic cutoff."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    PENDING = "PENDING"


class BottleneckKind(StrEnum):
    """Locally observed pressure signal, never a production diagnosis."""

    SERVICE_CAPACITY = "SERVICE_CAPACITY"
    QUEUE_WAIT = "QUEUE_WAIT"
    RELIABILITY = "RELIABILITY"
    DEADLINE = "DEADLINE"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class SyntheticLoadProfile:
    """Versioned deterministic arrival and service pattern."""

    version: str
    unit_count: int
    recording_duration_ns: int
    camera_stream_durations_ns: tuple[int, ...]
    arrival_interval_ms: int
    arrival_batch_size: int
    service_time_pattern_ms: tuple[int, ...]
    deadline_budget_ms: int
    observation_window_ms: int | None = None
    failed_ordinals: tuple[int, ...] = ()
    skipped_ordinals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("version", self.version)
        _require_positive_int("unit_count", self.unit_count)
        _require_positive_int("recording_duration_ns", self.recording_duration_ns)
        if not self.camera_stream_durations_ns:
            raise ValueError("camera_stream_durations_ns must be nonempty")
        for duration in self.camera_stream_durations_ns:
            _require_positive_int("camera stream duration", duration)
        _require_nonnegative_int("arrival_interval_ms", self.arrival_interval_ms)
        _require_positive_int("arrival_batch_size", self.arrival_batch_size)
        if not self.service_time_pattern_ms:
            raise ValueError("service_time_pattern_ms must be nonempty")
        for service_time in self.service_time_pattern_ms:
            _require_positive_int("service time", service_time)
        _require_positive_int("deadline_budget_ms", self.deadline_budget_ms)
        if self.observation_window_ms is not None:
            _require_positive_int("observation_window_ms", self.observation_window_ms)
        failed = _validate_ordinals("failed_ordinals", self.failed_ordinals, self.unit_count)
        skipped = _validate_ordinals("skipped_ordinals", self.skipped_ordinals, self.unit_count)
        if failed.intersection(skipped):
            raise ValueError("failed_ordinals and skipped_ordinals must be disjoint")

    @property
    def profile_digest(self) -> str:
        """Content address the complete synthetic workload policy."""

        return semantic_sha256(
            {
                "version": self.version,
                "unit_count": self.unit_count,
                "recording_duration_ns": str(self.recording_duration_ns),
                "camera_stream_durations_ns": [
                    str(value) for value in self.camera_stream_durations_ns
                ],
                "arrival_interval_ms": self.arrival_interval_ms,
                "arrival_batch_size": self.arrival_batch_size,
                "service_time_pattern_ms": list(self.service_time_pattern_ms),
                "deadline_budget_ms": self.deadline_budget_ms,
                "observation_window_ms": self.observation_window_ms,
                "failed_ordinals": sorted(self.failed_ordinals),
                "skipped_ordinals": sorted(self.skipped_ordinals),
            }
        )


@dataclass(frozen=True, slots=True)
class SyntheticWorkUnit:
    """One generated unit with explicit arrival, service, and deadline."""

    work_id: str
    ordinal: int
    arrival_at_ms: int
    service_time_ms: int
    deadline_at_ms: int
    recording_duration_ns: int
    camera_video_duration_ns: int
    planned_outcome: SyntheticOutcome


@dataclass(frozen=True, slots=True)
class SyntheticObservation:
    """Scheduled and cutoff-observed lifecycle for one synthetic unit."""

    work: SyntheticWorkUnit
    worker_ordinal: int | None
    started_at_ms: int | None
    scheduled_completion_at_ms: int
    observed_terminal_at_ms: int | None
    outcome: SyntheticOutcome
    queue_wait_ms: int
    service_time_ms: int
    wall_time_ms: int | None
    deadline_missed: bool


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Nearest-rank latency statistics over one explicit population."""

    count: int
    mean_ms: float
    p50_ms: int
    p95_ms: int
    p99_ms: int


@dataclass(frozen=True, slots=True)
class SyntheticCapacityReport:
    """Reconciled local simulation report that cannot claim measured capacity."""

    profile_version: str
    profile_digest: str
    worker_count: int
    cutoff_ms: int
    observations: tuple[SyntheticObservation, ...]
    succeeded_count: int
    failed_count: int
    skipped_count: int
    pending_count: int
    deadline_miss_count: int
    backlog_peak: int
    backlog_end: int
    utilization: float
    offered_recording_hours: float
    offered_camera_video_hours: float
    completed_recording_hours: float
    completed_camera_video_hours: float
    recording_hours_per_wall_hour: float
    camera_video_hours_per_wall_hour: float
    offered_units_per_wall_hour: float
    nominal_service_capacity_units_per_hour: float
    queue_wait: LatencySummary | None
    service_time: LatencySummary | None
    wall_time: LatencySummary | None
    bottlenecks: tuple[BottleneckKind, ...]

    @property
    def evidence_class(self) -> str:
        return "SYNTHETIC_LOCAL"

    @property
    def measurement_status(self) -> str:
        return "NOT_MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False

    @property
    def total_count(self) -> int:
        return self.succeeded_count + self.failed_count + self.skipped_count + self.pending_count

    @property
    def terminal_failure_rate(self) -> float:
        terminal_attempts = self.succeeded_count + self.failed_count
        return 0.0 if terminal_attempts == 0 else self.failed_count / terminal_attempts

    @property
    def skipped_rate(self) -> float:
        return 0.0 if self.total_count == 0 else self.skipped_count / self.total_count

    @property
    def deadline_miss_rate(self) -> float:
        eligible = self.succeeded_count + self.failed_count + self.pending_count
        return 0.0 if eligible == 0 else self.deadline_miss_count / eligible


@dataclass(frozen=True, slots=True)
class LocalSloPolicy:
    """Versioned local thresholds; evaluation never grants promotion."""

    version: str
    maximum_terminal_failure_rate: float
    maximum_skipped_rate: float
    maximum_deadline_miss_rate: float
    maximum_p95_wall_time_ms: int
    require_empty_backlog: bool = True

    def __post_init__(self) -> None:
        _require_nonempty("version", self.version)
        _require_unit_interval(
            "maximum_terminal_failure_rate",
            self.maximum_terminal_failure_rate,
        )
        _require_unit_interval("maximum_skipped_rate", self.maximum_skipped_rate)
        _require_unit_interval(
            "maximum_deadline_miss_rate",
            self.maximum_deadline_miss_rate,
        )
        _require_positive_int("maximum_p95_wall_time_ms", self.maximum_p95_wall_time_ms)
        if not isinstance(self.require_empty_backlog, bool):
            raise TypeError("require_empty_backlog must be boolean")

    @property
    def policy_digest(self) -> str:
        return semantic_sha256(
            {
                "version": self.version,
                "maximum_terminal_failure_rate": self.maximum_terminal_failure_rate,
                "maximum_skipped_rate": self.maximum_skipped_rate,
                "maximum_deadline_miss_rate": self.maximum_deadline_miss_rate,
                "maximum_p95_wall_time_ms": self.maximum_p95_wall_time_ms,
                "require_empty_backlog": self.require_empty_backlog,
            }
        )


@dataclass(frozen=True, slots=True)
class LocalSloEvaluation:
    """Non-promotional comparison of one synthetic report to local thresholds."""

    policy_version: str
    policy_digest: str
    profile_digest: str
    within_local_thresholds: bool
    violations: tuple[str, ...]

    @property
    def measurement_status(self) -> str:
        return "NOT_MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class CapacityRegressionPolicy:
    """Versioned bounds for like-for-like local performance comparisons."""

    version: str
    minimum_throughput_ratio: float
    maximum_p95_wall_time_ratio: float
    maximum_failure_rate_increase: float
    maximum_deadline_miss_rate_increase: float
    maximum_backlog_end_increase: int

    def __post_init__(self) -> None:
        _require_nonempty("version", self.version)
        if (
            isinstance(self.minimum_throughput_ratio, bool)
            or not isinstance(self.minimum_throughput_ratio, (int, float))
            or not math.isfinite(self.minimum_throughput_ratio)
            or not 0 < self.minimum_throughput_ratio <= 1
        ):
            raise ValueError("minimum_throughput_ratio must be in (0, 1]")
        if (
            isinstance(self.maximum_p95_wall_time_ratio, bool)
            or not isinstance(self.maximum_p95_wall_time_ratio, (int, float))
            or not math.isfinite(self.maximum_p95_wall_time_ratio)
            or self.maximum_p95_wall_time_ratio < 1
        ):
            raise ValueError("maximum_p95_wall_time_ratio must be finite and at least 1")
        _require_unit_interval(
            "maximum_failure_rate_increase",
            self.maximum_failure_rate_increase,
        )
        _require_unit_interval(
            "maximum_deadline_miss_rate_increase",
            self.maximum_deadline_miss_rate_increase,
        )
        _require_nonnegative_int(
            "maximum_backlog_end_increase",
            self.maximum_backlog_end_increase,
        )

    @property
    def policy_digest(self) -> str:
        return semantic_sha256(
            {
                "version": self.version,
                "minimum_throughput_ratio": self.minimum_throughput_ratio,
                "maximum_p95_wall_time_ratio": self.maximum_p95_wall_time_ratio,
                "maximum_failure_rate_increase": self.maximum_failure_rate_increase,
                "maximum_deadline_miss_rate_increase": (self.maximum_deadline_miss_rate_increase),
                "maximum_backlog_end_increase": self.maximum_backlog_end_increase,
            }
        )


@dataclass(frozen=True, slots=True)
class CapacityRegressionResult:
    """Like-for-like local comparison with explicit non-promotional semantics."""

    policy_version: str
    policy_digest: str
    profile_digest: str
    within_local_thresholds: bool
    regressions: tuple[str, ...]
    throughput_ratio: float | None
    p95_wall_time_ratio: float | None
    failure_rate_increase: float
    deadline_miss_rate_increase: float
    backlog_end_increase: int

    @property
    def measurement_status(self) -> str:
        return "NOT_MEASURED"

    @property
    def production_eligible(self) -> bool:
        return False


def generate_synthetic_load(
    profile: SyntheticLoadProfile,
) -> tuple[SyntheticWorkUnit, ...]:
    """Generate stable IDs and burst-shaped arrivals without random state."""

    if not isinstance(profile, SyntheticLoadProfile):
        raise TypeError("profile must be SyntheticLoadProfile")
    failed = frozenset(profile.failed_ordinals)
    skipped = frozenset(profile.skipped_ordinals)
    camera_video_duration_ns = sum(profile.camera_stream_durations_ns)
    profile_digest = profile.profile_digest
    units: list[SyntheticWorkUnit] = []
    for ordinal in range(profile.unit_count):
        arrival_at_ms = (ordinal // profile.arrival_batch_size) * profile.arrival_interval_ms
        if ordinal in skipped:
            planned_outcome = SyntheticOutcome.SKIPPED
        elif ordinal in failed:
            planned_outcome = SyntheticOutcome.FAILED
        else:
            planned_outcome = SyntheticOutcome.SUCCEEDED
        units.append(
            SyntheticWorkUnit(
                work_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"robata:synthetic-capacity:{profile_digest}:{ordinal}",
                    )
                ),
                ordinal=ordinal,
                arrival_at_ms=arrival_at_ms,
                service_time_ms=profile.service_time_pattern_ms[
                    ordinal % len(profile.service_time_pattern_ms)
                ],
                deadline_at_ms=arrival_at_ms + profile.deadline_budget_ms,
                recording_duration_ns=profile.recording_duration_ns,
                camera_video_duration_ns=camera_video_duration_ns,
                planned_outcome=planned_outcome,
            )
        )
    return tuple(units)


def simulate_capacity(
    profile: SyntheticLoadProfile,
    *,
    worker_count: int,
) -> SyntheticCapacityReport:
    """Run a deterministic discrete-event queue simulation to one explicit cutoff."""

    if not isinstance(profile, SyntheticLoadProfile):
        raise TypeError("profile must be SyntheticLoadProfile")
    _require_positive_int("worker_count", worker_count)
    units = generate_synthetic_load(profile)
    if (
        profile.observation_window_ms is not None
        and profile.observation_window_ms < units[-1].arrival_at_ms
    ):
        raise ValueError("observation_window_ms must include the complete arrival schedule")
    workers = [(0, ordinal) for ordinal in range(worker_count)]
    heapq.heapify(workers)
    scheduled: list[tuple[SyntheticWorkUnit, int | None, int | None, int, int, int]] = []
    busy_ms = 0
    for unit in units:
        if unit.planned_outcome is SyntheticOutcome.SKIPPED:
            scheduled.append((unit, None, None, unit.arrival_at_ms, 0, 0))
            continue
        available_at_ms, worker_ordinal = heapq.heappop(workers)
        started_at_ms = max(unit.arrival_at_ms, available_at_ms)
        completion_at_ms = started_at_ms + unit.service_time_ms
        heapq.heappush(workers, (completion_at_ms, worker_ordinal))
        scheduled.append(
            (
                unit,
                worker_ordinal,
                started_at_ms,
                completion_at_ms,
                started_at_ms - unit.arrival_at_ms,
                unit.service_time_ms,
            )
        )

    natural_cutoff = max(item[3] for item in scheduled)
    cutoff_ms = (
        profile.observation_window_ms
        if profile.observation_window_ms is not None
        else max(1, natural_cutoff)
    )
    observations: list[SyntheticObservation] = []
    for (
        unit,
        scheduled_worker_ordinal,
        scheduled_start_ms,
        completion_at_ms,
        queue_ms,
        service_ms,
    ) in scheduled:
        if unit.planned_outcome is SyntheticOutcome.SKIPPED:
            outcome = SyntheticOutcome.SKIPPED
            observed_terminal_at_ms: int | None = unit.arrival_at_ms
            wall_time_ms: int | None = 0
        elif completion_at_ms > cutoff_ms:
            outcome = SyntheticOutcome.PENDING
            observed_terminal_at_ms = None
            wall_time_ms = None
        else:
            outcome = unit.planned_outcome
            observed_terminal_at_ms = completion_at_ms
            wall_time_ms = completion_at_ms - unit.arrival_at_ms
        cutoff_started_at_ms = (
            scheduled_start_ms
            if scheduled_start_ms is not None and scheduled_start_ms <= cutoff_ms
            else None
        )
        cutoff_worker_ordinal = (
            scheduled_worker_ordinal if cutoff_started_at_ms is not None else None
        )
        if unit.planned_outcome is SyntheticOutcome.SKIPPED:
            queue_wait_at_cutoff_ms = 0
        elif cutoff_started_at_ms is not None:
            queue_wait_at_cutoff_ms = queue_ms
        else:
            queue_wait_at_cutoff_ms = max(0, cutoff_ms - unit.arrival_at_ms)
        deadline_missed = (
            cutoff_ms >= unit.deadline_at_ms and completion_at_ms > unit.deadline_at_ms
        )
        if scheduled_start_ms is not None:
            busy_ms += max(0, min(completion_at_ms, cutoff_ms) - scheduled_start_ms)
        observations.append(
            SyntheticObservation(
                work=unit,
                worker_ordinal=cutoff_worker_ordinal,
                started_at_ms=cutoff_started_at_ms,
                scheduled_completion_at_ms=completion_at_ms,
                observed_terminal_at_ms=observed_terminal_at_ms,
                outcome=outcome,
                queue_wait_ms=queue_wait_at_cutoff_ms,
                service_time_ms=service_ms,
                wall_time_ms=wall_time_ms,
                deadline_missed=deadline_missed,
            )
        )

    counts = {
        outcome: sum(item.outcome is outcome for item in observations)
        for outcome in SyntheticOutcome
    }
    succeeded = tuple(item for item in observations if item.outcome is SyntheticOutcome.SUCCEEDED)
    terminal_work = tuple(
        item
        for item in observations
        if item.outcome in (SyntheticOutcome.SUCCEEDED, SyntheticOutcome.FAILED)
    )
    elapsed_hours = cutoff_ms / _MILLISECONDS_PER_HOUR
    offered_recording_hours = (
        sum(item.work.recording_duration_ns for item in observations) / _NANOSECONDS_PER_HOUR
    )
    offered_camera_video_hours = (
        sum(item.work.camera_video_duration_ns for item in observations) / _NANOSECONDS_PER_HOUR
    )
    completed_recording_hours = (
        sum(item.work.recording_duration_ns for item in succeeded) / _NANOSECONDS_PER_HOUR
    )
    completed_camera_video_hours = (
        sum(item.work.camera_video_duration_ns for item in succeeded) / _NANOSECONDS_PER_HOUR
    )
    service_demand = tuple(
        item for item in observations if item.work.planned_outcome is not SyntheticOutcome.SKIPPED
    )
    mean_service_ms = (
        sum(item.service_time_ms for item in service_demand) / len(service_demand)
        if service_demand
        else None
    )
    queue_wait = _latency_summary(tuple(item.queue_wait_ms for item in terminal_work))
    service_time = _latency_summary(tuple(item.service_time_ms for item in terminal_work))
    wall_time = _latency_summary(
        tuple(item.wall_time_ms for item in terminal_work if item.wall_time_ms is not None)
    )
    backlog_peak = _backlog_peak(observations, cutoff_ms)
    backlog_end = counts[SyntheticOutcome.PENDING]
    bottlenecks: list[BottleneckKind] = []
    offered_rate = len(observations) / elapsed_hours
    nominal_capacity = (
        0.0 if mean_service_ms is None else worker_count * _MILLISECONDS_PER_HOUR / mean_service_ms
    )
    service_arrival_rate = len(service_demand) / elapsed_hours
    if backlog_end > 0 or (
        service_demand and service_arrival_rate > nominal_capacity * (1 + 1e-12)
    ):
        bottlenecks.append(BottleneckKind.SERVICE_CAPACITY)
    if (
        queue_wait is not None
        and service_time is not None
        and queue_wait.p95_ms > service_time.p95_ms
    ):
        bottlenecks.append(BottleneckKind.QUEUE_WAIT)
    if counts[SyntheticOutcome.FAILED] > 0:
        bottlenecks.append(BottleneckKind.RELIABILITY)
    if any(item.deadline_missed for item in observations):
        bottlenecks.append(BottleneckKind.DEADLINE)
    if not bottlenecks:
        bottlenecks.append(BottleneckKind.NONE)

    report = SyntheticCapacityReport(
        profile_version=profile.version,
        profile_digest=profile.profile_digest,
        worker_count=worker_count,
        cutoff_ms=cutoff_ms,
        observations=tuple(observations),
        succeeded_count=counts[SyntheticOutcome.SUCCEEDED],
        failed_count=counts[SyntheticOutcome.FAILED],
        skipped_count=counts[SyntheticOutcome.SKIPPED],
        pending_count=counts[SyntheticOutcome.PENDING],
        deadline_miss_count=sum(item.deadline_missed for item in observations),
        backlog_peak=backlog_peak,
        backlog_end=backlog_end,
        utilization=min(1.0, busy_ms / (worker_count * cutoff_ms)),
        offered_recording_hours=offered_recording_hours,
        offered_camera_video_hours=offered_camera_video_hours,
        completed_recording_hours=completed_recording_hours,
        completed_camera_video_hours=completed_camera_video_hours,
        recording_hours_per_wall_hour=completed_recording_hours / elapsed_hours,
        camera_video_hours_per_wall_hour=completed_camera_video_hours / elapsed_hours,
        offered_units_per_wall_hour=offered_rate,
        nominal_service_capacity_units_per_hour=nominal_capacity,
        queue_wait=queue_wait,
        service_time=service_time,
        wall_time=wall_time,
        bottlenecks=tuple(bottlenecks),
    )
    if report.total_count != profile.unit_count:
        raise AssertionError("synthetic outcome ledger does not reconcile")
    return report


def evaluate_local_slo(
    report: SyntheticCapacityReport,
    policy: LocalSloPolicy,
) -> LocalSloEvaluation:
    """Compare synthetic observations without changing their evidence class."""

    if not isinstance(report, SyntheticCapacityReport):
        raise TypeError("report must be SyntheticCapacityReport")
    if not isinstance(policy, LocalSloPolicy):
        raise TypeError("policy must be LocalSloPolicy")
    violations: list[str] = []
    if report.terminal_failure_rate > policy.maximum_terminal_failure_rate:
        violations.append("TERMINAL_FAILURE_RATE")
    if report.skipped_rate > policy.maximum_skipped_rate:
        violations.append("SKIPPED_RATE")
    if report.deadline_miss_rate > policy.maximum_deadline_miss_rate:
        violations.append("DEADLINE_MISS_RATE")
    if report.wall_time is None or report.wall_time.p95_ms > policy.maximum_p95_wall_time_ms:
        violations.append("P95_WALL_TIME")
    if policy.require_empty_backlog and report.backlog_end != 0:
        violations.append("BACKLOG_NOT_DRAINED")
    return LocalSloEvaluation(
        policy_version=policy.version,
        policy_digest=policy.policy_digest,
        profile_digest=report.profile_digest,
        within_local_thresholds=not violations,
        violations=tuple(violations),
    )


def compare_capacity_reports(
    baseline: SyntheticCapacityReport,
    candidate: SyntheticCapacityReport,
    policy: CapacityRegressionPolicy,
) -> CapacityRegressionResult:
    """Detect local regressions only for the same content-addressed workload."""

    if not isinstance(baseline, SyntheticCapacityReport):
        raise TypeError("baseline must be SyntheticCapacityReport")
    if not isinstance(candidate, SyntheticCapacityReport):
        raise TypeError("candidate must be SyntheticCapacityReport")
    if not isinstance(policy, CapacityRegressionPolicy):
        raise TypeError("policy must be CapacityRegressionPolicy")
    if baseline.profile_digest != candidate.profile_digest:
        raise ValueError("capacity regression comparison requires the same workload profile")

    throughput_ratio = _ratio(
        candidate.recording_hours_per_wall_hour,
        baseline.recording_hours_per_wall_hour,
    )
    p95_wall_time_ratio = _optional_latency_ratio(
        candidate.wall_time,
        baseline.wall_time,
    )
    failure_rate_increase = candidate.terminal_failure_rate - baseline.terminal_failure_rate
    deadline_miss_rate_increase = candidate.deadline_miss_rate - baseline.deadline_miss_rate
    backlog_end_increase = candidate.backlog_end - baseline.backlog_end
    regressions: list[str] = []
    if throughput_ratio is None or throughput_ratio < policy.minimum_throughput_ratio:
        regressions.append("THROUGHPUT")
    if p95_wall_time_ratio is None or p95_wall_time_ratio > policy.maximum_p95_wall_time_ratio:
        regressions.append("P95_WALL_TIME")
    if failure_rate_increase > policy.maximum_failure_rate_increase:
        regressions.append("TERMINAL_FAILURE_RATE")
    if deadline_miss_rate_increase > policy.maximum_deadline_miss_rate_increase:
        regressions.append("DEADLINE_MISS_RATE")
    if backlog_end_increase > policy.maximum_backlog_end_increase:
        regressions.append("BACKLOG_END")
    return CapacityRegressionResult(
        policy_version=policy.version,
        policy_digest=policy.policy_digest,
        profile_digest=baseline.profile_digest,
        within_local_thresholds=not regressions,
        regressions=tuple(regressions),
        throughput_ratio=throughput_ratio,
        p95_wall_time_ratio=p95_wall_time_ratio,
        failure_rate_increase=failure_rate_increase,
        deadline_miss_rate_increase=deadline_miss_rate_increase,
        backlog_end_increase=backlog_end_increase,
    )


def _latency_summary(values: tuple[int, ...]) -> LatencySummary | None:
    if not values:
        return None
    return LatencySummary(
        count=len(values),
        mean_ms=sum(values) / len(values),
        p50_ms=_nearest_rank(values, 0.50),
        p95_ms=_nearest_rank(values, 0.95),
        p99_ms=_nearest_rank(values, 0.99),
    )


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _backlog_peak(
    observations: list[SyntheticObservation],
    cutoff_ms: int,
) -> int:
    events: list[tuple[int, int]] = []
    for item in observations:
        if item.work.planned_outcome is SyntheticOutcome.SKIPPED:
            continue
        if item.work.arrival_at_ms <= cutoff_ms:
            events.append((item.work.arrival_at_ms, 1))
        if item.scheduled_completion_at_ms <= cutoff_ms:
            events.append((item.scheduled_completion_at_ms, -1))
    current = 0
    peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def _ratio(candidate: float, baseline: float) -> float | None:
    if baseline == 0:
        return 1.0 if candidate == 0 else math.inf
    return candidate / baseline


def _optional_latency_ratio(
    candidate: LatencySummary | None,
    baseline: LatencySummary | None,
) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline.p95_ms == 0:
        return 1.0 if candidate.p95_ms == 0 else None
    return candidate.p95_ms / baseline.p95_ms


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_unit_interval(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _validate_ordinals(
    name: str,
    values: tuple[int, ...],
    unit_count: int,
) -> frozenset[int]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    for value in values:
        _require_nonnegative_int(name, value)
        if value >= unit_count:
            raise ValueError(f"{name} contains an ordinal outside the workload")
    return frozenset(values)


__all__ = [
    "BottleneckKind",
    "CapacityRegressionPolicy",
    "CapacityRegressionResult",
    "LatencySummary",
    "LocalSloEvaluation",
    "LocalSloPolicy",
    "SyntheticCapacityReport",
    "SyntheticLoadProfile",
    "SyntheticObservation",
    "SyntheticOutcome",
    "SyntheticWorkUnit",
    "compare_capacity_reports",
    "evaluate_local_slo",
    "generate_synthetic_load",
    "simulate_capacity",
]
