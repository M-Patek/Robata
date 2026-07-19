"""SLA, throughput, and GPU-capacity accounting for the production requirements.

These classes only account for measurements and assumptions; they never claim that a local fake
model meets production capacity.  Provider/model integration remains behind a separate boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import isfinite
from typing import Annotated

from pydantic import Field

from robata.contracts.common import StrictModel

Positive = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


class MeasurementStatus(StrEnum):
    MEASURED = "MEASURED"
    NOT_MEASURED = "NOT_MEASURED"
    ASSUMPTION = "ASSUMPTION"


class SLAStage(StrEnum):
    QA = "QA"
    ANNOTATION = "ANNOTATION"


class ThroughputTarget(StrictModel):
    recording_hours_per_day: Positive = 500.0
    qa_deadline_days: Positive = 1.0
    annotation_deadline_days: Positive = 3.0


class StageCapacity(StrictModel):
    stage: SLAStage
    recording_hours: NonNegative
    wall_seconds: Positive
    gpu_hours: NonNegative = 0.0
    measurement_status: MeasurementStatus = MeasurementStatus.MEASURED

    @property
    def recording_hours_per_wall_hour(self) -> float:
        return self.recording_hours / (self.wall_seconds / 3600.0)

    @property
    def projected_recording_hours_per_day(self) -> float:
        return self.recording_hours_per_wall_hour * 24.0


class CapacityReport(StrictModel):
    target: ThroughputTarget
    observations: tuple[StageCapacity, ...] = ()
    total_gpu_capacity_hours_per_day: NonNegative = 48.0
    planned_gpu_hours_per_day: NonNegative = 32.0
    production_eligible: bool = False

    @property
    def headroom_gpu_hours_per_day(self) -> float:
        return self.total_gpu_capacity_hours_per_day - self.planned_gpu_hours_per_day

    @property
    def target_hours_per_day(self) -> float:
        return self.target.recording_hours_per_day

    @property
    def minimum_observed_throughput(self) -> float | None:
        if not self.observations:
            return None
        return min(
            observation.projected_recording_hours_per_day for observation in self.observations
        )

    @property
    def meets_target(self) -> bool:
        return (
            self.minimum_observed_throughput is not None
            and self.minimum_observed_throughput >= self.target.recording_hours_per_day
            and self.production_eligible
        )


@dataclass(frozen=True, slots=True)
class CapacityScenario:
    """One explicit H100/model-size capacity assumption for calibration reports."""

    h100_count: int
    model_size: str
    qa_gpu_hours_per_day: float
    annotation_gpu_hours_per_day: float
    total_gpu_capacity_hours_per_day: float
    planned_gpu_hours_per_day: float

    @property
    def headroom_hours_per_day(self) -> float:
        return self.total_gpu_capacity_hours_per_day - self.planned_gpu_hours_per_day

    @property
    def fits(self) -> bool:
        return self.headroom_hours_per_day >= 0

    def as_dict(self) -> dict[str, object]:
        return {
            "h100_count": self.h100_count,
            "model_size": self.model_size,
            "qa_gpu_hours_per_day": self.qa_gpu_hours_per_day,
            "annotation_gpu_hours_per_day": self.annotation_gpu_hours_per_day,
            "total_gpu_capacity_hours_per_day": self.total_gpu_capacity_hours_per_day,
            "planned_gpu_hours_per_day": self.planned_gpu_hours_per_day,
            "headroom_hours_per_day": self.headroom_hours_per_day,
            "fits": self.fits,
            "measurement_status": MeasurementStatus.ASSUMPTION.value,
            "production_eligible": False,
        }


@dataclass(frozen=True, slots=True)
class CapacityPlanner:
    """Evaluate the documented two-H100/7B assumptions without running a model."""

    h100_count: int = 2
    gpu_hours_per_card_per_day: float = 24.0
    qa_gpu_hours_per_day: float = 2.0
    annotation_gpu_hours_per_day: float = 30.0
    embedding_gpu_hours_per_day: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.h100_count, bool)
            or not isinstance(self.h100_count, int)
            or self.h100_count <= 0
        ):
            raise ValueError("h100_count must be a positive integer")
        for name, value in (
            ("gpu_hours_per_card_per_day", self.gpu_hours_per_card_per_day),
            ("qa_gpu_hours_per_day", self.qa_gpu_hours_per_day),
            ("annotation_gpu_hours_per_day", self.annotation_gpu_hours_per_day),
            ("embedding_gpu_hours_per_day", self.embedding_gpu_hours_per_day),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def total_gpu_capacity_hours_per_day(self) -> float:
        return self.h100_count * self.gpu_hours_per_card_per_day

    @property
    def planned_gpu_hours_per_day(self) -> float:
        return (
            self.qa_gpu_hours_per_day
            + self.annotation_gpu_hours_per_day
            + self.embedding_gpu_hours_per_day
        )

    @property
    def headroom_hours_per_day(self) -> float:
        return self.total_gpu_capacity_hours_per_day - self.planned_gpu_hours_per_day

    @property
    def fits_documented_assumption(self) -> bool:
        return self.headroom_hours_per_day >= 0

    def report(
        self,
        observations: Iterable[StageCapacity] = (),
        *,
        target: ThroughputTarget | None = None,
        production_eligible: bool = False,
    ) -> CapacityReport:
        return CapacityReport(
            target=target or ThroughputTarget(),
            observations=tuple(observations),
            total_gpu_capacity_hours_per_day=self.total_gpu_capacity_hours_per_day,
            planned_gpu_hours_per_day=self.planned_gpu_hours_per_day,
            production_eligible=production_eligible,
        )


def calibrate_capacity_scenarios(
    *,
    h100_counts: Iterable[int] = (1, 2, 4),
    model_annotation_gpu_hours: dict[str, float] | None = None,
    qa_gpu_hours_per_day: float = 2.0,
    gpu_hours_per_card_per_day: float = 24.0,
) -> tuple[CapacityScenario, ...]:
    """Evaluate the documented 7B/32B planning assumptions without claiming production capacity."""

    annotation = model_annotation_gpu_hours or {"7B": 30.0, "32B": 100.0}
    if not annotation:
        raise ValueError("model_annotation_gpu_hours must be non-empty")
    if qa_gpu_hours_per_day < 0 or gpu_hours_per_card_per_day <= 0:
        raise ValueError("GPU assumptions must be non-negative/positive")
    scenarios: list[CapacityScenario] = []
    for count in h100_counts:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("h100_counts must contain positive integers")
        for model_size, annotation_hours in annotation.items():
            if annotation_hours < 0:
                raise ValueError("annotation GPU hours must be non-negative")
            planned = qa_gpu_hours_per_day + float(annotation_hours)
            scenarios.append(
                CapacityScenario(
                    h100_count=count,
                    model_size=model_size,
                    qa_gpu_hours_per_day=float(qa_gpu_hours_per_day),
                    annotation_gpu_hours_per_day=float(annotation_hours),
                    total_gpu_capacity_hours_per_day=count * gpu_hours_per_card_per_day,
                    planned_gpu_hours_per_day=planned,
                )
            )
    return tuple(scenarios)


@dataclass(frozen=True, slots=True)
class SLADeadline:
    recording_id: str
    uploaded_at: datetime
    qa_due_at: datetime
    annotation_due_at: datetime


class SLAPlanner:
    """Compute T+1 QA and T+3 annotation deadlines with timezone-aware timestamps."""

    def __init__(self, *, qa_days: float = 1.0, annotation_days: float = 3.0) -> None:
        if qa_days <= 0 or annotation_days <= 0:
            raise ValueError("SLA durations must be positive")
        self.qa_days = float(qa_days)
        self.annotation_days = float(annotation_days)

    def deadline(self, recording_id: str, uploaded_at: datetime) -> SLADeadline:
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError("recording_id must be non-empty")
        if not isinstance(uploaded_at, datetime):
            raise TypeError("uploaded_at must be datetime")
        if uploaded_at.tzinfo is None:
            uploaded_at = uploaded_at.replace(tzinfo=UTC)
        uploaded_at = uploaded_at.astimezone(UTC)
        return SLADeadline(
            recording_id=recording_id,
            uploaded_at=uploaded_at,
            qa_due_at=uploaded_at + timedelta(days=self.qa_days),
            annotation_due_at=uploaded_at + timedelta(days=self.annotation_days),
        )

    def is_on_time(self, stage: SLAStage, completed_at: datetime, deadline: SLADeadline) -> bool:
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=UTC)
        completed_at = completed_at.astimezone(UTC)
        due = deadline.qa_due_at if stage is SLAStage.QA else deadline.annotation_due_at
        return completed_at <= due


class ThroughputLedger:
    """Collect immutable stage observations and produce a capacity report."""

    def __init__(
        self, *, planner: CapacityPlanner | None = None, target: ThroughputTarget | None = None
    ) -> None:
        self.planner = planner or CapacityPlanner()
        self.target = target or ThroughputTarget()
        self._observations: list[StageCapacity] = []

    def record(
        self,
        stage: SLAStage,
        recording_hours: float,
        wall_seconds: float,
        *,
        gpu_hours: float = 0.0,
        measurement_status: MeasurementStatus = MeasurementStatus.MEASURED,
    ) -> StageCapacity:
        observation = StageCapacity(
            stage=stage,
            recording_hours=recording_hours,
            wall_seconds=wall_seconds,
            gpu_hours=gpu_hours,
            measurement_status=measurement_status,
        )
        self._observations.append(observation)
        return observation

    def report(self, *, production_eligible: bool = False) -> CapacityReport:
        return self.planner.report(
            self._observations, target=self.target, production_eligible=production_eligible
        )

    @property
    def observations(self) -> tuple[StageCapacity, ...]:
        return tuple(self._observations)


__all__ = [
    "CapacityPlanner",
    "CapacityReport",
    "CapacityScenario",
    "MeasurementStatus",
    "SLADeadline",
    "SLAPlanner",
    "SLAStage",
    "StageCapacity",
    "ThroughputLedger",
    "ThroughputTarget",
    "calibrate_capacity_scenarios",
]
