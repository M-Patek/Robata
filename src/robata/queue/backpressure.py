"""Backpressure control using queue age, arrival/service rates, and backlog slope."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from robata.contracts.common import StrictModel
from robata.queue.models import NonEmptyString, NonNegativeInt
from robata.queue.stage import Stage

NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class PressureClass(StrEnum):
    """Operational pressure classification; never a work-result outcome."""

    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    THROTTLED = "THROTTLED"


class BackpressureConfig(StrictModel):
    """Tunable thresholds for backpressure decisions.

    ``queue_depth_threshold`` triggers admission rejection when the queue
    exceeds this depth.  ``oldest_age_threshold_ms`` rejects when the oldest
    pending item exceeds this age.  ``backlog_slope_threshold`` triggers
    shedding when the backlog growth rate exceeds this value.
    """

    version: NonEmptyString
    queue_depth_threshold: NonNegativeInt
    oldest_age_threshold_ms: NonNegativeInt
    backlog_slope_threshold: NonNegativeFloat


class QueueMetrics(StrictModel):
    """Snapshot of queue health at a point in time."""

    depth: NonNegativeInt
    oldest_age_ms: NonNegativeInt
    arrival_rate: NonNegativeFloat
    service_rate: NonNegativeFloat
    backlog_slope: NonNegativeFloat
    provider_quota: NonNegativeInt
    worker_utilization: NonNegativeFloat


class AdmissionDecision(StrictModel):
    """Result of a single admission check."""

    admitted: bool
    policy_version: NonEmptyString = "unversioned-backpressure"
    pressure_class: PressureClass = PressureClass.NORMAL
    signals: tuple[NonEmptyString, ...] = ()
    shedding_actions: tuple[NonEmptyString, ...] = ()
    reason: NonEmptyString | None = None
    suggested_delay_ms: NonNegativeInt | None = None


class SheddingAction(StrictModel):
    """A concrete backpressure action with priority and description."""

    action: NonEmptyString
    priority: NonNegativeInt
    description: NonEmptyString


class BackpressureController:
    """Backpressure decisions using queue age, arrival/service rates, and backlog slope.

    The controller implements a tiered shedding strategy (Section 17.5):

    1. Stop accepting optional deep-processing work.
    2. Reduce or pause GPT random shadow routing.
    3. Defer embedding and eager clip generation.
    4. Auto-scale primary workers within quota and cost limits.
    5. Throttle release of new window work from the persistent ledger.
    6. Apply emergency sampling/policy versions only when benchmarks pass.
    7. Quarantine pathological candidate expansions.
    """

    def __init__(self, config: BackpressureConfig) -> None:
        self._config = config
        self._shedding_actions: list[SheddingAction] = self._init_shedding_actions()

    def _init_shedding_actions(self) -> list[SheddingAction]:
        """Initialize the ordered list of shedding actions per Section 17.5."""
        return [
            SheddingAction(
                action="STOP_OPTIONAL_DEEP",
                priority=1,
                description="Stop accepting optional depth processing work",
            ),
            SheddingAction(
                action="REDUCE_GPT_SHADOW",
                priority=2,
                description="Reduce or pause GPT random shadow routing",
            ),
            SheddingAction(
                action="DEFER_EMBEDDING",
                priority=3,
                description="Defer embedding and eager clip generation",
            ),
            SheddingAction(
                action="AUTO_SCALE",
                priority=4,
                description="Auto-scale primary workers within quota and cost limits",
            ),
            SheddingAction(
                action="THROTTLE_LEDGER",
                priority=5,
                description="Throttle release of new window work from persistent ledger",
            ),
            SheddingAction(
                action="EMERGENCY_SAMPLING",
                priority=6,
                description="Apply emergency sampling/policy versions only when benchmarks pass",
            ),
            SheddingAction(
                action="QUARANTINE_CANDIDATE",
                priority=7,
                description="Quarantine pathological candidate expansions",
            ),
        ]

    def should_admit(self, stage: Stage, metrics: QueueMetrics) -> AdmissionDecision:
        """Evaluate whether a work item for ``stage`` should be admitted.

        Returns an :class:`AdmissionDecision` with ``admitted`` set to ``True``
        when all thresholds are satisfied, or ``False`` with a reason and
        optional suggested delay otherwise.
        """
        signals: list[str] = []
        delays: list[int] = []
        if metrics.depth > self._config.queue_depth_threshold:
            signals.append("QUEUE_DEPTH")
            delays.append(1_000)
        if metrics.oldest_age_ms > self._config.oldest_age_threshold_ms:
            signals.append("OLDEST_AGE")
            delays.append(500)
        if metrics.backlog_slope > self._config.backlog_slope_threshold:
            signals.append("BACKLOG_SLOPE")
            delays.append(2_000)
        if signals:
            return AdmissionDecision(
                admitted=False,
                policy_version=self._config.version,
                pressure_class=PressureClass.THROTTLED,
                signals=tuple(signals),
                shedding_actions=("THROTTLE_LEDGER",),
                reason=f"{stage.value} admission is throttled by {','.join(signals)}",
                suggested_delay_ms=max(delays),
            )

        elevated = any(
            _at_least_three_quarters(value, threshold)
            for value, threshold in (
                (metrics.depth, self._config.queue_depth_threshold),
                (metrics.oldest_age_ms, self._config.oldest_age_threshold_ms),
                (metrics.backlog_slope, self._config.backlog_slope_threshold),
            )
        )
        if elevated:
            return AdmissionDecision(
                admitted=True,
                policy_version=self._config.version,
                pressure_class=PressureClass.ELEVATED,
                signals=("APPROACHING_LIMIT",),
                shedding_actions=("STOP_OPTIONAL_DEEP",),
                reason=f"{stage.value} admission is approaching a pressure limit",
            )
        return AdmissionDecision(
            admitted=True,
            policy_version=self._config.version,
            pressure_class=PressureClass.NORMAL,
        )

    def get_shedding_order(self) -> Sequence[SheddingAction]:
        """Return the ordered sequence of shedding actions.

        Actions are returned in priority order (lowest number first).
        """
        return tuple(self._shedding_actions)


def _at_least_three_quarters(value: int | float, threshold: int | float) -> bool:
    return threshold > 0 and value * 4 >= threshold * 3


__all__ = [
    "AdmissionDecision",
    "BackpressureConfig",
    "BackpressureController",
    "PressureClass",
    "QueueMetrics",
    "SheddingAction",
]
