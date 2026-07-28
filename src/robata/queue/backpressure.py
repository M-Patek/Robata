"""Backpressure control using queue age, arrival/service rates, and backlog slope."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from math import floor
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.common import StrictModel
from robata.queue.models import NonEmptyString, NonNegativeInt
from robata.queue.stage import Stage

NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
PositiveUnitInterval = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]


class PressureClass(StrEnum):
    """Operational pressure classification; never a work-result outcome."""

    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    THROTTLED = "THROTTLED"


class BackpressureControllerMode(StrEnum):
    """Runtime-only controller mode."""

    FIXED = "FIXED"
    AIMD = "AIMD"


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
    minimum_rate_observation_interval_ms: PositiveInt = 1_000
    controller_version: NonEmptyString = "fixed-backpressure-controller-v1"
    controller_key: NonEmptyString = "stream-window-admission"
    controller_mode: BackpressureControllerMode = BackpressureControllerMode.FIXED
    minimum_limit: PositiveInt = 1
    maximum_limit: PositiveInt = 256
    additive_increase: PositiveInt = 1
    multiplicative_decrease: Annotated[
        float, Field(strict=True, gt=0, lt=1, allow_inf_nan=False)
    ] = 0.5
    cooldown_ms: NonNegativeInt = 0
    worker_utilization_threshold: PositiveUnitInterval = 1.0

    @model_validator(mode="after")
    def validate_controller_bounds(self) -> Self:
        if self.minimum_limit > self.maximum_limit:
            raise ValueError("minimum_limit must not exceed maximum_limit")
        return self


class QueueMetrics(StrictModel):
    """Versioned snapshot of measured or explicitly unavailable queue health."""

    observation_version: Literal["queue-metrics-v2"] = "queue-metrics-v2"
    depth: NonNegativeInt
    oldest_age_ms: NonNegativeInt
    arrival_rate: NonNegativeFloat | None = None
    service_rate: NonNegativeFloat | None = None
    backlog_slope: FiniteFloat | None = None
    provider_quota: NonNegativeInt | None = None
    worker_utilization: UnitInterval | None = None

    @property
    def has_rate_observation(self) -> bool:
        """Whether arrival, service, and signed slope were measured together."""

        return (
            self.arrival_rate is not None
            and self.service_rate is not None
            and self.backlog_slope is not None
        )


class BackpressureRuntimeSignals(StrictModel):
    """Transient provider and executor observations outside durable queue facts."""

    observation_version: Literal["backpressure-runtime-signals-v1"] = (
        "backpressure-runtime-signals-v1"
    )
    provider_quota: NonNegativeInt | None = None
    worker_utilization: UnitInterval | None = None


class AdmissionDecision(StrictModel):
    """Result of a single admission check."""

    admitted: bool
    policy_version: NonEmptyString = "unversioned-backpressure"
    pressure_class: PressureClass = PressureClass.NORMAL
    signals: tuple[NonEmptyString, ...] = ()
    shedding_actions: tuple[NonEmptyString, ...] = ()
    reason: NonEmptyString | None = None
    suggested_delay_ms: NonNegativeInt | None = None
    controller_limit: PositiveInt | None = None
    controller_mode: BackpressureControllerMode = BackpressureControllerMode.FIXED


class BackpressureControllerState(StrictModel):
    """Versioned, restartable state for one timing-only controller."""

    schema_version: Literal["1.0"] = "1.0"
    controller_key: NonEmptyString
    policy_version: NonEmptyString
    controller_version: NonEmptyString
    current_limit: PositiveInt
    decision_count: NonNegativeInt = 0
    last_observed_at_ms: NonNegativeInt | None = None
    last_metrics: QueueMetrics | None = None
    last_arrival_count: NonNegativeInt | None = None
    last_service_count: NonNegativeInt | None = None
    last_backlog_depth: NonNegativeInt | None = None
    last_decision: AdmissionDecision | None = None

    @model_validator(mode="after")
    def validate_policy_binding(self) -> Self:
        if (
            self.last_decision is not None
            and self.last_decision.policy_version != self.policy_version
        ):
            raise ValueError("controller state decision does not match policy_version")
        return self


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

    @property
    def config(self) -> BackpressureConfig:
        """Return the immutable runtime controller configuration."""

        return self._config

    def initial_state(self, controller_key: str) -> BackpressureControllerState:
        """Create the fixed starting point for one persisted controller."""

        if not isinstance(controller_key, str) or not controller_key:
            raise ValueError("controller_key must be non-empty")
        requested = (
            self._config.minimum_limit
            if self._config.queue_depth_threshold == 0
            else self._config.queue_depth_threshold
        )
        return BackpressureControllerState(
            controller_key=controller_key,
            policy_version=self._config.version,
            controller_version=self._config.controller_version,
            current_limit=max(
                self._config.minimum_limit,
                min(self._config.maximum_limit, requested),
            ),
        )

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
        if not isinstance(stage, Stage):
            raise TypeError("stage must be Stage")
        if not isinstance(metrics, QueueMetrics):
            raise TypeError("metrics must be QueueMetrics")
        signals: list[str] = []
        delays: list[int] = []
        if metrics.depth > self._config.queue_depth_threshold:
            signals.append("QUEUE_DEPTH")
            delays.append(1_000)
        if metrics.oldest_age_ms > self._config.oldest_age_threshold_ms:
            signals.append("OLDEST_AGE")
            delays.append(500)
        if (
            metrics.worker_utilization is not None
            and metrics.worker_utilization >= self._config.worker_utilization_threshold
        ):
            signals.append("WORKER_UTILIZATION")
            delays.append(500)
        if (
            metrics.backlog_slope is not None
            and metrics.backlog_slope > self._config.backlog_slope_threshold
        ):
            signals.append("BACKLOG_SLOPE")
            delays.append(2_000)
        if metrics.provider_quota == 0:
            signals.append("PROVIDER_QUOTA")
            delays.append(1_000)
        if (
            metrics.arrival_rate is not None
            and metrics.service_rate is not None
            and metrics.backlog_slope is not None
            and metrics.arrival_rate > metrics.service_rate
            and metrics.backlog_slope > 0
        ):
            signals.append("ARRIVAL_EXCEEDS_SERVICE")
            delays.append(1_500)
        if signals:
            return AdmissionDecision(
                admitted=False,
                policy_version=self._config.version,
                pressure_class=PressureClass.THROTTLED,
                signals=tuple(signals),
                shedding_actions=("THROTTLE_LEDGER",),
                reason=f"{stage.value} admission is throttled by {','.join(signals)}",
                suggested_delay_ms=max(delays),
                controller_mode=self._config.controller_mode,
            )

        elevated_values: tuple[tuple[int | float, int | float], ...] = (
            (metrics.depth, self._config.queue_depth_threshold),
            (metrics.oldest_age_ms, self._config.oldest_age_threshold_ms),
        )
        if metrics.backlog_slope is not None:
            elevated_values += ((metrics.backlog_slope, self._config.backlog_slope_threshold),)
        if any(_at_least_three_quarters(value, threshold) for value, threshold in elevated_values):
            return AdmissionDecision(
                admitted=True,
                policy_version=self._config.version,
                pressure_class=PressureClass.ELEVATED,
                signals=("APPROACHING_LIMIT",),
                shedding_actions=("STOP_OPTIONAL_DEEP",),
                reason=f"{stage.value} admission is approaching a pressure limit",
                controller_mode=self._config.controller_mode,
            )
        return AdmissionDecision(
            admitted=True,
            policy_version=self._config.version,
            pressure_class=PressureClass.NORMAL,
            controller_mode=self._config.controller_mode,
        )

    def evaluate(
        self,
        stage: Stage,
        metrics: QueueMetrics,
        state: BackpressureControllerState,
        *,
        observed_at_ms: int,
    ) -> tuple[AdmissionDecision, BackpressureControllerState]:
        """Return a deterministic decision and persisted successor state."""

        if not isinstance(state, BackpressureControllerState):
            raise TypeError("state must be BackpressureControllerState")
        if state.policy_version != self._config.version:
            raise ValueError("controller state policy_version does not match configuration")
        if state.controller_version != self._config.controller_version:
            raise ValueError("controller state version does not match configuration")
        if not self._config.minimum_limit <= state.current_limit <= self._config.maximum_limit:
            raise ValueError("controller state current_limit is outside configuration bounds")
        if isinstance(observed_at_ms, bool) or not isinstance(observed_at_ms, int):
            raise TypeError("observed_at_ms must be an integer")
        if observed_at_ms < 0:
            raise ValueError("observed_at_ms must be nonnegative")
        if state.last_observed_at_ms is not None and observed_at_ms < state.last_observed_at_ms:
            raise ValueError("observed_at_ms must not precede the persisted observation")

        fixed = self.should_admit(stage, metrics)
        cooldown_elapsed = (
            state.last_observed_at_ms is None
            or observed_at_ms >= state.last_observed_at_ms + self._config.cooldown_ms
        )
        next_limit = state.current_limit
        if self._config.controller_mode is BackpressureControllerMode.AIMD:
            if not fixed.admitted:
                next_limit = max(
                    self._config.minimum_limit,
                    floor(state.current_limit * self._config.multiplicative_decrease),
                )
            elif cooldown_elapsed:
                next_limit = min(
                    self._config.maximum_limit,
                    state.current_limit + self._config.additive_increase,
                )

        decision = fixed.model_copy(
            update={
                "controller_limit": next_limit,
                "controller_mode": self._config.controller_mode,
            }
        )
        if decision.admitted and metrics.depth > next_limit:
            decision = AdmissionDecision(
                admitted=False,
                policy_version=self._config.version,
                pressure_class=PressureClass.THROTTLED,
                signals=("CONTROLLER_LIMIT",),
                shedding_actions=("THROTTLE_LEDGER",),
                reason=f"{stage.value} admission is throttled by CONTROLLER_LIMIT",
                suggested_delay_ms=1_000,
                controller_limit=next_limit,
                controller_mode=self._config.controller_mode,
            )
        next_state = BackpressureControllerState(
            controller_key=state.controller_key,
            policy_version=state.policy_version,
            controller_version=state.controller_version,
            current_limit=next_limit,
            decision_count=state.decision_count + 1,
            last_observed_at_ms=observed_at_ms,
            last_metrics=metrics,
            last_arrival_count=state.last_arrival_count,
            last_service_count=state.last_service_count,
            last_backlog_depth=state.last_backlog_depth,
            last_decision=decision,
        )
        return decision, next_state

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
    "BackpressureControllerMode",
    "BackpressureControllerState",
    "BackpressureRuntimeSignals",
    "PressureClass",
    "QueueMetrics",
    "SheddingAction",
]
