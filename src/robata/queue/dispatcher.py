"""Weighted-fair primary dispatcher with deadline-aware priority."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.queue.models import NonEmptyString, NonNegativeInt
from robata.queue.stage import Stage


PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class DispatcherConfig(StrictModel):
    """Configuration for the stage dispatcher.

    ``stage_weights`` maps each stage to its relative weight for fair-share
    scheduling.  ``max_concurrent`` limits total in-flight work items.
    ``fairness_window_ms`` defines the time window over which fair-share
    allocation is computed.
    """

    version: NonEmptyString
    stage_weights: dict[Stage, NonNegativeFloat]
    max_concurrent: PositiveInt
    fairness_window_ms: PositiveInt


class DispatchResult(StrictModel):
    """Outcome of a single dispatch attempt."""

    accepted: bool
    reason: NonEmptyString | None = None
    estimated_wait_ms: NonNegativeInt | None = None


class CapacityReservation(StrictModel):
    """A held capacity slot for a specific stage.

    Reservations are time-bounded and must be released explicitly or allowed
    to expire.
    """

    reservation_id: OpaqueUuid
    stage: Stage
    weight: NonNegativeFloat
    reserved_at: NonEmptyString
    expires_at: NonEmptyString


class StageDispatcher:
    """Weighted-fair primary dispatcher with deadline priority.

    The dispatcher maintains per-stage capacity reservations and admits work
    items according to a weighted-fair policy modulated by deadline urgency.
    """

    def __init__(self, config: DispatcherConfig) -> None:
        self._config = config
        self._reservations: dict[str, CapacityReservation] = {}

    def dispatch(self, stage: Stage, work_item_id: OpaqueUuid) -> DispatchResult:
        """Attempt to dispatch a work item for the given stage.

        Returns a :class:`DispatchResult` indicating whether the item was
        accepted and, if not, an estimated wait time.
        """
        # TODO: implement weighted-fair admission logic with deadline boost
        return DispatchResult(accepted=True)

    def reserve_capacity(self, stage: Stage, weight: float) -> CapacityReservation:
        """Reserve capacity for a stage, returning a time-bounded reservation."""
        # TODO: implement capacity reservation with expiration
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        reservation_id = "00000000-0000-0000-0000-000000000000"
        return CapacityReservation(
            reservation_id=reservation_id,
            stage=stage,
            weight=weight,
            reserved_at=now,
            expires_at=now,
        )

    def release_capacity(self, reservation: CapacityReservation) -> None:
        """Release a previously held capacity reservation."""
        self._reservations.pop(reservation.reservation_id, None)


__all__ = [
    "CapacityReservation",
    "DispatcherConfig",
    "DispatchResult",
    "StageDispatcher",
]
