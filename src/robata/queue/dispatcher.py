"""Deterministic local weighted-fair primary dispatcher.

This module deliberately does not select a broker or implement deadline
boosting.  It provides a single-process admission scaffold whose decisions
depend only on configured stage weights and active capacity reservations.
Production scheduling may layer a versioned deadline policy over the same
boundary after that policy has been specified and benchmarked.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Annotated
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, TypeAdapter, ValidationError

from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.queue.models import NonEmptyString, NonNegativeInt
from robata.queue.stage import Stage

PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]

Clock = Callable[[], datetime]
IdFactory = Callable[[], str]

_OPAQUE_UUID_ADAPTER = TypeAdapter(OpaqueUuid)
_FAIRNESS_EPSILON = 1e-12


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _require_now(clock: Clock) -> datetime:
    now = clock()
    if not isinstance(now, datetime):
        raise TypeError("clock must return datetime")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return now.astimezone(UTC)


def _rfc3339(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _positive_weight(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reservation weight must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("reservation weight must be finite and positive")
    return normalized


class DispatcherConfig(StrictModel):
    """Configuration for the stage dispatcher.

    ``stage_weights`` contains relative scheduling shares.  A missing or zero
    stage weight disables admission for that stage.  ``max_concurrent`` limits
    the number of active reservations, while ``fairness_window_ms`` is the
    default reservation lifetime for this local scaffold.
    """

    version: NonEmptyString
    stage_weights: dict[Stage, NonNegativeFloat]
    max_concurrent: PositiveInt
    fairness_window_ms: PositiveInt


class DispatchResult(StrictModel):
    """Outcome of a single admission check."""

    accepted: bool
    reason: NonEmptyString | None = None
    estimated_wait_ms: NonNegativeInt | None = None


class CapacityReservation(StrictModel):
    """A time-bounded resource-cost reservation for one stage."""

    reservation_id: OpaqueUuid
    stage: Stage
    weight: PositiveFloat
    reserved_at: Rfc3339Timestamp
    expires_at: Rfc3339Timestamp


class StageDispatcher:
    """Local weighted-fair admission and reservation coordinator.

    Fairness is evaluated using normalized active load::

        sum(active reservation weights for stage) / configured stage weight

    A stage at the minimum normalized load may reserve the next slot.  A stage
    above that minimum waits for one of its reservations to release or expire.
    This is intentionally deadline-neutral because the current contract has no
    versioned deadline-policy input.

    ``dispatch`` is a side-effect-free admission check.  Callers must follow an
    accepted result with ``reserve_capacity`` in the same local coordinator.
    Both methods apply the same checks; the reservation call is authoritative.
    """

    def __init__(
        self,
        config: DispatcherConfig,
        *,
        clock: Clock | None = None,
        id_factory: IdFactory | None = None,
        reservation_ttl_ms: int | None = None,
    ) -> None:
        if not isinstance(config, DispatcherConfig):
            raise TypeError("config must be DispatcherConfig")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if id_factory is not None and not callable(id_factory):
            raise TypeError("id_factory must be callable")
        if reservation_ttl_ms is None:
            reservation_ttl_ms = config.fairness_window_ms
        if (
            isinstance(reservation_ttl_ms, bool)
            or not isinstance(reservation_ttl_ms, int)
            or reservation_ttl_ms <= 0
        ):
            raise ValueError("reservation_ttl_ms must be a positive integer")

        self._config = config
        # StrictModel freezes attribute assignment but not a nested dict.
        self._stage_weights = dict(config.stage_weights)
        self._clock = clock if clock is not None else _default_clock
        self._id_factory = id_factory
        self._reservation_ttl_ms = reservation_ttl_ms
        self._reservations: dict[str, CapacityReservation] = {}
        self._reservation_expiry: dict[str, datetime] = {}
        self._next_reservation_sequence = 0
        self._last_observed_at: datetime | None = None
        self._lock = RLock()

    @property
    def active_reservation_count(self) -> int:
        """Return active reservation count after expiring stale entries."""

        with self._lock:
            self._expire_due(self._now())
            return len(self._reservations)

    def dispatch(self, stage: Stage, work_item_id: OpaqueUuid) -> DispatchResult:
        """Check whether ``stage`` may reserve the next concurrency slot.

        The work-item ID is validated against the canonical opaque-UUID
        contract, but it does not affect fairness or create hidden state.
        """

        try:
            _OPAQUE_UUID_ADAPTER.validate_python(work_item_id)
        except ValidationError:
            return DispatchResult(
                accepted=False,
                reason="work_item_id is not a canonical opaque UUID",
            )

        with self._lock:
            now = self._now()
            self._expire_due(now)
            return self._admission_result(stage, now)

    def reserve_capacity(self, stage: Stage, weight: float) -> CapacityReservation:
        """Atomically reserve one slot for a positively weighted stage.

        Raises ``ValueError`` when the stage is disabled/unconfigured, the
        resource weight is invalid, capacity is full, fairness rejects the
        stage, or the ID factory collides with an active reservation.
        """

        normalized_weight = _positive_weight(weight)
        with self._lock:
            now = self._now()
            self._expire_due(now)
            admission = self._admission_result(stage, now)
            if not admission.accepted:
                raise ValueError(admission.reason or "capacity reservation rejected")

            reservation_id = self._next_reservation_id(stage)
            expires_at = now + timedelta(milliseconds=self._reservation_ttl_ms)
            reservation = CapacityReservation(
                reservation_id=reservation_id,
                stage=stage,
                weight=normalized_weight,
                reserved_at=_rfc3339(now),
                expires_at=_rfc3339(expires_at),
            )
            if reservation.reservation_id in self._reservations:
                raise ValueError(f"active reservation ID collision: {reservation.reservation_id}")
            self._reservations[reservation.reservation_id] = reservation
            self._reservation_expiry[reservation.reservation_id] = expires_at
            self._next_reservation_sequence += 1
            return reservation

    def release_capacity(self, reservation: CapacityReservation) -> None:
        """Idempotently release an exact active reservation.

        A forged value reusing an active ID fails closed instead of releasing
        another caller's capacity.
        """

        if not isinstance(reservation, CapacityReservation):
            raise TypeError("reservation must be CapacityReservation")
        with self._lock:
            self._expire_due(self._now())
            existing = self._reservations.get(reservation.reservation_id)
            if existing is None:
                return
            if existing != reservation:
                raise ValueError("reservation does not match the active reservation")
            self._reservations.pop(reservation.reservation_id, None)
            self._reservation_expiry.pop(reservation.reservation_id, None)

    def sweep_expired(self) -> int:
        """Remove expired reservations and return the number released."""

        with self._lock:
            return self._expire_due(self._now())

    def list_active_reservations(
        self,
        stage: Stage | None = None,
    ) -> tuple[CapacityReservation, ...]:
        """Return an immutable deterministic snapshot of active reservations."""

        if stage is not None and not isinstance(stage, Stage):
            raise ValueError("stage must be a Stage")
        with self._lock:
            self._expire_due(self._now())
            reservations = (
                reservation
                for reservation in self._reservations.values()
                if stage is None or reservation.stage is stage
            )
            return tuple(
                sorted(
                    reservations,
                    key=lambda reservation: (
                        reservation.reserved_at,
                        reservation.reservation_id,
                    ),
                )
            )

    def _admission_result(self, stage: Stage, now: datetime) -> DispatchResult:
        if not isinstance(stage, Stage):
            return DispatchResult(accepted=False, reason="stage must be a Stage")
        configured_weight = self._stage_weights.get(stage)
        if configured_weight is None:
            return DispatchResult(accepted=False, reason="stage is not configured")
        if configured_weight <= 0:
            return DispatchResult(accepted=False, reason="stage weight is zero")
        if len(self._reservations) >= self._config.max_concurrent:
            return DispatchResult(
                accepted=False,
                reason="dispatcher capacity is full",
                estimated_wait_ms=self._earliest_wait_ms(now),
            )

        eligible_weights = {
            candidate: weight
            for candidate, weight in self._stage_weights.items()
            if weight > 0
        }
        if not eligible_weights:
            return DispatchResult(accepted=False, reason="no positively weighted stages")
        active_weight = {candidate: 0.0 for candidate in eligible_weights}
        for reservation in self._reservations.values():
            if reservation.stage in active_weight:
                active_weight[reservation.stage] += reservation.weight

        normalized_load = {
            candidate: active_weight[candidate] / weight
            for candidate, weight in eligible_weights.items()
        }
        minimum_load = min(normalized_load.values())
        if normalized_load[stage] - minimum_load > _FAIRNESS_EPSILON:
            return DispatchResult(
                accepted=False,
                reason="stage is above its weighted fair share",
                estimated_wait_ms=self._earliest_wait_ms(now, stage=stage),
            )
        return DispatchResult(accepted=True)

    def _next_reservation_id(self, stage: Stage) -> str:
        if self._id_factory is not None:
            return self._id_factory()
        material = (
            f"robata:dispatcher:{self._config.version}:"
            f"{stage.value}:{self._next_reservation_sequence}"
        )
        return str(uuid5(NAMESPACE_URL, material))

    def _now(self) -> datetime:
        now = _require_now(self._clock)
        if self._last_observed_at is not None and now < self._last_observed_at:
            raise ValueError("clock moved backwards")
        self._last_observed_at = now
        return now

    def _earliest_wait_ms(self, now: datetime, *, stage: Stage | None = None) -> int:
        expirations = [
            self._reservation_expiry[reservation_id]
            for reservation_id, reservation in self._reservations.items()
            if stage is None or reservation.stage is stage
        ]
        if not expirations:
            return self._reservation_ttl_ms
        remaining_ms = math.ceil((min(expirations) - now).total_seconds() * 1000)
        return max(1, remaining_ms)

    def _expire_due(self, now: datetime) -> int:
        expired_ids = [
            reservation_id
            for reservation_id, expires_at in self._reservation_expiry.items()
            if now >= expires_at
        ]
        for reservation_id in expired_ids:
            self._reservations.pop(reservation_id, None)
            self._reservation_expiry.pop(reservation_id, None)
        return len(expired_ids)


__all__ = [
    "CapacityReservation",
    "DispatchResult",
    "DispatcherConfig",
    "StageDispatcher",
]
