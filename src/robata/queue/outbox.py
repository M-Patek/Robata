"""Durable at-least-once relay contracts for primary-completion outbox facts."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol

from robata.contracts.hashing import exact_bytes_sha256

Clock = Callable[[], datetime]


class OutboxDeliveryError(RuntimeError):
    """A durable outbox delivery invariant or storage operation failed."""


class OutboxFenceError(OutboxDeliveryError):
    """A worker attempted a state transition with a stale delivery fence."""


class OutboxDeliveryStatus(StrEnum):
    """Operational lifecycle for one immutable primary outbox fact."""

    PENDING = "PENDING"
    LEASED = "LEASED"
    RETRY_WAIT = "RETRY_WAIT"
    DELIVERED = "DELIVERED"
    DEAD_LETTER = "DEAD_LETTER"


@dataclass(frozen=True, slots=True)
class OutboxRetryPolicy:
    """Versioned retry parameters bound when an outbox row is first discovered."""

    version: str
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("retry policy version must be a nonempty string")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (
            ("base_delay_seconds", self.base_delay_seconds),
            ("max_delay_seconds", self.max_delay_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be less than base_delay_seconds")

    def delay_after(self, attempt_count: int) -> timedelta:
        """Return bounded exponential delay after a failed publish attempt."""

        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or attempt_count <= 0
        ):
            raise ValueError("attempt_count must be a positive integer")
        exponent = min(attempt_count - 1, 62)
        seconds = min(self.base_delay_seconds * (2**exponent), self.max_delay_seconds)
        return timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class OutboxMessage:
    """Exact primary-outbox payload presented to a delivery sink."""

    outbox_id: str
    completion_run_id: str
    recording_identity: str
    outbox_ordinal: int
    topic: str
    key: str
    payload: bytes
    payload_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("outbox_id", self.outbox_id),
            ("completion_run_id", self.completion_run_id),
            ("recording_identity", self.recording_identity),
            ("topic", self.topic),
            ("key", self.key),
            ("payload_sha256", self.payload_sha256),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if (
            isinstance(self.outbox_ordinal, bool)
            or not isinstance(self.outbox_ordinal, int)
            or self.outbox_ordinal < 0
        ):
            raise ValueError("outbox_ordinal must be a nonnegative integer")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be exact bytes")
        if exact_bytes_sha256(self.payload) != self.payload_sha256:
            raise ValueError("payload_sha256 does not match exact payload bytes")


@dataclass(frozen=True, slots=True)
class OutboxDeliverySnapshot:
    """Recovery view for one delivery state row."""

    outbox_id: str
    status: OutboxDeliveryStatus
    attempt_count: int
    lease_epoch: int
    fencing_token: str | None
    claimed_by: str | None
    lease_expires_at: str | None
    next_attempt_at: str
    retry_policy: OutboxRetryPolicy
    last_error: str | None
    delivered_at: str | None
    dead_lettered_at: str | None


@dataclass(frozen=True, slots=True)
class OutboxDeliveryClaim:
    """A leased message and the fence required to complete its attempt."""

    message: OutboxMessage
    delivery: OutboxDeliverySnapshot

    def __post_init__(self) -> None:
        if self.message.outbox_id != self.delivery.outbox_id:
            raise ValueError("claim message and delivery IDs disagree")
        if (
            self.delivery.status is not OutboxDeliveryStatus.LEASED
            or self.delivery.fencing_token is None
        ):
            raise ValueError("claim requires a leased delivery with a fencing token")


class OutboxDeliveryStore(Protocol):
    """Lease/fence boundary over durable primary-outbox delivery state."""

    def claim(
        self,
        *,
        worker_id: str,
        lease_duration: timedelta,
    ) -> OutboxDeliveryClaim | None: ...

    def acknowledge(self, claim: OutboxDeliveryClaim) -> OutboxDeliverySnapshot: ...

    def record_failure(
        self,
        claim: OutboxDeliveryClaim,
        error: str,
    ) -> OutboxDeliverySnapshot: ...

    def get(self, outbox_id: str) -> OutboxDeliverySnapshot | None: ...

    def list_dead_letters(
        self,
        *,
        limit: int = 100,
    ) -> tuple[OutboxDeliverySnapshot, ...]: ...


class IdempotentOutboxSink(Protocol):
    """Destination that rejects same-ID/different-bytes conflicts."""

    def publish(self, message: OutboxMessage) -> None: ...


class OutboxRelay:
    """Claim one durable message, publish it, then fence its acknowledgement."""

    def __init__(
        self,
        *,
        store: OutboxDeliveryStore,
        sink: IdempotentOutboxSink,
        worker_id: str,
        lease_duration: timedelta,
    ) -> None:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a nonempty string")
        if not isinstance(lease_duration, timedelta) or lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        self._store = store
        self._sink = sink
        self._worker_id = worker_id
        self._lease_duration = lease_duration

    def deliver_once(self) -> OutboxDeliverySnapshot | None:
        """Deliver at most one message; sink failures enter retry or DLQ state."""

        claim = self._store.claim(
            worker_id=self._worker_id,
            lease_duration=self._lease_duration,
        )
        if claim is None:
            return None
        try:
            self._sink.publish(claim.message)
        except Exception as error:
            description = f"{type(error).__name__}: {error}"[:1000]
            return self._store.record_failure(claim, description)
        return self._store.acknowledge(claim)


__all__ = [
    "Clock",
    "IdempotentOutboxSink",
    "OutboxDeliveryClaim",
    "OutboxDeliveryError",
    "OutboxDeliverySnapshot",
    "OutboxDeliveryStatus",
    "OutboxDeliveryStore",
    "OutboxFenceError",
    "OutboxMessage",
    "OutboxRelay",
    "OutboxRetryPolicy",
]
