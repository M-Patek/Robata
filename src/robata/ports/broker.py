"""Broker projection port with an explicit fail-closed default."""

from __future__ import annotations

from enum import StrEnum
from typing import NoReturn, Protocol

from robata.contracts.broker import (
    BrokerClaim,
    BrokerDeliverySnapshot,
    BrokerEnvelope,
    BrokerLease,
    BrokerPublishReceipt,
)


class BrokerErrorCode(StrEnum):
    ADAPTER_UNAVAILABLE = "ADAPTER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    DUPLICATE_MESSAGE = "DUPLICATE_MESSAGE"
    MESSAGE_NOT_FOUND = "MESSAGE_NOT_FOUND"
    LEASE_NOT_FOUND = "LEASE_NOT_FOUND"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    STALE_FENCE = "STALE_FENCE"
    RETRYABLE = "RETRYABLE"
    DEAD_LETTER = "DEAD_LETTER"


class BrokerError(RuntimeError):
    """Stable broker projection failure."""

    def __init__(self, code: BrokerErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class BrokerPort(Protocol):
    """Lease/fence/retry boundary over a transport projection."""

    def publish(self, envelope: BrokerEnvelope) -> BrokerPublishReceipt: ...

    def claim(self, worker_id: str, lease_seconds: int) -> BrokerClaim | None: ...

    def heartbeat(
        self, lease: BrokerLease, lease_seconds: int | None = None
    ) -> BrokerLease | None: ...

    def acknowledge(self, claim: BrokerClaim) -> None: ...

    def reject(self, claim: BrokerClaim, reason: str) -> BrokerDeliverySnapshot: ...

    def inspect(self, message_id: str) -> BrokerDeliverySnapshot: ...

    def list_dead_letters(self) -> tuple[BrokerDeliverySnapshot, ...]: ...

    def reconcile(self) -> int: ...


class FailClosedBroker:
    """Default production boundary; transport must be explicitly injected."""

    @staticmethod
    def _unavailable() -> NoReturn:
        raise BrokerError(BrokerErrorCode.ADAPTER_UNAVAILABLE, "broker adapter is not configured")

    def publish(self, envelope: BrokerEnvelope) -> BrokerPublishReceipt:
        del envelope
        self._unavailable()

    def claim(self, worker_id: str, lease_seconds: int) -> BrokerClaim | None:
        del worker_id, lease_seconds
        self._unavailable()

    def heartbeat(self, lease: BrokerLease, lease_seconds: int | None = None) -> BrokerLease | None:
        del lease, lease_seconds
        self._unavailable()

    def acknowledge(self, claim: BrokerClaim) -> None:
        del claim
        self._unavailable()

    def reject(self, claim: BrokerClaim, reason: str) -> BrokerDeliverySnapshot:
        del claim, reason
        self._unavailable()

    def inspect(self, message_id: str) -> BrokerDeliverySnapshot:
        del message_id
        self._unavailable()

    def list_dead_letters(self) -> tuple[BrokerDeliverySnapshot, ...]:
        self._unavailable()

    def reconcile(self) -> int:
        self._unavailable()


Broker = BrokerPort
BrokerAdapter = BrokerPort
MessageBroker = BrokerPort

__all__ = [
    "Broker",
    "BrokerAdapter",
    "BrokerError",
    "BrokerErrorCode",
    "BrokerPort",
    "FailClosedBroker",
    "MessageBroker",
]
