"""Deterministic broker projection for local lease/fence/reconciliation proofs."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.broker import (
    BrokerClaim,
    BrokerDeliverySnapshot,
    BrokerEnvelope,
    BrokerLease,
    BrokerPublishReceipt,
    BrokerStatus,
)
from robata.ports.broker import BrokerError, BrokerErrorCode

Clock = Callable[[], datetime]


@dataclass(slots=True)
class _BrokerRecord:
    envelope: BrokerEnvelope
    status: BrokerStatus = BrokerStatus.PENDING
    attempt: int = 0
    lease: BrokerLease | None = None
    lease_expires_at: datetime | None = None
    lease_duration_seconds: int | None = None
    available_at: datetime | None = None
    reason: str | None = None
    sequence: int = 0


class FakeBroker:
    """In-memory at-least-once broker with deterministic failure/replay behavior."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        max_attempts: int = 3,
        retry_backoff: timedelta = timedelta(0),
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not isinstance(retry_backoff, timedelta) or retry_backoff < timedelta(0):
            raise ValueError("retry_backoff must be non-negative")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._max_attempts = max_attempts
        self._retry_backoff = retry_backoff
        self._records: dict[str, _BrokerRecord] = {}
        self._sequence = 0
        self._failures: dict[str, tuple[BrokerError, bool]] = {}
        self._metrics: dict[str, int] = {}

    def fail_next(
        self,
        operation: str,
        *,
        code: BrokerErrorCode = BrokerErrorCode.RETRYABLE,
        message: str = "injected broker failure",
        after_write: bool = False,
    ) -> None:
        if operation not in {"publish", "claim", "heartbeat", "acknowledge", "reject", "reconcile"}:
            raise ValueError("unsupported broker operation")
        if not isinstance(code, BrokerErrorCode):
            raise TypeError("code must be BrokerErrorCode")
        if not isinstance(message, str) or not message:
            raise ValueError("message must be non-empty")
        if not isinstance(after_write, bool):
            raise TypeError("after_write must be a boolean")
        if after_write and operation != "publish":
            raise ValueError("after_write is supported only for publish")
        self._failures[operation] = (BrokerError(code, message), after_write)

    def publish(self, envelope: BrokerEnvelope) -> BrokerPublishReceipt:
        if not isinstance(envelope, BrokerEnvelope):
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, "envelope must be BrokerEnvelope")
        self._count("publish")
        failure = self._take_failure("publish")
        if failure is not None and not failure[1]:
            raise failure[0]
        existing = self._records.get(envelope.message_id)
        if existing is not None:
            if existing.envelope != envelope:
                raise BrokerError(
                    BrokerErrorCode.DUPLICATE_MESSAGE,
                    "message ID is already bound to different immutable envelope bytes",
                )
            return BrokerPublishReceipt(message_id=envelope.message_id, duplicate=True)
        self._sequence += 1
        self._records[envelope.message_id] = _BrokerRecord(
            envelope=envelope,
            available_at=self._now(),
            sequence=self._sequence,
        )
        if failure is not None:
            # The record is durable, but the caller observes a lost response.
            raise failure[0]
        return BrokerPublishReceipt(message_id=envelope.message_id, duplicate=False)

    def claim(self, worker_id: str, lease_seconds: int) -> BrokerClaim | None:
        worker = self._nonempty(worker_id, "worker_id")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds <= 0
        ):
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, "lease_seconds must be positive")
        self._count("claim")
        self._raise_failure("claim")
        self.reconcile()
        now = self._now()
        candidates = sorted(
            (
                record
                for record in self._records.values()
                if record.status in {BrokerStatus.PENDING, BrokerStatus.RETRY}
                and record.available_at is not None
                and record.available_at <= now
            ),
            key=lambda item: (item.sequence, item.envelope.message_id),
        )
        if not candidates:
            return None
        record = candidates[0]
        record.attempt += 1
        epoch = record.attempt
        lease = BrokerLease(
            lease_id=str(
                uuid5(
                    NAMESPACE_URL, f"robata:fake-broker-lease:{record.envelope.message_id}:{epoch}"
                )
            ),
            worker_id=worker,
            lease_epoch=epoch,
            fencing_token=str(
                uuid5(
                    NAMESPACE_URL,
                    f"robata:fake-broker-fence:{record.envelope.message_id}:{epoch}",
                )
            ),
        )
        record.status = BrokerStatus.CLAIMED
        record.lease = lease
        record.lease_duration_seconds = lease_seconds
        record.lease_expires_at = now + timedelta(seconds=lease_seconds)
        record.reason = None
        return BrokerClaim(envelope=record.envelope, lease=lease)

    def heartbeat(self, lease: BrokerLease, lease_seconds: int | None = None) -> BrokerLease | None:
        self._require_lease(lease)
        if lease_seconds is not None and (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or lease_seconds <= 0
        ):
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, "lease_seconds must be positive")
        self._count("heartbeat")
        self._raise_failure("heartbeat")
        record = self._record_for_lease(lease)
        self._require_live(record, lease)
        duration = lease_seconds
        if duration is None:
            duration = record.lease_duration_seconds
            if duration is None:
                lease_expires_at = record.lease_expires_at
                if lease_expires_at is None:
                    raise BrokerError(BrokerErrorCode.LEASE_EXPIRED, "broker lease has expired")
                remaining = lease_expires_at - self._now()
                duration = max(1, math.ceil(remaining.total_seconds()))
        if duration <= 0:
            raise BrokerError(BrokerErrorCode.LEASE_EXPIRED, "broker lease has expired")
        record.lease_duration_seconds = duration
        record.lease_expires_at = self._now() + timedelta(seconds=duration)
        return lease

    def acknowledge(self, claim: BrokerClaim) -> None:
        self._require_claim(claim)
        self._count("acknowledge")
        self._raise_failure("acknowledge")
        record = self._records.get(claim.envelope.message_id)
        if record is None:
            raise BrokerError(BrokerErrorCode.MESSAGE_NOT_FOUND, "message was not published")
        self._require_claim_envelope(record, claim)
        self._require_live(record, claim.lease)
        record.status = BrokerStatus.ACKED
        record.lease = None
        record.lease_expires_at = None
        record.lease_duration_seconds = None
        record.available_at = None
        record.reason = None

    def reject(self, claim: BrokerClaim, reason: str) -> BrokerDeliverySnapshot:
        self._require_claim(claim)
        if not isinstance(reason, str) or not reason.strip():
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, "reason must be non-empty")
        self._count("reject")
        self._raise_failure("reject")
        record = self._records.get(claim.envelope.message_id)
        if record is None:
            raise BrokerError(BrokerErrorCode.MESSAGE_NOT_FOUND, "message was not published")
        self._require_claim_envelope(record, claim)
        self._require_live(record, claim.lease)
        record.reason = reason.strip()[:1000]
        record.lease = None
        record.lease_expires_at = None
        record.lease_duration_seconds = None
        if record.attempt >= self._max_attempts:
            record.status = BrokerStatus.DEAD_LETTER
            record.available_at = None
        else:
            record.status = BrokerStatus.RETRY
            record.available_at = self._now() + self._retry_backoff
        return self._snapshot(record)

    def inspect(self, message_id: str) -> BrokerDeliverySnapshot:
        message = self._nonempty(message_id, "message_id")
        record = self._records.get(message)
        if record is None:
            raise BrokerError(BrokerErrorCode.MESSAGE_NOT_FOUND, f"message not found: {message}")
        return self._snapshot(record)

    def reconcile(self) -> int:
        self._count("reconcile")
        self._raise_failure("reconcile")
        now = self._now()
        recovered = 0
        for record in self._records.values():
            if (
                record.status is BrokerStatus.CLAIMED
                and record.lease_expires_at is not None
                and record.lease_expires_at <= now
            ):
                record.lease = None
                record.lease_expires_at = None
                record.lease_duration_seconds = None
                record.reason = "broker lease expired before acknowledgement"
                if record.attempt >= self._max_attempts:
                    record.status = BrokerStatus.DEAD_LETTER
                    record.available_at = None
                else:
                    record.status = BrokerStatus.RETRY
                    record.available_at = now + self._retry_backoff
                recovered += 1
        return recovered

    def list_dead_letters(self) -> tuple[BrokerDeliverySnapshot, ...]:
        return tuple(
            self._snapshot(record)
            for record in sorted(self._records.values(), key=lambda item: item.sequence)
            if record.status is BrokerStatus.DEAD_LETTER
        )

    def depth(self) -> int:
        self.reconcile()
        return sum(
            record.status in {BrokerStatus.PENDING, BrokerStatus.CLAIMED, BrokerStatus.RETRY}
            for record in self._records.values()
        )

    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def _record_for_lease(self, lease: BrokerLease) -> _BrokerRecord:
        record = self._records.get(
            next((key for key, value in self._records.items() if value.lease == lease), "")
        )
        if record is None:
            raise BrokerError(BrokerErrorCode.LEASE_NOT_FOUND, "broker lease is unknown")
        return record

    @staticmethod
    def _require_claim_envelope(record: _BrokerRecord, claim: BrokerClaim) -> None:
        if record.envelope != claim.envelope:
            raise BrokerError(
                BrokerErrorCode.STALE_FENCE,
                "broker claim envelope does not match the immutable published message",
            )

    def _require_live(self, record: _BrokerRecord, lease: BrokerLease) -> None:
        if record.status is not BrokerStatus.CLAIMED or record.lease != lease:
            raise BrokerError(BrokerErrorCode.STALE_FENCE, "broker lease fence is stale")
        if record.lease_expires_at is None or record.lease_expires_at <= self._now():
            self.reconcile()
            raise BrokerError(BrokerErrorCode.LEASE_EXPIRED, "broker lease has expired")

    @staticmethod
    def _require_claim(claim: BrokerClaim) -> None:
        if not isinstance(claim, BrokerClaim):
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, "claim must be BrokerClaim")

    @staticmethod
    def _require_lease(lease: BrokerLease) -> None:
        if not isinstance(lease, BrokerLease):
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, "lease must be BrokerLease")

    @staticmethod
    def _nonempty(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise BrokerError(BrokerErrorCode.INVALID_REQUEST, f"{label} must be non-empty")
        return value.strip()

    def _snapshot(self, record: _BrokerRecord) -> BrokerDeliverySnapshot:
        return BrokerDeliverySnapshot(
            message_id=record.envelope.message_id,
            status=record.status,
            attempt=max(1, record.attempt),
            lease=record.lease,
            reason=record.reason,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _count(self, operation: str) -> None:
        self._metrics[operation] = self._metrics.get(operation, 0) + 1

    def _take_failure(self, operation: str) -> tuple[BrokerError, bool] | None:
        return self._failures.pop(operation, None)

    def _raise_failure(self, operation: str) -> None:
        failure = self._take_failure(operation)
        if failure is not None:
            raise failure[0]


InMemoryBroker = FakeBroker
FakeBrokerPort = FakeBroker

__all__ = ["FakeBroker", "FakeBrokerPort", "InMemoryBroker"]
