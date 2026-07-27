"""Provider-neutral broker envelope contracts for production projections.

Broker metadata is deliberately separate from scheduler authority.  The envelope
carries the scheduler-issued work identity and fence so a transport cannot turn an
acknowledgement into a terminal state by itself.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]

BROKER_CONTRACT_VERSION: Literal["1.0"] = "1.0"


class BrokerStatus(StrEnum):
    """Delivery state observed at the broker projection boundary."""

    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    ACKED = "ACKED"
    RETRY = "RETRY"
    DEAD_LETTER = "DEAD_LETTER"


class BrokerEnvelope(StrictModel):
    """Immutable exact payload plus the authority fence it was projected from."""

    contract_version: Literal["1.0"] = BROKER_CONTRACT_VERSION
    message_id: NonEmptyString
    topic: NonEmptyString
    key: NonEmptyString
    schema_id: NonEmptyString
    schema_version: NonEmptyString
    payload: bytes
    payload_sha256: Sha256Digest
    work_logical_key: NonEmptyString
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    attempt: PositiveInt

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        if exact_bytes_sha256(self.payload) != self.payload_sha256:
            raise ValueError("payload_sha256 must match exact payload bytes")
        return self

    @property
    def content_sha256(self) -> str:
        return self.payload_sha256

    @property
    def idempotency_key(self) -> str:
        return self.message_id


class BrokerLease(StrictModel):
    """Opaque capability required for broker-local acknowledgement."""

    lease_id: NonEmptyString
    worker_id: NonEmptyString
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString


class BrokerPublishReceipt(StrictModel):
    """Idempotent publish acknowledgement; duplicate means exact replay."""

    message_id: NonEmptyString
    duplicate: bool = False


class BrokerClaim(StrictModel):
    """Envelope and the lease capability returned by a claim."""

    envelope: BrokerEnvelope
    lease: BrokerLease


class BrokerDeliverySnapshot(StrictModel):
    """Small deterministic reconciliation view for one broker message."""

    message_id: NonEmptyString
    status: BrokerStatus
    attempt: PositiveInt
    lease: BrokerLease | None = None
    reason: NonEmptyString | None = None


__all__ = [
    "BROKER_CONTRACT_VERSION",
    "BrokerClaim",
    "BrokerDeliverySnapshot",
    "BrokerEnvelope",
    "BrokerLease",
    "BrokerPublishReceipt",
    "BrokerStatus",
]
