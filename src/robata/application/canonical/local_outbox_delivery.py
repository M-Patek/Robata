"""Best-effort local delivery after authoritative primary completion."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import NonNegativeInt, model_validator

from robata.adapters.sqlite_outbox import (
    SQLiteIdempotentOutboxSink,
    SQLitePrimaryOutboxDeliveryStore,
)
from robata.contracts.common import StrictModel
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRegistry
from robata.event_pipeline.identity_registry import (
    EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
    EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
    EventIdentityOutboxRecord,
    EventIdentityOutboxWireRecord,
    validate_registered_event_identity_outbox_wire_record,
)
from robata.queue.outbox import (
    IdempotentOutboxSink,
    OutboxDeliveryError,
    OutboxDeliverySnapshot,
    OutboxDeliveryStatus,
    OutboxMessage,
    OutboxRelay,
    OutboxRetryPolicy,
)
from robata.runtime.observability import RuntimeObserver

LOCAL_OUTBOX_DELIVERY_MODEL_VERSION: Final = "canonical-local-outbox-delivery-v1"
LOCAL_OUTBOX_RETRY_POLICY_VERSION: Final = "canonical-local-outbox-retry-v1"
LOCAL_OUTBOX_WORKER_ID: Final = "canonical-local-outbox-relay-v1"
LOCAL_OUTBOX_MAX_DELIVERY_ATTEMPTS: Final = 1024
LOCAL_OUTBOX_LEASE_DURATION: Final = timedelta(seconds=30)


class LocalOutboxDeliveryOutcome(StrEnum):
    """Operator outcome that never changes primary-completion authority."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    DELIVERED = "DELIVERED"
    PENDING = "PENDING"
    DEAD_LETTER = "DEAD_LETTER"
    FAILED = "FAILED"


class LocalOutboxDeliverySummary(StrictModel):
    """Bounded post-commit delivery observation for one completion's outbox."""

    model_version: Literal["canonical-local-outbox-delivery-v1"]
    outcome: LocalOutboxDeliveryOutcome
    outbox_ids: tuple[str, ...]
    relay_attempt_count: NonNegativeInt
    pending_count: NonNegativeInt
    leased_count: NonNegativeInt
    retry_wait_count: NonNegativeInt
    delivered_count: NonNegativeInt
    dead_letter_count: NonNegativeInt
    unknown_count: NonNegativeInt
    budget_exhausted: bool
    last_error: str | None

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if len(set(self.outbox_ids)) != len(self.outbox_ids):
            raise ValueError("outbox delivery summary IDs must be unique")
        observed = (
            self.pending_count
            + self.leased_count
            + self.retry_wait_count
            + self.delivered_count
            + self.dead_letter_count
            + self.unknown_count
        )
        if observed != len(self.outbox_ids):
            raise ValueError("outbox delivery counts must exactly cover the completion outbox")
        if self.outcome is LocalOutboxDeliveryOutcome.NOT_APPLICABLE:
            if self.outbox_ids or observed or self.last_error is not None:
                raise ValueError("NOT_APPLICABLE requires an empty outbox")
        elif self.outcome is LocalOutboxDeliveryOutcome.DELIVERED:
            if self.delivered_count != len(self.outbox_ids) or not self.outbox_ids:
                raise ValueError("DELIVERED requires every outbox row to be delivered")
        elif self.outcome is LocalOutboxDeliveryOutcome.DEAD_LETTER:
            if self.dead_letter_count == 0:
                raise ValueError("DEAD_LETTER requires a dead-lettered row")
        elif self.outcome is LocalOutboxDeliveryOutcome.FAILED:
            if self.last_error is None:
                raise ValueError("FAILED requires an error description")
        elif not self.outbox_ids:
            raise ValueError("PENDING requires at least one outbox row")
        return self


class _UnavailableSink(IdempotentOutboxSink):
    """Turns sink initialization failure into durable per-row retry state."""

    def __init__(self, description: str) -> None:
        self._description = description

    def publish(self, message: OutboxMessage) -> None:
        del message
        raise OutboxDeliveryError(self._description)


def failed_local_outbox_delivery(
    outbox: tuple[EventIdentityOutboxRecord, ...],
    error: BaseException,
) -> LocalOutboxDeliverySummary:
    """Preserve a committed primary result while exposing relay failure."""

    ids = tuple(item.outbox_id for item in outbox)
    description = f"{type(error).__name__}: {error}"[:1000]
    return LocalOutboxDeliverySummary(
        model_version=LOCAL_OUTBOX_DELIVERY_MODEL_VERSION,
        outcome=LocalOutboxDeliveryOutcome.FAILED,
        outbox_ids=ids,
        relay_attempt_count=0,
        pending_count=0,
        leased_count=0,
        retry_wait_count=0,
        delivered_count=0,
        dead_letter_count=0,
        unknown_count=len(ids),
        budget_exhausted=False,
        last_error=description,
    )


def reconcile_local_primary_outbox(
    *,
    primary_database_path: Path,
    sink_database_path: Path,
    outbox: tuple[EventIdentityOutboxRecord, ...],
    registry: SchemaRegistry,
    max_delivery_attempts: int = LOCAL_OUTBOX_MAX_DELIVERY_ATTEMPTS,
    runtime_observer: RuntimeObserver | None = None,
) -> LocalOutboxDeliverySummary:
    """Drain currently eligible rows and reconcile exact local sink bytes."""

    if not isinstance(primary_database_path, Path):
        raise TypeError("primary_database_path must be pathlib.Path")
    if not isinstance(sink_database_path, Path):
        raise TypeError("sink_database_path must be pathlib.Path")
    if not isinstance(outbox, tuple) or any(
        not isinstance(item, EventIdentityOutboxRecord) for item in outbox
    ):
        raise TypeError("outbox must be a tuple of EventIdentityOutboxRecord values")
    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be SchemaRegistry")
    if (
        isinstance(max_delivery_attempts, bool)
        or not isinstance(max_delivery_attempts, int)
        or max_delivery_attempts <= 0
    ):
        raise ValueError("max_delivery_attempts must be a positive integer")
    ids = tuple(item.outbox_id for item in outbox)
    if len(set(ids)) != len(ids):
        raise ValueError("completion outbox IDs must be unique")
    if not outbox:
        return _summary(ids=(), snapshots=(), relay_attempt_count=0)

    try:
        sink: IdempotentOutboxSink
        exact_sink: SQLiteIdempotentOutboxSink | None
        sink_initialization_error: OutboxDeliveryError | None = None
        try:
            exact_sink = SQLiteIdempotentOutboxSink(
                sink_database_path,
                runtime_observer=runtime_observer,
            )
            sink = exact_sink
        except OutboxDeliveryError as error:
            exact_sink = None
            sink_initialization_error = error
            sink = _UnavailableSink(f"local outbox sink is unavailable: {error}")
        delivered_payload_verifier: Callable[[str, bytes], None] | None
        if exact_sink is not None:
            delivered_payload_verifier = _local_delivered_payload_verifier(exact_sink)
        elif sink_initialization_error is not None:
            delivered_payload_verifier = _unavailable_local_delivered_payload_verifier(
                sink_initialization_error
            )
        else:  # pragma: no cover - exact_sink has only the two states above.
            delivered_payload_verifier = None
        return reconcile_primary_outbox_to_sink(
            primary_database_path=primary_database_path,
            sink=sink,
            outbox=outbox,
            registry=registry,
            max_delivery_attempts=max_delivery_attempts,
            runtime_observer=runtime_observer,
            delivered_payload_verifier=delivered_payload_verifier,
        )
    except Exception as error:
        return failed_local_outbox_delivery(outbox, error)


def reconcile_primary_outbox_to_sink(
    *,
    primary_database_path: Path,
    sink: IdempotentOutboxSink,
    outbox: tuple[EventIdentityOutboxRecord, ...],
    registry: SchemaRegistry,
    max_delivery_attempts: int = LOCAL_OUTBOX_MAX_DELIVERY_ATTEMPTS,
    worker_id: str = LOCAL_OUTBOX_WORKER_ID,
    lease_duration: timedelta = LOCAL_OUTBOX_LEASE_DURATION,
    retry_policy: OutboxRetryPolicy | None = None,
    runtime_observer: RuntimeObserver | None = None,
    delivered_payload_verifier: Callable[[str, bytes], None] | None = None,
) -> LocalOutboxDeliverySummary:
    """Reconcile authoritative SQLite outbox rows to an injected idempotent sink.

    The source SQLite completion database remains the authority for leases, retry
    state, fences, and acknowledgement. The sink is only acknowledged after its
    publish call succeeds, so a response lost after a durable destination write
    is retried with the same opaque outbox ID and exact bytes. Transport locators
    and settings are deliberately absent from the canonical payload and identity.

    delivered_payload_verifier is optional because remote sinks do not generally
    expose reads. The local SQLite wrapper supplies it to retain its historical
    post-delivery exact-byte check.
    """

    _validate_reconciliation_request(
        primary_database_path=primary_database_path,
        sink=sink,
        outbox=outbox,
        registry=registry,
        max_delivery_attempts=max_delivery_attempts,
        worker_id=worker_id,
        lease_duration=lease_duration,
        retry_policy=retry_policy,
        delivered_payload_verifier=delivered_payload_verifier,
    )
    ids = tuple(item.outbox_id for item in outbox)
    if not outbox:
        return _summary(ids=(), snapshots=(), relay_attempt_count=0)

    try:
        expected_payloads = _expected_payloads(outbox, registry)
        store = SQLitePrimaryOutboxDeliveryStore(
            primary_database_path,
            retry_policy=_local_retry_policy() if retry_policy is None else retry_policy,
            registry=registry,
            runtime_observer=runtime_observer,
        )
        relay = OutboxRelay(
            store=store,
            sink=sink,
            worker_id=worker_id,
            lease_duration=lease_duration,
        )
        relay_attempt_count = 0
        while relay_attempt_count < max_delivery_attempts:
            delivered = relay.deliver_once()
            if delivered is None:
                break
            relay_attempt_count += 1
        snapshots = tuple(store.get(outbox_id) for outbox_id in ids)
        budget_exhausted = relay_attempt_count == max_delivery_attempts and any(
            snapshot is None
            or snapshot.status
            not in (OutboxDeliveryStatus.DELIVERED, OutboxDeliveryStatus.DEAD_LETTER)
            for snapshot in snapshots
        )
        if delivered_payload_verifier is not None:
            for snapshot in snapshots:
                if snapshot is None or snapshot.status is not OutboxDeliveryStatus.DELIVERED:
                    continue
                delivered_payload_verifier(
                    snapshot.outbox_id,
                    expected_payloads[snapshot.outbox_id],
                )
        return _summary(
            ids=ids,
            snapshots=snapshots,
            relay_attempt_count=relay_attempt_count,
            budget_exhausted=budget_exhausted,
        )
    except Exception as error:
        return failed_local_outbox_delivery(outbox, error)


def _validate_reconciliation_request(
    *,
    primary_database_path: Path,
    sink: IdempotentOutboxSink,
    outbox: tuple[EventIdentityOutboxRecord, ...],
    registry: SchemaRegistry,
    max_delivery_attempts: int,
    worker_id: str,
    lease_duration: timedelta,
    retry_policy: OutboxRetryPolicy | None,
    delivered_payload_verifier: Callable[[str, bytes], None] | None,
) -> None:
    if not isinstance(primary_database_path, Path):
        raise TypeError("primary_database_path must be pathlib.Path")
    if not callable(getattr(sink, "publish", None)):
        raise TypeError("sink must implement IdempotentOutboxSink.publish")
    if not isinstance(outbox, tuple) or any(
        not isinstance(item, EventIdentityOutboxRecord) for item in outbox
    ):
        raise TypeError("outbox must be a tuple of EventIdentityOutboxRecord values")
    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be SchemaRegistry")
    if (
        isinstance(max_delivery_attempts, bool)
        or not isinstance(max_delivery_attempts, int)
        or max_delivery_attempts <= 0
    ):
        raise ValueError("max_delivery_attempts must be a positive integer")
    if not isinstance(worker_id, str) or not worker_id:
        raise ValueError("worker_id must be a nonempty string")
    if not isinstance(lease_duration, timedelta) or lease_duration <= timedelta(0):
        raise ValueError("lease_duration must be positive")
    if retry_policy is not None and not isinstance(retry_policy, OutboxRetryPolicy):
        raise TypeError("retry_policy must be OutboxRetryPolicy or None")
    if delivered_payload_verifier is not None and not callable(delivered_payload_verifier):
        raise TypeError("delivered_payload_verifier must be callable or None")
    ids = tuple(item.outbox_id for item in outbox)
    if len(set(ids)) != len(ids):
        raise ValueError("completion outbox IDs must be unique")


def _local_retry_policy() -> OutboxRetryPolicy:
    """Construct the unchanged policy used by local and default sink reconciliation."""

    return OutboxRetryPolicy(
        version=LOCAL_OUTBOX_RETRY_POLICY_VERSION,
        max_attempts=3,
        base_delay_seconds=1.0,
        max_delay_seconds=60.0,
    )


def _local_delivered_payload_verifier(
    sink: SQLiteIdempotentOutboxSink,
) -> Callable[[str, bytes], None]:
    def verify(outbox_id: str, expected: bytes) -> None:
        if sink.payload(outbox_id) != expected:
            raise OutboxDeliveryError(
                "delivered source acknowledgement lacks matching exact local sink bytes"
            )

    return verify


def _unavailable_local_delivered_payload_verifier(
    initialization_error: OutboxDeliveryError,
) -> Callable[[str, bytes], None]:
    def verify(outbox_id: str, expected: bytes) -> None:
        del outbox_id, expected
        raise OutboxDeliveryError(
            "cannot reconcile delivered rows while the local sink is unavailable: "
            f"{initialization_error}"
        )

    return verify


def _expected_payloads(
    outbox: tuple[EventIdentityOutboxRecord, ...],
    registry: SchemaRegistry,
) -> dict[str, bytes]:
    schema_ref = registry.resolve_version(
        EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_ID,
        EVENT_IDENTITY_OUTBOX_RECORD_SCHEMA_VERSION,
    ).ref
    result: dict[str, bytes] = {}
    for record in outbox:
        wire = EventIdentityOutboxWireRecord.from_record(record, schema_ref=schema_ref)
        checked = validate_registered_event_identity_outbox_wire_record(wire, registry)
        result[record.outbox_id] = canonical_json_bytes(checked)
    return result


def _summary(
    *,
    ids: tuple[str, ...],
    snapshots: tuple[OutboxDeliverySnapshot | None, ...],
    relay_attempt_count: int,
    budget_exhausted: bool = False,
) -> LocalOutboxDeliverySummary:
    counts = {status: 0 for status in OutboxDeliveryStatus}
    unknown_count = 0
    errors: list[str] = []
    for snapshot in snapshots:
        if snapshot is None:
            unknown_count += 1
            continue
        counts[snapshot.status] += 1
        if snapshot.last_error is not None:
            errors.append(snapshot.last_error)
    if not ids:
        outcome = LocalOutboxDeliveryOutcome.NOT_APPLICABLE
    elif counts[OutboxDeliveryStatus.DEAD_LETTER]:
        outcome = LocalOutboxDeliveryOutcome.DEAD_LETTER
    elif counts[OutboxDeliveryStatus.DELIVERED] == len(ids):
        outcome = LocalOutboxDeliveryOutcome.DELIVERED
    else:
        outcome = LocalOutboxDeliveryOutcome.PENDING
    return LocalOutboxDeliverySummary(
        model_version=LOCAL_OUTBOX_DELIVERY_MODEL_VERSION,
        outcome=outcome,
        outbox_ids=ids,
        relay_attempt_count=relay_attempt_count,
        pending_count=counts[OutboxDeliveryStatus.PENDING],
        leased_count=counts[OutboxDeliveryStatus.LEASED],
        retry_wait_count=counts[OutboxDeliveryStatus.RETRY_WAIT],
        delivered_count=counts[OutboxDeliveryStatus.DELIVERED],
        dead_letter_count=counts[OutboxDeliveryStatus.DEAD_LETTER],
        unknown_count=unknown_count,
        budget_exhausted=budget_exhausted,
        last_error=errors[0][:1000] if errors else None,
    )


__all__ = [
    "LOCAL_OUTBOX_DELIVERY_MODEL_VERSION",
    "LocalOutboxDeliveryOutcome",
    "LocalOutboxDeliverySummary",
    "failed_local_outbox_delivery",
    "reconcile_local_primary_outbox",
    "reconcile_primary_outbox_to_sink",
]
