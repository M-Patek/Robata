from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from robata.adapters import (
    SQLiteIdempotentOutboxSink,
    SQLitePrimaryCompletionRepository,
    SQLitePrimaryOutboxDeliveryStore,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.queue import (
    OutboxDeliveryError,
    OutboxDeliveryStatus,
    OutboxFenceError,
    OutboxMessage,
    OutboxRelay,
    OutboxRetryPolicy,
)
from robata.runtime.observability import RuntimeProfileRecorder
from tests.integration.test_sqlite_primary_completion import _run_case


@dataclass
class _MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: float) -> None:
        self.value += timedelta(**kwargs)


class _FailingSink:
    def __init__(self, message: str = "broker unavailable") -> None:
        self.message = message
        self.attempts = 0

    def publish(self, message: OutboxMessage) -> None:
        del message
        self.attempts += 1
        raise RuntimeError(self.message)


def _policy(
    *,
    version: str = "local-outbox-retry-v1",
    max_attempts: int = 3,
    base_delay_seconds: float = 5.0,
) -> OutboxRetryPolicy:
    return OutboxRetryPolicy(
        version=version,
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=60.0,
    )


def _committed_outbox(tmp_path: Path, *, run_value: int) -> tuple[Path, str, bytes]:
    _, repository, command = _run_case(tmp_path, run_value=run_value)
    committed = repository.commit(command).committed
    assert len(committed.outbox) == 1
    record = committed.outbox[0]
    connection = sqlite3.connect(repository.path)
    try:
        payload = bytes(
            connection.execute(
                "SELECT payload_json FROM primary_outbox WHERE outbox_id = ?",
                (record.outbox_id,),
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return repository.path, record.outbox_id, payload


def _counter_total(recorder: RuntimeProfileRecorder, name: str) -> int:
    return sum(counter.value for counter in recorder.snapshot().counters if counter.name == name)


def test_runtime_observation_counts_exact_outbox_transaction_outcomes(
    tmp_path: Path,
) -> None:
    database_path, _, _ = _committed_outbox(tmp_path, run_value=90_010)
    clock = _MutableClock(datetime(2026, 7, 21, 17, tzinfo=UTC))
    delivery_recorder = RuntimeProfileRecorder()
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
        clock=clock,
        runtime_observer=delivery_recorder,
    )

    abandoned = store.claim(worker_id="relay-a", lease_duration=timedelta(seconds=10))
    assert abandoned is not None
    clock.advance(seconds=11)
    recovered = store.claim(worker_id="relay-b", lease_duration=timedelta(seconds=10))
    assert recovered is not None
    with pytest.raises(OutboxFenceError, match="stale"):
        store.acknowledge(abandoned)

    delivery_snapshot = delivery_recorder.snapshot()
    assert _counter_total(delivery_recorder, "sqlite.outbox_delivery.transactions") == 3
    assert _counter_total(delivery_recorder, "sqlite.outbox_delivery.commits") == 2
    assert _counter_total(delivery_recorder, "sqlite.outbox_delivery.rollbacks") == 1
    assert (
        sum(span.name == "sqlite.outbox_delivery.transaction" for span in delivery_snapshot.spans)
        == 3
    )

    sink_recorder = RuntimeProfileRecorder()
    sink = SQLiteIdempotentOutboxSink(
        tmp_path / "observed-sink.sqlite3",
        clock=clock,
        runtime_observer=sink_recorder,
    )
    sink.publish(recovered.message)
    conflicting_payload = b"{}"
    conflicting = replace(
        recovered.message,
        payload=conflicting_payload,
        payload_sha256=exact_bytes_sha256(conflicting_payload),
    )
    with pytest.raises(OutboxDeliveryError, match="different bytes"):
        sink.publish(conflicting)

    sink_snapshot = sink_recorder.snapshot()
    assert _counter_total(sink_recorder, "sqlite.outbox_sink.transactions") == 3
    assert _counter_total(sink_recorder, "sqlite.outbox_sink.commits") == 2
    assert _counter_total(sink_recorder, "sqlite.outbox_sink.rollbacks") == 1
    assert sum(span.name == "sqlite.outbox_sink.transaction" for span in sink_snapshot.spans) == 3


def test_relay_refuses_to_create_a_missing_primary_authority(tmp_path: Path) -> None:
    missing = tmp_path / "missing-primary.sqlite3"

    with pytest.raises(OutboxDeliveryError, match="must already exist"):
        SQLitePrimaryOutboxDeliveryStore(
            missing,
            retry_policy=_policy(),
        )

    assert not missing.exists()


def test_relay_delivers_once_and_reopen_does_not_duplicate(tmp_path: Path) -> None:
    database_path, outbox_id, payload = _committed_outbox(tmp_path, run_value=90_001)
    clock = _MutableClock(datetime(2026, 7, 21, 12, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
        clock=clock,
    )
    sink_path = tmp_path / "delivered.sqlite3"
    sink = SQLiteIdempotentOutboxSink(sink_path, clock=clock)
    relay = OutboxRelay(
        store=store,
        sink=sink,
        worker_id="relay-a",
        lease_duration=timedelta(seconds=30),
    )

    result = relay.deliver_once()

    assert result is not None
    assert result.status is OutboxDeliveryStatus.DELIVERED
    assert result.attempt_count == 1
    assert result.delivered_at is not None
    assert sink.count() == 1
    assert sink.payload(outbox_id) == payload

    reopened_store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
        clock=clock,
    )
    reopened_sink = SQLiteIdempotentOutboxSink(sink_path, clock=clock)
    reopened_relay = OutboxRelay(
        store=reopened_store,
        sink=reopened_sink,
        worker_id="relay-b",
        lease_duration=timedelta(seconds=30),
    )
    assert reopened_relay.deliver_once() is None
    assert reopened_sink.count() == 1

    connection = sqlite3.connect(database_path)
    try:
        delivered_at = connection.execute(
            "SELECT delivered_at FROM primary_outbox WHERE outbox_id = ?",
            (outbox_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert delivered_at == result.delivered_at


def test_transient_failure_waits_for_bound_retry_then_succeeds(tmp_path: Path) -> None:
    database_path, outbox_id, _ = _committed_outbox(tmp_path, run_value=90_002)
    clock = _MutableClock(datetime(2026, 7, 21, 13, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(base_delay_seconds=5.0),
        clock=clock,
    )
    failing = _FailingSink()
    first_relay = OutboxRelay(
        store=store,
        sink=failing,
        worker_id="relay-a",
        lease_duration=timedelta(seconds=30),
    )

    first = first_relay.deliver_once()

    assert first is not None
    assert first.status is OutboxDeliveryStatus.RETRY_WAIT
    assert first.attempt_count == 1
    assert first.last_error == "RuntimeError: broker unavailable"
    assert first_relay.deliver_once() is None

    reopened_store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(
            version="changed-policy-must-not-rebind-v2",
            base_delay_seconds=60.0,
        ),
        clock=clock,
    )
    assert reopened_store.get(outbox_id) == first

    clock.advance(seconds=5)
    sink = SQLiteIdempotentOutboxSink(tmp_path / "delivered.sqlite3", clock=clock)
    second_relay = OutboxRelay(
        store=reopened_store,
        sink=sink,
        worker_id="relay-b",
        lease_duration=timedelta(seconds=30),
    )
    second = second_relay.deliver_once()

    assert second is not None
    assert second.status is OutboxDeliveryStatus.DELIVERED
    assert second.attempt_count == 2
    assert sink.count() == 1
    assert reopened_store.get(outbox_id) == second


def test_poison_message_reaches_durable_dead_letter_state(tmp_path: Path) -> None:
    database_path, outbox_id, _ = _committed_outbox(tmp_path, run_value=90_003)
    clock = _MutableClock(datetime(2026, 7, 21, 14, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(max_attempts=2, base_delay_seconds=0.0),
        clock=clock,
    )
    failing = _FailingSink("poison")
    relay = OutboxRelay(
        store=store,
        sink=failing,
        worker_id="relay-a",
        lease_duration=timedelta(seconds=30),
    )

    first = relay.deliver_once()
    second = relay.deliver_once()

    assert first is not None
    assert first.status is OutboxDeliveryStatus.RETRY_WAIT
    assert second is not None
    assert second.status is OutboxDeliveryStatus.DEAD_LETTER
    assert second.attempt_count == 2
    assert second.dead_lettered_at is not None
    assert relay.deliver_once() is None
    assert store.list_dead_letters() == (second,)
    assert store.get(outbox_id) == second


def test_crash_after_publish_replays_idempotently_under_new_fence(tmp_path: Path) -> None:
    database_path, _, _ = _committed_outbox(tmp_path, run_value=90_004)
    clock = _MutableClock(datetime(2026, 7, 21, 15, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
        clock=clock,
    )
    sink = SQLiteIdempotentOutboxSink(tmp_path / "delivered.sqlite3", clock=clock)

    abandoned = store.claim(
        worker_id="relay-crashed",
        lease_duration=timedelta(seconds=10),
    )
    assert abandoned is not None
    sink.publish(abandoned.message)
    assert sink.count() == 1
    assert store.claim(worker_id="relay-too-early", lease_duration=timedelta(seconds=10)) is None

    clock.advance(seconds=11)
    recovered = store.claim(
        worker_id="relay-recovered",
        lease_duration=timedelta(seconds=10),
    )
    assert recovered is not None
    assert recovered.delivery.lease_epoch == abandoned.delivery.lease_epoch + 1
    assert recovered.delivery.fencing_token != abandoned.delivery.fencing_token
    sink.publish(recovered.message)
    delivered = store.acknowledge(recovered)

    assert delivered.status is OutboxDeliveryStatus.DELIVERED
    assert delivered.attempt_count == 2
    assert sink.count() == 1
    with pytest.raises(OutboxFenceError, match="stale"):
        store.acknowledge(abandoned)


def test_sink_rejects_same_id_with_different_exact_bytes(tmp_path: Path) -> None:
    database_path, _, _ = _committed_outbox(tmp_path, run_value=90_005)
    clock = _MutableClock(datetime(2026, 7, 21, 16, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
        clock=clock,
    )
    claim = store.claim(worker_id="relay-a", lease_duration=timedelta(seconds=30))
    assert claim is not None
    sink = SQLiteIdempotentOutboxSink(tmp_path / "delivered.sqlite3", clock=clock)
    sink.publish(claim.message)
    conflicting_payload = b"{}"
    conflicting = replace(
        claim.message,
        payload=conflicting_payload,
        payload_sha256=exact_bytes_sha256(conflicting_payload),
    )

    with pytest.raises(OutboxDeliveryError, match="different bytes"):
        sink.publish(conflicting)


def test_relay_rejects_outbox_with_forged_exact_schema_pin(tmp_path: Path) -> None:
    database_path, outbox_id, payload = _committed_outbox(tmp_path, run_value=90_007)
    document = json.loads(payload)
    document["schema_ref"]["sha256"] = "0" * 64
    forged = canonical_json_bytes(document)

    connection = sqlite3.connect(database_path)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'primary_outbox_immutable_fields'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER primary_outbox_immutable_fields")
        connection.execute(
            """
            UPDATE primary_outbox
            SET payload_json = ?, payload_json_sha256 = ?
            WHERE outbox_id = ?
            """,
            (sqlite3.Binary(forged), exact_bytes_sha256(forged), outbox_id),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
    )
    with pytest.raises(OutboxDeliveryError, match="not a valid record"):
        store.claim(worker_id="relay-a", lease_duration=timedelta(seconds=30))


def test_relay_rejects_noncanonical_outbox_after_digest_recompute(tmp_path: Path) -> None:
    database_path, outbox_id, payload = _committed_outbox(tmp_path, run_value=90_008)
    noncanonical = payload + b" "

    connection = sqlite3.connect(database_path)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name = 'primary_outbox_immutable_fields'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER primary_outbox_immutable_fields")
        connection.execute(
            """
            UPDATE primary_outbox
            SET payload_json = ?, payload_json_sha256 = ?
            WHERE outbox_id = ?
            """,
            (
                sqlite3.Binary(noncanonical),
                exact_bytes_sha256(noncanonical),
                outbox_id,
            ),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()

    store = SQLitePrimaryOutboxDeliveryStore(database_path, retry_policy=_policy())
    with pytest.raises(OutboxDeliveryError, match="canonical JSON"):
        store.claim(worker_id="relay-a", lease_duration=timedelta(seconds=30))


def test_sink_reopen_rejects_dropped_append_only_trigger(tmp_path: Path) -> None:
    sink_path = tmp_path / "drifted-sink.sqlite3"
    SQLiteIdempotentOutboxSink(sink_path)
    connection = sqlite3.connect(sink_path)
    try:
        connection.execute("DROP TRIGGER delivered_outbox_messages_no_update")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(OutboxDeliveryError, match="DDL"):
        SQLiteIdempotentOutboxSink(sink_path)


def test_sink_maps_post_construction_database_corruption(tmp_path: Path) -> None:
    sink = SQLiteIdempotentOutboxSink(tmp_path / "corrupted-sink.sqlite3")
    sink.path.write_bytes(b"not-a-sqlite-database")

    with pytest.raises(OutboxDeliveryError, match="open or verify"):
        sink.count()


def test_sink_maps_existing_database_corruption_during_construction(tmp_path: Path) -> None:
    sink_path = tmp_path / "already-corrupted-sink.sqlite3"
    sink_path.write_bytes(b"not-a-sqlite-database")

    with pytest.raises(OutboxDeliveryError, match="cannot initialize"):
        SQLiteIdempotentOutboxSink(sink_path)


def test_delivery_store_maps_post_construction_database_corruption(tmp_path: Path) -> None:
    database_path, outbox_id, _payload = _committed_outbox(tmp_path, run_value=90_009)
    store = SQLitePrimaryOutboxDeliveryStore(database_path, retry_policy=_policy())
    database_path.write_bytes(b"not-a-sqlite-database")

    with pytest.raises(OutboxDeliveryError, match="open or verify"):
        store.get(outbox_id)


def test_primary_schema_v1_is_migrated_without_losing_outbox(tmp_path: Path) -> None:
    database_path, outbox_id, payload = _committed_outbox(tmp_path, run_value=90_006)
    legacy_document = json.loads(payload)
    assert legacy_document.pop("schema_ref")
    legacy_payload = canonical_json_bytes(legacy_document)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP TRIGGER primary_outbox_delivery_is_monotonic")
        connection.execute("DROP TRIGGER primary_outbox_immutable_fields")
        connection.execute("DROP TABLE primary_outbox_deliveries")
        connection.execute(
            """
            UPDATE primary_outbox
            SET payload_json = ?, payload_json_sha256 = ?
            WHERE outbox_id = ?
            """,
            (
                sqlite3.Binary(legacy_payload),
                exact_bytes_sha256(legacy_payload),
                outbox_id,
            ),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    reopened = SQLitePrimaryCompletionRepository(database_path)

    connection = sqlite3.connect(reopened.path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        table = connection.execute(
            """
            SELECT name FROM sqlite_schema
            WHERE type = 'table' AND name = 'primary_outbox_deliveries'
            """
        ).fetchone()
        stored_id, stored_payload, stored_digest = connection.execute(
            """
            SELECT outbox_id, payload_json, payload_json_sha256
            FROM primary_outbox WHERE outbox_id = ?
            """,
            (outbox_id,),
        ).fetchone()
    finally:
        connection.close()
    stored_bytes = bytes(stored_payload)
    stored_document = json.loads(stored_bytes)
    assert version == 2
    assert table == ("primary_outbox_deliveries",)
    assert stored_id == outbox_id
    assert set(stored_document["schema_ref"]) == {
        "schema_id",
        "version",
        "artifact_id",
        "sha256",
    }
    assert stored_digest == exact_bytes_sha256(stored_bytes)
    assert reopened.list_outbox(stored_document["recording_identity"])[0].outbox_id == outbox_id


def test_expired_final_attempt_enters_durable_dead_letter_without_reclaim(
    tmp_path: Path,
) -> None:
    database_path, outbox_id, _ = _committed_outbox(tmp_path, run_value=90_011)
    clock = _MutableClock(datetime(2026, 7, 21, 18, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(max_attempts=1),
        clock=clock,
    )

    claim = store.claim(
        worker_id="relay-crashed",
        lease_duration=timedelta(seconds=10),
    )
    assert claim is not None
    assert claim.delivery.attempt_count == 1
    clock.advance(seconds=11)

    # Claim maintenance is the recovery trigger. The abandoned final attempt
    # must become terminal DLQ rather than being eligible for another claim.
    assert store.claim(worker_id="relay-recovery", lease_duration=timedelta(seconds=10)) is None
    dead = store.get(outbox_id)
    assert dead is not None
    assert dead.status is OutboxDeliveryStatus.DEAD_LETTER
    assert dead.attempt_count == 1
    assert dead.last_error == "delivery lease expired after final attempt"
    assert dead.dead_lettered_at is not None
    assert store.list_dead_letters() == (dead,)


def test_record_failure_rejects_a_claim_superseded_after_lease_expiry(
    tmp_path: Path,
) -> None:
    database_path, outbox_id, _ = _committed_outbox(tmp_path, run_value=90_012)
    clock = _MutableClock(datetime(2026, 7, 21, 19, tzinfo=UTC))
    store = SQLitePrimaryOutboxDeliveryStore(
        database_path,
        retry_policy=_policy(),
        clock=clock,
    )

    abandoned = store.claim(
        worker_id="relay-old",
        lease_duration=timedelta(seconds=10),
    )
    assert abandoned is not None
    clock.advance(seconds=11)
    replacement = store.claim(
        worker_id="relay-new",
        lease_duration=timedelta(seconds=10),
    )
    assert replacement is not None
    assert replacement.delivery.lease_epoch == abandoned.delivery.lease_epoch + 1

    with pytest.raises(OutboxFenceError, match="stale"):
        store.record_failure(abandoned, "late publish failure")

    current = store.get(outbox_id)
    assert current is not None
    assert current.status is OutboxDeliveryStatus.LEASED
    assert current.fencing_token == replacement.delivery.fencing_token
    assert current.last_error is None
