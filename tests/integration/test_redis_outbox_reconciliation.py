from __future__ import annotations

from pathlib import Path

from robata.adapters.redis_outbox import RedisIdempotentOutboxSink
from robata.application.canonical.local_outbox_delivery import (
    LocalOutboxDeliveryOutcome,
    reconcile_primary_outbox_to_sink,
)
from robata.contracts.schema_registry import default_schema_registry
from robata.queue.outbox import OutboxRetryPolicy
from tests.integration.test_sqlite_primary_completion import _run_case


class _ResponseLostRedis:
    """Fake Redis that writes once before losing the first client response."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.set_calls = 0
        self._lose_first_response = True

    def set(self, name: str, value: bytes, *, nx: bool = False) -> bool:
        self.set_calls += 1
        if nx and name in self.values:
            return False
        self.values[name] = value
        if self._lose_first_response:
            self._lose_first_response = False
            raise ConnectionError("response lost after durable Redis write")
        return True

    def get(self, name: str) -> bytes | None:
        return self.values.get(name)


def _retry_policy() -> OutboxRetryPolicy:
    return OutboxRetryPolicy(
        version="redis-outbox-test-v1",
        max_attempts=3,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
    )


def test_redis_sink_reconciliation_recovers_after_lost_publish_response(
    tmp_path: Path,
) -> None:
    _, repository, command = _run_case(tmp_path, run_value=90_111)
    committed = repository.commit(command).committed
    assert len(committed.outbox) == 1

    client = _ResponseLostRedis()
    first = reconcile_primary_outbox_to_sink(
        primary_database_path=repository.path,
        sink=RedisIdempotentOutboxSink(client=client),
        outbox=committed.outbox,
        registry=default_schema_registry(),
        max_delivery_attempts=2,
        worker_id="redis-outbox-relay",
        retry_policy=_retry_policy(),
    )

    assert first.outcome is LocalOutboxDeliveryOutcome.DELIVERED
    assert first.relay_attempt_count == 2
    assert first.delivered_count == 1
    assert len(client.values) == 1
    assert client.set_calls == 2

    replay = reconcile_primary_outbox_to_sink(
        primary_database_path=repository.path,
        sink=RedisIdempotentOutboxSink(client=client),
        outbox=committed.outbox,
        registry=default_schema_registry(),
        worker_id="redis-outbox-relay-restart",
        retry_policy=_retry_policy(),
    )

    assert replay.outcome is LocalOutboxDeliveryOutcome.DELIVERED
    assert replay.relay_attempt_count == 0
    assert len(client.values) == 1