from __future__ import annotations

import pytest

from robata.adapters.redis_outbox import RedisIdempotentOutboxSink
from robata.contracts.hashing import exact_bytes_sha256
from robata.queue.outbox import OutboxDeliveryError, OutboxMessage


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def set(self, name: str, value: bytes, *, nx: bool = False) -> bool:
        if nx and name in self.values:
            return False
        self.values[name] = value
        return True

    def get(self, name: str) -> bytes | None:
        return self.values.get(name)


def _message(payload: bytes = b"canonical-payload") -> OutboxMessage:
    return OutboxMessage(
        outbox_id="outbox/1",
        completion_run_id="run-1",
        recording_identity="recording-1",
        outbox_ordinal=0,
        topic="robata.primary.completed",
        key="recording-1",
        payload=payload,
        payload_sha256=exact_bytes_sha256(payload),
    )


def test_redis_sink_retries_the_same_exact_message_without_duplicate() -> None:
    client = _FakeRedis()
    sink = RedisIdempotentOutboxSink(client=client)
    sink.publish(_message())
    sink.publish(_message())
    assert len(client.values) == 1


def test_redis_sink_rejects_same_id_with_different_exact_payload() -> None:
    client = _FakeRedis()
    sink = RedisIdempotentOutboxSink(client=client)
    sink.publish(_message())

    with pytest.raises(OutboxDeliveryError, match="different immutable message bytes"):
        sink.publish(_message(b"different"))


def test_redis_sink_rejects_corrupt_existing_acknowledgement() -> None:
    client = _FakeRedis()
    sink = RedisIdempotentOutboxSink(client=client)
    message = _message()
    sink.publish(message)
    key = next(iter(client.values))
    client.values[key] = b'{"payload":"not-base64","payload_sha256":"bad"}'

    with pytest.raises(OutboxDeliveryError, match="malformed or corrupt"):
        sink.publish(message)
