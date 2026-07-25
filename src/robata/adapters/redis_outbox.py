"""Redis-backed idempotent destination for authoritative outbox messages."""

from __future__ import annotations

import base64
import json
from typing import Any, Protocol, cast

from robata.contracts.hashing import exact_bytes_sha256
from robata.queue.outbox import OutboxDeliveryError, OutboxMessage


class RedisCommandClient(Protocol):
    def set(self, name: str, value: bytes, *, nx: bool = False) -> bool | None: ...

    def get(self, name: str) -> bytes | str | None: ...


class RedisIdempotentOutboxSink:
    """Store an immutable outbox message under its existing opaque ID."""

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        client: RedisCommandClient | None = None,
        key_prefix: str = "robata:outbox:v1",
    ) -> None:
        if not isinstance(key_prefix, str) or not key_prefix.strip():
            raise ValueError("key_prefix must be a non-empty string")
        if client is None:
            if not isinstance(redis_url, str) or not redis_url.strip():
                raise ValueError("redis_url is required when client is not supplied")
            try:
                import redis
            except ImportError as error:  # pragma: no cover
                raise OutboxDeliveryError(
                    "redis-py is required for Redis outbox delivery"
                ) from error
            client = cast(
                RedisCommandClient,
                redis.Redis.from_url(redis_url, decode_responses=False),
            )
        if client is None:
            raise AssertionError("Redis client resolution unexpectedly returned None")
        self._client = client
        self._key_prefix = key_prefix.rstrip(":")

    def publish(self, message: OutboxMessage) -> None:
        if not isinstance(message, OutboxMessage):
            raise TypeError("message must be OutboxMessage")
        key = f"{self._key_prefix}:{_key_component(message.outbox_id)}"
        encoded = self._encode(message)
        try:
            inserted = self._client.set(key, encoded, nx=True)
            if inserted:
                return
            existing = self._client.get(key)
        except Exception as error:
            raise OutboxDeliveryError(f"Redis outbox publish failed: {error}") from error
        if existing is None:
            raise OutboxDeliveryError(
                "Redis outbox acknowledgement disappeared before verification"
            )
        if self._decode(existing) != self._identity(message):
            raise OutboxDeliveryError(
                "Redis outbox ID is already bound to different immutable message bytes"
            )

    @staticmethod
    def _identity(message: OutboxMessage) -> dict[str, object]:
        return {
            "outbox_id": message.outbox_id,
            "topic": message.topic,
            "key": message.key,
            "payload_sha256": message.payload_sha256,
            "payload": base64.b64encode(message.payload).decode("ascii"),
        }

    @classmethod
    def _encode(cls, message: OutboxMessage) -> bytes:
        encoded = json.dumps(cls._identity(message), sort_keys=True, separators=(",", ":"))
        return encoded.encode("utf-8")

    @staticmethod
    def _decode(value: bytes | str) -> dict[str, object]:
        raw = value.encode("utf-8") if isinstance(value, str) else value
        try:
            parsed: Any = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("stored record is not an object")
            payload = parsed.get("payload")
            digest = parsed.get("payload_sha256")
            if not isinstance(payload, str) or not isinstance(digest, str):
                raise ValueError("stored record lacks exact payload fields")
            exact = base64.b64decode(payload, validate=True)
            if exact_bytes_sha256(exact) != digest:
                raise ValueError("stored record digest does not match payload")
            return parsed
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise OutboxDeliveryError("Redis outbox record is malformed or corrupt") from error


def _key_component(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


__all__ = ["RedisCommandClient", "RedisIdempotentOutboxSink"]
