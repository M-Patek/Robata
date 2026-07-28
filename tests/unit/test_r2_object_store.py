"""Contract coverage for the optional, injected S3-compatible R2 adapter."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import md5

import pytest

from robata.adapters.r2_object_store import (
    R2Credentials,
    R2ObjectStore,
    R2ObjectStoreConfig,
    create_boto3_r2_client,
)
from robata.contracts.hashing import exact_bytes_sha256
from robata.contracts.object_storage import ObjectByteRange, ObjectPutRequest, ObjectVisibility
from robata.ports.object_storage import ObjectStoreError, ObjectStoreErrorCode


class _S3Error(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


@dataclass
class _StoredObject:
    payload: bytes
    content_type: str
    metadata: dict[str, str]
    etag: str


class _S3Double:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], _StoredObject] = {}
        self.put_count = 0
        self.get_count = 0
        self.fail_after_write_once = False
        self.range_response_override: bytes | None = None
        self.range_content_range_override: str | None = None
        self.omit_range_content_range = False
        self.last_range: str | None = None

    def put_object(self, **kwargs: object) -> Mapping[str, object]:
        bucket = _required_text(kwargs, "Bucket")
        key = _required_text(kwargs, "Key")
        payload = kwargs["Body"]
        content_type = _required_text(kwargs, "ContentType")
        metadata = kwargs["Metadata"]
        if not isinstance(payload, bytes) or not isinstance(metadata, dict):
            raise TypeError("test double received an invalid S3 put request")
        storage_key = (bucket, key)
        if kwargs.get("IfNoneMatch") == "*" and storage_key in self.objects:
            raise _S3Error("PreconditionFailed")
        self.put_count += 1
        self.objects[storage_key] = _StoredObject(
            payload=payload,
            content_type=content_type,
            metadata=dict(metadata),
            etag=md5(payload, usedforsecurity=False).hexdigest(),
        )
        if self.fail_after_write_once:
            self.fail_after_write_once = False
            raise _S3Error("RequestTimeout")
        return {"ETag": f'"{self.objects[storage_key].etag}"'}

    def head_object(self, **kwargs: object) -> Mapping[str, object]:
        record = self._require(kwargs)
        return {
            "ContentLength": len(record.payload),
            "ContentType": record.content_type,
            "Metadata": dict(record.metadata),
            "ETag": f'"{record.etag}"',
        }

    def get_object(self, **kwargs: object) -> Mapping[str, object]:
        record = self._require(kwargs)
        self.get_count += 1
        payload = record.payload
        range_value = kwargs.get("Range")
        content_range: str | None = None
        if range_value is not None:
            self.last_range = range_value if isinstance(range_value, str) else None
            if not isinstance(range_value, str) or not range_value.startswith("bytes="):
                raise _S3Error("InvalidRange")
            start_text, end_text = range_value[6:].split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start or start >= len(payload):
                raise _S3Error("InvalidRange")
            content_range = f"bytes {start}-{end}/{len(payload)}"
            payload = payload[start : end + 1]
            if self.range_response_override is not None:
                payload = self.range_response_override
        response: dict[str, object] = {"Body": _Body(payload)}
        if content_range is not None and not self.omit_range_content_range:
            response["ContentRange"] = self.range_content_range_override or content_range
        return response

    def delete_object(self, **kwargs: object) -> Mapping[str, object]:
        bucket = _required_text(kwargs, "Bucket")
        key = _required_text(kwargs, "Key")
        if (bucket, key) not in self.objects:
            raise _S3Error("NoSuchKey")
        del self.objects[(bucket, key)]
        return {}

    def _require(self, kwargs: Mapping[str, object]) -> _StoredObject:
        bucket = _required_text(kwargs, "Bucket")
        key = _required_text(kwargs, "Key")
        try:
            return self.objects[(bucket, key)]
        except KeyError as error:
            raise _S3Error("NoSuchKey") from error


def _required_text(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be text")
    return value


def _config() -> R2ObjectStoreConfig:
    return R2ObjectStoreConfig(
        endpoint_url="https://account-id.r2.cloudflarestorage.com",
        bucket="robata-production",
        prefix="artifacts/",
    )


def _request(
    payload: bytes = b"camera-bytes",
    *,
    key: str = "recordings/one.mcap",
    version: str | None = None,
) -> ObjectPutRequest:
    return ObjectPutRequest(
        key=key,
        payload=payload,
        sha256=exact_bytes_sha256(payload),
        byte_count=len(payload),
        media_type="application/octet-stream",
        object_version=version,
    )


def test_put_uses_immutable_versioned_key_and_reconciles_exact_bytes() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)

    receipt = store.put(_request())

    assert receipt.visibility is ObjectVisibility.VISIBLE
    assert receipt.locator.uri == (
        "r2://robata-production/artifacts/recordings/one.mcap/.robata-versions/v1-"
        "9b867455c8b391b9e21cba13"
    )
    assert client.put_count == 1
    stored = next(iter(client.objects.values()))
    assert stored.metadata["robata-sha256"] == receipt.sha256
    assert stored.metadata["robata-byte-count"] == str(receipt.byte_count)
    assert store.get(receipt.locator) == b"camera-bytes"
    assert store.head(receipt.locator).verified is True


def test_replayed_exact_put_is_idempotent_and_conflicting_version_is_rejected() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)
    first = _request(version="capture-v1")

    first_receipt = store.put(first)
    replay_receipt = store.put(first)

    assert replay_receipt == first_receipt
    assert client.put_count == 1
    with pytest.raises(ObjectStoreError) as error:
        store.put(_request(b"different", version="capture-v1"))
    assert error.value.code is ObjectStoreErrorCode.CONFLICT


def test_head_get_range_reconcile_and_delete_preserve_port_semantics() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)
    receipt = store.put(_request(b"0123456789"))

    assert store.get(receipt.locator, ObjectByteRange(start=2, end=6)) == b"2345"
    assert store.reconcile(receipt.locator).visibility is ObjectVisibility.VISIBLE

    stored = next(iter(client.objects.values()))
    stored.payload = b"corrupted!"
    with pytest.raises(ObjectStoreError) as integrity_error:
        store.reconcile(receipt.locator)
    assert integrity_error.value.code is ObjectStoreErrorCode.INTEGRITY_ERROR

    stored.payload = b"0123456789"
    store.delete(receipt.locator)
    assert store.head(receipt.locator).visibility is ObjectVisibility.MISSING
    with pytest.raises(ObjectStoreError) as missing_error:
        store.delete(receipt.locator)
    assert missing_error.value.code is ObjectStoreErrorCode.NOT_FOUND


def test_config_and_credentials_require_explicit_non_secret_environment_values() -> None:
    config = R2ObjectStoreConfig.from_environment(
        {
            "R2_ACCOUNT_ID": "account-id",
            "R2_BUCKET": "robata-production",
            "R2_PREFIX": "camera-artifacts",
            "R2_CONNECT_TIMEOUT_SECONDS": "11",
        }
    )
    credentials = R2Credentials.from_environment(
        {"R2_ACCESS_KEY_ID": "access", "R2_SECRET_ACCESS_KEY": "super-secret"}
    )

    assert config.endpoint_url == "https://account-id.r2.cloudflarestorage.com"
    assert config.normalized_prefix == "camera-artifacts/"
    assert config.connect_timeout_seconds == 11
    assert "super-secret" not in repr(credentials)
    with pytest.raises(ValueError, match="R2_BUCKET"):
        R2ObjectStoreConfig.from_environment({"R2_ACCOUNT_ID": "account-id"})
    with pytest.raises(ValueError, match="R2_ACCESS_KEY_ID"):
        R2Credentials.from_environment({"R2_SECRET_ACCESS_KEY": "super-secret"})


def test_locator_cannot_escape_configured_bucket_or_prefix() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)
    receipt = store.put(_request())
    wrong_bucket = receipt.locator.model_copy(
        update={"uri": receipt.locator.uri.replace("robata-production", "other-bucket")}
    )

    with pytest.raises(ObjectStoreError) as error:
        store.head(wrong_bucket)
    assert error.value.code is ObjectStoreErrorCode.INVALID_REQUEST


def test_put_recovers_only_when_a_lost_response_has_durable_matching_bytes() -> None:
    client = _S3Double()
    client.fail_after_write_once = True
    store = R2ObjectStore(_config(), client)

    receipt = store.put(_request())

    assert receipt.visibility is ObjectVisibility.VISIBLE
    assert client.put_count == 1
    assert store.get(receipt.locator) == b"camera-bytes"


def test_range_get_rejects_out_of_bounds_and_unverified_provider_bytes() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)
    receipt = store.put(_request(b"0123456789"))

    assert store.get(receipt.locator, ObjectByteRange(start=2, end=6)) == b"2345"
    assert client.last_range == "bytes=2-5"
    with pytest.raises(ObjectStoreError) as out_of_bounds:
        store.get(receipt.locator, ObjectByteRange(start=9, end=11))
    assert out_of_bounds.value.code is ObjectStoreErrorCode.INVALID_REQUEST

    client.range_response_override = b"short"
    with pytest.raises(ObjectStoreError) as truncated:
        store.get(receipt.locator, ObjectByteRange(start=2, end=6))
    assert truncated.value.code is ObjectStoreErrorCode.INTEGRITY_ERROR

    client.range_response_override = None
    client.range_content_range_override = "bytes 0-3/10"
    with pytest.raises(ObjectStoreError) as wrong_offset:
        store.get(receipt.locator, ObjectByteRange(start=2, end=6))
    assert wrong_offset.value.code is ObjectStoreErrorCode.INTEGRITY_ERROR

    client.range_content_range_override = None
    client.omit_range_content_range = True
    with pytest.raises(ObjectStoreError) as missing_content_range:
        store.get(receipt.locator, ObjectByteRange(start=2, end=6))
    assert missing_content_range.value.code is ObjectStoreErrorCode.INTEGRITY_ERROR


def test_head_treats_content_type_metadata_disagreement_as_partial() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)
    receipt = store.put(_request())
    stored = next(iter(client.objects.values()))
    stored.content_type = "text/plain"

    assert store.head(receipt.locator).visibility is ObjectVisibility.PARTIAL
    with pytest.raises(ObjectStoreError) as unavailable:
        store.get(receipt.locator)
    assert unavailable.value.code is ObjectStoreErrorCode.VISIBILITY_UNKNOWN


def test_optional_boto3_factory_fails_closed_when_dependency_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(ObjectStoreError) as unavailable:
        create_boto3_r2_client(
            _config(),
            R2Credentials(access_key_id="access", secret_access_key="super-secret"),
        )

    assert unavailable.value.code is ObjectStoreErrorCode.ADAPTER_UNAVAILABLE


def test_idempotent_replay_rechecks_exact_durable_bytes() -> None:
    client = _S3Double()
    store = R2ObjectStore(_config(), client)
    request = _request()
    store.put(request)
    stored = next(iter(client.objects.values()))
    stored.payload = b"x" * len(request.payload)

    with pytest.raises(ObjectStoreError) as corrupt_replay:
        store.put(request)

    assert corrupt_replay.value.code is ObjectStoreErrorCode.INTEGRITY_ERROR
