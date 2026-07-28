"""Explicit S3-compatible Cloudflare R2 adapter for the object-store port.

The adapter deliberately has no implicit cloud selection.  A caller supplies an
``R2ObjectStoreConfig`` plus either an injected S3 client or explicit credentials
for the optional boto3 factory.  This keeps local composition fail-closed while
making the production boundary testable with an S3-compatible double.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import import_module
from typing import Annotated, Protocol, Self, cast
from urllib.parse import quote, unquote, urlsplit

from pydantic import StringConstraints, model_validator

from robata.contracts.common import StrictModel
from robata.contracts.hashing import exact_bytes_sha256
from robata.contracts.object_storage import (
    ObjectByteRange,
    ObjectHead,
    ObjectLocator,
    ObjectPutReceipt,
    ObjectPutRequest,
    ObjectVisibility,
)
from robata.ports.object_storage import ObjectStoreError, ObjectStoreErrorCode

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]

_METADATA_SHA256 = "robata-sha256"
_METADATA_BYTE_COUNT = "robata-byte-count"
_METADATA_MEDIA_TYPE = "robata-media-type"
_METADATA_OBJECT_VERSION = "robata-object-version"
_VERSION_SEGMENT = ".robata-versions"
_CONTENT_RANGE = re.compile(r"^bytes (?P<start>[0-9]+)-(?P<end>[0-9]+)/(?P<total>[0-9]+)$")


class R2S3Client(Protocol):
    """Narrow synchronous S3 surface used by :class:`R2ObjectStore`."""

    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def delete_object(self, **kwargs: object) -> Mapping[str, object]: ...


class R2ObjectStoreConfig(StrictModel):
    """Non-secret configuration for one S3-compatible R2 bucket."""

    endpoint_url: NonEmptyString
    bucket: NonEmptyString
    prefix: str = ""
    region_name: NonEmptyString = "auto"
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 120
    max_attempts: int = 3

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        try:
            endpoint = urlsplit(self.endpoint_url)
            port = endpoint.port
        except ValueError as exc:
            raise ValueError("endpoint_url must be a valid HTTPS URL") from exc
        if (
            endpoint.scheme != "https"
            or not endpoint.hostname
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.query
            or endpoint.fragment
            or endpoint.path not in {"", "/"}
        ):
            raise ValueError("endpoint_url must be an absolute HTTPS origin without credentials")
        if port is not None and not 1 <= port <= 65_535:
            raise ValueError("endpoint_url port is out of range")
        if self.bucket != self.bucket.strip() or "/" in self.bucket:
            raise ValueError("bucket must be a non-empty bucket name without slashes")
        if not isinstance(self.prefix, str):
            raise TypeError("prefix must be a string")
        if self.prefix != self.prefix.strip():
            raise ValueError("prefix must not start or end with whitespace")
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("R2 timeouts must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        return self

    @property
    def normalized_prefix(self) -> str:
        value = self.prefix.strip("/")
        return "" if not value else f"{value}/"

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self:
        """Build non-secret configuration from an explicit environment mapping."""

        endpoint_url = environment.get("R2_ENDPOINT_URL")
        account_id = environment.get("R2_ACCOUNT_ID")
        if not endpoint_url and account_id:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        if not endpoint_url:
            raise ValueError("R2_ENDPOINT_URL or R2_ACCOUNT_ID must be configured")
        bucket = environment.get("R2_BUCKET")
        if not bucket:
            raise ValueError("R2_BUCKET must be configured")
        return cls(
            endpoint_url=endpoint_url,
            bucket=bucket,
            prefix=environment.get("R2_PREFIX", ""),
            region_name=environment.get("R2_REGION", "auto"),
            connect_timeout_seconds=_environment_positive_int(
                environment, "R2_CONNECT_TIMEOUT_SECONDS", 10
            ),
            read_timeout_seconds=_environment_positive_int(
                environment, "R2_READ_TIMEOUT_SECONDS", 120
            ),
            max_attempts=_environment_positive_int(environment, "R2_MAX_ATTEMPTS", 3),
        )


class R2Credentials:
    """Opaque S3 credentials whose representations do not disclose secret material."""

    __slots__ = ("_access_key_id", "_secret_access_key")

    def __init__(self, access_key_id: str, secret_access_key: str) -> None:
        if not isinstance(access_key_id, str) or not access_key_id.strip():
            raise ValueError("access_key_id must be non-empty")
        if not isinstance(secret_access_key, str) or not secret_access_key:
            raise ValueError("secret_access_key must be non-empty")
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key

    @property
    def access_key_id(self) -> str:
        return self._access_key_id

    @property
    def secret_access_key(self) -> str:
        return self._secret_access_key

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self:
        access_key_id = environment.get("R2_ACCESS_KEY_ID")
        secret_access_key = environment.get("R2_SECRET_ACCESS_KEY")
        if not access_key_id or not secret_access_key:
            raise ValueError("R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY must be configured")
        return cls(access_key_id, secret_access_key)

    def __repr__(self) -> str:
        return "R2Credentials(access_key_id=REDACTED, secret_access_key=REDACTED)"

    __str__ = __repr__


class R2ObjectStore:
    """Fail-closed exact-byte adapter backed by a versioned S3-compatible bucket.

    Object bytes are stored under a versioned physical key.  The logical object
    key, semantic version, SHA-256, byte count and media type are persisted as
    provider metadata and are verified before a successful ``put`` returns.
    This is intentionally stronger than assuming a filesystem-style atomic rename.
    """

    def __init__(self, config: R2ObjectStoreConfig, client: R2S3Client) -> None:
        if not isinstance(config, R2ObjectStoreConfig):
            raise TypeError("config must be R2ObjectStoreConfig")
        for name in ("put_object", "head_object", "get_object", "delete_object"):
            if not callable(getattr(client, name, None)):
                raise TypeError("client must provide the required S3 object methods")
        self._config = config
        self._client = client

    @property
    def config(self) -> R2ObjectStoreConfig:
        return self._config

    def put(self, request: ObjectPutRequest) -> ObjectPutReceipt:
        if not isinstance(request, ObjectPutRequest):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, "request must be ObjectPutRequest"
            )
        version = request.object_version or f"v1-{request.sha256[:24]}"
        locator = self._locator(request.key, version)
        existing = self.head(locator)
        if existing.visibility is ObjectVisibility.VISIBLE:
            self._require_visible_exact(locator, request)
            return self._receipt(locator, request)
        if existing.visibility is ObjectVisibility.PARTIAL:
            raise ObjectStoreError(
                ObjectStoreErrorCode.CONFLICT,
                "R2 object key/version exists without complete Robata integrity metadata",
            )
        try:
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=self._physical_key(request.key, version),
                Body=request.payload,
                ContentType=request.media_type,
                Metadata={
                    _METADATA_SHA256: request.sha256,
                    _METADATA_BYTE_COUNT: str(request.byte_count),
                    _METADATA_MEDIA_TYPE: request.media_type,
                    _METADATA_OBJECT_VERSION: version,
                },
                IfNoneMatch="*",
            )
        except Exception as error:
            mapped = self._map_error(error)
            # A write response can be lost after R2 has durably accepted the
            # object. Re-read the immutable key for every provider failure,
            # but only recover when a full exact-byte check proves the same write.
            try:
                concurrent = self.head(locator)
            except ObjectStoreError:
                raise mapped from error
            if concurrent.visibility is ObjectVisibility.VISIBLE:
                self._require_visible_exact(locator, request)
                return self._receipt(locator, request)
            if concurrent.visibility is ObjectVisibility.PARTIAL:
                raise ObjectStoreError(
                    ObjectStoreErrorCode.VISIBILITY_UNKNOWN,
                    "R2 object has incomplete integrity metadata after PUT failure",
                ) from error
            raise mapped from error
        self._require_visible_exact(locator, request)
        return self._receipt(locator, request)

    def head(self, locator: ObjectLocator) -> ObjectHead:
        key = self._key_from_locator(locator)
        try:
            response = self._client.head_object(Bucket=self._config.bucket, Key=key)
        except Exception as error:
            mapped = self._map_error(error)
            if mapped.code is ObjectStoreErrorCode.NOT_FOUND:
                return ObjectHead(locator=locator, visibility=ObjectVisibility.MISSING)
            raise mapped from error
        if not isinstance(response, Mapping):
            raise ObjectStoreError(
                ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
                "R2 HEAD returned an invalid response",
            )
        return self._head_from_response(locator, response)

    def get(self, locator: ObjectLocator, byte_range: ObjectByteRange | None = None) -> bytes:
        if not isinstance(locator, ObjectLocator):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, "locator must be ObjectLocator"
            )
        if byte_range is not None and not isinstance(byte_range, ObjectByteRange):
            raise TypeError("byte_range must be ObjectByteRange or None")
        head = self.head(locator)
        if head.visibility is ObjectVisibility.MISSING:
            raise ObjectStoreError(ObjectStoreErrorCode.NOT_FOUND, "R2 object is missing")
        if head.visibility is not ObjectVisibility.VISIBLE:
            raise ObjectStoreError(
                ObjectStoreErrorCode.VISIBILITY_UNKNOWN,
                "R2 object is missing complete integrity metadata",
            )
        byte_count = head.byte_count
        if byte_count is None:
            raise ObjectStoreError(
                ObjectStoreErrorCode.VISIBILITY_UNKNOWN,
                "R2 visible object is missing a verified byte count",
            )
        if byte_range is not None and byte_range.end > byte_count:
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST,
                "requested R2 byte range exceeds the verified object length",
            )
        arguments: dict[str, object] = {
            "Bucket": self._config.bucket,
            "Key": self._key_from_locator(locator),
        }
        if byte_range is not None:
            arguments["Range"] = f"bytes={byte_range.start}-{byte_range.end - 1}"
        try:
            response = self._client.get_object(**arguments)
        except Exception as error:
            raise self._map_error(error) from error
        payload = _read_response_body(response)
        if byte_range is None:
            if len(payload) != byte_count or exact_bytes_sha256(payload) != head.sha256:
                raise ObjectStoreError(
                    ObjectStoreErrorCode.INTEGRITY_ERROR,
                    "R2 GET bytes do not match the persisted integrity metadata",
                )
        else:
            _require_exact_content_range(response, byte_range, byte_count)
            if len(payload) != byte_range.length:
                raise ObjectStoreError(
                    ObjectStoreErrorCode.INTEGRITY_ERROR,
                    "R2 range GET length does not match the requested byte range",
                )
        return payload

    def delete(self, locator: ObjectLocator) -> None:
        if not isinstance(locator, ObjectLocator):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, "locator must be ObjectLocator"
            )
        if self.head(locator).visibility is ObjectVisibility.MISSING:
            raise ObjectStoreError(ObjectStoreErrorCode.NOT_FOUND, "R2 object is missing")
        try:
            self._client.delete_object(
                Bucket=self._config.bucket, Key=self._key_from_locator(locator)
            )
        except Exception as error:
            raise self._map_error(error) from error

    def reconcile(self, locator: ObjectLocator) -> ObjectHead:
        head = self.head(locator)
        if head.visibility is ObjectVisibility.VISIBLE:
            self.get(locator)
        return head

    def _locator(self, logical_key: str, version: str) -> ObjectLocator:
        physical_key = self._physical_key(logical_key, version)
        return ObjectLocator(
            uri=f"r2://{self._config.bucket}/{quote(physical_key, safe='/._-~')}",
            object_version=version,
        )

    def _physical_key(self, logical_key: str, version: str) -> str:
        return f"{self._config.normalized_prefix}{logical_key}/{_VERSION_SEGMENT}/{version}"

    def _key_from_locator(self, locator: ObjectLocator) -> str:
        if not isinstance(locator, ObjectLocator):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, "locator must be ObjectLocator"
            )
        parsed = urlsplit(locator.uri)
        if (
            parsed.scheme != "r2"
            or parsed.netloc != self._config.bucket
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith("/")
        ):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST,
                "locator does not belong to the configured R2 bucket",
            )
        key = unquote(parsed.path[1:])
        if not key or (
            self._config.normalized_prefix and not key.startswith(self._config.normalized_prefix)
        ):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST,
                "locator does not belong to the configured R2 prefix",
            )
        expected_suffix = f"/{_VERSION_SEGMENT}/{locator.object_version}"
        if not key.endswith(expected_suffix):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST,
                "locator physical key does not match its object version",
            )
        return key

    def _head_from_response(
        self, locator: ObjectLocator, response: Mapping[str, object]
    ) -> ObjectHead:
        metadata_value = response.get("Metadata", {})
        metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
        sha256 = _metadata_text(metadata, _METADATA_SHA256)
        byte_count_text = _metadata_text(metadata, _METADATA_BYTE_COUNT)
        media_type = _metadata_text(metadata, _METADATA_MEDIA_TYPE)
        version = _metadata_text(metadata, _METADATA_OBJECT_VERSION)
        content_length = response.get("ContentLength")
        content_type = response.get("ContentType")
        if (
            sha256 is None
            or byte_count_text is None
            or media_type is None
            or version != locator.object_version
            or not isinstance(content_length, int)
            or content_type != media_type
        ):
            return ObjectHead(locator=locator, visibility=ObjectVisibility.PARTIAL)
        try:
            byte_count = int(byte_count_text)
        except ValueError:
            return ObjectHead(locator=locator, visibility=ObjectVisibility.PARTIAL)
        if byte_count < 0 or byte_count != content_length or len(sha256) != 64:
            return ObjectHead(locator=locator, visibility=ObjectVisibility.PARTIAL)
        etag = _normalized_etag(response.get("ETag"))
        try:
            visible_locator = locator.model_copy(update={"etag": etag}) if etag else locator
            return ObjectHead(
                locator=visible_locator,
                visibility=ObjectVisibility.VISIBLE,
                sha256=sha256,
                byte_count=byte_count,
                media_type=media_type,
                etag=etag,
            )
        except ValueError:
            return ObjectHead(locator=locator, visibility=ObjectVisibility.PARTIAL)

    @staticmethod
    def _receipt(locator: ObjectLocator, request: ObjectPutRequest) -> ObjectPutReceipt:
        return ObjectPutReceipt(
            locator=locator,
            sha256=request.sha256,
            byte_count=request.byte_count,
            visibility=ObjectVisibility.VISIBLE,
        )

    def _require_visible_exact(self, locator: ObjectLocator, request: ObjectPutRequest) -> None:
        head = self.reconcile(locator)
        if head.visibility is not ObjectVisibility.VISIBLE:
            raise ObjectStoreError(
                ObjectStoreErrorCode.VISIBILITY_UNKNOWN,
                f"R2 object is not visibly readable after PUT: {head.visibility.value}",
            )
        self._require_matching_existing(head, request)

    @staticmethod
    def _require_matching_existing(head: ObjectHead, request: ObjectPutRequest) -> None:
        if (
            head.sha256 != request.sha256
            or head.byte_count != request.byte_count
            or head.media_type != request.media_type
        ):
            raise ObjectStoreError(
                ObjectStoreErrorCode.CONFLICT,
                "R2 object key/version is already bound to different bytes or metadata",
            )

    @staticmethod
    def _map_error(error: Exception) -> ObjectStoreError:
        if isinstance(error, ObjectStoreError):
            return error
        code = _provider_error_code(error)
        if code in {"404", "nosuchkey", "notfound", "nosuchbucket"}:
            return ObjectStoreError(ObjectStoreErrorCode.NOT_FOUND, "R2 object is missing")
        if code in {"409", "412", "preconditionfailed", "conditionalrequestconflict"}:
            return ObjectStoreError(
                ObjectStoreErrorCode.CONFLICT, "R2 rejected an immutable object create"
            )
        if code in {"invalidrequest", "invalidargument", "malformedxml", "invalidrange"}:
            return ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, f"R2 rejected request: {code}"
            )
        if code in {"accessdenied", "signaturedoesnotmatch", "invalidaccesskeyid"}:
            return ObjectStoreError(
                ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
                "R2 credentials or bucket access were rejected",
            )
        return ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
            f"R2 S3 operation failed{f': {code}' if code else ''}",
        )


def create_boto3_r2_client(config: R2ObjectStoreConfig, credentials: R2Credentials) -> R2S3Client:
    """Construct a path-style boto3 client only when the optional extra is installed."""

    if not isinstance(config, R2ObjectStoreConfig):
        raise TypeError("config must be R2ObjectStoreConfig")
    if not isinstance(credentials, R2Credentials):
        raise TypeError("credentials must be R2Credentials")
    try:
        boto3_module = import_module("boto3")
        botocore_config_module = import_module("botocore.config")
    except ImportError as error:
        raise ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
            "R2 adapter requires the optional robata[r2] dependency group",
        ) from error
    client = getattr(boto3_module, "client", None)
    config_class = getattr(botocore_config_module, "Config", None)
    if not callable(client) or not callable(config_class):
        raise ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
            "R2 adapter requires boto3 and botocore.config.Config",
        )
    return cast(
        R2S3Client,
        client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
            aws_access_key_id=credentials.access_key_id,
            aws_secret_access_key=credentials.secret_access_key,
            config=config_class(
                connect_timeout=config.connect_timeout_seconds,
                read_timeout=config.read_timeout_seconds,
                retries={"max_attempts": config.max_attempts, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        ),
    )


def create_r2_object_store_from_environment(environment: Mapping[str, str]) -> R2ObjectStore:
    """Build the optional real adapter from an explicit, caller-owned environment map."""

    config = R2ObjectStoreConfig.from_environment(environment)
    return R2ObjectStore(
        config, create_boto3_r2_client(config, R2Credentials.from_environment(environment))
    )


def _environment_positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    value = environment.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _metadata_text(metadata: Mapping[object, object], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _normalized_etag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip('"')
    return normalized or None


def _provider_error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return str(code).lower() if code is not None else None


def _read_response_body(response: Mapping[str, object]) -> bytes:
    if not isinstance(response, Mapping):
        raise ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE, "R2 GET returned an invalid response"
        )
    body = response.get("Body")
    read = getattr(body, "read", None)
    if not callable(read):
        raise ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE, "R2 GET returned no readable body"
        )
    payload = read()
    if not isinstance(payload, bytes):
        raise ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE, "R2 GET body was not bytes"
        )
    return payload


def _require_exact_content_range(
    response: Mapping[str, object],
    byte_range: ObjectByteRange,
    total_byte_count: int,
) -> None:
    content_range = response.get("ContentRange")
    if not isinstance(content_range, str):
        raise ObjectStoreError(
            ObjectStoreErrorCode.INTEGRITY_ERROR,
            "R2 range GET did not return ContentRange metadata",
        )
    matched = _CONTENT_RANGE.fullmatch(content_range)
    if matched is None:
        raise ObjectStoreError(
            ObjectStoreErrorCode.INTEGRITY_ERROR,
            "R2 range GET returned malformed ContentRange metadata",
        )
    if (
        int(matched["start"]) != byte_range.start
        or int(matched["end"]) != byte_range.end - 1
        or int(matched["total"]) != total_byte_count
    ):
        raise ObjectStoreError(
            ObjectStoreErrorCode.INTEGRITY_ERROR,
            "R2 range GET ContentRange does not match the requested bytes",
        )


__all__ = [
    "R2Credentials",
    "R2ObjectStore",
    "R2ObjectStoreConfig",
    "R2S3Client",
    "create_boto3_r2_client",
    "create_r2_object_store_from_environment",
]
