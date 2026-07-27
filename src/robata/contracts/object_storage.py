"""Provider-neutral object-storage contracts used by the R2 boundary.

These models are an adapter contract, not a replacement for a published
artifact schema.  In particular, object versions, ETags, and signed URLs are
locator metadata.  They never participate in artifact or recording identity;
the exact content digest and byte count are the integrity authority.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.logical_nodes import Rfc3339Timestamp

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
ObjectKey = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=1024,
        pattern=r"^\S(?:.*\S)?$",
    ),
]
ObjectUri = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=4,
        max_length=4096,
        pattern=r"^[A-Za-z][A-Za-z0-9+.-]*://\S+$",
    ),
]
HttpUri = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=8,
        max_length=8192,
        pattern=r"^https?://\S+$",
    ),
]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

OBJECT_STORAGE_CONTRACT_VERSION: Literal["1.0"] = "1.0"


class ObjectVisibility(StrEnum):
    """Observation state returned by an object HEAD/reconciliation check."""

    VISIBLE = "VISIBLE"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class ObjectLocator(StrictModel):
    """Versioned locator metadata for one object.

    ``uri`` and ``object_version`` identify where an adapter found an object,
    not what its bytes mean.  ``etag`` can be a multipart/provider-specific
    value and is consequently not treated as a content digest.  A presigned
    URL is intentionally optional and short-lived; it is never persisted in a
    semantic projection.
    """

    contract_version: Literal["1.0"] = OBJECT_STORAGE_CONTRACT_VERSION
    uri: ObjectUri
    object_version: SchemaVersion
    etag: NonEmptyString | None = None
    presigned_url: HttpUri | None = None
    presigned_url_expires_at: Rfc3339Timestamp | None = None

    @model_validator(mode="after")
    def validate_signed_locator(self) -> Self:
        if self.presigned_url_expires_at is not None and self.presigned_url is None:
            raise ValueError("presigned_url_expires_at requires presigned_url")
        return self

    @property
    def metadata_projection(self) -> dict[str, str]:
        """Return transport metadata without a content or logical identity."""

        projection = {
            "contract_version": self.contract_version,
            "uri": self.uri,
            "object_version": self.object_version,
        }
        if self.etag is not None:
            projection["etag"] = self.etag
        return projection

    @property
    def content_identity(self) -> None:
        """Make accidental use of locator metadata as content identity obvious."""

        return None


class ObjectByteRange(StrictModel):
    """Half-open byte range for bounded GET requests."""

    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start >= self.end:
            raise ValueError("byte range start must be less than end")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class ObjectPutRequest(StrictModel):
    """Blob-first write request carrying exact bytes and expected integrity."""

    contract_version: Literal["1.0"] = OBJECT_STORAGE_CONTRACT_VERSION
    key: ObjectKey
    payload: bytes
    sha256: Sha256Digest
    byte_count: NonNegativeInt
    media_type: NonEmptyString
    object_version: SchemaVersion | None = None

    @model_validator(mode="after")
    def validate_payload_integrity(self) -> Self:
        from robata.contracts.hashing import exact_bytes_sha256

        if self.byte_count != len(self.payload):
            raise ValueError("byte_count must match payload length")
        if exact_bytes_sha256(self.payload) != self.sha256:
            raise ValueError("sha256 must match exact payload bytes")
        return self

    @property
    def size_bytes(self) -> int:
        return self.byte_count

    @property
    def content_sha256(self) -> str:
        return self.sha256


class ObjectHead(StrictModel):
    """Verified metadata observed after a PUT or HEAD operation."""

    contract_version: Literal["1.0"] = OBJECT_STORAGE_CONTRACT_VERSION
    locator: ObjectLocator
    visibility: ObjectVisibility
    sha256: Sha256Digest | None = None
    byte_count: NonNegativeInt | None = None
    media_type: NonEmptyString | None = None
    etag: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_visible_metadata(self) -> Self:
        if self.visibility is ObjectVisibility.VISIBLE:
            if self.sha256 is None or self.byte_count is None:
                raise ValueError("visible object requires exact sha256 and byte_count")
            if (
                self.etag is not None
                and self.locator.etag is not None
                and self.etag != self.locator.etag
            ):
                raise ValueError("head etag must match locator etag when both are present")
        return self

    @property
    def verified(self) -> bool:
        return self.visibility is ObjectVisibility.VISIBLE and self.sha256 is not None


class ObjectPutReceipt(StrictModel):
    """Provider acknowledgement after an exact blob write."""

    contract_version: Literal["1.0"] = OBJECT_STORAGE_CONTRACT_VERSION
    locator: ObjectLocator
    sha256: Sha256Digest
    byte_count: NonNegativeInt
    visibility: ObjectVisibility = ObjectVisibility.VISIBLE

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.visibility is ObjectVisibility.VISIBLE and self.byte_count < 0:
            raise ValueError("visible receipt byte_count must be non-negative")
        return self

    @property
    def size_bytes(self) -> int:
        return self.byte_count

    @property
    def content_sha256(self) -> str:
        return self.sha256


# Names used by adapters and design notes.  They are aliases, not new wire
# versions, so importing one cannot accidentally create a second identity.
R2ObjectLocator = ObjectLocator
ObjectLocatorMetadata = ObjectLocator
ObjectRange = ObjectByteRange
ObjectPut = ObjectPutRequest
ObjectPutResult = ObjectPutReceipt


__all__ = [
    "OBJECT_STORAGE_CONTRACT_VERSION",
    "HttpUri",
    "ObjectByteRange",
    "ObjectHead",
    "ObjectKey",
    "ObjectLocator",
    "ObjectLocatorMetadata",
    "ObjectPut",
    "ObjectPutReceipt",
    "ObjectPutRequest",
    "ObjectPutResult",
    "ObjectRange",
    "ObjectUri",
    "ObjectVisibility",
    "R2ObjectLocator",
]
