"""Object-storage port with an explicit fail-closed default."""

from __future__ import annotations

from enum import StrEnum
from typing import NoReturn, Protocol

from robata.contracts.object_storage import (
    ObjectByteRange,
    ObjectHead,
    ObjectLocator,
    ObjectPutReceipt,
    ObjectPutRequest,
)


class ObjectStoreErrorCode(StrEnum):
    ADAPTER_UNAVAILABLE = "ADAPTER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"
    VISIBILITY_UNKNOWN = "VISIBILITY_UNKNOWN"
    RETENTION_UNSUPPORTED = "RETENTION_UNSUPPORTED"


class ObjectStoreError(RuntimeError):
    """Stable object-store boundary failure."""

    def __init__(self, code: ObjectStoreErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class ObjectStore(Protocol):
    """Minimal exact-byte object boundary used by artifact adapters."""

    def put(self, request: ObjectPutRequest) -> ObjectPutReceipt: ...

    def head(self, locator: ObjectLocator) -> ObjectHead: ...

    def get(self, locator: ObjectLocator, byte_range: ObjectByteRange | None = None) -> bytes: ...

    def delete(self, locator: ObjectLocator) -> None: ...

    def reconcile(self, locator: ObjectLocator) -> ObjectHead: ...


class FailClosedObjectStore:
    """Default production boundary; no cloud SDK is silently selected."""

    @staticmethod
    def _unavailable() -> NoReturn:
        raise ObjectStoreError(
            ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
            "object-store adapter is not configured",
        )

    def put(self, request: ObjectPutRequest) -> ObjectPutReceipt:
        del request
        self._unavailable()

    def head(self, locator: ObjectLocator) -> ObjectHead:
        del locator
        self._unavailable()

    def get(self, locator: ObjectLocator, byte_range: ObjectByteRange | None = None) -> bytes:
        del locator, byte_range
        self._unavailable()

    def delete(self, locator: ObjectLocator) -> None:
        del locator
        self._unavailable()

    def reconcile(self, locator: ObjectLocator) -> ObjectHead:
        del locator
        self._unavailable()


ObjectStorePort = ObjectStore
R2ObjectStore = ObjectStore
BlobStore = ObjectStore

__all__ = [
    "BlobStore",
    "FailClosedObjectStore",
    "ObjectStore",
    "ObjectStoreError",
    "ObjectStoreErrorCode",
    "ObjectStorePort",
    "R2ObjectStore",
]
