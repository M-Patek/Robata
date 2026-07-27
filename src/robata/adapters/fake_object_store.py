"""Deterministic in-memory object store used for local R2 contract proofs.

The fake models the failure points that matter at the artifact boundary: exact-byte
writes, duplicate/replay, delayed visibility, range GETs, retention expiry, and a
response lost after a durable PUT.  It deliberately has no cloud dependency.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import md5
from typing import Final

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

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class ObjectReconciliationIssue:
    """One storage observation that needs operator reconciliation."""

    key: str
    object_version: str
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class ObjectReconciliationReport:
    """Deterministic object-store reconciliation summary."""

    checked_count: int
    visible_count: int
    missing_count: int
    issue_count: int
    issues: tuple[ObjectReconciliationIssue, ...]


@dataclass(slots=True)
class _ObjectRecord:
    request: ObjectPutRequest
    locator: ObjectLocator
    visible_at: datetime
    retention_until: datetime | None
    deleted: bool = False


class FakeObjectStore:
    """Failure-injectable object store implementing :class:`ObjectStore`."""

    _DEFAULT_SCHEME: Final[str] = "r2"

    def __init__(
        self,
        *,
        bucket: str = "robata-test",
        uri_scheme: str = _DEFAULT_SCHEME,
        clock: Clock | None = None,
        visibility_delay: timedelta = timedelta(0),
        retention: timedelta | None = None,
    ) -> None:
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("bucket must be a non-empty string")
        if not isinstance(uri_scheme, str) or not uri_scheme.strip():
            raise ValueError("uri_scheme must be a non-empty string")
        if not isinstance(visibility_delay, timedelta) or visibility_delay < timedelta(0):
            raise ValueError("visibility_delay must be non-negative")
        if retention is not None and (
            not isinstance(retention, timedelta) or retention <= timedelta(0)
        ):
            raise ValueError("retention must be positive when supplied")
        self.bucket = bucket.strip()
        self.uri_scheme = uri_scheme.strip()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._visibility_delay = visibility_delay
        self._retention = retention
        self._objects: dict[tuple[str, str], _ObjectRecord] = {}
        self._failures: dict[str, tuple[ObjectStoreErrorCode, str, bool]] = {}
        self._counters: Counter[str] = Counter()

    @property
    def objects(self) -> tuple[ObjectLocator, ...]:
        """Return non-deleted locators in stable key/version order."""

        return tuple(
            record.locator
            for record in sorted(
                self._objects.values(),
                key=lambda item: (item.request.key, item.locator.object_version),
            )
            if not record.deleted
        )

    def fail_next(
        self,
        operation: str,
        *,
        code: ObjectStoreErrorCode = ObjectStoreErrorCode.ADAPTER_UNAVAILABLE,
        message: str = "injected object-store failure",
        after_write: bool = False,
    ) -> None:
        """Inject one failure; ``after_write`` simulates a lost PUT response."""

        if operation not in {"put", "head", "get", "delete", "reconcile"}:
            raise ValueError("unsupported object-store operation")
        if not isinstance(code, ObjectStoreErrorCode):
            raise TypeError("code must be ObjectStoreErrorCode")
        if not isinstance(message, str) or not message:
            raise ValueError("message must be a non-empty string")
        if not isinstance(after_write, bool):
            raise TypeError("after_write must be a boolean")
        self._failures[operation] = (code, message, after_write)

    def operation_counts(self) -> dict[str, int]:
        return dict(self._counters)

    def put(self, request: ObjectPutRequest) -> ObjectPutReceipt:
        self._require_request(request)
        self._counters["put"] += 1
        failure = self._take_failure("put")
        # A normal transport failure occurs before the provider accepts the
        # write. ``after_write`` is the explicit response-loss simulation and
        # is the only path that may leave durable bytes behind.
        if failure is not None and not failure[2]:
            raise ObjectStoreError(failure[0], failure[1])
        now = self._now()
        version = request.object_version or f"v1-{request.sha256[:24]}"
        key = (request.key, version)
        existing = self._objects.get(key)
        if existing is not None and not existing.deleted:
            if not self._same_payload(existing.request, request):
                raise ObjectStoreError(
                    ObjectStoreErrorCode.CONFLICT,
                    f"object key/version is already bound to different bytes: {request.key}",
                )
            receipt = self._receipt(existing, now)
            if failure is not None:
                raise ObjectStoreError(failure[0], failure[1])
            return receipt

        locator = ObjectLocator(
            uri=f"{self.uri_scheme}://{self.bucket}/{request.key}",
            object_version=version,
            etag=md5(request.payload, usedforsecurity=False).hexdigest(),
        )
        record = _ObjectRecord(
            request=request,
            locator=locator,
            visible_at=now + self._visibility_delay,
            retention_until=(None if self._retention is None else now + self._retention),
        )
        self._objects[key] = record
        if failure is not None:
            raise ObjectStoreError(failure[0], failure[1])
        return self._receipt(record, now)

    def head(self, locator: ObjectLocator) -> ObjectHead:
        self._require_locator(locator)
        self._counters["head"] += 1
        self._maybe_fail("head")
        record = self._record(locator)
        if record is None:
            return ObjectHead(locator=locator, visibility=ObjectVisibility.MISSING)
        now = self._now()
        if record.retention_until is not None and now >= record.retention_until:
            record.deleted = True
            return ObjectHead(locator=locator, visibility=ObjectVisibility.MISSING)
        if now < record.visible_at:
            return ObjectHead(locator=locator, visibility=ObjectVisibility.PARTIAL)
        request = record.request
        return ObjectHead(
            locator=record.locator,
            visibility=ObjectVisibility.VISIBLE,
            sha256=request.sha256,
            byte_count=request.byte_count,
            media_type=request.media_type,
            etag=record.locator.etag,
        )

    def get(self, locator: ObjectLocator, byte_range: ObjectByteRange | None = None) -> bytes:
        self._require_locator(locator)
        if byte_range is not None and not isinstance(byte_range, ObjectByteRange):
            raise TypeError("byte_range must be ObjectByteRange or None")
        self._counters["get"] += 1
        self._maybe_fail("get")
        record = self._record(locator)
        if record is None:
            raise ObjectStoreError(ObjectStoreErrorCode.NOT_FOUND, "object is missing")
        head = self.head(locator)
        if head.visibility is ObjectVisibility.MISSING:
            raise ObjectStoreError(ObjectStoreErrorCode.NOT_FOUND, "object is missing")
        if head.visibility is not ObjectVisibility.VISIBLE:
            raise ObjectStoreError(
                ObjectStoreErrorCode.VISIBILITY_UNKNOWN,
                f"object is not visible: {head.visibility.value}",
            )
        payload = record.request.payload
        if (
            exact_bytes_sha256(payload) != record.request.sha256
            or len(payload) != record.request.byte_count
        ):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INTEGRITY_ERROR, "stored object bytes are corrupt"
            )
        return payload if byte_range is None else payload[byte_range.start : byte_range.end]

    def delete(self, locator: ObjectLocator) -> None:
        self._require_locator(locator)
        self._counters["delete"] += 1
        self._maybe_fail("delete")
        record = self._record(locator)
        if record is None:
            raise ObjectStoreError(ObjectStoreErrorCode.NOT_FOUND, "object is missing")
        record.deleted = True

    def reconcile(self, locator: ObjectLocator) -> ObjectHead:
        self._require_locator(locator)
        self._counters["reconcile"] += 1
        self._maybe_fail("reconcile")
        head = self.head(locator)
        if head.visibility is ObjectVisibility.VISIBLE:
            try:
                payload = self.get(locator)
            except ObjectStoreError:
                raise
            if exact_bytes_sha256(payload) != head.sha256 or len(payload) != head.byte_count:
                raise ObjectStoreError(
                    ObjectStoreErrorCode.INTEGRITY_ERROR,
                    "HEAD metadata disagrees with exact GET bytes",
                )
        return head

    def reconcile_all(self) -> ObjectReconciliationReport:
        """Inspect every stored object and classify visibility/retention/integrity."""

        issues: list[ObjectReconciliationIssue] = []
        visible = 0
        missing = 0
        locators = self.objects
        for locator in locators:
            try:
                head = self.reconcile(locator)
            except ObjectStoreError as error:
                issues.append(
                    ObjectReconciliationIssue(
                        key=locator.uri,
                        object_version=locator.object_version,
                        kind=error.code.value,
                        detail=str(error),
                    )
                )
                continue
            if head.visibility is ObjectVisibility.VISIBLE:
                visible += 1
            elif head.visibility is ObjectVisibility.MISSING:
                missing += 1
                issues.append(
                    ObjectReconciliationIssue(
                        key=locator.uri,
                        object_version=locator.object_version,
                        kind="MISSING",
                        detail="object is not retained",
                    )
                )
            else:
                issues.append(
                    ObjectReconciliationIssue(
                        key=locator.uri,
                        object_version=locator.object_version,
                        kind=head.visibility.value,
                        detail="object is not yet visibly readable",
                    )
                )
        return ObjectReconciliationReport(
            checked_count=len(locators),
            visible_count=visible,
            missing_count=missing,
            issue_count=len(issues),
            issues=tuple(issues),
        )

    def _receipt(self, record: _ObjectRecord, now: datetime) -> ObjectPutReceipt:
        visible = now >= record.visible_at
        return ObjectPutReceipt(
            locator=record.locator,
            sha256=record.request.sha256,
            byte_count=record.request.byte_count,
            visibility=(ObjectVisibility.VISIBLE if visible else ObjectVisibility.PARTIAL),
        )

    def _record(self, locator: ObjectLocator) -> _ObjectRecord | None:
        record = next(
            (
                value
                for value in self._objects.values()
                if value.locator.uri == locator.uri
                and value.locator.object_version == locator.object_version
                and not value.deleted
            ),
            None,
        )
        return record

    def _take_failure(self, operation: str) -> tuple[ObjectStoreErrorCode, str, bool] | None:
        return self._failures.pop(operation, None)

    def _maybe_fail(self, operation: str) -> None:
        failure = self._take_failure(operation)
        if failure is not None:
            raise ObjectStoreError(failure[0], failure[1])

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _same_payload(left: ObjectPutRequest, right: ObjectPutRequest) -> bool:
        return (
            left.sha256 == right.sha256
            and left.byte_count == right.byte_count
            and left.media_type == right.media_type
            and left.payload == right.payload
        )

    @staticmethod
    def _require_request(request: ObjectPutRequest) -> None:
        if not isinstance(request, ObjectPutRequest):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, "request must be ObjectPutRequest"
            )

    @staticmethod
    def _require_locator(locator: ObjectLocator) -> None:
        if not isinstance(locator, ObjectLocator):
            raise ObjectStoreError(
                ObjectStoreErrorCode.INVALID_REQUEST, "locator must be ObjectLocator"
            )


InMemoryObjectStore = FakeObjectStore
R2FakeObjectStore = FakeObjectStore

__all__ = [
    "FakeObjectStore",
    "InMemoryObjectStore",
    "ObjectReconciliationIssue",
    "ObjectReconciliationReport",
    "R2FakeObjectStore",
]
