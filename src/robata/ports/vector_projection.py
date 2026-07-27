"""Ports for asynchronous embedding/vector projection.

The default boundary is fail-closed.  ``InMemoryVectorProjectionStore`` is a
small deterministic substitute used by local contract/replay tests; it models
enqueue -> drain, duplicate writes, changed-vector conflicts, and cosine search
without requiring Postgres, Supabase, or pgvector.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import NoReturn, Protocol

from robata.contracts.retrieval import (
    EmbeddingBackfillRequest,
    EmbeddingEncoderInput,
    EncodedEmbedding,
    VectorBackend,
    VectorLocator,
    VectorProjection,
    VectorProjectionReceipt,
    VectorProjectionRequest,
    VectorProjectionStatus,
    VectorProjectionSubject,
    VectorSearchHit,
    VectorSearchQuery,
)


class VectorProjectionErrorCode(StrEnum):
    ADAPTER_UNAVAILABLE = "ADAPTER_UNAVAILABLE"
    ENCODER_UNAVAILABLE = "ENCODER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    DUPLICATE_CONFLICT = "DUPLICATE_CONFLICT"
    CONFLICT = "CONFLICT"
    DIMENSION_MISMATCH = "DIMENSION_MISMATCH"
    INDEX_UNAVAILABLE = "INDEX_UNAVAILABLE"
    RLS_DENIED = "RLS_DENIED"
    RETRYABLE = "RETRYABLE"
    RETENTION_UNSUPPORTED = "RETENTION_UNSUPPORTED"


class VectorProjectionError(RuntimeError):
    """Stable error at the optional vector projection boundary."""

    def __init__(self, code: VectorProjectionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class VectorProjectionStore(Protocol):
    """Async/idempotent storage/search boundary over a vector backend."""

    def enqueue(self, request: VectorProjectionRequest) -> VectorProjectionReceipt: ...

    def upsert(self, request: VectorProjectionRequest) -> VectorProjectionReceipt: ...

    def get(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        *,
        tenant_id: str | None = None,
    ) -> VectorProjection | None: ...

    def search(self, query: VectorSearchQuery) -> tuple[VectorSearchHit, ...]: ...

    def drain(self, limit: int | None = None) -> int: ...

    def backfill(self, request: EmbeddingBackfillRequest) -> int: ...

    def mark_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        reason: str,
    ) -> VectorProjectionReceipt: ...

    def retry_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
    ) -> VectorProjectionReceipt: ...


class EmbeddingEncoder(Protocol):
    """Explicit CPU/API/RunPod encoder boundary; pgvector does not encode."""

    def encode(self, request: EmbeddingEncoderInput) -> EncodedEmbedding: ...


class FailClosedVectorProjectionStore:
    """No cloud/database adapter is selected implicitly."""

    @staticmethod
    def _unavailable() -> NoReturn:
        raise VectorProjectionError(
            VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
            "vector projection adapter is not configured",
        )

    def enqueue(self, request: VectorProjectionRequest) -> VectorProjectionReceipt:
        del request
        self._unavailable()

    def upsert(self, request: VectorProjectionRequest) -> VectorProjectionReceipt:
        del request
        self._unavailable()

    def get(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        *,
        tenant_id: str | None = None,
    ) -> VectorProjection | None:
        del subject, embedding_id, tenant_id
        self._unavailable()

    def search(self, query: VectorSearchQuery) -> tuple[VectorSearchHit, ...]:
        del query
        self._unavailable()

    def drain(self, limit: int | None = None) -> int:
        del limit
        self._unavailable()

    def backfill(self, request: EmbeddingBackfillRequest) -> int:
        del request
        self._unavailable()

    def mark_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        reason: str,
    ) -> VectorProjectionReceipt:
        del subject, embedding_id, reason
        self._unavailable()

    def retry_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
    ) -> VectorProjectionReceipt:
        del subject, embedding_id
        self._unavailable()

    publish = enqueue
    put = upsert
    query = search
    enqueue_backfill = backfill


class FailClosedEmbeddingEncoder:
    """Encoder must be explicitly configured; storage never creates vectors."""

    @staticmethod
    def encode(request: EmbeddingEncoderInput) -> EncodedEmbedding:
        del request
        raise VectorProjectionError(
            VectorProjectionErrorCode.ENCODER_UNAVAILABLE,
            "embedding encoder is not configured",
        )


class InMemoryVectorProjectionStore:
    """Deterministic local fake for async vector writes and replay tests."""

    def __init__(self) -> None:
        self._pending: dict[str, VectorProjectionRequest] = {}
        self._attempts: dict[str, int] = {}
        self._rows: dict[str, VectorProjection] = {}

    @staticmethod
    def _projection_id(request: VectorProjectionRequest) -> str:
        return f"vector:{request.idempotency_key}"

    @staticmethod
    def _same_request(left: VectorProjectionRequest, right: VectorProjectionRequest) -> bool:
        # ``requested_at`` is an enqueue/replay observation, not semantic
        # vector content.  A retry at a later wall-clock time must therefore
        # reuse the same row; all other request facts remain immutable and a
        # changed vector/policy is a duplicate conflict.
        left_projection = left.model_dump(mode="json")
        right_projection = right.model_dump(mode="json")
        left_projection.pop("requested_at", None)
        right_projection.pop("requested_at", None)
        return left_projection == right_projection

    @staticmethod
    def _identity_scope(request: VectorProjectionRequest) -> tuple[str, str, str, str]:
        subject = request.subject
        # Keep identity components separate; concatenating IDs with a delimiter
        # would permit ambiguous collisions when a caller's opaque ID contains
        # that delimiter.
        return (
            subject.subject_type.value,
            subject.event_revision_id,
            subject.artifact_id or "",
            request.embedding.embedding_id,
        )

    @classmethod
    def _ensure_identity_not_conflicted(
        cls,
        request: VectorProjectionRequest,
        *,
        key: str,
        pending: dict[str, VectorProjectionRequest],
        rows: dict[str, VectorProjection],
    ) -> None:
        scope = cls._identity_scope(request)
        for existing_key, existing_request in pending.items():
            if existing_key != key and cls._identity_scope(existing_request) == scope:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                    "vector subject/embedding identity was reused with different lineage",
                )
        for existing_key, row in rows.items():
            if existing_key != key and cls._identity_scope(row.request) == scope:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                    "vector subject/embedding identity was reused with different lineage",
                )

    @staticmethod
    def _receipt(
        request: VectorProjectionRequest,
        *,
        status: VectorProjectionStatus,
        duplicate: bool,
        queued: bool,
    ) -> VectorProjectionReceipt:
        return VectorProjectionReceipt(
            idempotency_key=request.idempotency_key,
            projection_id=InMemoryVectorProjectionStore._projection_id(request),
            status=status,
            duplicate=duplicate,
            queued=queued,
        )

    def enqueue(self, request: VectorProjectionRequest) -> VectorProjectionReceipt:
        if not isinstance(request, VectorProjectionRequest):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "enqueue requires VectorProjectionRequest",
            )
        key = request.idempotency_key
        self._ensure_identity_not_conflicted(
            request,
            key=key,
            pending=self._pending,
            rows=self._rows,
        )
        existing_request = self._pending.get(key)
        if existing_request is not None:
            if not self._same_request(existing_request, request):
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                    "vector projection idempotency key was reused with different bytes",
                )
            return self._receipt(
                request,
                status=VectorProjectionStatus.PENDING,
                duplicate=True,
                queued=True,
            )
        existing = self._rows.get(key)
        if existing is not None:
            if not self._same_request(existing.request, request):
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                    "vector projection idempotency key was reused with different bytes",
                )
            return self._receipt(
                request,
                status=existing.status,
                duplicate=True,
                queued=existing.status is VectorProjectionStatus.PENDING,
            )
        self._pending[key] = request
        self._attempts[key] = 1
        return self._receipt(
            request,
            status=VectorProjectionStatus.PENDING,
            duplicate=False,
            queued=True,
        )

    # ``submit`` is a descriptive alias used by a few local callers.
    submit = enqueue

    def upsert(self, request: VectorProjectionRequest) -> VectorProjectionReceipt:
        receipt = self.enqueue(request)
        if receipt.status is VectorProjectionStatus.PENDING:
            self.drain()
            return self._receipt(
                request,
                status=VectorProjectionStatus.READY,
                duplicate=receipt.duplicate,
                queued=False,
            )
        return receipt

    def get(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        *,
        tenant_id: str | None = None,
    ) -> VectorProjection | None:
        if not isinstance(subject, VectorProjectionSubject) or (
            not isinstance(embedding_id, str) or not embedding_id
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "get requires a VectorProjectionSubject and non-empty embedding_id",
            )
        if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "tenant_id must be a non-empty string when supplied",
            )
        for row in self._rows.values():
            if (
                row.request.subject == subject
                and row.request.embedding.embedding_id == embedding_id
            ):
                policy = row.request.access_policy
                if policy is not None and (
                    tenant_id is None or policy.tenant_id != tenant_id
                ):
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.RLS_DENIED,
                        "vector projection is not visible to this tenant",
                    )
                return row
        return None

    def search(self, query: VectorSearchQuery) -> tuple[VectorSearchHit, ...]:
        if not isinstance(query, VectorSearchQuery):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "search requires VectorSearchQuery",
            )
        rows = [
            row
            for row in self._rows.values()
            if row.status is VectorProjectionStatus.READY
            and row.request.embedding.embedding_id == query.embedding_id
            and (
                # A tenant-bound query can only see rows carrying the same
                # policy.  An unbound query is deliberately limited to rows
                # without an access policy; protected rows fail closed rather
                # than becoming a cross-tenant admin shortcut.
                (
                    query.tenant_id is not None
                    and row.request.access_policy is not None
                    and row.request.access_policy.tenant_id == query.tenant_id
                )
                or (query.tenant_id is None and row.request.access_policy is None)
            )
        ]
        candidate_revisions = set(query.candidate_event_revision_ids)
        candidate_subjects = set(query.candidate_subject_ids)
        if candidate_revisions:
            rows = [
                row
                for row in rows
                if row.request.subject.event_revision_id in candidate_revisions
            ]
        if candidate_subjects:
            rows = [
                row for row in rows if row.request.subject.projection_key in candidate_subjects
            ]
        # Dimension mismatches are adapter errors, not silent zero-score rows.
        for row in rows:
            if len(row.request.vector) != len(query.query_vector):
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DIMENSION_MISMATCH,
                    "query vector dimension does not match stored embedding",
                )
        query_norm = math.sqrt(sum(item * item for item in query.query_vector))
        scored: list[tuple[VectorProjection, float]] = []
        for row in rows:
            vector = row.request.vector
            row_norm = math.sqrt(sum(item * item for item in vector))
            score = 0.0
            if query_norm and row_norm:
                score = sum(a * b for a, b in zip(query.query_vector, vector, strict=True)) / (
                    query_norm * row_norm
                )
            if query.min_score is None or score >= query.min_score:
                scored.append((row, score))
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].request.subject.event_revision_id,
                item[0].request.subject.artifact_id or "",
            )
        )
        return tuple(
            VectorSearchHit(
                projection_id=row.projection_id,
                event_revision_id=row.request.subject.event_revision_id,
                artifact_id=row.request.subject.artifact_id,
                embedding_id=row.request.embedding.embedding_id,
                score=score,
                rank=rank,
                locator=row.locator,
            )
            for rank, (row, score) in enumerate(scored[: query.limit])
        )

    def drain(self, limit: int | None = None) -> int:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "drain limit must be a non-negative integer",
            )
        keys = list(self._pending)
        if limit is not None:
            keys = keys[:limit]
        for key in keys:
            request = self._pending.pop(key)
            self._rows[key] = VectorProjection(
                projection_id=self._projection_id(request),
                request=request,
                status=VectorProjectionStatus.READY,
                attempts=self._attempts.get(key, 1),
                locator=VectorLocator(
                    backend=VectorBackend.MEMORY,
                    relation="vector_projection",
                    row_key=key,
                ),
            )
        return len(keys)

    def mark_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        reason: str,
    ) -> VectorProjectionReceipt:
        if not isinstance(subject, VectorProjectionSubject) or (
            not isinstance(embedding_id, str) or not embedding_id
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "mark_failed requires a VectorProjectionSubject and non-empty embedding_id",
            )
        if not isinstance(reason, str) or not reason or len(reason) > 4096:
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "failure reason must be a non-empty string",
            )
        for _key, row in tuple(self._rows.items()):
            if (
                row.request.subject == subject
                and row.request.embedding.embedding_id == embedding_id
            ):
                if row.status is VectorProjectionStatus.FAILED:
                    if row.error != reason:
                        raise VectorProjectionError(
                            VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                            "FAILED vector projection reason cannot be mutated",
                        )
                    return VectorProjectionReceipt(
                        idempotency_key=row.idempotency_key,
                        projection_id=row.projection_id,
                        status=row.status,
                        duplicate=True,
                        queued=False,
                    )
                if row.status is VectorProjectionStatus.RETIRED:
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.INVALID_REQUEST,
                        "RETIRED vector projections cannot be failed",
                    )
                raise VectorProjectionError(
                    VectorProjectionErrorCode.CONFLICT,
                    "READY vector projections cannot be failed",
                )
        # A worker may report failure after enqueue but before the local fake
        # drains its pending queue. Materialize that failure as a durable row
        # so retry_failed() and attempts remain observable and replay-safe.
        for key, request in tuple(self._pending.items()):
            if (
                request.subject == subject
                and request.embedding.embedding_id == embedding_id
            ):
                self._pending.pop(key)
                failed = VectorProjection(
                    projection_id=self._projection_id(request),
                    request=request,
                    status=VectorProjectionStatus.FAILED,
                    error=reason,
                    attempts=self._attempts.get(key, 1),
                )
                self._rows[key] = failed
                return VectorProjectionReceipt(
                    idempotency_key=failed.idempotency_key,
                    projection_id=failed.projection_id,
                    status=failed.status,
                    duplicate=False,
                    queued=False,
                )
        raise VectorProjectionError(
            VectorProjectionErrorCode.NOT_FOUND,
            "vector projection was not found",
        )

    def retry_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
    ) -> VectorProjectionReceipt:
        if not isinstance(subject, VectorProjectionSubject) or (
            not isinstance(embedding_id, str) or not embedding_id
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "retry_failed requires a VectorProjectionSubject and non-empty embedding_id",
            )
        for key, row in tuple(self._rows.items()):
            if (
                row.request.subject == subject
                and row.request.embedding.embedding_id == embedding_id
            ):
                if row.status is not VectorProjectionStatus.FAILED:
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.INVALID_REQUEST,
                        "only FAILED vector projections may be retried",
                    )
                self._rows.pop(key)
                self._pending[key] = row.request
                self._attempts[key] = row.attempts + 1
                return self._receipt(
                    row.request,
                    status=VectorProjectionStatus.PENDING,
                    duplicate=True,
                    queued=True,
                )
        raise VectorProjectionError(
            VectorProjectionErrorCode.NOT_FOUND,
            "vector projection was not found",
        )

    def backfill(self, request: EmbeddingBackfillRequest) -> int:
        """Return target count; actual encoding remains an explicit adapter concern."""

        if not isinstance(request, EmbeddingBackfillRequest):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "backfill requires EmbeddingBackfillRequest",
            )
        if request.cursor is not None:
            raise VectorProjectionError(
                VectorProjectionErrorCode.INDEX_UNAVAILABLE,
                "cursor backfill requires a configured structured index",
            )
        return len(request.event_revision_ids) + len(request.artifact_ids)

    # Adapter spelling aliases retain the same idempotency semantics.
    publish = enqueue
    put = upsert
    query = search
    enqueue_backfill = backfill

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def ready_count(self) -> int:
        return sum(row.status is VectorProjectionStatus.READY for row in self._rows.values())


# Common adapter spellings; aliases do not create a second wire version.
VectorProjectionPort = VectorProjectionStore
VectorStore = VectorProjectionStore


class PgVectorProjectionStore(FailClosedVectorProjectionStore):
    """Explicit placeholder until a Postgres/Supabase adapter is qualified.

    Instantiating this name is safe but every operation fails closed rather than
    silently selecting a database or claiming pgvector support.
    """


InMemoryVectorStore = InMemoryVectorProjectionStore


__all__ = [
    "EmbeddingEncoder",
    "FailClosedEmbeddingEncoder",
    "FailClosedVectorProjectionStore",
    "InMemoryVectorProjectionStore",
    "InMemoryVectorStore",
    "PgVectorProjectionStore",
    "VectorProjectionError",
    "VectorProjectionErrorCode",
    "VectorProjectionPort",
    "VectorProjectionStore",
    "VectorStore",
]



