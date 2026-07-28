"""Ports for asynchronous embedding/vector projection.

The default boundary is fail-closed.  ``InMemoryVectorProjectionStore`` is a
small deterministic substitute used by local contract/replay tests; it models
enqueue -> drain, duplicate writes, changed-vector conflicts, and cosine search
without requiring Postgres, Supabase, or pgvector.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from enum import StrEnum
from typing import NoReturn, Protocol

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
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
                if policy is not None and (tenant_id is None or policy.tenant_id != tenant_id):
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
                row for row in rows if row.request.subject.event_revision_id in candidate_revisions
            ]
        if candidate_subjects:
            rows = [row for row in rows if row.request.subject.projection_key in candidate_subjects]
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
            if request.subject == subject and request.embedding.embedding_id == embedding_id:
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


PgVectorSqlRow = Mapping[str, object] | Sequence[object]


class PgVectorSqlCursor(Protocol):
    """Narrow DB-API surface used by PgVectorProjectionStore."""

    def execute(self, query: str, params: Sequence[object] = ()) -> object: ...

    def fetchone(self) -> PgVectorSqlRow | None: ...

    def fetchall(self) -> Sequence[PgVectorSqlRow]: ...

    def close(self) -> object: ...


class PgVectorSqlConnection(Protocol):
    """Explicit transaction-capable connection used by the physical adapter."""

    def cursor(self) -> PgVectorSqlCursor: ...

    def commit(self) -> object: ...

    def rollback(self) -> object: ...

    def close(self) -> object: ...


_PG_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PG_SETTING = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_RETRYABLE_SQLSTATES = frozenset({"40001", "40P01", "55P03", "57014", "08000", "08003", "08006"})
_INDEX_SQLSTATES = frozenset({"0A000", "42703", "42704", "42883", "42P01"})
_PG_PROJECTION_COLUMNS = (
    "projection_id, idempotency_key, request_json::text, status, attempts, error"
)


class PgVectorProjectionStore:
    """Explicit PostgreSQL/pgvector implementation of VectorProjectionStore.

    The constructor accepts a connection factory instead of a URL so importing
    or constructing the adapter never selects a database implicitly. Each
    caller operation starts a short transaction, sets a tenant context, and
    applies a tenant predicate in addition to database RLS. Worker operations
    require a separate, explicitly named worker role when RLS is enabled.

    provision_schema() creates only the relation, pgvector column, indexes,
    and RLS flags. The deployment must apply rls_policy_statement() using
    its trusted tenant-claim mechanism and then pass verify_backend().
    """

    def __init__(
        self,
        connection_factory: Callable[[], PgVectorSqlConnection],
        *,
        dimension: int,
        relation: str = "robata_vector_projection",
        vector_column: str = "embedding",
        backend: VectorBackend = VectorBackend.POSTGRES,
        index_name: str | None = None,
        require_rls: bool = True,
        worker_connection_factory: Callable[[], PgVectorSqlConnection] | None = None,
        worker_role: str | None = None,
        rls_policy_name: str | None = None,
        tenant_context_setting: str = "robata.tenant_id",
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if backend not in {VectorBackend.POSTGRES, VectorBackend.SUPABASE}:
            raise ValueError("physical pgvector adapter requires POSTGRES or SUPABASE backend")
        if not isinstance(require_rls, bool):
            raise TypeError("require_rls must be bool")
        self._relation_parts = _parse_relation(relation)
        self._relation = ".".join(self._relation_parts)
        self._relation_sql = ".".join(_quote_identifier(part) for part in self._relation_parts)
        self._vector_column = _parse_identifier(vector_column, "vector_column")
        self._vector_column_sql = _quote_identifier(self._vector_column)
        self._backfill_relation_parts = (
            *self._relation_parts[:-1],
            _derived_identifier(self._relation_parts[-1], "backfill"),
        )
        self._backfill_relation_sql = ".".join(
            _quote_identifier(part) for part in self._backfill_relation_parts
        )
        self._dimension = dimension
        self._backend = backend
        self._index_name = (
            None if index_name is None else _parse_identifier(index_name, "index_name")
        )
        self._connection_factory = connection_factory
        self._worker_connection_factory = worker_connection_factory
        self._worker_role = (
            None if worker_role is None else _parse_identifier(worker_role, "worker_role")
        )
        self._rls_policy_name = (
            _derived_identifier(self._relation_parts[-1], "tenant_access")
            if rls_policy_name is None
            else _parse_identifier(rls_policy_name, "rls_policy_name")
        )
        if not isinstance(tenant_context_setting, str) or not _PG_SETTING.fullmatch(
            tenant_context_setting
        ):
            raise ValueError("tenant_context_setting must be a dotted lowercase PostgreSQL setting")
        self._tenant_context_setting = tenant_context_setting
        self._require_rls = require_rls

    @property
    def relation(self) -> str:
        """Configured physical relation, without a connection side effect."""

        return self._relation

    @property
    def dimension(self) -> int:
        """Dimension pinned by the physical pgvector column."""

        return self._dimension

    def ddl_statements(self) -> tuple[str, ...]:
        """Return auditable, non-executing base DDL for this adapter."""

        state_index = _quote_identifier(_derived_identifier(self._relation_parts[-1], "state"))
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            (
                f"CREATE TABLE IF NOT EXISTS {self._relation_sql} ("
                "projection_id TEXT PRIMARY KEY, "
                "idempotency_key TEXT NOT NULL UNIQUE, "
                "identity_scope_sha256 TEXT NOT NULL UNIQUE, "
                "subject_key TEXT NOT NULL, "
                "subject_type TEXT NOT NULL, "
                "event_revision_id TEXT NOT NULL, "
                "artifact_id TEXT NULL, "
                "embedding_id TEXT NOT NULL, "
                "tenant_id TEXT NULL, "
                "request_json JSONB NOT NULL, "
                f"{self._vector_column_sql} vector({self._dimension}) NOT NULL, "
                "status TEXT NOT NULL CHECK (status IN ('PENDING', 'READY', 'FAILED', 'RETIRED')), "
                "attempts INTEGER NOT NULL CHECK (attempts > 0), "
                "error TEXT NULL, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ")"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS {state_index} ON {self._relation_sql} "
                "(embedding_id, status, event_revision_id)"
            ),
            (
                f"CREATE TABLE IF NOT EXISTS {self._backfill_relation_sql} ("
                "target_key TEXT PRIMARY KEY, "
                "request_key TEXT NOT NULL, "
                "embedding_json JSONB NOT NULL, "
                "target_kind TEXT NOT NULL, "
                "target_id TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'PENDING', "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ")"
            ),
            f"ALTER TABLE {self._relation_sql} ENABLE ROW LEVEL SECURITY",
            f"ALTER TABLE {self._relation_sql} FORCE ROW LEVEL SECURITY",
        ]
        if self._index_name is not None:
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {_quote_identifier(self._index_name)} "
                f"ON {self._relation_sql} USING hnsw "
                f"({self._vector_column_sql} vector_cosine_ops)"
            )
        return tuple(statements)

    def rls_policy_statement(self) -> str:
        """Return the deployment-owned policy template required by this adapter."""

        setting = self._tenant_context_setting.replace("'", "''")
        predicate = (
            f"tenant_id IS NULL OR tenant_id = NULLIF(current_setting('{setting}', true), '')"
        )
        if self._worker_role is not None:
            worker_role = self._worker_role.replace("'", "''")
            predicate = f"({predicate}) OR pg_has_role(current_user, '{worker_role}', 'member')"
        policy = _quote_identifier(self._rls_policy_name)
        return (
            f"CREATE POLICY {policy} ON {self._relation_sql} FOR ALL "
            f"USING ({predicate}) WITH CHECK ({predicate})"
        )

    def provision_schema(self) -> None:
        """Execute only base pgvector relation and RLS-flag DDL.

        Calling this does not mark the adapter qualified: deployment still has
        to apply a tenant policy and execute verify_backend() on the target.
        """

        with self._transaction(tenant_id=None, verify_backend=False) as cursor:
            for statement in self.ddl_statements():
                cursor.execute(statement)

    def verify_backend(self) -> None:
        """Fail closed unless pgvector, dimension, and configured RLS exist."""

        with self._transaction(tenant_id=None, verify_backend=False) as cursor:
            self._verify_backend_cursor(cursor)

    def enqueue(self, request: VectorProjectionRequest) -> VectorProjectionReceipt:
        self._validate_request(request)
        tenant_id = self._request_tenant(request)
        with self._transaction(tenant_id=tenant_id) as cursor:
            existing = self._select_by_idempotency(cursor, request.idempotency_key, for_update=True)
            if existing is not None:
                return self._duplicate_receipt(existing, request)
            identity_scope = self._identity_scope_sha256(request)
            conflict = self._select_by_identity(cursor, identity_scope, for_update=True)
            if conflict is not None:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                    "vector subject/embedding identity was reused with different lineage",
                )
            cursor.execute(
                (
                    f"INSERT INTO {self._relation_sql} ("
                    "projection_id, idempotency_key, identity_scope_sha256, subject_key, "
                    "subject_type, event_revision_id, artifact_id, embedding_id, tenant_id, "
                    f"request_json, {self._vector_column_sql}, status, attempts, error"
                    ") VALUES ("
                    "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s, %s, %s"
                    ") ON CONFLICT (idempotency_key) DO NOTHING RETURNING idempotency_key"
                ),
                (
                    self._projection_id(request),
                    request.idempotency_key,
                    identity_scope,
                    request.subject.projection_key,
                    request.subject.subject_type.value,
                    request.subject.event_revision_id,
                    request.subject.artifact_id,
                    request.embedding.embedding_id,
                    tenant_id,
                    _canonical_request_json(request),
                    _pgvector_literal(request.vector),
                    VectorProjectionStatus.PENDING.value,
                    1,
                    None,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is None:
                # A hidden global key collision must not expose another tenant's
                # request or be mislabeled as a safe idempotent replay.
                existing = self._select_by_idempotency(
                    cursor,
                    request.idempotency_key,
                    for_update=True,
                )
                if existing is None:
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.RLS_DENIED,
                        "vector projection write was not visible to this tenant",
                    )
                return self._duplicate_receipt(existing, request)
        return self._receipt(
            request,
            status=VectorProjectionStatus.PENDING,
            duplicate=False,
            queued=True,
        )

    # Retain local adapter spellings without introducing a second contract.
    submit = enqueue
    publish = enqueue

    def upsert(self, request: VectorProjectionRequest) -> VectorProjectionReceipt:
        receipt = self.enqueue(request)
        if receipt.status is not VectorProjectionStatus.PENDING:
            return receipt
        with self._transaction(tenant_id=self._request_tenant(request)) as cursor:
            cursor.execute(
                (
                    f"UPDATE {self._relation_sql} "
                    "SET status = %s, updated_at = CURRENT_TIMESTAMP "
                    "WHERE idempotency_key = %s AND status = %s "
                    f"RETURNING {_PG_PROJECTION_COLUMNS}"
                ),
                (
                    VectorProjectionStatus.READY.value,
                    request.idempotency_key,
                    VectorProjectionStatus.PENDING.value,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                row = self._select_by_idempotency(cursor, request.idempotency_key, for_update=True)
            if row is None:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.RLS_DENIED,
                    "vector projection was not visible while completing upsert",
                )
            projection = self._projection_from_row(row)
            if not self._same_request(projection.request, request):
                raise VectorProjectionError(
                    VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                    "vector projection idempotency key was reused with different bytes",
                )
            return self._receipt(
                request,
                status=projection.status,
                duplicate=receipt.duplicate,
                queued=projection.status is VectorProjectionStatus.PENDING,
            )

    put = upsert

    def get(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        *,
        tenant_id: str | None = None,
    ) -> VectorProjection | None:
        self._validate_subject_lookup(subject, embedding_id, tenant_id)
        with self._transaction(tenant_id=tenant_id) as cursor:
            if tenant_id is None:
                tenant_clause = "tenant_id IS NULL"
                params: tuple[object, ...] = (subject.projection_key, embedding_id)
            else:
                tenant_clause = "(tenant_id = %s OR tenant_id IS NULL)"
                params = (subject.projection_key, embedding_id, tenant_id)
            cursor.execute(
                (
                    f"SELECT {_PG_PROJECTION_COLUMNS} "
                    f"FROM {self._relation_sql} WHERE subject_key = %s AND embedding_id = %s "
                    f"AND {tenant_clause} ORDER BY idempotency_key LIMIT 1"
                ),
                params,
            )
            row = cursor.fetchone()
            return None if row is None else self._projection_from_row(row)

    def search(self, query: VectorSearchQuery) -> tuple[VectorSearchHit, ...]:
        if not isinstance(query, VectorSearchQuery):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "search requires a VectorSearchQuery",
            )
        if len(query.query_vector) != self._dimension:
            raise VectorProjectionError(
                VectorProjectionErrorCode.DIMENSION_MISMATCH,
                "query vector dimension does not match configured pgvector column",
            )
        vector = _pgvector_literal(query.query_vector)
        distance = f"{self._vector_column_sql} <=> %s::vector"
        clauses = ["status = %s", "embedding_id = %s"]
        where_params: list[object] = [VectorProjectionStatus.READY.value, query.embedding_id]
        if query.tenant_id is None:
            # An unbound request can only see explicitly public rows. It is
            # never a backdoor around tenant-scoped RLS.
            clauses.append("tenant_id IS NULL")
        else:
            clauses.append("tenant_id = %s")
            where_params.append(query.tenant_id)
        if query.candidate_event_revision_ids:
            placeholders = ", ".join("%s" for _ in query.candidate_event_revision_ids)
            clauses.append(f"event_revision_id IN ({placeholders})")
            where_params.extend(query.candidate_event_revision_ids)
        if query.candidate_subject_ids:
            placeholders = ", ".join("%s" for _ in query.candidate_subject_ids)
            clauses.append(f"subject_key IN ({placeholders})")
            where_params.extend(query.candidate_subject_ids)
        if query.min_score is not None:
            clauses.append(f"1.0 - ({distance}) >= %s")
            where_params.extend((vector, query.min_score))
        statement = (
            "SELECT projection_id, idempotency_key, event_revision_id, artifact_id, embedding_id, "
            f"1.0 - ({distance}) AS score FROM {self._relation_sql} WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY {distance} ASC, event_revision_id ASC, COALESCE(artifact_id, '') ASC "
            "LIMIT %s"
        )
        params = (vector, *where_params, vector, query.limit)
        with self._transaction(tenant_id=query.tenant_id) as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()
        hits: list[VectorSearchHit] = []
        for rank, row in enumerate(rows):
            raw_score = _row_float(row, "score", 5)
            if raw_score < -1.000000001 or raw_score > 1.000000001:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.CONFLICT,
                    "pgvector cosine score is outside the valid range",
                )
            hits.append(
                VectorSearchHit(
                    projection_id=_row_text(row, "projection_id", 0),
                    event_revision_id=_row_text(row, "event_revision_id", 2),
                    artifact_id=_row_optional_text(row, "artifact_id", 3),
                    embedding_id=_row_text(row, "embedding_id", 4),
                    score=max(-1.0, min(1.0, raw_score)),
                    rank=rank,
                    locator=self._locator(_row_text(row, "idempotency_key", 1)),
                )
            )
        return tuple(hits)

    query = search

    def drain(self, limit: int | None = None) -> int:
        self._validate_limit(limit)
        limit_clause = "" if limit is None else " LIMIT %s"
        parameters: list[object] = [VectorProjectionStatus.PENDING.value]
        if limit is not None:
            parameters.append(limit)
        parameters.append(VectorProjectionStatus.READY.value)
        statement = (
            "WITH claimed AS ("
            f"SELECT idempotency_key FROM {self._relation_sql} "
            "WHERE status = %s ORDER BY idempotency_key"
            f"{limit_clause} FOR UPDATE SKIP LOCKED"
            ") "
            f"UPDATE {self._relation_sql} AS projection SET status = %s, "
            "updated_at = CURRENT_TIMESTAMP FROM claimed "
            "WHERE projection.idempotency_key = claimed.idempotency_key "
            "RETURNING projection.idempotency_key"
        )
        with self._worker_transaction() as cursor:
            cursor.execute(statement, tuple(parameters))
            return len(cursor.fetchall())

    def mark_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
        reason: str,
    ) -> VectorProjectionReceipt:
        self._validate_subject_lookup(subject, embedding_id, None)
        if not isinstance(reason, str) or not reason or len(reason) > 4096:
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "failure reason must be a non-empty string",
            )
        with self._worker_transaction() as cursor:
            row = self._select_by_subject(cursor, subject, embedding_id, for_update=True)
            if row is None:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.NOT_FOUND,
                    "vector projection was not found",
                )
            projection = self._projection_from_row(row)
            if projection.status is VectorProjectionStatus.FAILED:
                if projection.error != reason:
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.DUPLICATE_CONFLICT,
                        "FAILED vector projection reason cannot be mutated",
                    )
                return self._receipt(
                    projection.request,
                    status=projection.status,
                    duplicate=True,
                    queued=False,
                )
            if projection.status is VectorProjectionStatus.RETIRED:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.INVALID_REQUEST,
                    "RETIRED vector projections cannot be failed",
                )
            if projection.status is VectorProjectionStatus.READY:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.CONFLICT,
                    "READY vector projections cannot be failed",
                )
            cursor.execute(
                (
                    f"UPDATE {self._relation_sql} SET status = %s, error = %s, "
                    "updated_at = CURRENT_TIMESTAMP WHERE idempotency_key = %s"
                ),
                (VectorProjectionStatus.FAILED.value, reason, projection.idempotency_key),
            )
            return self._receipt(
                projection.request,
                status=VectorProjectionStatus.FAILED,
                duplicate=False,
                queued=False,
            )

    def retry_failed(
        self,
        subject: VectorProjectionSubject,
        embedding_id: str,
    ) -> VectorProjectionReceipt:
        self._validate_subject_lookup(subject, embedding_id, None)
        with self._worker_transaction() as cursor:
            row = self._select_by_subject(cursor, subject, embedding_id, for_update=True)
            if row is None:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.NOT_FOUND,
                    "vector projection was not found",
                )
            projection = self._projection_from_row(row)
            if projection.status is not VectorProjectionStatus.FAILED:
                raise VectorProjectionError(
                    VectorProjectionErrorCode.INVALID_REQUEST,
                    "only FAILED vector projections may be retried",
                )
            cursor.execute(
                (
                    f"UPDATE {self._relation_sql} SET status = %s, error = NULL, "
                    "attempts = attempts + 1, updated_at = CURRENT_TIMESTAMP "
                    "WHERE idempotency_key = %s"
                ),
                (VectorProjectionStatus.PENDING.value, projection.idempotency_key),
            )
            return self._receipt(
                projection.request,
                status=VectorProjectionStatus.PENDING,
                duplicate=True,
                queued=True,
            )

    def backfill(self, request: EmbeddingBackfillRequest) -> int:
        if not isinstance(request, EmbeddingBackfillRequest):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "backfill requires an EmbeddingBackfillRequest",
            )
        if request.embedding.dimension != self._dimension:
            raise VectorProjectionError(
                VectorProjectionErrorCode.DIMENSION_MISMATCH,
                "backfill embedding dimension does not match configured pgvector column",
            )
        if request.cursor is not None:
            # The port does not carry a structured-index cursor resolver. A
            # physical deployment must provide one explicitly rather than scan
            # a hidden cross-tenant source table from this optional adapter.
            raise VectorProjectionError(
                VectorProjectionErrorCode.INDEX_UNAVAILABLE,
                "cursor backfill requires a configured structured-index resolver",
            )
        targets = (
            *(("EVENT_REVISION", target) for target in request.event_revision_ids),
            *(("CLIP_ARTIFACT", target) for target in request.artifact_ids),
        )
        with self._worker_transaction() as cursor:
            for target_kind, target_id in targets:
                target_key = semantic_sha256(
                    {
                        "backfill_request": request.idempotency_key,
                        "target_kind": target_kind,
                        "target_id": target_id,
                    }
                )
                cursor.execute(
                    (
                        f"INSERT INTO {self._backfill_relation_sql} ("
                        "target_key, request_key, embedding_json, target_kind, target_id, status"
                        ") VALUES (%s, %s, %s::jsonb, %s, %s, %s) "
                        "ON CONFLICT (target_key) DO NOTHING"
                    ),
                    (
                        target_key,
                        request.idempotency_key,
                        canonical_json_bytes(request.embedding).decode("utf-8"),
                        target_kind,
                        target_id,
                        VectorProjectionStatus.PENDING.value,
                    ),
                )
        # This is a bounded requested population, not a claim that an encoder
        # has already produced vectors. Duplicate target rows remain safe.
        return len(targets)

    enqueue_backfill = backfill

    def _validate_request(self, request: VectorProjectionRequest) -> None:
        if not isinstance(request, VectorProjectionRequest):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "enqueue requires VectorProjectionRequest",
            )
        if request.embedding.dimension != self._dimension or len(request.vector) != self._dimension:
            raise VectorProjectionError(
                VectorProjectionErrorCode.DIMENSION_MISMATCH,
                "projection vector dimension does not match configured pgvector column",
            )
        if (
            self._require_rls
            and request.access_policy is not None
            and not request.access_policy.row_level_security_required
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.RLS_DENIED,
                "physical pgvector adapter requires row-level security for tenant rows",
            )

    @staticmethod
    def _validate_subject_lookup(
        subject: VectorProjectionSubject,
        embedding_id: str,
        tenant_id: str | None,
    ) -> None:
        if not isinstance(subject, VectorProjectionSubject) or (
            not isinstance(embedding_id, str) or not embedding_id
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "lookup requires a VectorProjectionSubject and non-empty embedding_id",
            )
        if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "tenant_id must be a non-empty string when supplied",
            )

    @staticmethod
    def _validate_limit(limit: int | None) -> None:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INVALID_REQUEST,
                "drain limit must be a non-negative integer",
            )

    @staticmethod
    def _same_request(left: VectorProjectionRequest, right: VectorProjectionRequest) -> bool:
        left_projection = left.model_dump(mode="json")
        right_projection = right.model_dump(mode="json")
        left_projection.pop("requested_at", None)
        right_projection.pop("requested_at", None)
        return left_projection == right_projection

    @staticmethod
    def _projection_id(request: VectorProjectionRequest) -> str:
        return f"pgvector:{request.idempotency_key}"

    @staticmethod
    def _request_tenant(request: VectorProjectionRequest) -> str | None:
        return None if request.access_policy is None else request.access_policy.tenant_id

    @staticmethod
    def _identity_scope_sha256(request: VectorProjectionRequest) -> str:
        subject = request.subject
        return semantic_sha256(
            {
                "subject_type": subject.subject_type.value,
                "event_revision_id": subject.event_revision_id,
                "artifact_id": subject.artifact_id,
                "embedding_id": request.embedding.embedding_id,
            }
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
            projection_id=PgVectorProjectionStore._projection_id(request),
            status=status,
            duplicate=duplicate,
            queued=queued,
        )

    def _duplicate_receipt(
        self,
        row: PgVectorSqlRow,
        request: VectorProjectionRequest,
    ) -> VectorProjectionReceipt:
        existing = self._projection_from_row(row)
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

    def _locator(self, row_key: str) -> VectorLocator:
        return VectorLocator(
            backend=self._backend,
            relation=self._relation,
            vector_column=self._vector_column,
            row_key=row_key,
            index_name=self._index_name,
        )

    def _projection_from_row(self, row: PgVectorSqlRow) -> VectorProjection:
        try:
            raw_request = _row_value(row, "request_json", 2)
            if isinstance(raw_request, bytes):
                raw_request = raw_request.decode("utf-8")
            if isinstance(raw_request, str):
                request = VectorProjectionRequest.model_validate_json(raw_request, strict=True)
            elif isinstance(raw_request, Mapping):
                request = VectorProjectionRequest.model_validate(raw_request, strict=True)
            else:
                raise TypeError("request_json is not JSON text or object")
            status = VectorProjectionStatus(_row_text(row, "status", 3))
            error = _row_optional_text(row, "error", 5)
            return VectorProjection(
                projection_id=_row_text(row, "projection_id", 0),
                request=request,
                status=status,
                locator=(
                    self._locator(_row_text(row, "idempotency_key", 1))
                    if status is VectorProjectionStatus.READY
                    else None
                ),
                error=error,
                attempts=_row_positive_int(row, "attempts", 4),
            )
        except VectorProjectionError:
            raise
        except Exception as error:
            raise VectorProjectionError(
                VectorProjectionErrorCode.CONFLICT,
                f"invalid persisted pgvector projection row: {error}",
            ) from error

    def _select_by_idempotency(
        self,
        cursor: PgVectorSqlCursor,
        key: str,
        *,
        for_update: bool,
    ) -> PgVectorSqlRow | None:
        cursor.execute(
            (
                f"SELECT {_PG_PROJECTION_COLUMNS} "
                f"FROM {self._relation_sql} WHERE idempotency_key = %s"
                + (" FOR UPDATE" if for_update else "")
            ),
            (key,),
        )
        return cursor.fetchone()

    def _select_by_identity(
        self,
        cursor: PgVectorSqlCursor,
        identity_scope: str,
        *,
        for_update: bool,
    ) -> PgVectorSqlRow | None:
        cursor.execute(
            (
                f"SELECT {_PG_PROJECTION_COLUMNS} "
                f"FROM {self._relation_sql} WHERE identity_scope_sha256 = %s"
                + (" FOR UPDATE" if for_update else "")
            ),
            (identity_scope,),
        )
        return cursor.fetchone()

    def _select_by_subject(
        self,
        cursor: PgVectorSqlCursor,
        subject: VectorProjectionSubject,
        embedding_id: str,
        *,
        for_update: bool,
    ) -> PgVectorSqlRow | None:
        cursor.execute(
            (
                f"SELECT {_PG_PROJECTION_COLUMNS} "
                f"FROM {self._relation_sql} WHERE subject_key = %s AND embedding_id = %s"
                + (" FOR UPDATE" if for_update else "")
            ),
            (subject.projection_key, embedding_id),
        )
        return cursor.fetchone()

    @contextmanager
    def _transaction(
        self,
        *,
        tenant_id: str | None,
        verify_backend: bool = True,
        worker: bool = False,
    ) -> Iterator[PgVectorSqlCursor]:
        connection: PgVectorSqlConnection | None = None
        cursor: PgVectorSqlCursor | None = None
        try:
            factory = self._connection_factory
            if worker:
                if self._require_rls and (
                    self._worker_connection_factory is None or self._worker_role is None
                ):
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.ADAPTER_UNAVAILABLE,
                        "RLS-enabled pgvector workers require an explicit "
                        "worker connection and role",
                    )
                factory = self._worker_connection_factory or self._connection_factory
            connection = factory()
            cursor = connection.cursor()
            cursor.execute("BEGIN")
            self._set_tenant_context(cursor, tenant_id)
            if verify_backend:
                self._verify_backend_cursor(cursor)
            if worker and self._worker_role is not None:
                cursor.execute(
                    "SELECT pg_has_role(current_user, %s, 'member') AS is_worker",
                    (self._worker_role,),
                )
                if not _row_bool(cursor.fetchone(), "is_worker", 0):
                    raise VectorProjectionError(
                        VectorProjectionErrorCode.RLS_DENIED,
                        "configured pgvector worker connection lacks the required database role",
                    )
            yield cursor
            connection.commit()
        except VectorProjectionError:
            _rollback_quietly(connection)
            raise
        except Exception as error:
            _rollback_quietly(connection)
            raise _map_pg_error(error) from error
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def _worker_transaction(self) -> AbstractContextManager[PgVectorSqlCursor]:
        return self._transaction(tenant_id=None, worker=True)

    def _set_tenant_context(self, cursor: PgVectorSqlCursor, tenant_id: str | None) -> None:
        cursor.execute(
            "SELECT set_config(%s, %s, true)",
            (self._tenant_context_setting, "" if tenant_id is None else tenant_id),
        )

    def _verify_backend_cursor(self, cursor: PgVectorSqlCursor) -> None:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector') "
            "AS vector_extension_installed"
        )
        if not _row_bool(cursor.fetchone(), "vector_extension_installed", 0):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INDEX_UNAVAILABLE,
                "pgvector extension is not installed",
            )
        cursor.execute(
            "SELECT c.relrowsecurity AS rls_enabled, "
            "c.relforcerowsecurity AS rls_forced, "
            "a.atttypid = 'vector'::regtype AS vector_column_valid, "
            "a.atttypmod - 4 AS vector_dimension "
            "FROM pg_catalog.pg_class AS c "
            "JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid "
            "WHERE c.oid = %s::regclass AND a.attname = %s AND NOT a.attisdropped",
            (self._relation, self._vector_column),
        )
        relation = cursor.fetchone()
        if relation is None:
            raise VectorProjectionError(
                VectorProjectionErrorCode.INDEX_UNAVAILABLE,
                "configured pgvector relation or vector column does not exist",
            )
        if not _row_bool(relation, "vector_column_valid", 2):
            raise VectorProjectionError(
                VectorProjectionErrorCode.INDEX_UNAVAILABLE,
                "configured pgvector column does not use the vector type",
            )
        if _row_int(relation, "vector_dimension", 3) != self._dimension:
            raise VectorProjectionError(
                VectorProjectionErrorCode.DIMENSION_MISMATCH,
                "configured pgvector column dimension differs from adapter configuration",
            )
        if not self._require_rls:
            return
        if not _row_bool(relation, "rls_enabled", 0) or not _row_bool(
            relation,
            "rls_forced",
            1,
        ):
            raise VectorProjectionError(
                VectorProjectionErrorCode.RLS_DENIED,
                "configured pgvector relation does not enforce row-level security",
            )
        cursor.execute(
            "SELECT COUNT(*) AS policy_count FROM pg_catalog.pg_policy "
            "WHERE polrelid = %s::regclass AND polname = %s",
            (self._relation, self._rls_policy_name),
        )
        if _row_int(cursor.fetchone(), "policy_count", 0) < 1:
            raise VectorProjectionError(
                VectorProjectionErrorCode.RLS_DENIED,
                "configured pgvector relation has no database RLS policy",
            )


def _parse_identifier(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not _PG_IDENTIFIER.fullmatch(value)
        or len(value.encode("utf-8")) > 63
    ):
        raise ValueError(f"{label} must be a PostgreSQL identifier of at most 63 bytes")
    return value


def _parse_relation(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("relation must be a string")
    parts = tuple(value.split("."))
    if len(parts) not in {1, 2}:
        raise ValueError("relation must be an unquoted table or schema.table identifier")
    return tuple(_parse_identifier(part, "relation component") for part in parts)


def _derived_identifier(base: str, suffix: str) -> str:
    suffix_value = f"_{_parse_identifier(suffix, 'suffix')}"
    available = 63 - len(suffix_value.encode("utf-8"))
    if available < 1:
        raise ValueError("derived PostgreSQL identifier has no base space")
    return f"{base[:available]}{suffix_value}"


def _quote_identifier(value: str) -> str:
    return f'"{value}"'


def _canonical_request_json(request: VectorProjectionRequest) -> str:
    return canonical_json_bytes(request).decode("utf-8")


def _pgvector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(float(item), ".17g") for item in vector) + "]"


def _row_value(row: PgVectorSqlRow, name: str, index: int) -> object:
    if isinstance(row, Mapping):
        if name not in row:
            raise ValueError(f"row does not include {name}")
        return row[name]
    if isinstance(row, (str, bytes)) or index >= len(row):
        raise ValueError(f"row does not include {name}")
    return row[index]


def _row_text(row: PgVectorSqlRow, name: str, index: int) -> str:
    value = _row_value(row, name, index)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {name} must be non-empty text")
    return value


def _row_optional_text(row: PgVectorSqlRow, name: str, index: int) -> str | None:
    value = _row_value(row, name, index)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {name} must be text or null")
    return value


def _row_int(row: PgVectorSqlRow | None, name: str, index: int) -> int:
    if row is None:
        raise ValueError(f"row does not include {name}")
    value = _row_value(row, name, index)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"row {name} must be integer")
    return value


def _row_positive_int(row: PgVectorSqlRow, name: str, index: int) -> int:
    value = _row_int(row, name, index)
    if value <= 0:
        raise ValueError(f"row {name} must be positive")
    return value


def _row_float(row: PgVectorSqlRow, name: str, index: int) -> float:
    value = _row_value(row, name, index)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"row {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row {name} must be finite")
    return result


def _row_bool(row: PgVectorSqlRow | None, name: str, index: int) -> bool:
    if row is None:
        raise ValueError(f"row does not include {name}")
    value = _row_value(row, name, index)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"row {name} must be boolean")


def _rollback_quietly(connection: PgVectorSqlConnection | None) -> None:
    if connection is None:
        return
    with suppress(Exception):
        connection.rollback()


def _close_quietly(value: object | None) -> None:
    if value is None:
        return
    closer = getattr(value, "close", None)
    if not callable(closer):
        return
    with suppress(Exception):
        closer()


def _map_pg_error(error: Exception) -> VectorProjectionError:
    state = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
    if state in _RETRYABLE_SQLSTATES:
        code = VectorProjectionErrorCode.RETRYABLE
    elif state in {"28000", "42501"}:
        code = VectorProjectionErrorCode.RLS_DENIED
    elif state == "23505":
        code = VectorProjectionErrorCode.DUPLICATE_CONFLICT
    elif state in _INDEX_SQLSTATES:
        code = VectorProjectionErrorCode.INDEX_UNAVAILABLE
    else:
        code = VectorProjectionErrorCode.ADAPTER_UNAVAILABLE
    return VectorProjectionError(code, str(error) or type(error).__name__)


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
