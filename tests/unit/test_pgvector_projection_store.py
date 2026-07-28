"""Unit contract tests for the explicit PostgreSQL/pgvector adapter.

These tests intentionally use a recording DB-API double instead of a local
Postgres server. They assert the physical SQL boundary and fail-closed error
mapping without claiming a real pgvector or RLS qualification run.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.retrieval import (
    EmbeddingBackfillRequest,
    EmbeddingModality,
    EmbeddingNormalization,
    EmbeddingProvider,
    EmbeddingSpec,
    VectorAccessPolicy,
    VectorProjectionRequest,
    VectorProjectionStatus,
    VectorProjectionSubject,
    VectorRetentionPolicy,
    VectorSearchQuery,
)
from robata.ports.vector_projection import (
    PgVectorProjectionStore,
    VectorProjectionError,
    VectorProjectionErrorCode,
)


class _RecordingCursor:
    def __init__(
        self,
        *,
        one_results: Sequence[object | None] = (),
        all_results: Sequence[Sequence[object]] = (),
        execute_error: Exception | None = None,
    ) -> None:
        self._one_results = list(one_results)
        self._all_results = [tuple(items) for items in all_results]
        self._execute_error = execute_error
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    def execute(self, query: str, params: Sequence[object] = ()) -> None:
        self.statements.append((query, tuple(params)))
        if self._execute_error is not None:
            raise self._execute_error

    def fetchone(self) -> object | None:
        return self._one_results.pop(0) if self._one_results else None

    def fetchall(self) -> tuple[object, ...]:
        return self._all_results.pop(0) if self._all_results else ()

    def close(self) -> None:
        self.closed = True


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> _RecordingCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class _RetryablePgError(RuntimeError):
    sqlstate = "40001"


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        embedding_id="physical-text-v1",
        model="deterministic-test",
        model_version="1.0",
        modality=EmbeddingModality.TEXT,
        dimension=3,
        normalization=EmbeddingNormalization.NONE,
        encoder_provider=EmbeddingProvider.CPU,
        index_policy_version="hnsw-v1",
    )


def _request(
    *,
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
    access_policy: VectorAccessPolicy | None = None,
) -> VectorProjectionRequest:
    return VectorProjectionRequest(
        subject=VectorProjectionSubject(event_revision_id="revision-1"),
        embedding=_spec(),
        vector=vector,
        retention=VectorRetentionPolicy(retention_policy_version="retention-v1", ttl_days=7),
        access_policy=access_policy,
    )


def _verified_responses(*after: object | None) -> tuple[object | None, ...]:
    # RLS is disabled only for the recording executor; real default instances
    # additionally require row security enabled, forced, and policy-backed.
    return (
        (True,),
        (False, False, True, 3),
        *after,
    )


def _store(
    connection: _RecordingConnection,
    *,
    require_rls: bool = False,
    worker_connection: _RecordingConnection | None = None,
    worker_role: str | None = None,
) -> PgVectorProjectionStore:
    return PgVectorProjectionStore(
        lambda: connection,
        dimension=3,
        relation="public.event_vector_projection",
        index_name="event_vector_projection_hnsw",
        require_rls=require_rls,
        worker_connection_factory=(
            None if worker_connection is None else lambda: worker_connection
        ),
        worker_role=worker_role,
    )


def _stored_row(
    request: VectorProjectionRequest,
    *,
    status: VectorProjectionStatus,
    error: str | None = None,
) -> tuple[object, ...]:
    return (
        f"pgvector:{request.idempotency_key}",
        request.idempotency_key,
        canonical_json_bytes(request).decode("utf-8"),
        status.value,
        1,
        error,
    )


def _matching_statement(cursor: _RecordingCursor, prefix: str) -> tuple[str, tuple[object, ...]]:
    return next(item for item in cursor.statements if item[0].startswith(prefix))


def test_pgvector_constructor_is_explicit_and_ddl_keeps_rls_deployment_owned() -> None:
    cursor = _RecordingCursor()
    connection = _RecordingConnection(cursor)

    with pytest.raises(TypeError, match="connection_factory"):
        PgVectorProjectionStore(None, dimension=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="relation component"):
        PgVectorProjectionStore(lambda: connection, dimension=3, relation="public; DROP TABLE x")

    store = _store(connection, require_rls=True, worker_role="robata_vector_worker")
    ddl = store.ddl_statements()
    assert any(statement == "CREATE EXTENSION IF NOT EXISTS vector" for statement in ddl)
    assert any('"embedding" vector(3)' in statement for statement in ddl)
    assert any("ENABLE ROW LEVEL SECURITY" in statement for statement in ddl)
    assert any("FORCE ROW LEVEL SECURITY" in statement for statement in ddl)
    assert "CREATE POLICY" in store.rls_policy_statement()
    assert "pg_has_role" in store.rls_policy_statement()
    assert cursor.statements == []


def test_enqueue_records_tenant_scoped_pgvector_sql_and_idempotency() -> None:
    request = _request(
        access_policy=VectorAccessPolicy(policy_version="tenant-v1", tenant_id="tenant-a")
    )
    cursor = _RecordingCursor(
        one_results=_verified_responses(None, None, (request.idempotency_key,))
    )
    connection = _RecordingConnection(cursor)

    receipt = _store(connection).enqueue(request)

    assert receipt.status is VectorProjectionStatus.PENDING
    assert receipt.duplicate is False
    assert connection.commits == 1
    setting = _matching_statement(cursor, "SELECT set_config")
    assert setting[1] == ("robata.tenant_id", "tenant-a")
    insert = _matching_statement(cursor, "INSERT INTO")
    assert "ON CONFLICT (idempotency_key) DO NOTHING" in insert[0]
    assert insert[1][8] == "tenant-a"
    assert insert[1][10] == "[1,0,0]"


def test_search_binds_pgvector_parameters_in_sql_order_and_unbound_is_public_only() -> None:
    bound_cursor = _RecordingCursor(
        one_results=_verified_responses(),
        all_results=[
            (
                (
                    "pgvector:one",
                    "one",
                    "revision-1",
                    None,
                    "physical-text-v1",
                    0.9,
                ),
            )
        ],
    )
    bound = _store(_RecordingConnection(bound_cursor))
    result = bound.search(
        VectorSearchQuery(
            embedding_id="physical-text-v1",
            tenant_id="tenant-a",
            query_vector=(1.0, 0.0, 0.0),
            candidate_event_revision_ids=("revision-1",),
            limit=1,
        )
    )

    assert result[0].event_revision_id == "revision-1"
    search = _matching_statement(bound_cursor, "SELECT projection_id")
    assert "tenant_id = %s" in search[0]
    assert search[1] == (
        "[1,0,0]",
        VectorProjectionStatus.READY.value,
        "physical-text-v1",
        "tenant-a",
        "revision-1",
        "[1,0,0]",
        1,
    )

    public_cursor = _RecordingCursor(one_results=_verified_responses(), all_results=[()])
    public = _store(_RecordingConnection(public_cursor))
    assert (
        public.search(
            VectorSearchQuery(
                embedding_id="physical-text-v1",
                query_vector=(1.0, 0.0, 0.0),
            )
        )
        == ()
    )
    public_search = _matching_statement(public_cursor, "SELECT projection_id")
    assert "tenant_id IS NULL" in public_search[0]
    assert "tenant_id = %s" not in public_search[0]
    assert _matching_statement(public_cursor, "SELECT set_config")[1] == (
        "robata.tenant_id",
        "",
    )


def test_get_and_idempotency_conflict_stay_tenant_scoped_on_physical_rows() -> None:
    request = _request(
        access_policy=VectorAccessPolicy(policy_version="tenant-v1", tenant_id="tenant-a")
    )
    get_cursor = _RecordingCursor(
        one_results=_verified_responses(_stored_row(request, status=VectorProjectionStatus.READY))
    )
    projection = _store(_RecordingConnection(get_cursor)).get(
        request.subject,
        request.embedding.embedding_id,
        tenant_id="tenant-a",
    )
    assert projection is not None
    assert projection.status is VectorProjectionStatus.READY
    get_statement = _matching_statement(get_cursor, "SELECT projection_id")
    assert "(tenant_id = %s OR tenant_id IS NULL)" in get_statement[0]
    assert get_statement[1][-1] == "tenant-a"

    changed = _request(
        vector=(0.0, 1.0, 0.0),
        access_policy=VectorAccessPolicy(policy_version="tenant-v1", tenant_id="tenant-a"),
    )
    conflict_cursor = _RecordingCursor(
        one_results=_verified_responses(_stored_row(request, status=VectorProjectionStatus.PENDING))
    )
    with pytest.raises(VectorProjectionError) as conflict:
        _store(_RecordingConnection(conflict_cursor)).enqueue(changed)
    assert conflict.value.code is VectorProjectionErrorCode.DUPLICATE_CONFLICT


def test_upsert_and_failed_lifecycle_keep_projection_contract_on_physical_rows() -> None:
    request = _request()
    upsert_cursor = _RecordingCursor(
        one_results=_verified_responses(
            None,
            None,
            (request.idempotency_key,),
            *_verified_responses(_stored_row(request, status=VectorProjectionStatus.READY)),
        )
    )
    upsert_connection = _RecordingConnection(upsert_cursor)

    receipt = _store(upsert_connection).upsert(request)

    assert receipt.status is VectorProjectionStatus.READY
    assert _matching_statement(upsert_cursor, "UPDATE")[1][0] == VectorProjectionStatus.READY.value

    pending_row = _stored_row(request, status=VectorProjectionStatus.PENDING)
    failed_cursor = _RecordingCursor(one_results=_verified_responses(pending_row))
    failed_receipt = _store(_RecordingConnection(failed_cursor)).mark_failed(
        request.subject,
        request.embedding.embedding_id,
        "adapter offline",
    )
    assert failed_receipt.status is VectorProjectionStatus.FAILED
    failed_update = _matching_statement(failed_cursor, "UPDATE")
    assert failed_update[1] == (
        VectorProjectionStatus.FAILED.value,
        "adapter offline",
        request.idempotency_key,
    )

    failed_row = _stored_row(
        request,
        status=VectorProjectionStatus.FAILED,
        error="adapter offline",
    )
    retry_cursor = _RecordingCursor(one_results=_verified_responses(failed_row))
    retry_receipt = _store(_RecordingConnection(retry_cursor)).retry_failed(
        request.subject,
        request.embedding.embedding_id,
    )
    assert retry_receipt.status is VectorProjectionStatus.PENDING
    assert _matching_statement(retry_cursor, "UPDATE")[1] == (
        VectorProjectionStatus.PENDING.value,
        request.idempotency_key,
    )


def test_worker_drain_and_backfill_are_explicit_and_cursor_backfill_fails_closed() -> None:
    cursor = _RecordingCursor(
        one_results=_verified_responses(),
        all_results=[(("one",), ("two",))],
    )
    store = _store(_RecordingConnection(cursor))
    assert store.drain(limit=2) == 2
    drain = _matching_statement(cursor, "WITH claimed")
    assert "FOR UPDATE SKIP LOCKED" in drain[0]
    assert drain[1] == (VectorProjectionStatus.PENDING.value, 2, VectorProjectionStatus.READY.value)

    backfill_cursor = _RecordingCursor(one_results=_verified_responses())
    backfill = _store(_RecordingConnection(backfill_cursor))
    request = EmbeddingBackfillRequest(
        embedding=_spec(),
        event_revision_ids=("revision-1",),
        artifact_ids=("artifact-1",),
    )
    assert backfill.backfill(request) == 2
    inserts = [item for item in backfill_cursor.statements if item[0].startswith("INSERT INTO")]
    assert len(inserts) == 2
    assert "ON CONFLICT (target_key) DO NOTHING" in inserts[0][0]

    with pytest.raises(VectorProjectionError) as missing_resolver:
        backfill.backfill(
            EmbeddingBackfillRequest(
                embedding=_spec(),
                cursor="structured-index-cursor-1",
            )
        )
    assert missing_resolver.value.code is VectorProjectionErrorCode.INDEX_UNAVAILABLE


def test_rls_and_database_error_paths_fail_closed() -> None:
    accepted_cursor = _RecordingCursor(
        one_results=[
            (True,),
            (True, True, True, 3),
            (1,),
        ]
    )
    _store(_RecordingConnection(accepted_cursor), require_rls=True).verify_backend()
    policy_check = _matching_statement(accepted_cursor, "SELECT COUNT(*)")
    assert policy_check[1] == (
        "public.event_vector_projection",
        "event_vector_projection_tenant_access",
    )

    rls_cursor = _RecordingCursor(
        one_results=[
            (True,),
            (False, True, True, 3),
        ]
    )
    rls_connection = _RecordingConnection(rls_cursor)
    with pytest.raises(VectorProjectionError) as rls_error:
        _store(rls_connection, require_rls=True).verify_backend()
    assert rls_error.value.code is VectorProjectionErrorCode.RLS_DENIED
    assert rls_connection.rollbacks == 1
    worker_executor = _RecordingCursor(
        one_results=[
            (True,),
            (True, True, True, 3),
            (1,),
            (True,),
        ],
        all_results=[(("worker-row",),)],
    )
    worker_connection = _RecordingConnection(worker_executor)
    configured_worker = _store(
        _RecordingConnection(_RecordingCursor()),
        require_rls=True,
        worker_connection=worker_connection,
        worker_role="robata_vector_worker",
    )
    assert configured_worker.drain() == 1
    assert any(
        statement.startswith("SELECT pg_has_role")
        for statement, _params in worker_executor.statements
    )

    worker_cursor = _RecordingCursor()
    worker_store = _store(_RecordingConnection(worker_cursor), require_rls=True)
    with pytest.raises(VectorProjectionError) as worker_error:
        worker_store.drain()
    assert worker_error.value.code is VectorProjectionErrorCode.ADAPTER_UNAVAILABLE
    assert worker_cursor.statements == []

    retry_cursor = _RecordingCursor(execute_error=_RetryablePgError("serialization failure"))
    with pytest.raises(VectorProjectionError) as retryable:
        _store(_RecordingConnection(retry_cursor)).enqueue(_request())
    assert retryable.value.code is VectorProjectionErrorCode.RETRYABLE
