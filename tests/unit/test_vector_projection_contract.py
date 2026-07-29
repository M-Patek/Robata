from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.contracts.retrieval import (
    EmbeddingBackfillRequest,
    EmbeddingEncoderInput,
    EmbeddingModality,
    EmbeddingNormalization,
    EmbeddingProvider,
    EmbeddingSpec,
    VectorAccessPolicy,
    VectorBackend,
    VectorLocator,
    VectorProjectionRequest,
    VectorProjectionStatus,
    VectorProjectionSubject,
    VectorRetentionPolicy,
    VectorSearchQuery,
)
from robata.ports.vector_projection import (
    FailClosedEmbeddingEncoder,
    FailClosedVectorProjectionStore,
    InMemoryVectorProjectionStore,
    VectorProjectionError,
    VectorProjectionErrorCode,
)


def _spec(*, dimension: int = 3) -> EmbeddingSpec:
    return EmbeddingSpec(
        embedding_id="text-test-v1",
        model="deterministic-test",
        model_version="1.0",
        modality=EmbeddingModality.TEXT,
        dimension=dimension,
        normalization=EmbeddingNormalization.NONE,
        encoder_provider=EmbeddingProvider.CPU,
        index_policy_version="none-v1",
    )


def _request(
    revision: str = "revision-1",
    vector: tuple[float, ...] = (1.0, 0.0, 0.0),
    access_policy: VectorAccessPolicy | None = None,
    subject: VectorProjectionSubject | None = None,
):
    return VectorProjectionRequest(
        subject=subject or VectorProjectionSubject(event_revision_id=revision),
        embedding=_spec(dimension=len(vector)),
        vector=vector,
        retention=VectorRetentionPolicy(retention_policy_version="retention-v1", ttl_days=7),
        access_policy=access_policy,
    )


def test_embedding_metadata_and_locator_are_versioned_and_transport_independent() -> None:
    spec = _spec()
    assert spec.contract_version == "1.0"
    assert spec.semantic_sha256 == spec.model_copy().semantic_sha256
    locator = VectorLocator(
        backend=VectorBackend.SUPABASE,
        relation="event_embeddings",
        row_key="row-1",
        index_name="event_embeddings_hnsw",
    )
    assert locator.content_identity is None
    assert locator.metadata_projection["backend"] == "SUPABASE"
    assert "row_key" in locator.metadata_projection


def test_projection_request_binds_dimension_and_lineage() -> None:
    with pytest.raises(ValidationError, match="dimension"):
        VectorProjectionRequest(
            subject=VectorProjectionSubject(event_revision_id="revision-1"),
            embedding=_spec(dimension=3),
            vector=(1.0, 0.0),
            retention=VectorRetentionPolicy(retention_policy_version="retention-v1", ttl_days=7),
        )
    with pytest.raises(ValidationError, match="artifact_id and artifact_sha256"):
        VectorProjectionSubject(event_revision_id="revision-1", artifact_id="clip-1")


def test_async_store_is_idempotent_and_search_is_structured_candidate_bound() -> None:
    store = InMemoryVectorProjectionStore()
    first = _request("revision-1", (1.0, 0.0, 0.0))
    second = _request("revision-2", (0.0, 1.0, 0.0))
    assert store.enqueue(first).status is VectorProjectionStatus.PENDING
    assert store.enqueue(first).duplicate
    assert store.pending_count == 1
    assert store.drain() == 1
    assert store.upsert(second).status is VectorProjectionStatus.READY

    hits = store.search(
        VectorSearchQuery(
            embedding_id="text-test-v1",
            vector=(1.0, 0.0, 0.0),
            candidate_event_revision_ids=("revision-1",),
        )
    )
    assert len(hits) == 1
    assert hits[0].event_revision_id == "revision-1"
    assert hits[0].score == pytest.approx(1.0)


def test_changed_vector_reusing_subject_is_a_conflict_not_a_second_revision() -> None:
    store = InMemoryVectorProjectionStore()
    request = _request()
    store.enqueue(request)
    changed = _request(vector=(0.0, 1.0, 0.0))
    # Idempotency is based on event revision + embedding metadata, not vector bytes.
    assert changed.idempotency_key == request.idempotency_key
    with pytest.raises(VectorProjectionError) as error:
        store.enqueue(changed)
    assert error.value.code is VectorProjectionErrorCode.DUPLICATE_CONFLICT


def test_subject_keys_are_injective_and_failed_rows_are_replay_safe() -> None:
    unbound = VectorProjectionSubject(event_revision_id="x:artifact:y")
    bound = VectorProjectionSubject(
        event_revision_id="x",
        artifact_id="y",
        artifact_sha256="a" * 64,
    )
    assert unbound.projection_key != bound.projection_key
    assert (
        _request(unbound.event_revision_id).idempotency_key
        != _request(
            subject=bound,
        ).idempotency_key
    )

    store = InMemoryVectorProjectionStore()
    request = _request()
    store.enqueue(request)
    first = store.mark_failed(request.subject, request.embedding.embedding_id, "offline")
    assert first.status is VectorProjectionStatus.FAILED
    assert first.duplicate is False
    assert store.mark_failed(request.subject, request.embedding.embedding_id, "offline").duplicate
    with pytest.raises(VectorProjectionError) as changed:
        store.mark_failed(request.subject, request.embedding.embedding_id, "different")
    assert changed.value.code is VectorProjectionErrorCode.DUPLICATE_CONFLICT

    store.retry_failed(request.subject, request.embedding.embedding_id)
    store.drain()
    with pytest.raises(VectorProjectionError) as ready:
        store.mark_failed(request.subject, request.embedding.embedding_id, "late")
    assert ready.value.code is VectorProjectionErrorCode.CONFLICT


def test_protected_get_requires_matching_tenant() -> None:
    policy = VectorAccessPolicy(policy_version="rls-v1", tenant_id="tenant-a")
    store = InMemoryVectorProjectionStore()
    request = _request(access_policy=policy)
    store.upsert(request)
    assert store.get(request.subject, request.embedding.embedding_id, tenant_id="tenant-a")
    with pytest.raises(VectorProjectionError) as denied:
        store.get(request.subject, request.embedding.embedding_id)
    assert denied.value.code is VectorProjectionErrorCode.RLS_DENIED


def test_fail_closed_adapters_and_encoder_are_explicit() -> None:
    request = _request()
    with pytest.raises(VectorProjectionError, match="not configured") as store_error:
        FailClosedVectorProjectionStore().enqueue(request)
    assert store_error.value.code is VectorProjectionErrorCode.ADAPTER_UNAVAILABLE

    encoder_input = EmbeddingEncoderInput(
        subject=request.subject,
        embedding=_spec(),
        text="right hand grasps cup",
    )
    with pytest.raises(VectorProjectionError, match="not configured") as encoder_error:
        FailClosedEmbeddingEncoder.encode(encoder_input)
    assert encoder_error.value.code is VectorProjectionErrorCode.ENCODER_UNAVAILABLE


def test_backfill_is_bounded_and_encoder_input_is_modality_specific() -> None:
    request = EmbeddingBackfillRequest(
        embedding=_spec(),
        event_revision_ids=("revision-1", "revision-2"),
        batch_size=1,
    )
    assert request.idempotency_key
    assert len(request.event_revision_ids) == 2
    with pytest.raises(ValidationError, match="TEXT encoder input requires text"):
        EmbeddingEncoderInput(
            subject=VectorProjectionSubject(event_revision_id="r"),
            embedding=_spec(),
        )
