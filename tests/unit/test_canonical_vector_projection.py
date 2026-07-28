"""Tests for detached P14 revision-bound vector projection dispatch."""

from __future__ import annotations

import pytest

from robata.application.canonical.vector_projection import (
    CanonicalVectorIndexParameter,
    CanonicalVectorProjectionBridge,
    CanonicalVectorProjectionDispatchStatus,
    CanonicalVectorProjectionIntent,
    CanonicalVectorProjectionPolicy,
)
from robata.contracts.retrieval import (
    EmbeddingModality,
    EmbeddingNormalization,
    EmbeddingProvider,
    EmbeddingSpec,
    EncodedEmbedding,
    VectorAccessPolicy,
    VectorIndexKind,
    VectorProjectionStatus,
    VectorProjectionSubject,
    VectorRetentionPolicy,
)
from robata.ports.vector_projection import (
    FailClosedVectorProjectionStore,
    InMemoryVectorProjectionStore,
    VectorProjectionErrorCode,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _embedding() -> EmbeddingSpec:
    return EmbeddingSpec(
        embedding_id="canonical-p14-text-v1",
        model="deterministic-encoder",
        model_version="1.0",
        modality=EmbeddingModality.TEXT,
        dimension=2,
        normalization=EmbeddingNormalization.L2,
        encoder_provider=EmbeddingProvider.CPU,
        index_policy_version="hnsw-v1",
        index_kind=VectorIndexKind.HNSW,
    )


def _policy(embedding: EmbeddingSpec) -> CanonicalVectorProjectionPolicy:
    return CanonicalVectorProjectionPolicy.create(
        policy_version="canonical-p14-policy-v1",
        embedding_id=embedding.embedding_id,
        embedding_semantic_sha256=embedding.semantic_sha256,
        preprocessing_policy_version="text-preprocess-v1",
        preprocessing_semantic_sha256=_digest(1),
        index_kind=embedding.index_kind,
        index_parameters=(
            CanonicalVectorIndexParameter(name="ef_construction", value="64"),
            CanonicalVectorIndexParameter(name="m", value="16"),
        ),
    )


def _intent(
    *,
    source_digest: str = _digest(2),
) -> CanonicalVectorProjectionIntent:
    embedding = _embedding()
    return CanonicalVectorProjectionIntent.create(
        event_revision_id="event-revision-1",
        event_revision_semantic_sha256=source_digest,
        policy=_policy(embedding),
        encoded_embedding=EncodedEmbedding(
            subject=VectorProjectionSubject(
                event_revision_id="event-revision-1",
                source_semantic_sha256=source_digest,
            ),
            embedding=embedding,
            vector=(1.0, 0.0),
            encoder_run_id="encoder-run-1",
        ),
        retention=VectorRetentionPolicy(retention_policy_version="retention-v1", ttl_days=7),
        access_policy=VectorAccessPolicy(policy_version="tenant-policy-v1", tenant_id="tenant-a"),
    )


def test_bridge_enqueues_a_revision_bound_projection_without_completion_state() -> None:
    intent = _intent()
    store = InMemoryVectorProjectionStore()
    bridge = CanonicalVectorProjectionBridge(store=store)

    first = bridge.enqueue(intent)
    replay = bridge.enqueue(intent)

    assert first.status is CanonicalVectorProjectionDispatchStatus.QUEUED
    assert first.receipt is not None
    assert first.receipt.status is VectorProjectionStatus.PENDING
    assert replay.receipt is not None
    assert replay.receipt.duplicate is True
    assert store.drain() == 1
    assert (
        store.get(
            intent.encoded_embedding.subject,
            intent.encoded_embedding.embedding.embedding_id,
            tenant_id="tenant-a",
        )
        is not None
    )


def test_intent_rejects_a_subject_without_the_canonical_revision_digest() -> None:
    embedding = _embedding()
    with pytest.raises(ValueError, match="source digest"):
        CanonicalVectorProjectionIntent.create(
            event_revision_id="event-revision-1",
            event_revision_semantic_sha256=_digest(2),
            policy=_policy(embedding),
            encoded_embedding=EncodedEmbedding(
                subject=VectorProjectionSubject(
                    event_revision_id="event-revision-1",
                    source_semantic_sha256=_digest(3),
                ),
                embedding=embedding,
                vector=(1.0, 0.0),
            ),
            retention=VectorRetentionPolicy(retention_policy_version="retention-v1", ttl_days=7),
        )


def test_unavailable_adapter_is_an_observable_optional_projection_failure() -> None:
    dispatch = CanonicalVectorProjectionBridge(store=FailClosedVectorProjectionStore()).enqueue(
        _intent()
    )

    assert dispatch.status is CanonicalVectorProjectionDispatchStatus.FAILED
    assert dispatch.error_code is VectorProjectionErrorCode.ADAPTER_UNAVAILABLE
    assert dispatch.receipt is None
