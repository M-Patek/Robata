"""Tests for durable, detached P14 vector projection handoff dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.vector_projection import (
    CanonicalVectorIndexParameter,
    CanonicalVectorProjectionDispatch,
    CanonicalVectorProjectionDispatchStatus,
    CanonicalVectorProjectionIntent,
    CanonicalVectorProjectionPolicy,
)
from robata.application.canonical.vector_projection_dispatch import (
    CanonicalVectorProjectionDispatchBridge,
    CanonicalVectorProjectionDispatchExecutionStatus,
    CanonicalVectorProjectionDispatchRecord,
    CanonicalVectorProjectionDispatchStore,
    CanonicalVectorProjectionDispatchWorker,
    CanonicalVectorProjectionJobStatus,
    VectorProjectionDispatchRetryNotAllowed,
    VectorProjectionDispatchStorageError,
)
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.retrieval import (
    EmbeddingModality,
    EmbeddingNormalization,
    EmbeddingProvider,
    EmbeddingSpec,
    EncodedEmbedding,
    VectorAccessPolicy,
    VectorIndexKind,
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
        embedding_id="canonical-p14-dispatch-text-v1",
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
        policy_version="canonical-p14-dispatch-policy-v1",
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
    requested_at: str | None = None,
) -> CanonicalVectorProjectionIntent:
    embedding = _embedding()
    return CanonicalVectorProjectionIntent.create(
        event_revision_id="event-revision-p14-dispatch-1",
        event_revision_semantic_sha256=_digest(2),
        policy=_policy(embedding),
        encoded_embedding=EncodedEmbedding(
            subject=VectorProjectionSubject(
                event_revision_id="event-revision-p14-dispatch-1",
                source_semantic_sha256=_digest(2),
            ),
            embedding=embedding,
            vector=(1.0, 0.0),
            encoder_run_id="encoder-run-p14-dispatch-1",
        ),
        retention=VectorRetentionPolicy(retention_policy_version="retention-v1", ttl_days=7),
        access_policy=VectorAccessPolicy(policy_version="tenant-policy-v1", tenant_id="tenant-a"),
        requested_at=requested_at,
    )


def test_durable_handoff_normalizes_wall_clock_observation_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    intent = _intent(requested_at="2026-07-28T00:00:00Z")
    intent_bytes = canonical_json_bytes(intent)
    sidecar = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    bridge = CanonicalVectorProjectionDispatchBridge(sidecar)

    queued = bridge.enqueue(intent)

    assert queued.status is CanonicalVectorProjectionJobStatus.ENQUEUED
    assert queued.replayed is False
    assert queued.job.intent.requested_at is None
    assert canonical_json_bytes(intent) == intent_bytes
    assert sidecar.job_path(queued.job.semantic_sha256).read_bytes() == canonical_json_bytes(
        queued.job
    )

    vectors = InMemoryVectorProjectionStore()
    first = CanonicalVectorProjectionDispatchWorker(
        store=sidecar,
        vector_store=vectors,
    ).drain()

    assert len(first) == 1
    assert first[0].status is CanonicalVectorProjectionDispatchExecutionStatus.RECORDED
    assert first[0].record.dispatch.status is CanonicalVectorProjectionDispatchStatus.QUEUED
    assert first[0].record.dispatch.receipt is not None
    assert first[0].record.dispatch.receipt.duplicate is False
    assert first[0].record.production_eligible is False
    assert vectors.pending_count == 1

    restarted = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    replay = CanonicalVectorProjectionDispatchWorker(
        store=restarted,
        vector_store=vectors,
    ).drain()

    assert len(replay) == 1
    assert replay[0].status is CanonicalVectorProjectionDispatchExecutionStatus.REPLAYED
    assert replay[0].record == first[0].record
    assert vectors.pending_count == 1
    assert vectors.drain() == 1
    assert (
        vectors.get(
            queued.job.intent.encoded_embedding.subject,
            queued.job.intent.encoded_embedding.embedding.embedding_id,
            tenant_id="tenant-a",
        )
        is not None
    )


def test_recovery_retries_an_unrecorded_adapter_handoff_idempotently(tmp_path: Path) -> None:
    sidecar = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    queued = CanonicalVectorProjectionDispatchBridge(sidecar).enqueue(_intent())
    vectors = InMemoryVectorProjectionStore()

    # Simulate a process exit after the physical enqueue but before it could
    # write the sidecar record.  The restarted worker must safely observe the
    # duplicate acknowledgement and then seal it.
    accepted = vectors.enqueue(queued.job.intent.to_request())
    assert accepted.duplicate is False
    recovered = CanonicalVectorProjectionDispatchWorker(
        store=CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar"),
        vector_store=vectors,
    ).drain()

    assert len(recovered) == 1
    receipt = recovered[0].record.dispatch.receipt
    assert recovered[0].status is CanonicalVectorProjectionDispatchExecutionStatus.RECORDED
    assert recovered[0].record.dispatch.status is CanonicalVectorProjectionDispatchStatus.QUEUED
    assert receipt is not None and receipt.duplicate is True
    assert vectors.pending_count == 1


def test_failed_handoff_is_immutable_and_retry_is_a_distinct_recoverable_job(
    tmp_path: Path,
) -> None:
    sidecar = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    bridge = CanonicalVectorProjectionDispatchBridge(sidecar)
    initial = bridge.enqueue(_intent())

    failed = CanonicalVectorProjectionDispatchWorker(
        store=sidecar,
        vector_store=FailClosedVectorProjectionStore(),
    ).drain()[0]

    assert failed.record.dispatch.status is CanonicalVectorProjectionDispatchStatus.FAILED
    assert failed.record.dispatch.error_code is VectorProjectionErrorCode.ADAPTER_UNAVAILABLE
    assert failed.record.production_eligible is False

    retry = bridge.retry_failed(initial.job)
    assert retry.status is CanonicalVectorProjectionJobStatus.ENQUEUED
    assert retry.job.retry_ordinal == 1
    assert retry.job.retry_of_job_semantic_sha256 == initial.job.semantic_sha256
    assert retry.job.retry_of_dispatch_semantic_sha256 == failed.record.semantic_sha256
    assert bridge.retry_failed(initial.job).status is CanonicalVectorProjectionJobStatus.REPLAYED

    vectors = InMemoryVectorProjectionStore()
    executions = CanonicalVectorProjectionDispatchWorker(
        store=sidecar,
        vector_store=vectors,
    ).drain()
    by_job = {execution.job.semantic_sha256: execution for execution in executions}

    assert by_job[initial.job.semantic_sha256].replayed is True
    retried = by_job[retry.job.semantic_sha256]
    assert retried.status is CanonicalVectorProjectionDispatchExecutionStatus.RECORDED
    assert retried.record.dispatch.status is CanonicalVectorProjectionDispatchStatus.QUEUED
    assert vectors.pending_count == 1


def test_existing_physical_failed_row_becomes_a_retryable_sidecar_failure(
    tmp_path: Path,
) -> None:
    sidecar = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    bridge = CanonicalVectorProjectionDispatchBridge(sidecar)
    initial = bridge.enqueue(_intent())
    vectors = InMemoryVectorProjectionStore()
    vectors.enqueue(initial.job.intent.to_request())
    vectors.mark_failed(
        initial.job.intent.encoded_embedding.subject,
        initial.job.intent.encoded_embedding.embedding.embedding_id,
        "offline",
    )

    failed = CanonicalVectorProjectionDispatchWorker(
        store=sidecar,
        vector_store=vectors,
    ).drain()[0]

    assert failed.record.dispatch.status is CanonicalVectorProjectionDispatchStatus.FAILED
    assert failed.record.dispatch.error_code is VectorProjectionErrorCode.RETRYABLE
    retry = bridge.retry_failed(initial.job)

    executions = CanonicalVectorProjectionDispatchWorker(
        store=sidecar,
        vector_store=vectors,
    ).drain()
    by_job = {execution.job.semantic_sha256: execution for execution in executions}

    assert by_job[retry.job.semantic_sha256].record.dispatch.status is (
        CanonicalVectorProjectionDispatchStatus.QUEUED
    )
    assert vectors.pending_count == 1


def test_nonretryable_dispatch_failure_cannot_be_rescheduled(tmp_path: Path) -> None:
    sidecar = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    bridge = CanonicalVectorProjectionDispatchBridge(sidecar)
    queued = bridge.enqueue(_intent())
    dispatch = CanonicalVectorProjectionDispatch(
        intent_logical_key=queued.job.intent.logical_key,
        intent_semantic_sha256=queued.job.intent.semantic_sha256,
        idempotency_key=queued.job.intent.to_request().idempotency_key,
        status=CanonicalVectorProjectionDispatchStatus.FAILED,
        error_code=VectorProjectionErrorCode.DUPLICATE_CONFLICT,
        error_message="immutable lineage conflict",
    )
    record = CanonicalVectorProjectionDispatchRecord.create(job=queued.job, dispatch=dispatch)
    sidecar.put_or_get_record(job=queued.job, record=record)

    with pytest.raises(VectorProjectionDispatchRetryNotAllowed, match="not retryable"):
        bridge.retry_failed(queued.job)


def test_sidecar_rejects_tampered_record_bytes(tmp_path: Path) -> None:
    sidecar = CanonicalVectorProjectionDispatchStore(tmp_path / "sidecar")
    queued = CanonicalVectorProjectionDispatchBridge(sidecar).enqueue(_intent())
    CanonicalVectorProjectionDispatchWorker(
        store=sidecar,
        vector_store=InMemoryVectorProjectionStore(),
    ).drain()

    sidecar.record_path(queued.job.semantic_sha256).write_bytes(b"{}")
    with pytest.raises(
        VectorProjectionDispatchStorageError,
        match="invalid vector dispatch record",
    ):
        sidecar.get_record(queued.job.semantic_sha256)
