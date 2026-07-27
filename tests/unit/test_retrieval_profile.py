from __future__ import annotations

import asyncio

import pytest

from robata.benchmark.retrieval import (
    BackfillWriteDisposition,
    RetrievalBackfillCounters,
    RetrievalBackfillTarget,
    RetrievalCostProfile,
    RetrievalFilterObservation,
    RetrievalLatencyProfile,
    build_filter_metrics,
    build_recall_profile,
    build_retrieval_profile,
    calculate_vector_recall,
    run_embedding_backfill,
)


def _target(revision: str, artifact: str = "artifact-1") -> RetrievalBackfillTarget:
    return RetrievalBackfillTarget(event_revision_id=revision, artifact_identity=artifact)


def test_embedding_backfill_coalesces_duplicates_and_reuses_sink_rows() -> None:
    existing: set[str] = set()

    assert _target("x:artifact:y").identity_key != _target("x", "y").identity_key

    async def encoder(target: RetrievalBackfillTarget) -> tuple[float, ...]:
        return (float(len(target.event_revision_id)), 0.5)

    async def writer(write: object) -> BackfillWriteDisposition:
        key = write.target.identity_key  # type: ignore[attr-defined]
        if key in existing:
            return BackfillWriteDisposition.REUSED
        existing.add(key)
        return BackfillWriteDisposition.WRITTEN

    targets = (_target("revision-1"), _target("revision-1"), _target("revision-2"))
    first = asyncio.run(
        run_embedding_backfill(
            targets,
            encoder=encoder,
            writer=writer,
            encoder_name="offline-encoder",
            model_version="embedding-v1",
            dimension=2,
            concurrency=2,
        )
    )
    second = asyncio.run(
        run_embedding_backfill(
            targets,
            encoder=encoder,
            writer=writer,
            encoder_name="offline-encoder",
            model_version="embedding-v1",
            dimension=2,
            concurrency=2,
        )
    )

    assert (
        first.target_count,
        first.unique_target_count,
        first.duplicate_target_count,
    ) == (3, 2, 1)
    assert (first.written_count, first.reused_count, first.failed_count) == (2, 0, 0)
    assert (second.written_count, second.reused_count, second.failed_count) == (0, 2, 0)


def test_embedding_backfill_records_failure_without_blocking_other_targets() -> None:
    async def encoder(target: RetrievalBackfillTarget) -> tuple[float, ...]:
        if target.event_revision_id == "bad":
            raise RuntimeError("encoder unavailable")
        return (1.0,)

    async def writer(write: object) -> bool:
        del write
        return True

    report = asyncio.run(
        run_embedding_backfill(
            (_target("bad"), _target("good")),
            encoder=encoder,
            writer=writer,
            encoder_name="offline",
            model_version="v1",
            dimension=1,
        )
    )
    assert report.failed_count == 1
    assert report.written_count == 1
    assert any(item.status == "FAILED" and item.error for item in report.observations)


def test_structured_filter_metrics_and_vector_recall_are_bounded() -> None:
    metrics = build_filter_metrics(
        (
            RetrievalFilterObservation(
                query_id="q-1",
                structured_ids=("a", "b"),
                vector_ids=("a",),
                result_ids=("a",),
            ),
            RetrievalFilterObservation(
                query_id="q-2",
                structured_ids=("c", "d"),
                vector_ids=("c",),
                result_ids=("c",),
            ),
        )
    )
    assert metrics.structured_first is True
    assert metrics.vector_selectivity == pytest.approx(0.5)
    assert metrics.structured_candidate_count == 4
    assert metrics.vector_candidate_count == 2

    points = calculate_vector_recall(
        {"q-1": ("a", "b"), "q-2": ("c",)},
        {"q-1": ("a",), "q-2": ("other", "c")},
        cutoffs=(1, 2),
    )
    assert [point.recall for point in points] == [pytest.approx(1 / 3), pytest.approx(2 / 3)]

    with pytest.raises(ValueError, match="drawn from structured"):
        RetrievalFilterObservation(
            query_id="bad",
            structured_ids=("a",),
            vector_ids=("b",),
            result_ids=(),
        )


def test_recall_latency_cost_and_profile_digest_are_replayable() -> None:
    recall = build_recall_profile(
        (
            (("a",), ("a", "b")),
            (("other", "b"), ("b",)),
        ),
        ks=(1, 2),
        filtered_candidate_count=3,
    )
    assert recall.query_count == 2
    assert recall.recall_at_k["1"] == pytest.approx(0.25)
    assert recall.recall_at_k["2"] == pytest.approx(0.75)

    latency = RetrievalLatencyProfile.from_samples((10, 20, 100, 200))
    assert (latency.sample_count, latency.p50_ms, latency.p95_ms, latency.p99_ms) == (
        4,
        20.0,
        200.0,
        200.0,
    )
    cost = RetrievalCostProfile(
        structured_queries=2,
        vector_queries=2,
        vector_rows_scanned=5,
        encoder_calls=1,
        provider_tokens=100,
        cpu_ms=3.0,
        estimated_cost=0.01,
    )
    first = build_retrieval_profile(
        backfill=RetrievalBackfillCounters(requested=2, enqueued=2, completed=2),
        observations=((("a",), ("a",)),),
        latency_samples_ms=(1.0, 2.0),
        cost=cost,
        ks=(1,),
        filtered_candidate_count=1,
    )
    second = build_retrieval_profile(
        backfill=RetrievalBackfillCounters(requested=2, enqueued=2, completed=2),
        observations=((("a",), ("a",)),),
        latency_samples_ms=(1.0, 2.0),
        cost=cost,
        ks=(1,),
        filtered_candidate_count=1,
    )
    assert first == second
    assert first.profile_digest
    assert first.structured_authoritative is True
    assert first.embeddings_non_blocking is True
    assert first.external_database_status == "NOT_MEASURED"
    assert first.production_eligible is False


def test_recall_rejects_invalid_cutoff() -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_vector_recall({"q": ("a",)}, {"q": ("a",)}, cutoffs=(0,))
