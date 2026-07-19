from __future__ import annotations

from datetime import UTC, datetime

import pytest

from robata.annotation import (
    AnnotationPipeline,
    DeterministicAnnotationPrincipal,
    StructuredLabels,
)
from robata.application import LocalRequirementsPipeline
from robata.capacity import (
    CapacityPlanner,
    MeasurementStatus,
    SLAPlanner,
    SLAStage,
    ThroughputLedger,
)
from robata.frame_cache import FrameCacheCapacityEstimate, FramePayload, SharedFrameCache
from robata.qa import ClipMark, QAClassifier, QAIssue, QAStatus
from robata.search import ClipSearchIndex, VerbNormalizer


def test_qa_retains_all_local_marks_and_only_fail_deletes() -> None:
    result = QAClassifier().assess(
        "video-1",
        20,
        [
            ClipMark(start_sec=2, end_sec=4, issue=QAIssue.BLACK_SCREEN, confidence=0.9),
            ClipMark(start_sec=10, end_sec=12, issue=QAIssue.HAIR_BLOCKING_VIEW, confidence=0.8),
        ],
    )
    assert result.status is QAStatus.WARNING
    assert result.retained is True
    assert result.delete_source is False
    assert len(result.clip_marks) == 2
    assert result.effective_duration_sec == pytest.approx(16)

    failed = QAClassifier().assess(
        "video-2",
        20,
        [ClipMark(start_sec=0, end_sec=20, issue=QAIssue.BLACK_SCREEN, confidence=1.0)],
    )
    assert failed.status is QAStatus.FAIL
    assert failed.retained is False
    assert failed.delete_source is True


def test_annotation_excludes_fail_and_carries_warning_marks() -> None:
    classifier = QAClassifier()
    warning = classifier.assess(
        "warning-video",
        12,
        [ClipMark(start_sec=2, end_sec=3, issue=QAIssue.GLITCHED_SCREEN, confidence=0.7)],
    )
    passed = classifier.assess("pass-video", 3)
    failed = classifier.assess(
        "fail-video",
        3,
        [ClipMark(start_sec=0, end_sec=3, issue=QAIssue.BLURRY_LENS, confidence=0.99)],
    )
    output = AnnotationPipeline(DeterministicAnnotationPrincipal(segment_duration_sec=5)).run(
        [warning, passed, failed]
    )
    assert output.skipped_fail_video_ids == ("fail-video",)
    assert output.accepted_video_ids == ("warning-video", "pass-video")
    assert any(d.qa_clip_marks for d in output.drafts if d.video_id == "warning-video")
    assert all(d.video_id != "fail-video" for d in output.drafts)


def test_feed_once_decodes_only_once(tmp_path) -> None:
    calls = 0

    def decode():
        nonlocal calls
        calls += 1
        return [FramePayload(0.0, b"a"), FramePayload(0.5, b"b")]

    cache = SharedFrameCache(tmp_path)
    first = cache.feed_once("video", "local://video", decode)
    second = cache.feed_once("video", "local://video", decode)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 1
    assert cache.stats().decode_attempts == 1
    assert cache.read_frame(first.manifest.frames[0]) == b"a"


def test_zero_gpu_search_normalizes_synonyms_and_returns_playback_target() -> None:
    assert VerbNormalizer.normalize("scrubbing") == "clean"
    labels = StructuredLabels(
        verb="wipe", noun="table", attributes=("wooden",), location="center", hand="right"
    )
    index = ClipSearchIndex(
        [
            {
                "clip_id": "clip-1",
                "video_id": "video-1",
                "start_sec": 1,
                "end_sec": 4,
                "structured_labels": labels,
                "source_uri": "https://media/video-1.mp4",
            }
        ]
    )
    hits = index.search("scrub table")
    assert len(hits) == 1
    assert hits[0].verb_family == "clean"
    assert "start=1" in hits[0].playback_target
    assert "end=4" in hits[0].playback_target


def test_capacity_and_sla_are_explicitly_non_certifying() -> None:
    planner = CapacityPlanner()
    assert planner.total_gpu_capacity_hours_per_day == 48
    assert planner.planned_gpu_hours_per_day == 32
    assert planner.fits_documented_assumption is True
    assert 5.0 <= FrameCacheCapacityEstimate().estimated_terabytes <= 7.0
    ledger = ThroughputLedger(planner=planner)
    ledger.record(SLAStage.QA, 500, 24 * 3600, measurement_status=MeasurementStatus.NOT_MEASURED)
    assert ledger.report().production_eligible is False
    assert ledger.report().meets_target is False
    deadline = SLAPlanner().deadline("video", datetime(2026, 7, 19, tzinfo=UTC))
    assert deadline.qa_due_at.day == 20
    assert SLAPlanner().is_on_time(SLAStage.QA, deadline.qa_due_at, deadline)


def test_full_requirements_pipeline_publishes_all_non_model_artifacts(tmp_path) -> None:
    from robata.frame_cache import FramePayload, SharedFrameCache

    root = tmp_path / "pipeline"
    pipeline = LocalRequirementsPipeline(cache=SharedFrameCache(root / "cache"))
    result = pipeline.run(
        "video",
        4.0,
        source_uri="local://video",
        decoder=lambda: [FramePayload(0.0, b"a"), FramePayload(1.0, b"b")],
        output_directory=root / "out",
    )
    assert result.provider_requests == 0
    assert result.production_eligible is False
    assert (root / "out" / "requirements-run.json").exists()
    assert result.search_entries > 0
