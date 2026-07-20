from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from robata.contracts import CameraId, NanosecondInterval
from robata.contracts.pipeline import (
    CameraQAClaim,
    CameraQAResult,
    CameraQAStatus,
    RecordingQAStatus,
)
from robata.qa_pipeline.aggregate import QAAggregationPolicy, QAAggregator
from robata.qa_pipeline.suspicion_reducer import (
    SuspiciousInterval,
    SuspiciousIntervalReducer,
)


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:test:{label}"))


def _camera_result(
    camera_id: CameraId,
    status: CameraQAStatus,
    *,
    score: float | None = None,
    mcap_id: str | None = None,
    start_ns: int = 0,
    end_ns: int = 10_000,
) -> CameraQAResult:
    return CameraQAResult(
        qa_result_id=_id(f"qa:{camera_id.value}:{status.value}:{start_ns}:{end_ns}"),
        mcap_id=mcap_id or _id("mcap"),
        package_id=_id(f"package:{camera_id.value}:{start_ns}:{end_ns}"),
        inference_id=_id(f"inference:{camera_id.value}:{start_ns}:{end_ns}"),
        camera_id=camera_id,
        claim=CameraQAClaim(
            camera_id=camera_id,
            observed_interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
            status=status,
            issues=(),
            reported_score=score,
            frame_ordinals=(),
        ),
        evidence_frame_ids=(),
    )


def test_qa_aggregate_is_canonical_incomplete_and_non_promotable() -> None:
    statuses = {
        CameraId.CAM_01: CameraQAStatus.GOOD,
        CameraId.CAM_02: CameraQAStatus.GOOD,
        CameraId.CAM_03: CameraQAStatus.GOOD,
        CameraId.CAM_04: CameraQAStatus.GOOD,
        CameraId.CAM_05: CameraQAStatus.GOOD,
        CameraId.CAM_06: CameraQAStatus.INCOMPLETE,
    }
    results = [
        _camera_result(camera_id, statuses[camera_id], score=index / 10)
        for index, camera_id in enumerate(reversed(tuple(CameraId)), start=1)
    ]

    aggregate = QAAggregator().aggregate_camera_results(results)

    assert aggregate.overall_status is RecordingQAStatus.INCOMPLETE
    assert aggregate.usable_camera_count == 5
    assert aggregate.camera_result_ids == tuple(
        next(result.qa_result_id for result in results if result.camera_id is camera_id)
        for camera_id in CameraId
    )
    assert aggregate.model_score == pytest.approx(0.35)
    assert aggregate.deterministic_quality is None
    assert aggregate.policy_version == "local-development-v1"
    assert aggregate.promotion_eligible is False


def test_qa_aggregate_rejects_mixed_recordings_and_scopes() -> None:
    results = [_camera_result(camera_id, CameraQAStatus.GOOD) for camera_id in CameraId]
    results[-1] = _camera_result(
        CameraId.CAM_06,
        CameraQAStatus.GOOD,
        mcap_id=_id("other-mcap"),
    )
    with pytest.raises(ValueError, match="one MCAP"):
        QAAggregator().aggregate_camera_results(results)

    results[-1] = _camera_result(
        CameraId.CAM_06,
        CameraQAStatus.GOOD,
        start_ns=1,
    )
    with pytest.raises(ValueError, match="exact scope"):
        QAAggregator().aggregate_camera_results(results)


def test_unresolved_qa_policy_cannot_claim_promotion() -> None:
    with pytest.raises(ValidationError, match="cannot be promotable"):
        QAAggregationPolicy(
            version="unapproved-v1",
            degraded_min_usable=4,
            status_quality={status: 0.0 for status in CameraQAStatus},
            promotion_eligible=True,
        )


def test_suspicious_reduction_is_cross_camera_deterministic_and_clipped() -> None:
    intervals = (
        SuspiciousInterval(
            start_ns=100,
            end_ns=200,
            camera_id=CameraId.CAM_02,
            issue_type="BLUR",
            confidence=0.8,
        ),
        SuspiciousInterval(
            start_ns=210,
            end_ns=300,
            camera_id=CameraId.CAM_01,
            issue_type="OCCLUSION",
            confidence=0.9,
        ),
        SuspiciousInterval(
            start_ns=900,
            end_ns=980,
            camera_id=CameraId.CAM_03,
            issue_type="EXPOSURE",
            confidence=0.7,
        ),
    )
    reducer = SuspiciousIntervalReducer()

    first = reducer.reduce(
        intervals,
        padding_ns=50,
        max_gap_ns=20,
        recording_duration_ns=1_000,
        policy_version="qa-reduce-v2",
    )
    replay = reducer.reduce(
        tuple(reversed(intervals)),
        padding_ns=50,
        max_gap_ns=20,
        recording_duration_ns=1_000,
        policy_version="qa-reduce-v2",
    )

    assert first == replay
    assert [(item.start_ns, item.end_ns) for item in first] == [(50, 350), (850, 1_000)]
    assert first[0].cameras == (CameraId.CAM_01, CameraId.CAM_02)
    assert first[0].merged_from_count == 2
    assert len({source.interval_id for source in first[0].source_intervals}) == 2
    assert first[0].reduction_policy_version.max_gap_ns == 20
    assert first[0].reduction_policy_version.clip_to_recording_bounds is True


def test_suspicious_interval_and_reducer_fail_closed() -> None:
    with pytest.raises(ValidationError, match="must be non-empty"):
        SuspiciousInterval(
            start_ns=5,
            end_ns=5,
            camera_id=CameraId.CAM_01,
            issue_type="BLUR",
            confidence=0.5,
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        SuspiciousIntervalReducer().reduce((), padding_ns=-1)
