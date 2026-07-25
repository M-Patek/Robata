from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.application.canonical.media_quality import (
    FrameQualityObservation,
    FrameTimingEvidence,
    LocalQualityFlag,
    build_local_media_quality_report,
)
from robata.application.canonical.product_qa import (
    product_qa_context_from_media_quality_report,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.qa import (
    ClipMark,
    ProductQAConfidenceKind,
    ProductQAEvidenceScope,
    ProductQAIssue,
    ProductQAIssueEvidence,
    ProductQAScopeKind,
    QAAssessment,
    QAClassifier,
    QAIssue,
    QAStatus,
)
from robata.qa_pipeline.product import (
    ProductQACascadeProjector,
    ProductQACascadeStatus,
    ProductQAClassState,
)


def _mark(
    start_sec: float,
    end_sec: float,
    issue: ProductQAIssue = ProductQAIssue.BLACK_SCREEN,
) -> ClipMark:
    return ClipMark(
        start_sec=start_sec,
        end_sec=end_sec,
        issue=issue,
        confidence=0.8,
    )


def _evidence(
    *,
    issue: ProductQAIssue = ProductQAIssue.BLACK_SCREEN,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    note: str | None = None,
) -> ProductQAIssueEvidence:
    return ProductQAIssueEvidence(
        issue=issue,
        scope=ProductQAEvidenceScope(
            kind=ProductQAScopeKind.CAMERA_INTERVAL,
            subject_refs=("recording-1", "camera-1"),
            camera_id=CameraId.CAM_01,
        ),
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        confidence=0.75,
        confidence_kind=ProductQAConfidenceKind.DETECTOR_REPORTED,
        evidence_refs=("frame-1",),
        note=note,
    )


def test_product_issue_vocabulary_is_exact_and_internal_taxonomy_is_retained() -> None:
    assert tuple(issue.value for issue in ProductQAIssue) == (
        "BLACK_SCREEN",
        "GLITCHED_SCREEN",
        "BLURRY_LENS",
        "EXCESSIVE_SPEED",
        "EGO_DEVICE_WORN_BACKWARDS",
        "EGO_HAND_NOT_CENTERED",
        "CAMERA_STATIONARY_OVER_5S",
        "HAIR_BLOCKING_VIEW",
        "IRRELEVANT_ACTION_PARTIAL_SEGMENT",
        "TASK_IRRELEVANT_ACTION",
        "ARM_HAND_OBSTRUCTED",
        "HAND_OVERLAP_CONTACT_CROSSING",
        "INCOMPLETE_TASK",
        "LACK_OF_DIVERSITY",
        "LACK_OF_AUTHENTICITY",
        "VIDEO_ABNORMAL_ENDING",
        "TOO_DARK_OR_OVEREXPOSED",
        "UNAUTHORIZED_PERSON_OR_ANIMAL",
        "REVEALING_OUTFIT",
        "PERFORMED_OTHER_EXISTING_TASK",
        "OTHER",
    )
    assert QAIssue.MOTION_BLUR.value == "MOTION_BLUR"
    assert QAIssue.SCENE_CHANGE.value == "SCENE_CHANGE"


@pytest.mark.parametrize("note", [None, "   "])
def test_other_evidence_requires_a_meaningful_note(note: str | None) -> None:
    with pytest.raises(ValidationError, match="OTHER issue evidence requires a nonempty note"):
        _evidence(issue=ProductQAIssue.OTHER, note=note)


def test_camera_recording_scope_requires_a_camera_id() -> None:
    with pytest.raises(ValidationError, match="camera evidence scopes require camera_id"):
        ProductQAEvidenceScope(
            kind=ProductQAScopeKind.CAMERA_RECORDING,
            subject_refs=("recording-1",),
        )


@pytest.mark.parametrize(
    "subject_refs",
    [("recording-1", "recording-1"), ("recording-1", "   ")],
)
def test_scope_references_must_be_unique_and_nonblank(
    subject_refs: tuple[str, str],
) -> None:
    with pytest.raises(ValidationError, match="subject_refs"):
        ProductQAEvidenceScope(
            kind=ProductQAScopeKind.TASK_INTERVAL,
            subject_refs=subject_refs,
        )


def test_rich_evidence_projects_to_exact_clip_mark_shape() -> None:
    evidence = _evidence(issue=ProductQAIssue.ARM_HAND_OBSTRUCTED)

    assert evidence.to_clip_mark().model_dump(mode="json") == {
        "start_sec": 1.0,
        "end_sec": 2.0,
        "issue": "ARM_HAND_OBSTRUCTED",
        "confidence": 0.75,
    }


def test_non_temporal_evidence_cannot_be_projected_to_a_clip_mark() -> None:
    evidence = ProductQAIssueEvidence(
        issue=ProductQAIssue.LACK_OF_DIVERSITY,
        scope=ProductQAEvidenceScope(
            kind=ProductQAScopeKind.CROSS_RECORDING_SEQUENCE,
            subject_refs=("cohort-1",),
        ),
        confidence=0.5,
        confidence_kind=ProductQAConfidenceKind.MODEL_REPORTED,
        evidence_refs=("comparison-1",),
    )

    with pytest.raises(ValueError, match="temporal interval is required"):
        evidence.to_clip_mark()

    result = QAClassifier().assess_evidence(
        "recording-1",
        10_000_000_000,
        (evidence,),
    )
    assert result.assessment.status is QAStatus.WARNING
    assert result.assessment.clip_marks == ()
    assert result.issue_evidence == (evidence,)
    assert result.unprojected_issue_evidence == (evidence,)


def test_clip_mark_requires_a_nonempty_interval() -> None:
    with pytest.raises(ValidationError, match="start_sec must be less than end_sec"):
        _mark(2.0, 2.0)


def test_classifier_pass_is_local_retained_and_not_production_eligible() -> None:
    result = QAClassifier().assess("recording-1", 10.0)

    assert result.status is QAStatus.PASS
    assert result.effective_duration_sec == 10.0
    assert result.retained is True
    assert result.delete_source is False
    assert result.production_eligible is False


def test_classifier_subtracts_overlapping_intervals_once() -> None:
    result = QAClassifier().assess(
        "recording-1",
        10.0,
        (
            _mark(3.0, 6.0, ProductQAIssue.HAIR_BLOCKING_VIEW),
            _mark(1.0, 4.0, ProductQAIssue.ARM_HAND_OBSTRUCTED),
        ),
    )

    assert result.status is QAStatus.WARNING
    assert result.effective_duration_sec == 5.0
    assert tuple(mark.start_sec for mark in result.clip_marks) == (1.0, 3.0)


def test_recording_failure_issue_does_not_authorize_source_deletion() -> None:
    result = QAClassifier().assess(
        "recording-1",
        10.0,
        (_mark(2.0, 3.0, ProductQAIssue.BLURRY_LENS),),
    )

    assert result.status is QAStatus.FAIL
    assert result.retained is True
    assert result.delete_source is False
    assert result.production_eligible is False


def test_full_coverage_black_screen_is_fail_but_partial_coverage_is_warning() -> None:
    classifier = QAClassifier()

    complete = classifier.assess(
        "recording-1",
        10.0,
        (_mark(4.0, 10.0), _mark(0.0, 4.0)),
    )
    partial = classifier.assess("recording-1", 10.0, (_mark(0.0, 9.9),))

    assert complete.status is QAStatus.FAIL
    assert complete.effective_duration_sec == 0.0
    assert partial.status is QAStatus.WARNING
    assert partial.effective_duration_sec == pytest.approx(0.1)


def test_recording_scope_projects_to_the_complete_duration() -> None:
    evidence = ProductQAIssueEvidence(
        issue=ProductQAIssue.BLURRY_LENS,
        scope=ProductQAEvidenceScope(
            kind=ProductQAScopeKind.CAMERA_RECORDING,
            subject_refs=("recording-1", "camera-1"),
            camera_id=CameraId.CAM_01,
        ),
        confidence=0.7,
        confidence_kind=ProductQAConfidenceKind.MODEL_REPORTED,
        evidence_refs=("inference-1",),
    )

    result = QAClassifier().assess_evidence(
        "recording-1",
        10_000_000_000,
        (evidence,),
    )

    assert result.assessment.status is QAStatus.FAIL
    assert result.assessment.clip_marks == (
        _mark(0.0, 10.0, ProductQAIssue.BLURRY_LENS).model_copy(update={"confidence": 0.7}),
    )
    assert result.assessment.effective_duration_sec == 0.0
    assert result.issue_evidence == (evidence,)


def test_assess_evidence_rejects_intervals_outside_the_recording() -> None:
    with pytest.raises(ValueError, match="issue interval must lie within"):
        QAClassifier().assess_evidence(
            "recording-1",
            1_500_000_000,
            (_evidence(),),
        )


def test_assessment_flags_cannot_be_promoted_or_changed_to_deletion() -> None:
    fields = {
        "recording_id": "recording-1",
        "duration_sec": 10.0,
        "status": QAStatus.PASS,
        "effective_duration_sec": 10.0,
    }

    with pytest.raises(ValidationError, match="production_eligible"):
        QAAssessment(**fields, production_eligible=True)
    with pytest.raises(ValidationError, match="delete_source"):
        QAAssessment(**fields, delete_source=True)



def test_complete_product_cascade_has_one_deterministic_state_for_every_class() -> None:
    result = ProductQACascadeProjector().project(
        recording_id="recording-1",
        recording_duration_ns=10_000_000_000,
    )

    assert result.status is ProductQACascadeStatus.COMPLETE
    assert tuple(item.issue for item in result.class_coverage) == tuple(ProductQAIssue)
    assert {item.state for item in result.class_coverage} == {ProductQAClassState.NO_ISSUE}
    assert result.product_result.assessment.status is QAStatus.PASS


def test_product_cascade_preserves_observed_issue_while_source_gap_marks_rest_incomplete() -> None:
    black = _evidence(issue=ProductQAIssue.BLACK_SCREEN)

    result = ProductQACascadeProjector().project(
        recording_id="recording-1",
        recording_duration_ns=10_000_000_000,
        observed_evidence=(black,),
        incomplete_reason_codes=("SOURCE_CADENCE_GAP:cam_01:1000000000",),
    )
    coverage = {item.issue: item for item in result.class_coverage}

    assert result.status is ProductQACascadeStatus.INCOMPLETE_INPUT
    assert coverage[ProductQAIssue.BLACK_SCREEN].state is ProductQAClassState.OBSERVED
    assert coverage[ProductQAIssue.BLACK_SCREEN].evidence == (black,)
    assert coverage[ProductQAIssue.ARM_HAND_OBSTRUCTED].state is (
        ProductQAClassState.INCOMPLETE_INPUT
    )
    assert coverage[ProductQAIssue.ARM_HAND_OBSTRUCTED].reason_codes == (
        "SOURCE_CADENCE_GAP:cam_01:1000000000",
    )
    assert result.product_result.issue_evidence == (black,)


def test_product_cascade_supports_a_selected_abstained_class_without_erasing_no_issue() -> None:
    result = ProductQACascadeProjector().project(
        recording_id="recording-1",
        recording_duration_ns=10_000_000_000,
        abstained_issues=(ProductQAIssue.ARM_HAND_OBSTRUCTED,),
    )
    coverage = {item.issue: item for item in result.class_coverage}

    assert result.status is ProductQACascadeStatus.ABSTAINED
    assert coverage[ProductQAIssue.ARM_HAND_OBSTRUCTED].state is ProductQAClassState.ABSTAINED
    assert coverage[ProductQAIssue.ARM_HAND_OBSTRUCTED].reason_codes == ("SEMANTIC_ABSTAINED",)
    assert coverage[ProductQAIssue.BLACK_SCREEN].state is ProductQAClassState.NO_ISSUE


def test_media_quality_context_maps_direct_and_proxy_evidence_at_original_timestamp() -> None:
    duration_ns = 10_000_000_000
    timestamp_ns = 2_000_000_000
    interval = NanosecondInterval(start_ns=0, end_ns=duration_ns)
    timings = {
        camera_id: (
            FrameTimingEvidence(
                camera_id=camera_id,
                packet_index=0,
                aligned_timestamp_ns=0,
                source_timestamp_ns=1_000_000_000,
                source_sequence=0,
            ),
        )
        for camera_id in CAMERA_IDS
    }
    flags = tuple(
        sorted(
            (
                LocalQualityFlag.OBSERVED_BLACK_LUMA,
                LocalQualityFlag.OBSERVED_OVEREXPOSED_LUMA,
                LocalQualityFlag.PROXY_LOW_EDGE_ENERGY,
                LocalQualityFlag.PROXY_FROZEN_CONTENT,
            ),
            key=lambda value: value.value,
        )
    )
    observations = {camera_id: () for camera_id in CAMERA_IDS}
    observations[CameraId.CAM_01] = (
        FrameQualityObservation(
            camera_id=CameraId.CAM_01,
            packet_index=2,
            aligned_timestamp_ns=timestamp_ns,
            source_timestamp_ns=3_000_000_000,
            grayscale_sha256="0" * 64,
            mean_luma_milli=127_000,
            black_fraction_ppm=1_000_000,
            overexposed_fraction_ppm=1_000_000,
            edge_energy_milli=0,
            frame_delta_milli=0,
            flags=flags,
        ),
    )
    report = build_local_media_quality_report(
        requested_max_duration_ns=duration_ns,
        recording_duration_ns=duration_ns,
        requested_interval=interval,
        timings=timings,
        frame_observations=observations,
    )

    context = product_qa_context_from_media_quality_report(report)
    by_issue = {item.issue: item for item in context.observed_evidence}

    assert context.media_quality_report_semantic_sha256 == report.semantic_sha256
    assert set(by_issue) == {
        ProductQAIssue.BLACK_SCREEN,
        ProductQAIssue.TOO_DARK_OR_OVEREXPOSED,
        ProductQAIssue.BLURRY_LENS,
        ProductQAIssue.CAMERA_STATIONARY_OVER_5S,
    }
    assert all(
        item.interval == NanosecondInterval(start_ns=timestamp_ns, end_ns=timestamp_ns + 1)
        for item in by_issue.values()
    )
    assert (
        by_issue[ProductQAIssue.BLACK_SCREEN].confidence_kind
        is ProductQAConfidenceKind.DETECTOR_REPORTED
    )
    assert (
        by_issue[ProductQAIssue.BLURRY_LENS].confidence_kind
        is ProductQAConfidenceKind.POLICY_DERIVED
    )
    assert "media-quality:" + report.semantic_sha256 in by_issue[
        ProductQAIssue.BLACK_SCREEN
    ].evidence_refs[0]
