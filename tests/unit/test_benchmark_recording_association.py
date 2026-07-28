from __future__ import annotations

import pytest

from robata.benchmark.association import (
    AssociationFixtureTruthCluster,
    build_recording_association_fixture_metrics,
)
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.event_pipeline.recording_association import (
    AssociationAcceptedEvidenceRef,
    AssociationSourceActionRef,
    CompletedRecordingAssociationBinding,
    RecordingAssociationEngine,
    RecordingAssociationInput,
    RecordingAssociationOutcome,
    RecordingAssociationPolicy,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _key(namespace: str, value: int) -> str:
    return f"{namespace}:{_digest(value)}"


def _recording() -> CompletedRecordingAssociationBinding:
    return CompletedRecordingAssociationBinding(
        completed_run_id="00000000-0000-0000-0000-000000000001",
        completed_recording_logical_key=_key("completed-recording", 1),
        completed_recording_semantic_sha256=_digest(1),
        completed_recording_exact_sha256=_digest(2),
        mcap_id="00000000-0000-0000-0000-000000000003",
        source_content_sha256=_digest(4),
        camera_mapping_semantic_sha256=_digest(5),
        alignment_semantic_sha256=_digest(6),
    )


def _source(value: int) -> AssociationSourceActionRef:
    return AssociationSourceActionRef(
        source_action_logical_key=_key("accepted-action", value),
        source_action_semantic_sha256=_digest(value),
    )


def _input(value: int, start_ns: int, end_ns: int) -> RecordingAssociationInput:
    recording = _recording()
    source = _source(value)
    evidence = AssociationAcceptedEvidenceRef(
        source_action=source,
        accepted_evidence_logical_key=_key("accepted-camera-evidence", 100 + value),
        accepted_evidence_semantic_sha256=_digest(100 + value),
        accepted_evidence_exact_sha256=_digest(200 + value),
        camera_id=CameraId.CAM_01,
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        label="turn",
        confidence_millionths=900_000,
    )
    return RecordingAssociationInput.create(
        source_action=source,
        mcap_id=recording.mcap_id,
        source_content_sha256=recording.source_content_sha256,
        camera_mapping_semantic_sha256=recording.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=recording.alignment_semantic_sha256,
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        label="turn",
        accepted_evidence=(evidence,),
    )


def _report(*inputs: RecordingAssociationInput):
    return RecordingAssociationEngine(
        RecordingAssociationPolicy.create(version="fixture-association-metrics-v1")
    ).derive(recording=_recording(), inputs=inputs)


def _truth_cluster(*inputs: RecordingAssociationInput) -> AssociationFixtureTruthCluster:
    return AssociationFixtureTruthCluster(
        source_actions=tuple(
            sorted(
                (item.source_action for item in inputs),
                key=lambda item: item.source_action_logical_key,
            )
        )
    )


def test_fixture_metrics_are_content_addressed_and_explicitly_not_representative() -> None:
    first = _input(10, 0, 10)
    second = _input(11, 10, 20)
    report = _report(first, second)

    metrics = build_recording_association_fixture_metrics(
        fixture_id="p11-touching-pair",
        report=report,
        expected_clusters=(_truth_cluster(first, second),),
    )
    replay = build_recording_association_fixture_metrics(
        fixture_id="p11-touching-pair",
        report=report,
        expected_clusters=(_truth_cluster(second, first),),
    )

    assert report.outcome is RecordingAssociationOutcome.ASSOCIATIONS_DERIVED
    assert metrics == replay
    assert metrics.true_positive_pair_count == 1
    assert metrics.precision_millionths == 1_000_000
    assert metrics.recall_millionths == 1_000_000
    assert metrics.f1_millionths == 1_000_000
    assert metrics.association_coverage_millionths == 1_000_000
    assert metrics.representative_measurement_status == "NOT_MEASURED"
    assert metrics.production_eligible is False
    assert metrics.logical_key.endswith(metrics.semantic_sha256)


def test_fixture_metrics_keep_false_positive_and_false_negative_populations_separate() -> None:
    first = _input(20, 0, 10)
    second = _input(21, 10, 20)
    third = _input(22, 50, 60)
    report = _report(first, second, third)

    metrics = build_recording_association_fixture_metrics(
        fixture_id="p11-mismatched-pair",
        report=report,
        expected_clusters=(_truth_cluster(second, third),),
    )

    assert metrics.predicted_associated_pair_count == 1
    assert metrics.expected_associated_pair_count == 1
    assert metrics.true_positive_pair_count == 0
    assert metrics.false_positive_pair_count == 1
    assert metrics.false_negative_pair_count == 1
    assert metrics.precision_millionths == 0
    assert metrics.recall_millionths == 0
    assert metrics.f1_millionths == 0
    assert metrics.association_coverage_millionths == 666_666


def test_fixture_metrics_record_valid_no_association_without_an_event_absence_claim() -> None:
    report = _report(_input(30, 0, 10))
    metrics = build_recording_association_fixture_metrics(
        fixture_id="p11-singleton",
        report=report,
        expected_clusters=(),
    )

    assert report.outcome is RecordingAssociationOutcome.NO_ASSOCIATIONS
    assert metrics.predicted_associated_pair_count == 0
    assert metrics.expected_associated_pair_count == 0
    assert metrics.precision_millionths == 1_000_000
    assert metrics.recall_millionths == 1_000_000
    assert metrics.association_coverage_millionths == 0
    assert "NO_EVENTS" not in str(metrics.model_dump(mode="json"))


def test_fixture_metrics_reject_truth_that_is_not_bound_to_report_inputs() -> None:
    first = _input(40, 0, 10)
    second = _input(41, 10, 20)
    foreign = _input(42, 20, 30)

    with pytest.raises(ValueError, match="absent from the report"):
        build_recording_association_fixture_metrics(
            fixture_id="p11-foreign-truth",
            report=_report(first, second),
            expected_clusters=(_truth_cluster(first, foreign),),
        )
