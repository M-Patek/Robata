from __future__ import annotations

from pathlib import Path

import pytest

from robata.application.canonical.recording_association import (
    CanonicalRecordingAssociationBridge,
    RecordingAssociationPublicationStatus,
    RecordingAssociationReportConflict,
    RecordingAssociationReportStorageError,
    RecordingAssociationReportStore,
)
from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes
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


def _logical_key(namespace: str, value: int) -> str:
    return f"{namespace}:{_digest(value)}"


def _recording() -> CompletedRecordingAssociationBinding:
    return CompletedRecordingAssociationBinding(
        completed_run_id="00000000-0000-4000-8000-000000000001",
        completed_recording_logical_key=_logical_key("completed-recording", 1),
        completed_recording_semantic_sha256=_digest(1),
        completed_recording_exact_sha256=_digest(2),
        mcap_id="00000000-0000-4000-8000-000000000002",
        source_content_sha256=_digest(3),
        camera_mapping_semantic_sha256=_digest(4),
        alignment_semantic_sha256=_digest(5),
    )


def _source_action(ordinal: int) -> AssociationSourceActionRef:
    digest = _digest(100 + ordinal)
    return AssociationSourceActionRef(
        source_action_logical_key=f"accepted-action:{digest}",
        source_action_semantic_sha256=digest,
    )


def _input(
    ordinal: int,
    *,
    start_ns: int,
    end_ns: int,
    camera_id: CameraId = CameraId.CAM_01,
    label: str = "turn",
) -> RecordingAssociationInput:
    source_action = _source_action(ordinal)
    evidence_digest = _digest(200 + ordinal)
    evidence = AssociationAcceptedEvidenceRef(
        source_action=source_action,
        accepted_evidence_logical_key=f"accepted-camera-evidence:{evidence_digest}",
        accepted_evidence_semantic_sha256=evidence_digest,
        accepted_evidence_exact_sha256=_digest(300 + ordinal),
        camera_id=camera_id,
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        label=label,
        confidence_millionths=900_000,
    )
    recording = _recording()
    return RecordingAssociationInput.create(
        source_action=source_action,
        mcap_id=recording.mcap_id,
        source_content_sha256=recording.source_content_sha256,
        camera_mapping_semantic_sha256=recording.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=recording.alignment_semantic_sha256,
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        label=label,
        accepted_evidence=(evidence,),
    )


def _bridge(
    tmp_path: Path,
    *,
    max_gap_ns: int = 0,
) -> tuple[RecordingAssociationReportStore, CanonicalRecordingAssociationBridge]:
    policy = RecordingAssociationPolicy.create(
        version="canonical-recording-association-test-v1",
        max_gap_ns=max_gap_ns,
    )
    store = RecordingAssociationReportStore(tmp_path / "association-reports")
    return store, CanonicalRecordingAssociationBridge(RecordingAssociationEngine(policy), store)


def test_detached_bridge_publishes_exact_accepted_evidence_after_completed_run(
    tmp_path: Path,
) -> None:
    store, bridge = _bridge(tmp_path)
    recording = _recording()
    first = _input(1, start_ns=0, end_ns=10)
    second = _input(2, start_ns=10, end_ns=20)

    result = bridge.derive_and_publish(
        recording=recording,
        inputs=(second, first),
        bridge_evidence=(),
    )

    assert result.status is RecordingAssociationPublicationStatus.PUBLISHED
    assert result.replayed is False
    assert result.report is not None
    assert result.report.recording == recording
    assert result.report.outcome is RecordingAssociationOutcome.ASSOCIATIONS_DERIVED
    assert result.report.inputs[0].accepted_evidence
    assert result.report.inputs[1].accepted_evidence
    path = store.report_path(result.report.semantic_sha256)
    assert path.read_bytes() == canonical_json_bytes(result.report)
    assert store.get(result.report.semantic_sha256) == result.report


def test_restart_replays_same_derived_report_bytes_without_reduction_dependency(
    tmp_path: Path,
) -> None:
    store, bridge = _bridge(tmp_path)
    recording = _recording()
    inputs = (_input(1, start_ns=0, end_ns=10), _input(2, start_ns=10, end_ns=20))
    first = bridge.derive_and_publish(
        recording=recording,
        inputs=inputs,
        bridge_evidence=(),
    )
    assert first.report is not None

    restarted_store, restarted_bridge = _bridge(tmp_path)
    replay = restarted_bridge.derive_and_publish(
        recording=recording,
        inputs=tuple(reversed(inputs)),
        bridge_evidence=(),
    )

    assert replay.status is RecordingAssociationPublicationStatus.REPLAYED
    assert replay.replayed is True
    assert replay.report is not None
    assert replay.report == first.report
    assert restarted_store.get(first.report.semantic_sha256) == first.report
    assert restarted_store.report_path(first.report.semantic_sha256).read_bytes() == (
        store.report_path(first.report.semantic_sha256).read_bytes()
    )


def test_store_rejects_different_content_and_tampered_bytes(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)
    recording = _recording()
    report = bridge.derive_and_publish(
        recording=recording,
        inputs=(_input(1, start_ns=0, end_ns=10), _input(2, start_ns=10, end_ns=20)),
        bridge_evidence=(),
    ).report
    assert report is not None
    conflicting_report = RecordingAssociationEngine(
        RecordingAssociationPolicy.create(
            version="canonical-recording-association-other-policy-v1",
            max_gap_ns=1,
        )
    ).derive(
        recording=recording,
        inputs=(_input(3, start_ns=100, end_ns=110),),
    )
    path = store.report_path(report.semantic_sha256)
    path.write_bytes(canonical_json_bytes(conflicting_report))

    with pytest.raises(RecordingAssociationReportConflict, match="different immutable"):
        store.put_or_get(report)

    path.write_bytes(b"{}")
    with pytest.raises(RecordingAssociationReportStorageError, match="invalid report"):
        store.get(report.semantic_sha256)


def test_no_associations_is_persisted_as_an_event_neutral_derived_result(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)
    result = bridge.derive_and_publish(
        recording=_recording(),
        inputs=(
            _input(1, start_ns=0, end_ns=10, camera_id=CameraId.CAM_01),
            _input(2, start_ns=100, end_ns=110, camera_id=CameraId.CAM_02),
        ),
        bridge_evidence=(),
    )
    assert result.report is not None

    assert result.report.outcome is RecordingAssociationOutcome.NO_ASSOCIATIONS
    assert result.report.clusters == ()
    stored_bytes = store.report_path(result.report.semantic_sha256).read_bytes()
    assert b"NO_EVENTS" not in stored_bytes
    assert result.report.production_eligible is False


def test_no_accepted_evidence_does_not_publish_or_claim_an_event_outcome(tmp_path: Path) -> None:
    store, bridge = _bridge(tmp_path)

    result = bridge.derive_and_publish(
        recording=_recording(),
        inputs=(),
        bridge_evidence=(),
    )

    assert result.status is RecordingAssociationPublicationStatus.NO_ACCEPTED_EVIDENCE
    assert result.report is None
    assert result.replayed is False
    assert tuple((store.root / "reports").iterdir()) == ()
