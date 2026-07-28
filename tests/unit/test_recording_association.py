from __future__ import annotations

import pytest

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.event_pipeline.recording_association import (
    AssociationAcceptedEvidenceRef,
    AssociationBridgeKind,
    AssociationInputDisposition,
    AssociationPairDisposition,
    AssociationReasonCode,
    AssociationSourceActionRef,
    CompletedRecordingAssociationBinding,
    RecordingAssociationBridgeEvidence,
    RecordingAssociationEngine,
    RecordingAssociationError,
    RecordingAssociationInput,
    RecordingAssociationOutcome,
    RecordingAssociationPolicy,
    verify_recording_association_report,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _key(namespace: str, value: int) -> str:
    return f"{namespace}:{_digest(value)}"


def _source(value: int) -> AssociationSourceActionRef:
    return AssociationSourceActionRef(
        source_action_logical_key=_key("source-action", value),
        source_action_semantic_sha256=_digest(value),
    )


def _recording() -> CompletedRecordingAssociationBinding:
    return CompletedRecordingAssociationBinding(
        completed_run_id="00000000-0000-0000-0000-000000000001",
        completed_recording_logical_key=_key("completed-recording", 2),
        completed_recording_semantic_sha256=_digest(2),
        completed_recording_exact_sha256=_digest(3),
        mcap_id="00000000-0000-0000-0000-000000000004",
        source_content_sha256=_digest(5),
        camera_mapping_semantic_sha256=_digest(6),
        alignment_semantic_sha256=_digest(7),
    )


def _input(
    value: int,
    start_ns: int,
    end_ns: int,
    *,
    label: str = "turn",
    cameras: tuple[CameraId, ...] = (CameraId.CAM_01,),
    confidence: int = 900_000,
) -> RecordingAssociationInput:
    recording = _recording()
    source = _source(value)
    evidence = tuple(
        AssociationAcceptedEvidenceRef(
            source_action=source,
            accepted_evidence_logical_key=_key("accepted-evidence", value * 10 + ordinal),
            accepted_evidence_semantic_sha256=_digest(value * 10 + ordinal),
            accepted_evidence_exact_sha256=_digest(value * 100 + ordinal),
            camera_id=camera_id,
            interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
            label=label,
            confidence_millionths=confidence,
        )
        for ordinal, camera_id in enumerate(cameras, start=1)
    )
    return RecordingAssociationInput.create(
        source_action=source,
        mcap_id=recording.mcap_id,
        source_content_sha256=recording.source_content_sha256,
        camera_mapping_semantic_sha256=recording.camera_mapping_semantic_sha256,
        alignment_semantic_sha256=recording.alignment_semantic_sha256,
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        label=label,
        accepted_evidence=evidence,
    )


def _bridge(
    value: int,
    left: RecordingAssociationInput,
    right: RecordingAssociationInput,
    *,
    start_ns: int,
    end_ns: int,
    kind: AssociationBridgeKind = AssociationBridgeKind.GAP_CONTINUITY,
    confidence: int = 900_000,
) -> RecordingAssociationBridgeEvidence:
    return RecordingAssociationBridgeEvidence(
        source_actions=tuple(
            sorted(
                (left.source_action, right.source_action),
                key=lambda item: item.source_action_logical_key,
            )
        ),
        accepted_bridge_evidence_logical_key=_key("accepted-bridge-evidence", value),
        accepted_bridge_evidence_semantic_sha256=_digest(value),
        accepted_bridge_evidence_exact_sha256=_digest(value + 1_000),
        camera_id=CameraId.CAM_01,
        interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        confidence_millionths=confidence,
        kind=kind,
    )


def _engine(**kwargs: object) -> RecordingAssociationEngine:
    policy = RecordingAssociationPolicy.create(version="fixture-v1", **kwargs)
    return RecordingAssociationEngine(policy)


def test_touching_and_overlap_are_deterministic_and_replayable() -> None:
    first = _input(10, 0, 10)
    second = _input(11, 10, 30)

    report = _engine().derive(recording=_recording(), inputs=(second, first))
    replay = _engine().derive(recording=_recording(), inputs=(first, second))

    assert report.model_dump(mode="json") == replay.model_dump(mode="json")
    assert report.outcome is RecordingAssociationOutcome.ASSOCIATIONS_DERIVED
    assert report.pair_decisions[0].disposition is AssociationPairDisposition.MERGED
    assert report.pair_decisions[0].reason_codes == (
        AssociationReasonCode.TOUCHING_OR_OVERLAPPING_SHARED_CAMERA,
    )
    assert report.clusters[0].interval == NanosecondInterval(start_ns=0, end_ns=30)
    assert report.production_eligible is False
    assert "NO_EVENTS" not in {item.value for item in RecordingAssociationOutcome}
    assert verify_recording_association_report(report) == report


def test_gap_and_label_transition_require_explicit_bridge_evidence() -> None:
    first = _input(20, 0, 10)
    second = _input(21, 12, 20)
    engine = _engine(max_gap_ns=3)

    unbridged = engine.derive(recording=_recording(), inputs=(first, second))
    assert unbridged.outcome is RecordingAssociationOutcome.NO_ASSOCIATIONS
    assert unbridged.pair_decisions[0].reason_codes == (
        AssociationReasonCode.GAP_REQUIRES_BRIDGE_EVIDENCE,
    )

    bridged = engine.derive(
        recording=_recording(),
        inputs=(first, second),
        bridge_evidence=(_bridge(201, first, second, start_ns=10, end_ns=12),),
    )
    assert bridged.pair_decisions[0].temporal_gap_ns == 2
    assert bridged.pair_decisions[0].reason_codes == (
        AssociationReasonCode.GAP_CONTINUITY_EVIDENCE,
    )

    transition = _input(22, 21, 30, label="exit")
    transition_bridge = _bridge(
        202,
        second,
        transition,
        start_ns=20,
        end_ns=21,
        kind=AssociationBridgeKind.LABEL_TRANSITION,
    )
    disabled = _engine(max_gap_ns=2).derive(
        recording=_recording(), inputs=(second, transition), bridge_evidence=(transition_bridge,)
    )
    assert disabled.pair_decisions[0].reason_codes == (
        AssociationReasonCode.LABEL_TRANSITIONS_DISABLED,
    )
    enabled = _engine(max_gap_ns=2, allow_label_transitions=True).derive(
        recording=_recording(), inputs=(second, transition), bridge_evidence=(transition_bridge,)
    )
    assert enabled.clusters[0].labels == ("exit", "turn")
    assert enabled.pair_decisions[0].reason_codes == (
        AssociationReasonCode.LABEL_TRANSITION_EVIDENCE,
    )


def test_camera_incompatibility_and_gap_limit_produce_explained_split() -> None:
    first = _input(30, 0, 10, cameras=(CameraId.CAM_01,))
    second = _input(31, 10, 20, cameras=(CameraId.CAM_02,))
    report = _engine().derive(recording=_recording(), inputs=(first, second))
    assert report.outcome is RecordingAssociationOutcome.NO_ASSOCIATIONS
    assert report.pair_decisions[0].reason_codes == (
        AssociationReasonCode.INSUFFICIENT_SHARED_CAMERA_SUPPORT,
    )

    distant = _input(32, 30, 40)
    gap_report = _engine(max_gap_ns=3).derive(recording=_recording(), inputs=(first, distant))
    assert gap_report.pair_decisions[0].reason_codes == (AssociationReasonCode.GAP_EXCEEDS_POLICY,)


def test_tied_bridges_are_ambiguity_and_not_a_silent_merge() -> None:
    first = _input(40, 0, 10)
    second = _input(41, 12, 20)
    third = _input(42, 12, 20)
    report = _engine(max_gap_ns=3).derive(
        recording=_recording(),
        inputs=(third, first, second),
        bridge_evidence=(
            _bridge(401, first, second, start_ns=10, end_ns=12),
            _bridge(402, first, third, start_ns=10, end_ns=12),
        ),
    )

    assert (
        sum(
            item.disposition is AssociationPairDisposition.AMBIGUOUS
            for item in report.pair_decisions
        )
        == 2
    )
    assert report.input_decisions[0].disposition is AssociationInputDisposition.AMBIGUOUS
    assert report.clusters[0].source_actions == (second.source_action, third.source_action)


def test_duplicate_tampered_and_incomplete_bindings_are_rejected() -> None:
    first = _input(50, 0, 10)
    second = _input(51, 10, 20)
    engine = _engine()
    with pytest.raises(RecordingAssociationError, match="duplicate"):
        engine.derive(recording=_recording(), inputs=(first, first))

    foreign_recording = _recording().model_copy(update={"source_content_sha256": _digest(999)})
    with pytest.raises(RecordingAssociationError, match="completed recording"):
        engine.derive(recording=foreign_recording, inputs=(first, second))

    tampered = first.model_copy(update={"semantic_sha256": _digest(998)})
    with pytest.raises(ValueError, match="semantic identity"):
        engine.derive(recording=_recording(), inputs=(tampered, second))

    foreign_evidence = AssociationAcceptedEvidenceRef(
        source_action=_source(999),
        accepted_evidence_logical_key=_key("accepted-evidence", 999),
        accepted_evidence_semantic_sha256=_digest(999),
        accepted_evidence_exact_sha256=_digest(998),
        camera_id=CameraId.CAM_01,
        interval=NanosecondInterval(start_ns=0, end_ns=10),
        label="turn",
        confidence_millionths=900_000,
    )
    with pytest.raises(ValueError, match="bind the input source action"):
        RecordingAssociationInput.create(
            source_action=first.source_action,
            mcap_id=first.mcap_id,
            source_content_sha256=first.source_content_sha256,
            camera_mapping_semantic_sha256=first.camera_mapping_semantic_sha256,
            alignment_semantic_sha256=first.alignment_semantic_sha256,
            interval=NanosecondInterval(start_ns=0, end_ns=10),
            label="turn",
            accepted_evidence=(foreign_evidence,),
        )


def test_no_associations_is_not_a_no_events_claim() -> None:
    report = _engine().derive(recording=_recording(), inputs=(_input(60, 0, 10),))
    assert report.outcome is RecordingAssociationOutcome.NO_ASSOCIATIONS
    assert report.pair_decisions == ()
    assert report.input_decisions[0].disposition is AssociationInputDisposition.UNASSOCIATED
    assert report.input_decisions[0].reason_codes == (
        AssociationReasonCode.NO_SUPPORTED_ASSOCIATION,
    )
