from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from robata.contracts import CameraId, NanosecondInterval, SixCameraMap
from robata.contracts.perception_stream import (
    CameraEvidenceRelation,
    CameraObservationEvidence,
    CognitionGateSignal,
    MageActionObservation,
    MageObservation,
    RefineReason,
    RefineTargetField,
    StorageSegmentReference,
    create_perception_context_manifest,
    create_perception_refine_request,
)
from robata.perception.projectors import EventProjector
from tests.support.perception_stream import digest, make_context, make_observation


def test_nonoverlap_context_identity_is_deterministic_and_rejects_overlap() -> None:
    first = make_context()
    replay = make_context()

    assert first == replay
    assert first.context_manifest_key.endswith(first.context_manifest_semantic_sha256)

    segment_one = first.ordered_segments[0]
    second_digest = digest("overlap")
    overlapping = StorageSegmentReference(
        segment_ordinal=segment_one.segment_ordinal + 1,
        segment_key=f"stream-segment-vnext:{second_digest}",
        segment_semantic_sha256=second_digest,
        interval=NanosecondInterval(start_ns=7_000_000_000, end_ns=9_000_000_000),
    )
    payload = first.model_dump(mode="python")
    payload["context_interval"] = NanosecondInterval(start_ns=0, end_ns=9_000_000_000)
    payload["ordered_segments"] = (segment_one, overlapping)
    for camera_id in CameraId:
        camera = payload["cameras"][camera_id]
        camera["segment_semantic_sha256_values"] = (
            segment_one.segment_semantic_sha256,
            second_digest,
        )

    with pytest.raises(ValidationError, match="must not overlap"):
        create_perception_context_manifest(
            source_recording_key=payload["source_recording_key"],
            source_recording_exact_sha256=payload["source_recording_exact_sha256"],
            context_interval=payload["context_interval"],
            ordered_segments=payload["ordered_segments"],
            focus_segment_ordinal=1,
            cameras=SixCameraMap(payload["cameras"]),
            codec_policy_version=payload["codec_policy_version"],
            context_policy_version=payload["context_policy_version"],
        )


def test_logical_inference_identity_is_stable_while_artifact_semantics_are_distinct() -> None:
    context = make_context()
    first = make_observation(context=context, inference_artifact_seed="attempt-a")
    recomputed = make_observation(
        context=context,
        inference_artifact_seed="attempt-b",
        created_at="2026-08-07T00:01:00Z",
    )

    assert first.observation_id == recomputed.observation_id
    assert first.observation_logical_key == recomputed.observation_logical_key
    assert first.observation_identity_sha256 == recomputed.observation_identity_sha256
    assert first.observation_semantic_sha256 != recomputed.observation_semantic_sha256


def test_observation_rejects_non_observable_camera_claims_and_semantic_tamper() -> None:
    context = make_context(selected_cameras=(CameraId.CAM_01,))
    observation = make_observation(context=context)
    payload = deepcopy(observation.model_dump(mode="python"))
    action = MageActionObservation.model_validate(payload["observations"][0], strict=True)
    camera_map = {
        camera_id: CameraObservationEvidence.model_validate(value, strict=True)
        for camera_id, value in action.camera_evidence.model_dump(mode="python").items()
    }
    camera_map[CameraId.CAM_02] = CameraObservationEvidence(
        camera_id=CameraId.CAM_02,
        relation=CameraEvidenceRelation.SUPPORTS,
        visibility=0.8,
        observed_interval=context.context_interval,
        evidence_semantic_sha256_values=(digest("forbidden-evidence"),),
    )
    payload["observations"] = (
        action.model_copy(update={"camera_evidence": SixCameraMap(camera_map)}),
    )

    with pytest.raises(ValidationError, match="unavailable or unselected"):
        MageObservation.model_validate(payload, strict=True)

    tampered = observation.model_dump(mode="python")
    tampered["inference_artifact_exact_sha256"] = digest("tampered")
    with pytest.raises(ValidationError, match="semantic digest"):
        MageObservation.model_validate(tampered, strict=True)


def test_cognition_gate_is_a_shadow_signal_not_an_admission_decision() -> None:
    with pytest.raises(ValidationError, match="must be derived"):
        CognitionGateSignal(
            score=0.25,
            threshold=0.5,
            would_admit=True,
            gate_policy_version="mage-gate-shadow-v1",
        )

    observation = make_observation()
    assert observation.cognition_gate.mode == "SHADOW_ONLY"
    assert observation.cognition_gate.would_admit is False
    assert len(EventProjector().project(observation).hypotheses) == 1


def test_refinement_identity_is_narrow_ordered_and_deterministic() -> None:
    observation = make_observation()
    hypothesis = EventProjector().project(observation).hypotheses[0]
    first = create_perception_refine_request(
        source_observation_logical_key=observation.observation_logical_key,
        source_observation_semantic_sha256=observation.observation_semantic_sha256,
        target_hypothesis_logical_key=hypothesis.hypothesis_logical_key,
        target_hypothesis_semantic_sha256=hypothesis.hypothesis_semantic_sha256,
        reason=RefineReason.BOUNDARY,
        target_fields=(RefineTargetField.END_BOUNDARY, RefineTargetField.START_BOUNDARY),
        refine_interval=NanosecondInterval(start_ns=500_000_000, end_ns=2_500_000_000),
        refine_policy_version="bounded-refine-v1",
        prompt_version="boundary-only-v1",
    )
    replay = create_perception_refine_request(
        source_observation_logical_key=observation.observation_logical_key,
        source_observation_semantic_sha256=observation.observation_semantic_sha256,
        target_hypothesis_logical_key=hypothesis.hypothesis_logical_key,
        target_hypothesis_semantic_sha256=hypothesis.hypothesis_semantic_sha256,
        reason=RefineReason.BOUNDARY,
        target_fields=(RefineTargetField.START_BOUNDARY, RefineTargetField.END_BOUNDARY),
        refine_interval=NanosecondInterval(start_ns=500_000_000, end_ns=2_500_000_000),
        refine_policy_version="bounded-refine-v1",
        prompt_version="boundary-only-v1",
    )

    assert first == replay
    assert first.target_fields == (
        RefineTargetField.END_BOUNDARY,
        RefineTargetField.START_BOUNDARY,
    )
