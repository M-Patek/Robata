"""Factories shared by perception-stream contract and reducer tests."""

from __future__ import annotations

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.perception_stream import (
    ActorObservation,
    BoundaryAssessment,
    CameraAbsenceReason,
    CameraContextBinding,
    CameraEvidenceRelation,
    CameraObservationEvidence,
    CognitionGateSignal,
    MageActionObservation,
    MageObservation,
    ObjectObservation,
    PerceptionContextManifest,
    SemanticCameraQa,
    SemanticQaDisposition,
    StorageSegmentReference,
    create_mage_observation,
    create_perception_context_manifest,
)
from robata.perception.projectors import (
    MediaHealthCameraFact,
    MediaHealthDisposition,
    MediaHealthReport,
    create_media_health_report,
)


def digest(seed: object) -> str:
    return semantic_sha256({"seed": seed})


def make_context(
    *,
    start_ns: int = 0,
    end_ns: int = 8_000_000_000,
    segment_ordinal: int = 0,
    selected_cameras: tuple[CameraId, ...] = CAMERA_IDS,
    unavailable_cameras: tuple[CameraId, ...] = (),
    source_recording_key: str = "recording:fixture",
    source_recording_seed: object = "recording",
    codec_policy_version: str = "mage-native-codec-v1",
    context_policy_version: str = "non-overlap-segment-v1",
) -> PerceptionContextManifest:
    segment_digest = digest(("segment", segment_ordinal, start_ns, end_ns))
    segments = (
        StorageSegmentReference(
            segment_ordinal=segment_ordinal,
            segment_key=f"stream-segment-vnext:{segment_digest}",
            segment_semantic_sha256=segment_digest,
            interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        ),
    )
    cameras: dict[CameraId, CameraContextBinding] = {}
    for camera_id in CAMERA_IDS:
        unavailable = camera_id in unavailable_cameras
        cameras[camera_id] = CameraContextBinding(
            camera_id=camera_id,
            available=not unavailable,
            selected_for_inference=(camera_id in selected_cameras and not unavailable),
            codec_stream_exact_sha256=(
                None if unavailable else digest(("codec", camera_id.value, start_ns, end_ns))
            ),
            segment_semantic_sha256_values=() if unavailable else (segment_digest,),
            absence_reason=CameraAbsenceReason.MISSING if unavailable else None,
        )
    return create_perception_context_manifest(
        source_recording_key=source_recording_key,
        source_recording_exact_sha256=digest(source_recording_seed),
        context_interval=NanosecondInterval(start_ns=start_ns, end_ns=end_ns),
        ordered_segments=segments,
        focus_segment_ordinal=segment_ordinal,
        cameras=SixCameraMap(cameras),
        codec_policy_version=codec_policy_version,
        context_policy_version=context_policy_version,
    )


def make_observation(
    *,
    context: PerceptionContextManifest | None = None,
    local_ref: str = "o1",
    action: str = "pick_up_cup",
    action_start_ns: int = 1_000_000_000,
    action_end_ns: int = 2_000_000_000,
    started_before_context: bool = False,
    continues_after_context: bool = False,
    start_confidence: float = 0.86,
    end_confidence: float = 0.82,
    contradicting_cameras: tuple[CameraId, ...] = (),
    unusable_qa_cameras: tuple[CameraId, ...] = (),
    inference_artifact_seed: object = "artifact",
    created_at: str = "2026-08-07T00:00:00Z",
) -> MageObservation:
    context = context or make_context()
    semantic_qa = SixCameraMap(
        {
            camera_id: SemanticCameraQa(
                camera_id=camera_id,
                disposition=(
                    SemanticQaDisposition.UNUSABLE
                    if camera_id in unusable_qa_cameras
                    else (
                        SemanticQaDisposition.USABLE
                        if (
                            context.cameras[camera_id].available
                            and context.cameras[camera_id].selected_for_inference
                        )
                        else SemanticQaDisposition.UNKNOWN
                    )
                ),
                issues=(),
                confidence=(
                    0.2
                    if camera_id in unusable_qa_cameras
                    else (
                        0.95
                        if (
                            context.cameras[camera_id].available
                            and context.cameras[camera_id].selected_for_inference
                        )
                        else None
                    )
                ),
            )
            for camera_id in CAMERA_IDS
        }
    )
    observed_interval = NanosecondInterval(
        start_ns=context.context_interval.start_ns,
        end_ns=context.context_interval.end_ns,
    )
    evidence = SixCameraMap(
        {
            camera_id: CameraObservationEvidence(
                camera_id=camera_id,
                relation=(
                    CameraEvidenceRelation.CONTRADICTS
                    if camera_id in contradicting_cameras
                    else (
                        CameraEvidenceRelation.SUPPORTS
                        if context.cameras[camera_id].selected_for_inference
                        else CameraEvidenceRelation.NOT_OBSERVABLE
                    )
                ),
                visibility=(0.9 if context.cameras[camera_id].selected_for_inference else None),
                observed_interval=(
                    observed_interval if context.cameras[camera_id].selected_for_inference else None
                ),
                evidence_semantic_sha256_values=(
                    (digest((local_ref, camera_id.value)),)
                    if context.cameras[camera_id].selected_for_inference
                    else ()
                ),
            )
            for camera_id in CAMERA_IDS
        }
    )
    action_observation = MageActionObservation(
        local_ref=local_ref,
        action=action,
        interval=NanosecondInterval(start_ns=action_start_ns, end_ns=action_end_ns),
        confidence=0.93,
        actor=ActorObservation(hand="right", actor_type="robot_hand"),
        object=ObjectObservation(object_type="cup"),
        camera_evidence=evidence,
        boundary=BoundaryAssessment(
            start_confidence=start_confidence,
            end_confidence=end_confidence,
            started_before_context=started_before_context,
            continues_after_context=continues_after_context,
        ),
    )
    return create_mage_observation(
        observation_schema_version="mage-observation-v1",
        context=context,
        model_family="mage_vl",
        model_revision="5c78cab61938e73859b63724d9bf5cb88c477eaa",
        model_artifact_manifest_sha256=digest("mage-model-manifest"),
        prompt_version="mage-unified-observation-prompt-v1",
        inference_artifact_exact_sha256=digest(inference_artifact_seed),
        cognition_gate=CognitionGateSignal(
            score=0.25,
            threshold=0.5,
            would_admit=False,
            gate_policy_version="mage-gate-shadow-v1",
        ),
        semantic_qa=semantic_qa,
        observations=(action_observation,),
        created_at=created_at,
    )


def make_media_health(
    context: PerceptionContextManifest,
    *,
    degraded_camera: CameraId | None = None,
) -> MediaHealthReport:
    cameras: dict[CameraId, MediaHealthCameraFact] = {}
    for camera_id in CAMERA_IDS:
        context_camera = context.cameras[camera_id]
        if not context_camera.available:
            disposition = MediaHealthDisposition.UNAVAILABLE
            issue_codes = ("camera_missing",)
            interval = None
        elif camera_id is degraded_camera:
            disposition = MediaHealthDisposition.DEGRADED
            issue_codes = ("exposure_anomaly",)
            interval = context.context_interval
        else:
            disposition = MediaHealthDisposition.HEALTHY
            issue_codes = ()
            interval = context.context_interval
        cameras[camera_id] = MediaHealthCameraFact(
            camera_id=camera_id,
            disposition=disposition,
            issue_codes=issue_codes,
            observed_interval=interval,
        )
    return create_media_health_report(
        context_manifest_semantic_sha256=context.context_manifest_semantic_sha256,
        policy_version="deterministic-media-health-v1",
        cameras=SixCameraMap(cameras),
    )
