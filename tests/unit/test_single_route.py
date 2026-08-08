from __future__ import annotations

from pathlib import Path

import pytest

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.perception.single_route import (
    CameraEncoderInput,
    SingleCameraAuthority,
    SingleCameraAuthorityPolicy,
    SmallEncoderActionCandidate,
    SmallEncoderObservation,
    SmallEncoderQuality,
    SmallEncoderShadowComparison,
    SmallEncoderShadowMode,
    small_encoder_observation_semantic_sha256,
)


def _candidate_observation(**overrides: object) -> SmallEncoderObservation:
    base: dict[str, object] = {
        "encoder_id": "small_encoder_v0",
        "encoder_revision": "local-test-1",
        "camera_id": CameraId.CAM_01,
        "context_manifest_key": f"perception-context-v1:{'a' * 64}",
        "context_manifest_semantic_sha256": "a" * 64,
        "segment_semantic_sha256": "b" * 64,
        "interval": NanosecondInterval(start_ns=0, end_ns=8_000_000_000),
        "quality": SmallEncoderQuality.USABLE,
        "confidence": 0.8,
        "candidate_actions": (
            SmallEncoderActionCandidate(
                label="pick_up_object",
                interval=NanosecondInterval(start_ns=1_000_000_000, end_ns=2_000_000_000),
                confidence=0.7,
            ),
        ),
        "feature_artifact_exact_sha256": "c" * 64,
        "source_frame_exact_sha256_values": ("d" * 64,),
    }
    base.update(overrides)
    digest = semantic_sha256(
        SmallEncoderObservation.model_construct(
            schema_version="small-encoder-observation-v1",
            shadow_only=True,
            semantic_sha256="",
            **base,
        ).model_dump(mode="json", exclude={"semantic_sha256"})
    )
    return SmallEncoderObservation(semantic_sha256=digest, **base)


def test_single_camera_policy_keeps_native_mage_authority_and_one_worker() -> None:
    authority = SingleCameraAuthority(SingleCameraAuthorityPolicy())
    assert authority.as_projection() == {
        "policy_version": "single-camera-mage-authority-v1",
        "camera_id": "cam_01",
        "authority_provider": "MAGE_NATIVE",
        "shadow_encoder_mode": "DISABLED",
        "worker_count": 1,
        "generation_concurrency": 1,
        "max_inflight_observations": 1,
        "raw_refine_provider": "MAGE_NATIVE",
    }


def test_small_encoder_observation_is_shadow_only_and_identity_bound() -> None:
    observation = _candidate_observation()
    assert observation.shadow_only is True
    assert observation.semantic_sha256 == small_encoder_observation_semantic_sha256(observation)
    assert observation.candidate_actions[0].interval.end_ns == 2_000_000_000


def test_small_encoder_rejects_candidate_outside_bounded_segment() -> None:
    with pytest.raises(ValueError, match="inside its segment"):
        _candidate_observation(
            candidate_actions=(
                SmallEncoderActionCandidate(
                    label="pick_up_object",
                    interval=NanosecondInterval(
                        start_ns=7_000_000_000,
                        end_ns=9_000_000_000,
                    ),
                ),
            )
        )


def test_shadow_comparison_cannot_claim_publication() -> None:
    comparison = SmallEncoderShadowComparison(
        context_manifest_semantic_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        authority_observation_logical_key="mage-observation-v1:authority",
        authority_observation_semantic_sha256="b" * 64,
        candidate_observation_semantic_sha256="c" * 64,
    )
    assert comparison.status == "SHADOW_ONLY"
    assert SmallEncoderShadowMode.SHADOW_ONLY.value == "SHADOW_ONLY"


def test_camera_encoder_input_is_bounded_and_explicit(tmp_path: Path) -> None:
    source = tmp_path / "segment.mp4"
    source.write_bytes(b"segment")
    value = CameraEncoderInput(
        context_manifest_key=f"perception-context-v1:{'a' * 64}",
        context_manifest_semantic_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        segment_semantic_sha256="b" * 64,
        interval=NanosecondInterval(start_ns=0, end_ns=8_000_000_000),
        durable_path=str(source),
    )
    assert value.camera_id is CameraId.CAM_01
    assert value.durable_path == str(source)
