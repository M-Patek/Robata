from __future__ import annotations

import pytest
from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.perception.single_route import SingleCameraAuthority, SingleCameraAuthorityPolicy

EXPECTED_PROJECTION_SHA256 = "ace58fab03745bbb57f9013af8378fa8df28aa0c96600c3ee15d8d5d58bbd7db"


def test_single_camera_policy_keeps_pure_native_mage_authority_and_identity() -> None:
    authority = SingleCameraAuthority(SingleCameraAuthorityPolicy())
    projection = authority.as_projection()

    assert projection == {
        "policy_version": "single-camera-mage-authority-v1",
        "camera_id": "cam_01",
        "authority_provider": "MAGE_NATIVE",
        "shadow_encoder_mode": "DISABLED",
        "worker_count": 1,
        "generation_concurrency": 1,
        "max_inflight_observations": 1,
        "raw_refine_provider": "MAGE_NATIVE",
    }
    assert exact_bytes_sha256(canonical_json_bytes(projection)) == EXPECTED_PROJECTION_SHA256


def test_small_encoder_shadow_route_is_not_admitted() -> None:
    with pytest.raises(ValidationError, match="shadow_encoder_mode"):
        SingleCameraAuthorityPolicy.model_validate(
            {"shadow_encoder_mode": "SHADOW_ONLY"},
            strict=True,
        )
