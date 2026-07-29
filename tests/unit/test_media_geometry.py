from __future__ import annotations

import pytest

from robata.application.canonical.media_geometry import (
    CameraCalibrationProfile,
    DerivedGeometryArtifact,
    FramePreprocessPolicy,
    GeometryMapCache,
    GeometryProcessingError,
    GeometryView,
    materialize_geometry_view,
)
from robata.contracts.cameras import CameraId
from robata.contracts.hashing import semantic_sha256
from robata.ports.decoded_frame import DecodedFrameView


def _profile(camera_id: CameraId = CameraId.CAM_01) -> CameraCalibrationProfile:
    return CameraCalibrationProfile(
        profile_id="calibration-test",
        version="calibration-v1",
        camera_id=camera_id,
        source_width=2,
        source_height=2,
        intrinsic_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        distortion_coefficients=(0.1, -0.01),
    )


def _frame() -> DecodedFrameView:
    return DecodedFrameView(timestamp_ns=20, width=2, height=2, gray_pixels=b"\x01\x02\x03\x04")


def test_raw_view_is_deterministic_and_does_not_require_calibration() -> None:
    cache = GeometryMapCache(max_entries=2)
    policy = FramePreprocessPolicy()
    first = materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
        policy=policy,
        map_cache=cache,
    )
    replay = materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
        policy=policy,
        map_cache=cache,
    )

    assert isinstance(first, DerivedGeometryArtifact)
    assert first.payload == _frame().gray_pixels
    assert first.fact == replay.fact
    assert first.fact.calibration_sha256 is None
    assert first.fact.lineage_sha256 == semantic_sha256(
        {
            "projection_version": "canonical-derived-geometry-lineage-v1",
            "source_frame_id": "source-frame",
            "source_content_sha256": "a" * 64,
            "camera_id": "cam_01",
            "source_timestamp_ns": "10",
            "aligned_timestamp_ns": "20",
            "view": "RAW",
            "calibration_sha256": None,
            "preprocess_policy_version": policy.version,
            "preprocess_policy_sha256": policy.semantic_sha256,
            "source_width": 2,
            "source_height": 2,
            "output_width": 2,
            "output_height": 2,
            "exact_bytes_sha256": first.fact.exact_bytes_sha256,
            "bytes": 4,
        }
    )
    assert cache.size == 1


def test_supplemental_geometry_requires_calibration_and_binds_every_identity_input() -> None:
    policy = FramePreprocessPolicy(
        version="preprocess-test-v1",
        view=GeometryView.FISHEYE_UNDISTORTED,
        output_width=1,
        output_height=1,
    )
    profile = _profile()
    result = materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
        policy=policy,
        calibration=profile,
        map_cache=GeometryMapCache(max_entries=2),
    )

    assert result.payload == b"\x01"
    assert result.fact.view is GeometryView.FISHEYE_UNDISTORTED
    assert result.fact.calibration_sha256 == profile.semantic_sha256
    assert result.fact.output_width == 1
    assert result.fact.output_height == 1
    assert result.fact.frame_id != "source-frame"

    changed_time = materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=11,
        aligned_timestamp_ns=20,
        policy=policy,
        calibration=profile,
        map_cache=GeometryMapCache(max_entries=2),
    )
    changed_policy = materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
        policy=policy.model_copy(update={"version": "preprocess-test-v2"}),
        calibration=profile,
        map_cache=GeometryMapCache(max_entries=2),
    )
    assert changed_time.fact.frame_id != result.fact.frame_id
    assert changed_policy.fact.frame_id != result.fact.frame_id

    with pytest.raises(GeometryProcessingError, match="require calibration"):
        materialize_geometry_view(
            _frame(),
            source_frame_id="source-frame",
            source_content_sha256="a" * 64,
            camera_id=CameraId.CAM_01,
            source_timestamp_ns=10,
            aligned_timestamp_ns=20,
            policy=policy,
        )


def test_geometry_map_cache_is_bounded_and_profile_camera_dimensions_are_checked() -> None:
    cache = GeometryMapCache(max_entries=1)
    first = _profile()
    second = first.model_copy(update={"profile_id": "calibration-test-2"})
    policy = FramePreprocessPolicy(view=GeometryView.PERSPECTIVE)
    materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
        policy=policy,
        calibration=first,
        map_cache=cache,
    )
    materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
        policy=policy,
        calibration=second,
        map_cache=cache,
    )
    assert cache.size == 1

    with pytest.raises(GeometryProcessingError, match="dimensions"):
        materialize_geometry_view(
            DecodedFrameView(timestamp_ns=20, width=1, height=4, gray_pixels=b"\x01\x02\x03\x04"),
            source_frame_id="source-frame",
            source_content_sha256="a" * 64,
            camera_id=CameraId.CAM_01,
            source_timestamp_ns=10,
            aligned_timestamp_ns=20,
            policy=policy,
            calibration=first,
        )


def test_geometry_lineage_and_aligned_timestamp_are_self_consistent() -> None:
    artifact = materialize_geometry_view(
        _frame(),
        source_frame_id="source-frame",
        source_content_sha256="a" * 64,
        camera_id=CameraId.CAM_01,
        source_timestamp_ns=10,
        aligned_timestamp_ns=20,
    )
    tampered = artifact.fact.model_dump(mode="python")
    tampered["lineage_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="lineage_sha256"):
        type(artifact.fact).model_validate(tampered, strict=True)

    with pytest.raises(GeometryProcessingError, match="aligned geometry timestamp"):
        materialize_geometry_view(
            _frame(),
            source_frame_id="source-frame",
            source_content_sha256="a" * 64,
            camera_id=CameraId.CAM_01,
            source_timestamp_ns=10,
            aligned_timestamp_ns=21,
        )
