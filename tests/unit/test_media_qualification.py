from __future__ import annotations

import pytest

from robata.benchmark.media_qualification import (
    MediaBackend,
    MediaExecutionMode,
    MediaParityStatus,
    MediaQualificationMeasurement,
    MediaSourceProfile,
    build_media_qualification_report,
    measure_media_callable,
)

_DIGEST = "a" * 64
_TIMESTAMP_DIGEST = "b" * 64
_ARTIFACT_DIGEST = "c" * 64
_SELECTED_FRAME_DIGEST = "d" * 64
_DIMENSION_DIGEST = "e" * 64
_EXACT_BYTE_DIGEST = "f" * 64
_SEMANTIC_DIGEST = "0" * 64
_NVDEC_RUNTIME_PROVENANCE_DIGEST = "1" * 64


def _profile(label: str = "h264-1080p") -> MediaSourceProfile:
    return MediaSourceProfile.create(
        codec="h264",
        width=1920,
        height=1080,
        fps_num=30,
        gop_frames=30,
        transfer_path="object-store-to-device",
        source_manifest_digest=_DIGEST,
    )


def _measurement(
    backend: MediaBackend,
    *,
    profile: MediaSourceProfile | None = None,
    execution_mode: MediaExecutionMode = MediaExecutionMode.FRESH,
    timestamp_digest: str | None = _TIMESTAMP_DIGEST,
    artifact_digest: str | None = _ARTIFACT_DIGEST,
    camera_seconds: float = 150.0,
    wall_time_ns: int = 1_000_000_000,
    fallback_count: int = 0,
    runtime_provenance_digest: str | None = _NVDEC_RUNTIME_PROVENANCE_DIGEST,
) -> MediaQualificationMeasurement:
    return MediaQualificationMeasurement(
        workload_manifest_digest=_DIGEST,
        run_namespace=f"media-{backend.value.lower()}",
        backend=backend,
        execution_mode=execution_mode,
        source_profile=profile or _profile(),
        camera_seconds=camera_seconds,
        wall_time_ns=wall_time_ns,
        decoded_frames=4500,
        resized_frames=4500,
        materialized_frames=4500,
        timestamp_parity=MediaParityStatus.MATCH,
        artifact_parity=MediaParityStatus.MATCH,
        timestamp_contract_digest=timestamp_digest,
        artifact_contract_digest=artifact_digest,
        selected_frame_parity=MediaParityStatus.MATCH,
        dimension_parity=MediaParityStatus.MATCH,
        exact_byte_parity=MediaParityStatus.MATCH,
        semantic_parity=MediaParityStatus.MATCH,
        selected_frame_contract_digest=_SELECTED_FRAME_DIGEST,
        dimension_contract_digest=_DIMENSION_DIGEST,
        exact_byte_contract_digest=_EXACT_BYTE_DIGEST,
        semantic_contract_digest=_SEMANTIC_DIGEST,
        media_runtime_provenance_digest=(
            runtime_provenance_digest if backend is MediaBackend.NVDEC else None
        ),
        nvdec_fallback_count=fallback_count,
    )


def test_complete_fresh_pair_meets_125_average_and_150_margin() -> None:
    report = build_media_qualification_report(
        (_measurement(MediaBackend.CPU), _measurement(MediaBackend.NVDEC)),
        target_hardware_status="MEASURED",
    )

    assert report.safe_envelope is True
    assert report.average_camera_seconds_per_second == pytest.approx(150.0)
    assert report.minimum_camera_seconds_per_second == pytest.approx(150.0)
    assert report.production_eligible is False


def test_missing_profile_pair_is_not_safe() -> None:
    report = build_media_qualification_report(
        (_measurement(MediaBackend.CPU),),
        target_hardware_status="MEASURED",
    )

    assert report.safe_envelope is False
    assert "MEDIA_MATRIX_INCOMPLETE" in report.envelope.unmet_requirements
    assert "REQUIRED_BACKEND_NOT_MEASURED" in report.envelope.unmet_requirements


def test_parity_requires_equal_non_null_contract_digests() -> None:
    report = build_media_qualification_report(
        (
            _measurement(MediaBackend.CPU),
            _measurement(MediaBackend.NVDEC, timestamp_digest="d" * 64),
        ),
        target_hardware_status="MEASURED",
    )

    assert report.envelope.parity_complete is False
    assert "MEDIA_CONTRACT_PARITY_NOT_PROVEN" in report.envelope.unmet_requirements
    assert report.safe_envelope is False


def test_target_matrix_requires_selected_frame_dimension_byte_and_semantic_parity() -> None:
    nvdec = _measurement(MediaBackend.NVDEC).model_copy(
        update={"exact_byte_contract_digest": "1" * 64}
    )
    report = build_media_qualification_report(
        (_measurement(MediaBackend.CPU), nvdec),
        target_hardware_status="MEASURED",
    )

    assert report.envelope.parity_complete is False
    assert "MEDIA_CONTRACT_PARITY_NOT_PROVEN" in report.envelope.unmet_requirements


def test_target_matrix_requires_runtime_provenance_for_nvdec_parity() -> None:
    report = build_media_qualification_report(
        (
            _measurement(MediaBackend.CPU),
            _measurement(MediaBackend.NVDEC, runtime_provenance_digest=None),
        ),
        target_hardware_status="MEASURED",
    )

    assert report.envelope.parity_complete is False
    assert "MEDIA_CONTRACT_PARITY_NOT_PROVEN" in report.envelope.unmet_requirements
    assert report.safe_envelope is False


def test_fresh_media_measurements_require_unique_run_namespaces() -> None:
    cpu = _measurement(MediaBackend.CPU)
    nvdec = _measurement(MediaBackend.NVDEC).model_copy(update={"run_namespace": cpu.run_namespace})

    with pytest.raises(ValueError, match="unique run namespaces"):
        build_media_qualification_report(
            (cpu, nvdec),
            target_hardware_status="MEASURED",
        )


def test_match_parity_requires_contract_digests() -> None:
    with pytest.raises(ValueError, match="contract digest"):
        _measurement(MediaBackend.CPU, timestamp_digest=None)


def test_target_hardware_and_fresh_execution_are_explicit_gates() -> None:
    report = build_media_qualification_report(
        (
            _measurement(MediaBackend.CPU, execution_mode=MediaExecutionMode.REPLAY),
            _measurement(MediaBackend.NVDEC, execution_mode=MediaExecutionMode.REPLAY),
        ),
    )

    assert report.safe_envelope is False
    assert "TARGET_HARDWARE_NOT_MEASURED" in report.envelope.unmet_requirements
    assert "FRESH_MEDIA_RUN_NOT_MEASURED" in report.envelope.unmet_requirements


def test_nvdec_fallback_cannot_claim_parity() -> None:
    with pytest.raises(ValueError, match="fallback observations"):
        _measurement(MediaBackend.NVDEC, fallback_count=1)


def test_measure_callable_records_elapsed_time_and_counters() -> None:
    ticks = iter((10.0, 11.0))
    measurement = measure_media_callable(
        lambda: {"decoded_frames": 3, "resized_frames": 2, "materialized_frames": 1},
        workload_manifest_digest=_DIGEST,
        run_namespace="media-callable",
        backend=MediaBackend.CPU,
        execution_mode=MediaExecutionMode.FRESH,
        source_profile=_profile(),
        camera_seconds=1.0,
        clock=lambda: next(ticks),
    )

    assert measurement.wall_time_ns == 1_000_000_000
    assert measurement.decoded_frames == 3
    assert measurement.resized_frames == 2
    assert measurement.materialized_frames == 1


def test_measure_callable_records_direct_publication_durability_facts() -> None:
    ticks = iter((20.0, 21.0))
    measurement = measure_media_callable(
        lambda: {"decoded_frames": 1, "resized_frames": 1, "materialized_frames": 1},
        workload_manifest_digest=_DIGEST,
        run_namespace="media-publication",
        backend=MediaBackend.CPU,
        execution_mode=MediaExecutionMode.FRESH,
        source_profile=_profile(),
        camera_seconds=1.0,
        stats={
            "fsync_count": 2,
            "write_duration_ns": 11,
            "fsync_duration_ns": 13,
            "rename_duration_ns": 17,
            "directory_sync_duration_ns": 19,
            "publication_duration_ns": 23,
            "end_to_end_duration_ns": 29,
        },
        clock=lambda: next(ticks),
    )

    assert measurement.fsync_count == 2
    assert measurement.write_duration_ns == 11
    assert measurement.fsync_duration_ns == 13
    assert measurement.rename_duration_ns == 17
    assert measurement.directory_sync_duration_ns == 19
    assert measurement.publication_duration_ns == 23
    assert measurement.end_to_end_duration_ns == 29
    assert measurement.durability_measured is True


def test_report_digest_detects_tampering() -> None:
    report = build_media_qualification_report(
        (_measurement(MediaBackend.CPU), _measurement(MediaBackend.NVDEC)),
        target_hardware_status="MEASURED",
    )
    tampered = report.model_dump(mode="python")
    tampered["workload_manifest_digest"] = "d" * 64
    with pytest.raises(ValueError, match=r"workload manifest|report_sha256"):
        type(report).model_validate(tampered, strict=True)
