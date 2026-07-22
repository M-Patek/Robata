from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from robata.application.canonical.media_quality import (
    FrameQualityObservation,
    FrameTimingEvidence,
    LocalMediaQualityReport,
    LocalQualityFlag,
    build_local_media_quality_report,
    registered_local_media_quality_report_document,
)
from robata.application.canonical.media_quality_binding import (
    LocalMediaQualityBinding,
    derive_local_media_quality_binding,
    derive_local_media_quality_binding_document,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.schema_registry import SchemaRegistry


def _timing(camera_id: CameraId, packet_index: int) -> FrameTimingEvidence:
    timestamp_ns = packet_index * 1_000_000_000
    return FrameTimingEvidence(
        camera_id=camera_id,
        packet_index=packet_index,
        aligned_timestamp_ns=timestamp_ns,
        source_timestamp_ns=timestamp_ns + 10_000,
        source_sequence=packet_index,
    )


def _observation(
    camera_id: CameraId,
    packet_index: int,
    *flags: LocalQualityFlag,
) -> FrameQualityObservation:
    timestamp_ns = packet_index * 1_000_000_000
    return FrameQualityObservation(
        camera_id=camera_id,
        packet_index=packet_index,
        aligned_timestamp_ns=timestamp_ns,
        source_timestamp_ns=timestamp_ns + 10_000,
        grayscale_sha256=f"{packet_index + 1:064x}",
        mean_luma_milli=1_000,
        black_fraction_ppm=0,
        overexposed_fraction_ppm=0,
        edge_energy_milli=0,
        frame_delta_milli=None,
        flags=tuple(flags),
    )


def _report(*, with_flags: bool) -> LocalMediaQualityReport:
    timings = {
        camera_id: (_timing(camera_id, 0), _timing(camera_id, 1)) for camera_id in CAMERA_IDS
    }
    observations = {camera_id: () for camera_id in CAMERA_IDS}
    if with_flags:
        observations[CameraId.CAM_01] = (
            _observation(
                CameraId.CAM_01,
                0,
                LocalQualityFlag.PROXY_LOW_EDGE_ENERGY,
                LocalQualityFlag.OBSERVED_BLACK_LUMA,
            ),
            _observation(
                CameraId.CAM_01,
                1,
                LocalQualityFlag.PROXY_LOW_EDGE_ENERGY,
            ),
        )
    return build_local_media_quality_report(
        requested_max_duration_ns=2_000_000_000,
        recording_duration_ns=2_000_000_000,
        requested_interval=NanosecondInterval(start_ns=0, end_ns=2_000_000_000),
        timings=timings,
        frame_observations=observations,
    )


def test_binding_is_deterministic_and_counts_exact_flags() -> None:
    report = _report(with_flags=True)

    binding = derive_local_media_quality_binding(report)
    replay = derive_local_media_quality_binding(report)

    assert binding == replay
    assert binding.report_semantic_sha256 == report.semantic_sha256
    assert (
        binding.supplemental_target_plan_semantic_sha256
        == report.supplemental_targets.semantic_sha256
    )
    assert binding.evidence_class == "LOCAL_CONFORMANCE"
    assert binding.production_eligible is False
    assert binding.requires_review is True
    assert tuple((item.flag, item.occurrence_count) for item in binding.flag_counts) == (
        (LocalQualityFlag.OBSERVED_BLACK_LUMA, 1),
        (LocalQualityFlag.PROXY_LOW_EDGE_ENERGY, 2),
    )


def test_persisted_document_derives_the_same_binding_for_recovery() -> None:
    report = _report(with_flags=True)
    registry = SchemaRegistry()
    document = registered_local_media_quality_report_document(report, registry)

    recovered = derive_local_media_quality_binding_document(document, registry)

    assert recovered == derive_local_media_quality_binding(report)


def test_low_edge_proxy_requires_review_without_becoming_semantic_occlusion() -> None:
    binding = derive_local_media_quality_binding(_report(with_flags=True))

    assert LocalQualityFlag.PROXY_LOW_EDGE_ENERGY in {item.flag for item in binding.flag_counts}
    assert "OCCLUSION" not in binding.model_dump_json()


def test_clean_report_does_not_require_quality_review() -> None:
    binding = derive_local_media_quality_binding(_report(with_flags=False))

    assert binding.flag_counts == ()
    assert binding.requires_review is False


def test_derivation_rejects_tampered_report_digest() -> None:
    tampered = replace(_report(with_flags=True), semantic_sha256="0" * 64)

    with pytest.raises(ValueError, match="media quality report semantic digest"):
        derive_local_media_quality_binding(tampered)


def test_binding_rejects_review_and_digest_inconsistency() -> None:
    binding = derive_local_media_quality_binding(_report(with_flags=True))
    values = binding.model_dump(mode="python")
    values["requires_review"] = False

    with pytest.raises(ValidationError, match="requires_review"):
        LocalMediaQualityBinding.model_validate(values, strict=True)

    values = binding.model_dump(mode="python")
    values["semantic_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="semantic_sha256"):
        LocalMediaQualityBinding.model_validate(values, strict=True)
