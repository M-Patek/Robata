from __future__ import annotations

from dataclasses import replace

import pytest

from robata.application.canonical.media_quality import (
    FrameTimingEvidence,
    build_local_media_quality_report,
)
from robata.application.canonical.media_quality_source_binding import (
    bind_registered_media_quality_source,
)
from robata.application.canonical.media_quality_supplemental import (
    freeze_registered_media_quality_targets,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.schema_registry import SchemaRegistry
from tests.unit.test_sampling_materializer import _digest


def _timing(
    camera_id: CameraId,
    packet_index: int,
    timestamp_ns: int,
    sequence: int,
) -> FrameTimingEvidence:
    return FrameTimingEvidence(
        camera_id=camera_id,
        packet_index=packet_index,
        aligned_timestamp_ns=timestamp_ns,
        source_timestamp_ns=1_000_000_000 + timestamp_ns,
        source_sequence=sequence,
    )


def _report(*, degraded: bool):
    interval = NanosecondInterval(start_ns=0, end_ns=600_000_000)
    base_times = (
        (0, 100_000_000, 200_000_000, 500_000_000)
        if degraded
        else (0, 100_000_000, 200_000_000, 300_000_000)
    )
    sequences = (0, 1, 3, 4) if degraded else (0, 1, 2, 3)
    timings = {
        camera_id: tuple(
            _timing(
                camera_id,
                packet_index,
                timestamp_ns + (camera_index * 1_000_000 if degraded else 0),
                sequences[packet_index],
            )
            for packet_index, timestamp_ns in enumerate(base_times)
        )
        for camera_index, camera_id in enumerate(CAMERA_IDS)
    }
    return build_local_media_quality_report(
        requested_max_duration_ns=600_000_000,
        recording_duration_ns=1_000_000_000,
        requested_interval=interval,
        timings=timings,
        frame_observations={camera_id: () for camera_id in CAMERA_IDS},
    )


def _freeze(report):
    registry = SchemaRegistry()
    source_binding = bind_registered_media_quality_source(
        report,
        registry=registry,
        source_content_sha256=_digest("source-content"),
        camera_mapping_semantic_sha256=_digest("mapping"),
        alignment_semantic_sha256=_digest("alignment"),
    )
    return freeze_registered_media_quality_targets(
        report,
        registry=registry,
        source_binding=source_binding,
        selection_tolerance_ns=100_000_000,
        tie_break_policy_version="nearest-v1",
        dedupe_policy_version="one-source-frame-v1",
    )


def test_registered_report_targets_freeze_without_coordinate_reinterpretation() -> None:
    report = _report(degraded=True)
    plan = _freeze(report)

    assert plan is not None
    assert plan.source_report_semantic_sha256 == report.semantic_sha256
    assert plan.source_binding.semantic_sha256
    assert plan.source_target_plan_semantic_sha256 == report.supplemental_targets.semantic_sha256
    assert tuple((item.camera_id, item.target_ns) for item in plan.targets) == tuple(
        (item.camera_id, item.target_ns) for item in report.supplemental_targets.targets
    )
    assert plan.target_policy_version == report.supplemental_targets.policy_version


def test_clean_registered_report_does_not_create_an_empty_qa_package_plan() -> None:
    assert _freeze(_report(degraded=False)) is None


def test_tampered_report_digest_cannot_be_frozen() -> None:
    report = replace(_report(degraded=True), semantic_sha256=_digest("tampered"))

    with pytest.raises(ValueError, match="semantic digest is inconsistent"):
        _freeze(report)


def test_foreign_source_binding_cannot_freeze_another_report() -> None:
    registry = SchemaRegistry()
    degraded = _report(degraded=True)
    clean = _report(degraded=False)
    binding = bind_registered_media_quality_source(
        clean,
        registry=registry,
        source_content_sha256=_digest("source-content"),
        camera_mapping_semantic_sha256=_digest("mapping"),
        alignment_semantic_sha256=_digest("alignment"),
    )

    with pytest.raises(ValueError, match="does not bind the report"):
        freeze_registered_media_quality_targets(
            degraded,
            registry=registry,
            source_binding=binding,
            selection_tolerance_ns=100_000_000,
            tie_break_policy_version="nearest-v1",
            dedupe_policy_version="one-source-frame-v1",
        )
