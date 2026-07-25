from __future__ import annotations

import av
import pytest

from robata.application.canonical.media_quality import (
    FrameTimingEvidence,
    LocalFrameQualityAnalyzer,
    LocalMediaQualityPolicy,
    LocalQualityFlag,
    NeighborTargetPolicy,
    QualityTriggerProvenance,
    QualityTriggerSource,
    build_local_media_quality_report,
    plan_neighbor_targets,
    pyav_decoded_frame_view,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.ports.decoded_frame import DecodedFrameView


def _gray_frame(rows: tuple[tuple[int, ...], ...]) -> av.VideoFrame:
    height = len(rows)
    width = len(rows[0])
    assert all(len(row) == width for row in rows)
    frame = av.VideoFrame(width=width, height=height, format="gray")
    plane = frame.planes[0]
    contents = bytearray(plane.buffer_size)
    for row_index, row in enumerate(rows):
        start = row_index * plane.line_size
        contents[start : start + width] = bytes(row)
    plane.update(contents)
    return frame


def _decoded_view(
    rows: tuple[tuple[int, ...], ...],
    *,
    timestamp_ns: int,
    analysis_width: int = 64,
) -> DecodedFrameView:
    return pyav_decoded_frame_view(
        _gray_frame(rows),
        timestamp_ns=timestamp_ns,
        analysis_width=analysis_width,
    )


def _timing(
    camera_id: CameraId,
    packet_index: int,
    aligned_timestamp_ns: int,
    source_sequence: int | None = None,
) -> FrameTimingEvidence:
    return FrameTimingEvidence(
        camera_id=camera_id,
        packet_index=packet_index,
        aligned_timestamp_ns=aligned_timestamp_ns,
        source_timestamp_ns=1_000_000_000 + aligned_timestamp_ns,
        source_sequence=packet_index if source_sequence is None else source_sequence,
    )


def test_frame_analyzer_observes_luma_and_keeps_blur_as_a_proxy() -> None:
    analyzer = LocalFrameQualityAnalyzer(CameraId.CAM_01)
    black = analyzer.observe(
        _decoded_view(((0, 0, 0, 0),) * 4, timestamp_ns=0),
        _timing(CameraId.CAM_01, 0, 0),
    )
    overexposed = analyzer.observe(
        _decoded_view(((255, 255, 255, 255),) * 4, timestamp_ns=1_000_000_000),
        _timing(CameraId.CAM_01, 1, 1_000_000_000),
    )
    checkerboard = analyzer.observe(
        _decoded_view(
            (
                (0, 255, 0, 255),
                (255, 0, 255, 0),
                (0, 255, 0, 255),
                (255, 0, 255, 0),
            ),
            timestamp_ns=2_000_000_000,
        ),
        _timing(CameraId.CAM_01, 2, 2_000_000_000),
    )

    assert LocalQualityFlag.OBSERVED_BLACK_LUMA in black.flags
    assert LocalQualityFlag.PROXY_LOW_EDGE_ENERGY in black.flags
    assert LocalQualityFlag.OBSERVED_OVEREXPOSED_LUMA in overexposed.flags
    assert LocalQualityFlag.PROXY_LOW_EDGE_ENERGY in overexposed.flags
    assert LocalQualityFlag.PROXY_LOW_EDGE_ENERGY not in checkerboard.flags
    assert checkerboard.edge_energy_milli > black.edge_energy_milli


def test_frame_analyzer_marks_only_sustained_stable_content_as_freeze_proxy() -> None:
    policy = LocalMediaQualityPolicy(
        freeze_delta_milli=0,
        freeze_min_duration_ns=2_000_000_000,
    )
    analyzer = LocalFrameQualityAnalyzer(CameraId.CAM_01, policy)
    rows = ((80, 80, 80, 80),) * 4

    first = analyzer.observe(_decoded_view(rows, timestamp_ns=0), _timing(CameraId.CAM_01, 0, 0))
    second = analyzer.observe(
        _decoded_view(rows, timestamp_ns=1_000_000_000),
        _timing(CameraId.CAM_01, 1, 1_000_000_000),
    )
    third = analyzer.observe(
        _decoded_view(rows, timestamp_ns=2_000_000_000),
        _timing(CameraId.CAM_01, 2, 2_000_000_000),
    )

    assert LocalQualityFlag.PROXY_FROZEN_CONTENT not in first.flags
    assert LocalQualityFlag.PROXY_FROZEN_CONTENT not in second.flags
    assert LocalQualityFlag.PROXY_FROZEN_CONTENT in third.flags
    assert third.frame_delta_milli == 0


def test_frame_analyzer_resets_freeze_proxy_after_content_changes() -> None:
    policy = LocalMediaQualityPolicy(
        freeze_delta_milli=0,
        freeze_min_duration_ns=2_000_000_000,
    )
    analyzer = LocalFrameQualityAnalyzer(CameraId.CAM_01, policy)
    first_rows = ((80, 80, 80, 80),) * 4
    changed_rows = ((160, 160, 160, 160),) * 4

    analyzer.observe(
        _decoded_view(first_rows, timestamp_ns=0),
        _timing(CameraId.CAM_01, 0, 0),
    )
    analyzer.observe(
        _decoded_view(first_rows, timestamp_ns=1_000_000_000),
        _timing(CameraId.CAM_01, 1, 1_000_000_000),
    )
    changed = analyzer.observe(
        _decoded_view(changed_rows, timestamp_ns=2_000_000_000),
        _timing(CameraId.CAM_01, 2, 2_000_000_000),
    )
    stable_once = analyzer.observe(
        _decoded_view(changed_rows, timestamp_ns=3_000_000_000),
        _timing(CameraId.CAM_01, 3, 3_000_000_000),
    )
    stable_long_enough = analyzer.observe(
        _decoded_view(changed_rows, timestamp_ns=4_000_000_000),
        _timing(CameraId.CAM_01, 4, 4_000_000_000),
    )

    assert LocalQualityFlag.PROXY_FROZEN_CONTENT not in changed.flags
    assert LocalQualityFlag.PROXY_FROZEN_CONTENT not in stable_once.flags
    assert LocalQualityFlag.PROXY_FROZEN_CONTENT in stable_long_enough.flags


def test_report_observes_cadence_sequence_and_cross_camera_skew() -> None:
    interval = NanosecondInterval(start_ns=0, end_ns=600_000_000)
    timings = {
        camera_id: tuple(
            _timing(
                camera_id,
                packet_index,
                timestamp_ns + camera_index * 1_000_000,
                source_sequence=(0, 1, 3, 4)[packet_index],
            )
            for packet_index, timestamp_ns in enumerate((0, 100_000_000, 200_000_000, 500_000_000))
        )
        for camera_index, camera_id in enumerate(CAMERA_IDS)
    }
    observations = {camera_id: () for camera_id in CAMERA_IDS}
    policy = LocalMediaQualityPolicy(sync_skew_threshold_ns=2_000_000)

    report = build_local_media_quality_report(
        requested_max_duration_ns=600_000_000,
        recording_duration_ns=1_000_000_000,
        requested_interval=interval,
        timings=timings,
        frame_observations=observations,
        policy=policy,
    )
    replay = build_local_media_quality_report(
        requested_max_duration_ns=600_000_000,
        recording_duration_ns=1_000_000_000,
        requested_interval=interval,
        timings=timings,
        frame_observations=observations,
        policy=policy,
    )

    assert report.window_limited
    assert all(len(ledger.cadence_gaps) == 1 for ledger in report.camera_ledgers)
    assert all(len(ledger.sequence_gaps) == 1 for ledger in report.camera_ledgers)
    assert report.cross_camera_skew.max_ns == 5_000_000
    assert report.cross_camera_skew.flags == (LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW,)
    assert report.supplemental_targets.targets
    assert report.semantic_sha256 == replay.semantic_sha256
    assert (
        report.supplemental_targets.semantic_sha256 == replay.supplemental_targets.semantic_sha256
    )


def test_neighbor_targets_preserve_unclipped_before_and_after_coordinates() -> None:
    trigger = QualityTriggerProvenance(
        camera_id=CameraId.CAM_03,
        trigger_timestamp_ns=500,
        source=QualityTriggerSource.FRAME,
        flag=LocalQualityFlag.OBSERVED_BLACK_LUMA,
        packet_index=7,
    )
    plan = plan_neighbor_targets(
        (trigger,),
        interval=NanosecondInterval(start_ns=0, end_ns=1_000),
        policy=NeighborTargetPolicy(
            offsets_ns=(-100, 100),
            max_targets_per_camera=4,
            max_targets_total=4,
        ),
    )

    assert plan.clipped_count == 0
    assert tuple(target.target_ns for target in plan.targets) == (400, 600)
    assert tuple(
        (item.offset_ns, item.requested_target_ns, item.clipped)
        for target in plan.targets
        for item in target.provenance
    ) == ((-100, 400, False), (100, 600, False))


def test_neighbor_targets_clip_dedupe_budget_and_preserve_provenance() -> None:
    interval = NanosecondInterval(start_ns=0, end_ns=1_000)
    triggers = (
        QualityTriggerProvenance(
            camera_id=CameraId.CAM_01,
            trigger_timestamp_ns=50,
            source=QualityTriggerSource.FRAME,
            flag=LocalQualityFlag.OBSERVED_BLACK_LUMA,
            packet_index=1,
        ),
        QualityTriggerProvenance(
            camera_id=CameraId.CAM_01,
            trigger_timestamp_ns=50,
            source=QualityTriggerSource.FRAME,
            flag=LocalQualityFlag.PROXY_LOW_EDGE_ENERGY,
            packet_index=1,
        ),
    )
    policy = NeighborTargetPolicy(
        offsets_ns=(-100, 100),
        max_targets_per_camera=1,
        max_targets_total=1,
    )

    plan = plan_neighbor_targets(triggers, interval=interval, policy=policy)
    replay = plan_neighbor_targets(tuple(reversed(triggers)), interval=interval, policy=policy)

    assert plan.candidate_count == 4
    assert plan.clipped_count == 2
    assert plan.deduplicated_count == 2
    assert plan.dropped_by_per_camera_budget == 1
    assert plan.dropped_by_total_budget == 0
    assert tuple((target.camera_id, target.target_ns) for target in plan.targets) == (
        (CameraId.CAM_01, 0),
    )
    assert len(plan.targets[0].provenance) == 2
    assert all(item.clipped for item in plan.targets[0].provenance)
    assert plan.semantic_sha256 == replay.semantic_sha256


def test_pyav_decoded_frame_view_is_row_major_and_strips_plane_padding() -> None:
    rows = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )

    view = _decoded_view(rows, timestamp_ns=123)
    downscaled = _decoded_view(rows, timestamp_ns=124, analysis_width=2)

    assert (view.timestamp_ns, view.width, view.height) == (123, 4, 2)
    assert view.gray_pixels == bytes(range(8))
    assert (downscaled.width, downscaled.height, downscaled.pixel_count) == (2, 1, 2)


def test_frame_analyzer_requires_a_view_with_matching_exact_timestamp() -> None:
    analyzer = LocalFrameQualityAnalyzer(CameraId.CAM_01)
    timing = _timing(CameraId.CAM_01, 0, 0)

    with pytest.raises(TypeError, match="DecodedFrameView"):
        analyzer.observe(_gray_frame(((0,),)), timing)
    with pytest.raises(ValueError, match="timestamp differs"):
        analyzer.observe(_decoded_view(((0,),), timestamp_ns=1), timing)


def test_decoded_frame_view_requires_exact_row_major_dimensions() -> None:
    with pytest.raises(ValueError, match=r"width \* height"):
        DecodedFrameView(timestamp_ns=0, width=2, height=2, gray_pixels=b"\x00")
    with pytest.raises(TypeError, match="immutable bytes"):
        DecodedFrameView(
            timestamp_ns=0,
            width=1,
            height=1,
            gray_pixels=bytearray(b"\x00"),  # type: ignore[arg-type]
        )


def test_frame_analyzer_rejects_changed_dimensions_even_when_pixel_count_matches() -> None:
    analyzer = LocalFrameQualityAnalyzer(CameraId.CAM_01)
    analyzer.observe(
        DecodedFrameView(timestamp_ns=0, width=2, height=2, gray_pixels=bytes(4)),
        _timing(CameraId.CAM_01, 0, 0),
    )

    with pytest.raises(ValueError, match="dimensions must remain stable"):
        analyzer.observe(
            DecodedFrameView(timestamp_ns=1, width=1, height=4, gray_pixels=bytes(4)),
            _timing(CameraId.CAM_01, 1, 1),
        )
