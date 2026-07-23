from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

import pytest

from robata.application.canonical.bounded_media import (
    BoundedMediaPolicy,
    BoundedSinglePassMediaPlanner,
    EncodedMediaPacket,
    PlannerEmission,
    WindowClosureReason,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    CameraAbsenceReason,
    StreamIntervalAbsence,
    StreamSegmentSequence,
)

_DIGEST = "a" * 64
_TIMES = (0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000, 2_500_000_000)


def _policy(**updates: object) -> BoundedMediaPolicy:
    values: dict[str, object] = {
        "source_scope_digest": _DIGEST,
        "mapping_semantic_sha256": "b" * 64,
        "alignment_semantic_sha256": "c" * 64,
        "allowed_lateness_ns": 0,
        "ring_max_bytes_per_camera": 64,
    }
    values.update(updates)
    return BoundedMediaPolicy(**values)


def _packet(
    traversal_index: int,
    camera_id: CameraId,
    timestamp_ns: int,
    *,
    source_order: int | None = None,
    source_sequence: int | None = None,
    source_locator: str | None = None,
    payload: bytes | None = None,
) -> EncodedMediaPacket:
    ordinal = timestamp_ns // 500_000_000 if source_order is None else source_order
    return EncodedMediaPacket(
        traversal_index=traversal_index,
        camera_id=camera_id,
        source_order=ordinal,
        source_sequence=ordinal if source_sequence is None else source_sequence,
        source_timestamp_ns=timestamp_ns,
        aligned_timestamp_ns=timestamp_ns,
        source_locator=source_locator or f"mcap://{camera_id.value}/{ordinal}",
        payload=payload or f"{camera_id.value}:{timestamp_ns}".encode(),
    )


def _schema_ref() -> SchemaRef:
    return SchemaRef(
        schema_id="https://schemas.robata.dev/test",
        version="1.0.0",
        artifact_id="00000000-0000-4000-8000-000000000001",
        sha256="d" * 64,
    )


def _feed(
    planner: BoundedSinglePassMediaPlanner,
    *,
    cameras: Iterable[CameraId] = CAMERA_IDS,
    times: Iterable[int] = _TIMES,
) -> list[PlannerEmission]:
    emissions: list[PlannerEmission] = []
    traversal = 0
    for timestamp_ns in times:
        for camera_id in cameras:
            emissions.append(planner.push(_packet(traversal, camera_id, timestamp_ns)))
            traversal += 1
    return emissions


def test_single_pass_emits_ordered_segments_windows_and_bounded_rings() -> None:
    planner = BoundedSinglePassMediaPlanner(_policy())
    emissions = _feed(planner)
    finish = planner.finish(3_000_000_000)

    windows = [window for emission in emissions for window in emission.windows]
    windows.extend(finish.windows)
    assert tuple(window.ordinal for window in windows) == (0, 1, 2)
    assert windows[0].closure_reason is WindowClosureReason.WATERMARK
    assert windows[-1].closure_reason is WindowClosureReason.EOS
    assert windows[-1].effective_interval.end_ns == 3_000_000_000
    assert all(
        tuple(plan.camera_id for plan in window.camera_plans) == CAMERA_IDS for window in windows
    )
    assert all(len(plan.members) == 2 for window in windows[:2] for plan in window.camera_plans)
    assert all(snapshot.total_bytes <= 64 for snapshot in planner.ring_snapshots())
    assert all(snapshot.evicted_packet_count > 0 for snapshot in planner.ring_snapshots())


def test_partial_source_gap_is_explicit_for_each_logical_segment_bucket() -> None:
    planner = BoundedSinglePassMediaPlanner(_policy())
    _feed(planner, cameras=CAMERA_IDS[:-1])
    finish = planner.finish(3_000_000_000)

    first = finish.windows[0]
    missing = first.camera_plans[-1]
    assert missing.camera_id is CameraId.CAM_06
    assert all(member.segment is None for member in missing.members)
    assert all(member.absence_reason is CameraAbsenceReason.ABSENT for member in missing.members)
    assert all(member.absence_evidence_sha256 is not None for member in missing.members)
    mapped = first.to_incremental_window(_schema_ref())
    mapped_missing = mapped.camera_closure[-1]
    assert isinstance(mapped_missing, StreamSegmentSequence)
    assert all(
        isinstance(member, StreamIntervalAbsence) for member in mapped_missing.ordered_members
    )


def test_quality_target_selection_is_nearest_and_quality_gaps_are_explicit() -> None:
    planner = BoundedSinglePassMediaPlanner(_policy(quality_selection_tolerance_ns=100_000_000))
    traversal = 0
    # Both packets belong to the first quality bucket; the 100 ms packet wins.
    for camera_id in CAMERA_IDS:
        planner.push(
            _packet(
                traversal,
                camera_id,
                100_000_000,
                source_order=0,
                source_sequence=0,
            )
        )
        traversal += 1
        planner.push(
            _packet(
                traversal,
                camera_id,
                400_000_000,
                source_order=1,
                source_sequence=1,
            )
        )
        traversal += 1
    finish = planner.finish(1_000_000_000)

    assert finish.quality_targets
    assert all(target.packet.source_order == 0 for target in finish.quality_targets)
    timing = finish.quality_targets[0].timing_evidence()
    assert timing.camera_id is finish.quality_targets[0].camera_id
    assert timing.packet_index == finish.quality_targets[0].packet.source_order
    assert finish.windows
    assert finish.windows[0].quality_gaps
    assert all(
        gap.reason == "NO_DECODABLE_TARGET_WITHIN_TOLERANCE"
        for gap in finish.windows[0].quality_gaps
    )


def test_segment_identity_excludes_locator_but_binds_payload_and_source_closure() -> None:
    def run(locator_prefix: str, payload_suffix: bytes = b"") -> tuple[str, str]:
        planner = BoundedSinglePassMediaPlanner(_policy())
        traversal = 0
        for timestamp_ns in (0, 1_000_000_000):
            for camera_id in CAMERA_IDS:
                planner.push(
                    _packet(
                        traversal,
                        camera_id,
                        timestamp_ns,
                        source_locator=f"{locator_prefix}/{traversal}",
                        payload=f"payload-{camera_id.value}-{timestamp_ns}".encode()
                        + payload_suffix,
                    )
                )
                traversal += 1
        finish = planner.finish(2_000_000_000)
        segment = finish.closed_segments[0]
        window = finish.windows[0].to_incremental_window(_schema_ref())
        return segment.semantic_sha256, window.window_semantic_sha256

    assert run("memory://first") == run("memory://different")
    assert run("memory://first", b"-changed") != run("memory://first")


def test_replay_is_deterministic_and_segment_reference_is_wire_compatible() -> None:
    first = BoundedSinglePassMediaPlanner(_policy())
    second = BoundedSinglePassMediaPlanner(_policy())
    _feed(first)
    _feed(second)
    first_finish = first.finish(3_000_000_000)
    second_finish = second.finish(3_000_000_000)

    assert first_finish.facts == second_finish.facts
    assert [item.semantic_sha256 for item in first_finish.closed_segments] == [
        item.semantic_sha256 for item in second_finish.closed_segments
    ]
    assert [item.planning_sha256 for item in first_finish.windows] == [
        item.planning_sha256 for item in second_finish.windows
    ]
    segment = first_finish.closed_segments[0]
    reference = segment.reference()
    assert reference.segment_key.endswith(reference.segment_semantic_sha256)
    manifest = segment.to_stream_segment_manifest(_schema_ref())
    assert manifest.segment_semantic_sha256 == segment.semantic_sha256
    expected_spool_digest = sha256(
        b"".join(
            len(packet.payload).to_bytes(8, "big") + packet.payload
            for packet in (
                _packet(0, CameraId.CAM_01, 2_000_000_000),
                _packet(1, CameraId.CAM_01, 2_500_000_000),
            )
        )
    ).hexdigest()
    assert segment.exact_content_sha256 == expected_spool_digest
    mapped = first_finish.windows[0].to_incremental_window(_schema_ref())
    assert all(isinstance(slot, StreamSegmentSequence) for slot in mapped.camera_closure)
    assert mapped.camera_closure[0].ordered_members[0].kind == "SEGMENT"


def test_partial_gap_maps_to_ordered_segment_and_interval_absence_members() -> None:
    planner = BoundedSinglePassMediaPlanner(_policy())
    traversal = 0
    for timestamp_ns in (0, 1_000_000_000, 2_000_000_000):
        for camera_id in CAMERA_IDS:
            if camera_id is CameraId.CAM_06 and timestamp_ns == 1_000_000_000:
                continue
            planner.push(_packet(traversal, camera_id, timestamp_ns))
            traversal += 1
    finish = planner.finish(3_000_000_000)

    first = finish.windows[0]
    assert [member.kind for member in first.camera_plans[-1].members] == [
        "ABSENCE",
        "SEGMENT",
    ]
    mapped = first.to_incremental_window(_schema_ref())
    mapped_slot = mapped.camera_closure[-1]
    assert isinstance(mapped_slot, StreamSegmentSequence)
    assert [member.kind for member in mapped_slot.ordered_members] == [
        "INTERVAL_ABSENCE",
        "SEGMENT",
    ]


def test_source_order_and_traversal_order_are_fail_closed() -> None:
    planner = BoundedSinglePassMediaPlanner(_policy())
    planner.push(_packet(0, CameraId.CAM_01, 0))
    with pytest.raises(ValueError, match="traversal indexes"):
        planner.push(_packet(0, CameraId.CAM_02, 0))

    planner.push(_packet(1, CameraId.CAM_01, 500_000_000))
    with pytest.raises(ValueError, match="source_order"):
        planner.push(
            _packet(
                2,
                CameraId.CAM_01,
                1_000_000_000,
                source_order=0,
                source_sequence=2,
            )
        )


def test_policy_rejects_unaligned_window_bounds() -> None:
    with pytest.raises(ValueError, match="align"):
        _policy(window_width_ns=1_500_000_000)
