from __future__ import annotations

import pytest

from robata.contracts import CameraId
from robata.perception.projectors import (
    EventProjector,
    EvidenceProjector,
    MediaHealthDisposition,
    ProjectedQaDisposition,
    QaProjector,
)
from robata.perception.tracking import (
    EventTrackPolicy,
    EventTrackReconciler,
    EventTrackState,
    finalize_event_track,
)
from tests.support.perception_stream import make_context, make_media_health, make_observation


def test_one_mage_observation_projects_qa_event_and_evidence_deterministically() -> None:
    context = make_context(selected_cameras=(CameraId.CAM_01, CameraId.CAM_03))
    observation = make_observation(context=context)
    health = make_media_health(context, degraded_camera=CameraId.CAM_03)

    qa = QaProjector().project(observation, health)
    events = EventProjector().project(observation)
    evidence = EvidenceProjector().project(observation, events)

    assert qa == QaProjector().project(observation, health)
    assert events == EventProjector().project(observation)
    assert evidence == EvidenceProjector().project(observation, events)
    assert qa.camera_facts[CameraId.CAM_01].disposition is ProjectedQaDisposition.USABLE
    assert qa.camera_facts[CameraId.CAM_03].disposition is ProjectedQaDisposition.DEGRADED
    assert (
        qa.camera_facts[CameraId.CAM_03].media_health_disposition is MediaHealthDisposition.DEGRADED
    )
    assert len(events.hypotheses) == 1
    assert len(evidence.event_evidence) == 1
    bundle = evidence.event_evidence[0]
    assert len(bundle.cameras) == 6
    assert bundle.cameras[CameraId.CAM_01].selected_for_inference is True
    assert bundle.cameras[CameraId.CAM_02].selected_for_inference is False


def test_qa_projection_keeps_missing_media_separate_from_semantic_qa() -> None:
    context = make_context(unavailable_cameras=(CameraId.CAM_06,))
    observation = make_observation(context=context)
    qa = QaProjector().project(observation, make_media_health(context))

    camera = qa.camera_facts[CameraId.CAM_06]
    assert camera.disposition is ProjectedQaDisposition.UNAVAILABLE
    assert camera.media_health_disposition is MediaHealthDisposition.UNAVAILABLE
    assert camera.issue_codes == ("camera_missing",)


def _projection_for_segment(
    *,
    segment_ordinal: int,
    start_ns: int,
    end_ns: int,
    action_start_ns: int,
    action_end_ns: int,
    started_before_context: bool,
    continues_after_context: bool,
):
    context = make_context(
        start_ns=start_ns,
        end_ns=end_ns,
        segment_ordinal=segment_ordinal,
    )
    observation = make_observation(
        context=context,
        local_ref=f"o{segment_ordinal}",
        action_start_ns=action_start_ns,
        action_end_ns=action_end_ns,
        started_before_context=started_before_context,
        continues_after_context=continues_after_context,
        inference_artifact_seed=f"artifact-{segment_ordinal}",
    )
    return EventProjector().project(observation)


def test_temporal_reconciler_turns_three_segment_observations_into_one_event_track() -> None:
    policy = EventTrackPolicy(version="event-track-policy-v1", max_merge_gap_ns=1)
    reconciler = EventTrackReconciler(policy)
    first = _projection_for_segment(
        segment_ordinal=1,
        start_ns=8_000_000_000,
        end_ns=16_000_000_000,
        action_start_ns=14_700_000_000,
        action_end_ns=16_000_000_000,
        started_before_context=False,
        continues_after_context=True,
    )
    second = _projection_for_segment(
        segment_ordinal=2,
        start_ns=16_000_000_000,
        end_ns=24_000_000_000,
        action_start_ns=16_000_000_000,
        action_end_ns=24_000_000_000,
        started_before_context=True,
        continues_after_context=True,
    )
    third = _projection_for_segment(
        segment_ordinal=3,
        start_ns=24_000_000_000,
        end_ns=32_000_000_000,
        action_start_ns=24_000_000_000,
        action_end_ns=24_400_000_000,
        started_before_context=True,
        continues_after_context=False,
    )

    r1 = reconciler.reconcile((), first)
    assert len(r1.current_tracks) == 1
    assert r1.current_tracks[0].state is EventTrackState.OPEN

    r2 = reconciler.reconcile(r1.current_tracks, second)
    assert len(r2.current_tracks) == 1
    assert r2.current_tracks[0].state is EventTrackState.UPDATED

    r3 = reconciler.reconcile(r2.current_tracks, third)
    assert len(r3.current_tracks) == 1
    track = r3.current_tracks[0]
    assert track.state is EventTrackState.CLOSED
    assert track.interval.start_ns == 14_700_000_000
    assert track.interval.end_ns == 24_400_000_000
    assert len(track.source_hypotheses) == 3
    assert track.event_track_id == r1.current_tracks[0].event_track_id
    assert track.event_track_key == r1.current_tracks[0].event_track_key

    finalized = finalize_event_track(
        track,
        resolved_event_semantic_sha256="f" * 64,
    )
    assert finalized.state is EventTrackState.FINALIZED
    assert finalized.event_track_id == track.event_track_id
    assert finalized.parent_revision_semantic_sha256 == track.revision_semantic_sha256


def test_adjacent_same_label_does_not_merge_without_continuation_evidence() -> None:
    policy = EventTrackPolicy(version="event-track-policy-v1", max_merge_gap_ns=1)
    reconciler = EventTrackReconciler(policy)
    first = _projection_for_segment(
        segment_ordinal=1,
        start_ns=0,
        end_ns=8_000_000_000,
        action_start_ns=7_000_000_000,
        action_end_ns=8_000_000_000,
        started_before_context=False,
        continues_after_context=True,
    )
    second = _projection_for_segment(
        segment_ordinal=2,
        start_ns=8_000_000_000,
        end_ns=16_000_000_000,
        action_start_ns=8_000_000_000,
        action_end_ns=9_000_000_000,
        started_before_context=False,
        continues_after_context=False,
    )

    r1 = reconciler.reconcile((), first)
    r2 = reconciler.reconcile(r1.current_tracks, second)

    assert len(r2.current_tracks) == 2
    assert len(r2.created_track_keys) == 1
    assert all(track.state is EventTrackState.CLOSED for track in r2.current_tracks)


def test_finalization_fails_closed_for_an_open_track() -> None:
    projection = _projection_for_segment(
        segment_ordinal=1,
        start_ns=0,
        end_ns=8_000_000_000,
        action_start_ns=7_000_000_000,
        action_end_ns=8_000_000_000,
        started_before_context=False,
        continues_after_context=True,
    )
    track = (
        EventTrackReconciler(EventTrackPolicy(version="event-track-policy-v1"))
        .reconcile((), projection)
        .current_tracks[0]
    )

    with pytest.raises(ValueError, match="only a closed track"):
        finalize_event_track(track, resolved_event_semantic_sha256="f" * 64)
