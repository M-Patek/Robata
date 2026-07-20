from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import pytest

from robata.contracts import CameraId, NanosecondInterval, SixCameraMap
from robata.contracts.pipeline import (
    BoundaryCameraClaim,
    BoundaryRefinement,
    BoundaryStatus,
    CameraActionClaim,
    CameraEvidenceStatus,
    CameraQAClaim,
    CameraQAResult,
    CameraQAStatus,
    CandidateEvent,
    CandidateEventStatus,
    EventProposal,
    ProposalCameraClaim,
    QAResultAggregate,
    RecordingQAStatus,
)
from robata.event_pipeline.adjudication import (
    AdjudicationPolicy,
    FusionAdjudicator,
)
from robata.event_pipeline.boundary import BoundaryRefiner
from robata.event_pipeline.fusion import FUSION_STAGES, FusionEngine, FusionPolicy


def _id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:event-test:{label}"))


def _candidate() -> CandidateEvent:
    cameras = {
        camera_id: ProposalCameraClaim(
            camera_id=camera_id,
            status=CameraEvidenceStatus.NO_EVENT,
            frame_ordinals=(),
        )
        for camera_id in CameraId
    }
    return CandidateEvent(
        candidate_event_id=_id("candidate"),
        mcap_id=_id("mcap"),
        source_package_id=_id("package"),
        source_inference_id=_id("proposal-inference"),
        proposal=EventProposal(
            ordinal=0,
            interval=NanosecondInterval(start_ns=100, end_ns=200),
            label_hint="grasp",
            reported_score=0.9,
            cameras=SixCameraMap(cameras),
        ),
        dense_interval=NanosecondInterval(start_ns=0, end_ns=300),
        ontology_version="ontology-v1",
        status=CandidateEventStatus.ACCEPTED,
    )


def _action_evidence() -> SixCameraMap[CameraActionClaim]:
    return SixCameraMap(
        {
            camera_id: CameraActionClaim(
                camera_id=camera_id,
                status=CameraEvidenceStatus.SUPPORTING,
                event_interval=NanosecondInterval(start_ns=120, end_ns=180),
                observed_interval=NanosecondInterval(start_ns=0, end_ns=300),
                visibility=0.9,
                observed_frame_count=1,
                coverage_fraction=0.9,
                reported_score=0.9,
                frame_ordinals=(0,),
                reason=None,
            )
            for camera_id in CameraId
        }
    )


def _boundary_evidence() -> BoundaryRefinement:
    return BoundaryRefinement(
        cameras=SixCameraMap(
            {
                camera_id: BoundaryCameraClaim(
                    camera_id=camera_id,
                    status=BoundaryStatus.OBSERVED,
                    observed_interval=NanosecondInterval(start_ns=0, end_ns=300),
                    onset_interval=NanosecondInterval(start_ns=100, end_ns=120),
                    offset_interval=NanosecondInterval(start_ns=180, end_ns=200),
                    reported_score=0.9,
                    frame_ordinals=(0,),
                    reason=None,
                )
                for camera_id in CameraId
            }
        )
    )


def _qa(candidate: CandidateEvent) -> QAResultAggregate:
    results = {
        camera_id: CameraQAResult(
            qa_result_id=_id(f"qa:{camera_id.value}"),
            mcap_id=candidate.mcap_id,
            package_id=candidate.source_package_id,
            inference_id=_id(f"qa-inference:{camera_id.value}"),
            camera_id=camera_id,
            claim=CameraQAClaim(
                camera_id=camera_id,
                observed_interval=candidate.dense_interval,
                status=CameraQAStatus.GOOD,
                issues=(),
                reported_score=0.9,
                frame_ordinals=(0,),
            ),
            evidence_frame_ids=(_id(f"frame:{camera_id.value}"),),
        )
        for camera_id in CameraId
    }
    return QAResultAggregate(
        aggregate_id=_id("qa-aggregate"),
        mcap_id=candidate.mcap_id,
        scope=candidate.dense_interval,
        overall_status=RecordingQAStatus.USABLE,
        usable_camera_count=6,
        camera_results=SixCameraMap(results),
        policy_version="qa-v1",
    )


def _policy() -> FusionPolicy:
    return FusionPolicy(
        version="fusion-v1",
        stages=FUSION_STAGES,
        weights={
            "visibility": 0.25,
            "coverage": 0.25,
            "reported_score": 0.25,
            "qa": 0.25,
        },
        minimum_supporting_cameras=2,
        final_confidence_threshold=0.5,
    )


def test_boundary_refinement_is_deterministic_and_tracks_uncertainty() -> None:
    candidate = _candidate()
    evidence = _boundary_evidence()
    first = BoundaryRefiner().refine(candidate, evidence)
    replay = BoundaryRefiner().refine(candidate, evidence)

    assert first == replay
    assert first.interval == NanosecondInterval(start_ns=110, end_ns=190)
    assert first.observed_camera_count == 6
    assert first.used_fallback is False
    assert first.production_eligible is False
    assert first.uncertainty_ns == 10


def test_boundary_refinement_falls_back_explicitly_when_evidence_is_missing() -> None:
    candidate = _candidate()
    missing = BoundaryRefinement(
        cameras=SixCameraMap(
            {
                camera_id: BoundaryCameraClaim(
                    camera_id=camera_id,
                    status=BoundaryStatus.MISSING,
                    observed_interval=None,
                    onset_interval=None,
                    offset_interval=None,
                    reported_score=None,
                    frame_ordinals=(),
                    reason="no boundary evidence",
                )
                for camera_id in CameraId
            }
        )
    )
    refined = BoundaryRefiner().refine(candidate, missing)
    assert refined.interval == candidate.proposal.interval
    assert refined.used_fallback is True
    assert refined.observed_camera_count == 0


def test_fusion_requires_policy_stages_and_preserves_local_only_status() -> None:
    candidate = _candidate()
    engine = FusionEngine(_policy())
    first = engine.fuse(
        candidate,
        _action_evidence(),
        _boundary_evidence(),
        _qa(candidate),
    )
    replay = engine.fuse(
        candidate,
        _action_evidence(),
        _boundary_evidence(),
        _qa(candidate),
    )

    assert first == replay
    assert len(first) == 1
    assert first[0].ambiguity_state == "RESOLVED"
    assert first[0].supporting_camera_count == 6
    assert first[0].interval == NanosecondInterval(start_ns=110, end_ns=190)
    assert first[0].production_eligible is False

    with pytest.raises(ValueError, match="canonical eight stages"):
        FusionPolicy(
            version="bad",
            stages=("NORMALIZE",),
            weights={key: 0.25 for key in ("visibility", "coverage", "reported_score", "qa")},
        )


def test_adjudicator_abstains_on_ambiguous_and_conflicting_decisions() -> None:
    candidate = _candidate()
    qa = _qa(candidate)
    decisions = FusionEngine(_policy()).fuse(
        candidate,
        _action_evidence(),
        _boundary_evidence(),
        qa,
    )
    winner = decisions[0]
    conflict = winner.model_copy(
        update={
            "event_id": _id("conflict"),
            "confidence": 0.8,
            "action_type": "reach",
        }
    )
    ambiguous = winner.model_copy(
        update={
            "event_id": _id("ambiguous"),
            "ambiguity_state": "LOW_UNCALIBRATED_CONFIDENCE",
        }
    )

    result = FusionAdjudicator().adjudicate(
        (winner, conflict, ambiguous),
        AdjudicationPolicy(
            version="adjudication-v1",
            conflict_resolution_strategy="ABSTAIN_ON_CONFLICT",
        ),
    )

    assert result.final_decisions == ()
    assert {item.event_id for item in result.abstained} == {
        winner.event_id,
        conflict.event_id,
        ambiguous.event_id,
    }
    assert "CONFLICT_ABSTAIN" in result.rationale
