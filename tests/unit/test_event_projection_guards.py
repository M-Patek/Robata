from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from robata.contracts.cameras import CAMERA_IDS, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.pipeline import CameraEvidenceStatus, ProposalCameraClaim
from robata.contracts.temporal import TemporalPackageSet
from robata.event_pipeline.candidate import (
    CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE,
    CANDIDATE_EVENT_UUID_DERIVATION_NAMESPACE,
    CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE,
    CandidateReductionPolicy,
    CandidateReductionResult,
    CanonicalCandidateEvent,
    candidate_event_semantic_sha256,
    candidate_policy_semantic_sha256,
    candidate_reduction_semantic_sha256,
)
from robata.event_pipeline.proposer import (
    EventProposalError,
    EventProposalOutcome,
    EventProposalOutputRef,
    EventProposalProjector,
    EventProposalResult,
    NormalizedEventProposal,
    event_proposal_result_semantic_sha256,
)
from robata.inference.enrichment import OrchestratorEnrichedOutput
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import VisionTask


def _uuid(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012d}"


def _camera_coverage() -> SixCameraMap[ProposalCameraClaim]:
    return SixCameraMap(
        {
            camera_id: ProposalCameraClaim(
                camera_id=camera_id,
                status=CameraEvidenceStatus.MISSING,
                frame_ordinals=(),
            )
            for camera_id in CAMERA_IDS
        }
    )


def _proposal() -> NormalizedEventProposal:
    return NormalizedEventProposal(
        source_proposal_logical_key=f"event-proposal:{'1' * 64}",
        source=EventProposalOutputRef(
            source_inference_id=_uuid(1),
            source_input_plan_semantic_sha256="2" * 64,
            claim_ordinal=0,
            enrichment_logical_key=f"orchestrator-enrichment:{'3' * 64}",
            claim_id=_uuid(2),
        ),
        mcap_id=_uuid(3),
        source_package_ids=(_uuid(4),),
        interval=NanosecondInterval(start_ns=10, end_ns=20),
        label="fixture-action",
        model_reported_score=None,
        camera_coverage=_camera_coverage(),
        evidence=(),
    )


def _proposal_result(proposal: NormalizedEventProposal) -> EventProposalResult:
    digest = event_proposal_result_semantic_sha256(
        input_plan_semantic_sha256="4" * 64,
        outcome=EventProposalOutcome.CLAIMS,
        proposal_logical_keys=(proposal.source_proposal_logical_key,),
    )
    return EventProposalResult(
        task=VisionTask.EVENT_PROPOSAL,
        input_plan_semantic_sha256="4" * 64,
        outcome=EventProposalOutcome.CLAIMS,
        proposals=(proposal,),
        semantic_sha256=digest,
        logical_key=f"event-proposal-result:{digest}",
    )


def _candidate(proposal: NormalizedEventProposal) -> CanonicalCandidateEvent:
    policy = CandidateReductionPolicy(version="candidate-reduction-test-v1")
    policy_digest = candidate_policy_semantic_sha256(policy)
    values = {
        "candidate_event_id": _uuid(5),
        "candidate_logical_key": f"{CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
        "mcap_id": proposal.mcap_id,
        "source_content_sha256": "5" * 64,
        "camera_mapping_semantic_sha256": "6" * 64,
        "alignment_semantic_sha256": "7" * 64,
        "source_package_ids": proposal.source_package_ids,
        "source_proposal_logical_keys": (proposal.source_proposal_logical_key,),
        "effective_interval": proposal.interval,
        "requested_dense_interval": NanosecondInterval(start_ns=5, end_ns=25),
        "label": proposal.label,
        "reducer_policy_version": policy.version,
        "reducer_policy_semantic_sha256": policy_digest,
        "ontology_version": policy.ontology_version,
        "generation": 0,
        "parent_candidate_logical_key": None,
        "camera_coverage": proposal.camera_coverage,
        "production_eligible": False,
    }
    draft = CanonicalCandidateEvent.model_construct(**values)
    digest = candidate_event_semantic_sha256(draft)
    values["candidate_event_id"] = str(
        uuid5(NAMESPACE_URL, f"{CANDIDATE_EVENT_UUID_DERIVATION_NAMESPACE}:{digest}")
    )
    values["candidate_logical_key"] = f"{CANDIDATE_EVENT_LOGICAL_KEY_NAMESPACE}:{digest}"
    return CanonicalCandidateEvent.model_validate(values, strict=True)


def _candidate_result(candidate: CanonicalCandidateEvent) -> CandidateReductionResult:
    policy = CandidateReductionPolicy(version=candidate.reducer_policy_version)
    policy_digest = candidate_policy_semantic_sha256(policy)
    source_digest = "8" * 64
    digest = candidate_reduction_semantic_sha256(
        source_event_proposal_result_semantic_sha256=source_digest,
        policy_semantic_sha256=policy_digest,
        candidate_logical_keys=(candidate.candidate_logical_key,),
    )
    return CandidateReductionResult(
        policy=policy,
        policy_semantic_sha256=policy_digest,
        source_event_proposal_result_semantic_sha256=source_digest,
        candidates=(candidate,),
        semantic_sha256=digest,
        logical_key=f"{CANDIDATE_REDUCTION_LOGICAL_KEY_NAMESPACE}:{digest}",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"merge_gap_ns": -1}, "nonnegative"),
        ({"dense_padding_before_ns": -1}, "nonnegative"),
        ({"dense_padding_after_ns": -1}, "nonnegative"),
        ({"max_candidates": 0}, "positive"),
    ],
)
def test_candidate_policy_rejects_invalid_standalone_construction(
    changes: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        CandidateReductionPolicy(version="candidate-reduction-test-v1", **changes)


def test_proposal_models_reject_standalone_production_promotion() -> None:
    proposal = _proposal()
    values = proposal.model_dump(mode="python")
    values["production_eligible"] = True
    with pytest.raises(ValidationError, match="production_eligible"):
        NormalizedEventProposal.model_validate(values, strict=True)

    result = _proposal_result(proposal)
    values = result.model_dump(mode="python")
    values["production_eligible"] = True
    with pytest.raises(ValidationError, match="production_eligible"):
        EventProposalResult.model_validate(values, strict=True)


def test_candidate_models_reject_standalone_production_promotion() -> None:
    candidate = _candidate(_proposal())
    values = candidate.model_dump(mode="python")
    values["production_eligible"] = True
    with pytest.raises(ValidationError, match="production_eligible"):
        CanonicalCandidateEvent.model_validate(values, strict=True)

    result = _candidate_result(candidate)
    values = result.model_dump(mode="python")
    values["production_eligible"] = True
    with pytest.raises(ValidationError, match="production_eligible"):
        CandidateReductionResult.model_validate(values, strict=True)


def test_event_proposal_projector_rejects_incomplete_call_part_coverage() -> None:
    package_id = _uuid(10)
    digest = "a" * 64
    input_plan = cast(
        InferenceInputPlan,
        SimpleNamespace(
            subject=SimpleNamespace(
                task=VisionTask.EVENT_PROPOSAL,
                packages=(SimpleNamespace(package_id=package_id),),
            ),
            request_catalog=SimpleNamespace(
                task=VisionTask.EVENT_PROPOSAL,
                semantic_sha256=digest,
            ),
            call_plan=SimpleNamespace(parts=(object(), object())),
            semantic_sha256=digest,
        ),
    )
    package_set = cast(
        TemporalPackageSet,
        SimpleNamespace(members=(SimpleNamespace(package_id=package_id),)),
    )
    output = cast(
        OrchestratorEnrichedOutput,
        SimpleNamespace(
            task=VisionTask.EVENT_PROPOSAL,
            input_plan_semantic_sha256=digest,
            request_catalog_sha256=digest,
            abstained=False,
            claims=(),
        ),
    )

    with pytest.raises(EventProposalError, match="one enriched output per call part"):
        EventProposalProjector().project(
            input_plan=input_plan,
            package_set=package_set,
            enriched_outputs=(output,),
        )
