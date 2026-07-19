from __future__ import annotations

import json

import pytest

from robata.adapters.fake_vision_model import (
    FAKE_MODEL_NAME,
    FAKE_MODEL_VERSION,
    FAKE_PROVIDER,
    DeterministicFakeVisionModelAdapter,
)
from robata.contracts.cameras import CAMERA_IDS, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.mainline import (
    ActionEvidence,
    BoundaryRefinement,
    BoundaryStatus,
    CameraEvidenceStatus,
    CameraQAStatus,
    EventProposalOutput,
    InferenceCameraInput,
    InferenceStatus,
    QAOutput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionTask,
)


def _uuid(value: int) -> str:
    return f"00000000-0000-0000-0000-{value:012x}"


def _request(task: VisionTask, *, frames_per_camera: int = 2) -> VisionInferenceRequest:
    candidate_id = (
        _uuid(4)
        if task
        in {
            VisionTask.ACTION_EVIDENCE,
            VisionTask.BOUNDARY_REFINEMENT,
            VisionTask.FUSION_ADJUDICATION,
        }
        else None
    )
    return VisionInferenceRequest(
        schema_version="1.0",
        inference_id=_uuid(1),
        request_id=_uuid(2),
        mcap_id=_uuid(3),
        package_id=_uuid(5),
        package_content_sha256="a" * 64,
        interval=NanosecondInterval(start_ns=100, end_ns=500),
        subject_candidate_id=candidate_id,
        task=task,
        provider=FAKE_PROVIDER,
        model_name=FAKE_MODEL_NAME,
        model_version=FAKE_MODEL_VERSION,
        prompt_version="1.0",
        output_contract_version="1.0",
        camera_inputs=SixCameraMap[InferenceCameraInput](
            {
                camera_id: InferenceCameraInput(
                    camera_id=camera_id,
                    frame_ids=tuple(
                        _uuid(100 + camera_index * 10 + frame_index)
                        for frame_index in range(frames_per_camera)
                    ),
                )
                for camera_index, camera_id in enumerate(CAMERA_IDS)
            }
        ),
        timeout_ms=1_000,
    )


@pytest.mark.parametrize(
    ("task", "output_type"),
    [
        (VisionTask.QA_COARSE, QAOutput),
        (VisionTask.QA_DENSE, QAOutput),
        (VisionTask.EVENT_PROPOSAL, EventProposalOutput),
        (VisionTask.ACTION_EVIDENCE, ActionEvidence),
        (VisionTask.BOUNDARY_REFINEMENT, BoundaryRefinement),
    ],
)
def test_all_primary_tasks_return_strict_deterministic_success(
    task: VisionTask,
    output_type: type[object],
) -> None:
    request = _request(task)
    adapter = DeterministicFakeVisionModelAdapter()

    first = adapter.infer(request)
    second = adapter.infer(request)

    assert isinstance(first, VisionInferenceSuccess)
    assert isinstance(first.output, output_type)
    assert first == second
    assert first.status is InferenceStatus.SUCCEEDED
    assert (first.provider, first.model_name, first.model_version) == (
        FAKE_PROVIDER,
        FAKE_MODEL_NAME,
        FAKE_MODEL_VERSION,
    )
    assert (first.inference_id, first.request_id, first.task) == (
        request.inference_id,
        request.request_id,
        task,
    )
    assert first.usage.input_frames == first.usage.input_images == 12
    assert first.usage.cost is None and first.latency_ms == 0
    assert first.raw_output_sha256 == exact_bytes_sha256(canonical_json_bytes(first.output))
    assert adapter.external_provider_requests == 0


def test_qa_claims_are_complete_for_all_six_views() -> None:
    outcome = DeterministicFakeVisionModelAdapter().infer(_request(VisionTask.QA_COARSE))

    assert isinstance(outcome, VisionInferenceSuccess)
    assert isinstance(outcome.output, QAOutput)
    assert tuple(outcome.output.cameras.keys()) == CAMERA_IDS
    for camera_id, claim in outcome.output.cameras.items():
        assert claim.camera_id is camera_id
        assert claim.status is CameraQAStatus.GOOD
        assert claim.observed_interval == NanosecondInterval(start_ns=100, end_ns=500)
        assert claim.frame_ordinals == (0, 1)
        assert claim.issues == ()


def test_default_event_and_boundaries_are_centered_and_have_no_authoritative_ids() -> None:
    adapter = DeterministicFakeVisionModelAdapter()
    proposal_outcome = adapter.infer(_request(VisionTask.EVENT_PROPOSAL))
    action_outcome = adapter.infer(_request(VisionTask.ACTION_EVIDENCE))
    boundary_outcome = adapter.infer(_request(VisionTask.BOUNDARY_REFINEMENT))

    assert isinstance(proposal_outcome, VisionInferenceSuccess)
    assert isinstance(proposal_outcome.output, EventProposalOutput)
    assert len(proposal_outcome.output.proposals) == 1
    assert proposal_outcome.output.proposals[0].interval == NanosecondInterval(
        start_ns=200,
        end_ns=400,
    )

    assert isinstance(action_outcome, VisionInferenceSuccess)
    assert isinstance(action_outcome.output, ActionEvidence)
    assert len(action_outcome.output.cross_view_hypotheses) == 1
    assert all(
        claim.status is CameraEvidenceStatus.SUPPORTING
        for claim in action_outcome.output.cameras.values()
    )

    assert isinstance(boundary_outcome, VisionInferenceSuccess)
    assert isinstance(boundary_outcome.output, BoundaryRefinement)
    assert all(
        claim.status is BoundaryStatus.OBSERVED
        for claim in boundary_outcome.output.cameras.values()
    )
    first_boundary = boundary_outcome.output.cameras[CAMERA_IDS[0]]
    assert first_boundary.onset_interval == NanosecondInterval(start_ns=200, end_ns=250)
    assert first_boundary.offset_interval == NanosecondInterval(start_ns=350, end_ns=400)

    provider_payloads = (
        proposal_outcome.output,
        action_outcome.output,
        boundary_outcome.output,
    )
    forbidden = {"candidate_event_id", "event_id", "inference_id", "package_id"}
    for payload in provider_payloads:
        keys = set(json.loads(canonical_json_bytes(payload)))
        assert keys.isdisjoint(forbidden)


def test_explicit_no_event_mode_is_consistent_across_event_tasks() -> None:
    adapter = DeterministicFakeVisionModelAdapter(no_event=True)
    proposal = adapter.infer(_request(VisionTask.EVENT_PROPOSAL))
    action = adapter.infer(_request(VisionTask.ACTION_EVIDENCE))
    boundary = adapter.infer(_request(VisionTask.BOUNDARY_REFINEMENT))

    assert isinstance(proposal, VisionInferenceSuccess)
    assert isinstance(proposal.output, EventProposalOutput)
    assert proposal.output.proposals == ()

    assert isinstance(action, VisionInferenceSuccess)
    assert isinstance(action.output, ActionEvidence)
    assert action.output.cross_view_hypotheses == ()
    assert all(
        claim.status is CameraEvidenceStatus.NO_EVENT
        and claim.event_interval is None
        and claim.observed_frame_count == 2
        for claim in action.output.cameras.values()
    )

    assert isinstance(boundary, VisionInferenceSuccess)
    assert isinstance(boundary.output, BoundaryRefinement)
    assert all(
        claim.status is BoundaryStatus.NO_BOUNDARY
        and claim.onset_interval is None
        and claim.offset_interval is None
        for claim in boundary.output.cameras.values()
    )
    assert adapter.external_provider_requests == 0


def test_empty_camera_input_returns_structured_permanent_failure() -> None:
    request = _request(VisionTask.EVENT_PROPOSAL, frames_per_camera=0)

    outcome = DeterministicFakeVisionModelAdapter().infer(request)

    assert isinstance(outcome, VisionInferenceFailure)
    assert outcome.status is InferenceStatus.FAILED
    assert outcome.output is None
    assert outcome.failure.code == "FAKE_INPUT_MISSING_FRAMES"
    assert outcome.failure.retryability.value == "PERMANENT"
    assert outcome.provider == FAKE_PROVIDER


def test_model_identity_substitution_fails_instead_of_silently_changing_provider() -> None:
    request = _request(VisionTask.QA_COARSE).model_copy(update={"provider": "qwen"})

    outcome = DeterministicFakeVisionModelAdapter().infer(request)

    assert isinstance(outcome, VisionInferenceFailure)
    assert outcome.failure.code == "FAKE_MODEL_IDENTITY_MISMATCH"
    assert outcome.provider == FAKE_PROVIDER
    assert outcome.model_name == FAKE_MODEL_NAME
    assert outcome.raw_output_sha256 is None
