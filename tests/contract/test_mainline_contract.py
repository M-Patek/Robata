from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from robata.contracts import (
    CAMERA_IDS,
    ActionEventStatus,
    ActionEvidence,
    BoundaryCameraClaim,
    BoundaryRefinement,
    BoundaryStatus,
    CameraActionClaim,
    CameraEventProvenance,
    CameraEvidenceStatus,
    CameraId,
    CameraPackage,
    CameraPackageStatus,
    CameraQAClaim,
    CameraQAResult,
    CameraQAStatus,
    CandidateEvent,
    CandidateEventStatus,
    CrossViewHypothesis,
    EventProposal,
    EventProposalOutput,
    FusedActionEvent,
    InferenceCameraInput,
    InferenceFailureDetail,
    InferenceStatus,
    MainlineBundle,
    MainlineRunReport,
    MainlineStage,
    MaterializedFrame,
    NanosecondInterval,
    ProposalCameraClaim,
    QAOutput,
    QAResultAggregate,
    RecordingQAStatus,
    Retryability,
    RunStatus,
    SamplingPurpose,
    SamplingStrategy,
    SamplingSummary,
    SixCameraMap,
    StageReport,
    StageStatus,
    TemporalVisualPackage,
    TemporalWindow,
    VisionInferenceFailure,
    VisionInferenceOutcome,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionTask,
    VisionUsage,
)

SECOND = 1_000_000_000


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _digest(number: int) -> str:
    return f"{number % 16:x}" * 64


def _interval(start_ns: int = 0, end_ns: int = 4 * SECOND) -> NanosecondInterval:
    return NanosecondInterval(start_ns=start_ns, end_ns=end_ns)


def _window(
    *,
    window_id: int = 10,
    purpose: SamplingPurpose = SamplingPurpose.QA_COARSE,
    parent_window_id: str | None = None,
    source_candidate_id: str | None = None,
    generation: int = 0,
) -> TemporalWindow:
    return TemporalWindow(
        schema_version="1.0",
        window_id=_uuid(window_id),
        mcap_id=_uuid(1),
        camera_mapping_run_id=None,
        alignment_id=None,
        requested_interval=_interval(),
        interval=_interval(),
        purpose=purpose,
        parent_window_id=parent_window_id,
        source_candidate_id=source_candidate_id,
        source_event_id=None,
        generation=generation,
    )


def _frame(camera_number: int, *, frame_id_base: int) -> MaterializedFrame:
    timestamp_ns = SECOND + camera_number
    return MaterializedFrame(
        frame_id=_uuid(frame_id_base + camera_number),
        ordinal=0,
        source_frame_index=camera_number,
        target_timestamp_ns=SECOND,
        aligned_timestamp_ns=timestamp_ns,
        source_timestamp_ns=1_710_000_000_000_000_000 + camera_number,
        delta_to_target_ns=camera_number,
        artifact_uri=f"artifact://frames/cam-{camera_number:02d}.png",
        artifact_sha256=_digest(camera_number),
        width=1600,
        height=1300,
        quality_flags=(),
    )


def _camera_package(
    camera_id: CameraId,
    *,
    frame_id_base: int,
) -> CameraPackage:
    camera_number = CAMERA_IDS.index(camera_id) + 1
    return CameraPackage(
        camera_id=camera_id,
        status=CameraPackageStatus.AVAILABLE,
        source_video_uri=f"artifact://videos/{camera_id.value}.mp4",
        frames=(_frame(camera_number, frame_id_base=frame_id_base),),
        sampling=SamplingSummary(
            strategy=SamplingStrategy.UNIFORM,
            target_fps=1.0,
            actual_fps=1.0,
            target_count=1,
            actual_count=1,
            missed_targets=0,
        ),
        missing_reason=None,
    )


def _package(
    window: TemporalWindow,
    *,
    package_id: int,
    frame_id_base: int,
    digest: int,
) -> TemporalVisualPackage:
    cameras = SixCameraMap[CameraPackage].model_validate(
        {
            camera_id: _camera_package(camera_id, frame_id_base=frame_id_base)
            for camera_id in CAMERA_IDS
        }
    )
    return TemporalVisualPackage(
        schema_version="1.0",
        package_id=_uuid(package_id),
        content_sha256=_digest(digest),
        mcap_id=window.mcap_id,
        window_id=window.window_id,
        purpose=window.purpose,
        interval=window.interval,
        cameras=cameras,
        frame_count_total=6,
        producer_version="local-materializer-v1",
        created_at="2026-07-18T12:00:00Z",
    )


def _camera_inputs(
    package: TemporalVisualPackage,
) -> SixCameraMap[InferenceCameraInput]:
    return SixCameraMap[InferenceCameraInput].model_validate(
        {
            camera_id: InferenceCameraInput(
                camera_id=camera_id,
                frame_ids=tuple(frame.frame_id for frame in package.cameras[camera_id].frames),
            )
            for camera_id in CAMERA_IDS
        }
    )


def _request(
    *,
    number: int,
    package: TemporalVisualPackage,
    task: VisionTask,
    candidate_id: str | None = None,
) -> VisionInferenceRequest:
    return VisionInferenceRequest(
        schema_version="1.0",
        inference_id=_uuid(100 + number),
        request_id=_uuid(200 + number),
        mcap_id=package.mcap_id,
        package_id=package.package_id,
        package_content_sha256=package.content_sha256,
        interval=package.interval,
        subject_candidate_id=candidate_id,
        task=task,
        provider="fake",
        model_name="deterministic-fixture",
        model_version="1.0",
        prompt_version="mainline-v0",
        output_contract_version="1.0",
        camera_inputs=_camera_inputs(package),
        timeout_ms=1_000,
    )


def _qa_output(interval: NanosecondInterval) -> QAOutput:
    return QAOutput(
        cameras=SixCameraMap[CameraQAClaim].model_validate(
            {
                camera_id: CameraQAClaim(
                    camera_id=camera_id,
                    observed_interval=interval,
                    status=CameraQAStatus.GOOD,
                    issues=(),
                    reported_score=None,
                    frame_ordinals=(0,),
                )
                for camera_id in CAMERA_IDS
            }
        )
    )


def _proposal_output(interval: NanosecondInterval) -> EventProposalOutput:
    return EventProposalOutput(
        proposals=(
            EventProposal(
                ordinal=0,
                interval=_interval(SECOND, 2 * SECOND),
                label_hint="grasp",
                reported_score=None,
                cameras=SixCameraMap[ProposalCameraClaim].model_validate(
                    {
                        camera_id: ProposalCameraClaim(
                            camera_id=camera_id,
                            status=CameraEvidenceStatus.SUPPORTING,
                            frame_ordinals=(0,),
                        )
                        for camera_id in CAMERA_IDS
                    }
                ),
            ),
        )
    )


def _action_output(interval: NanosecondInterval) -> ActionEvidence:
    return ActionEvidence(
        cameras=SixCameraMap[CameraActionClaim].model_validate(
            {
                camera_id: CameraActionClaim(
                    camera_id=camera_id,
                    status=CameraEvidenceStatus.SUPPORTING,
                    event_interval=_interval(SECOND, 2 * SECOND),
                    observed_interval=interval,
                    visibility=1.0,
                    observed_frame_count=1,
                    coverage_fraction=1.0,
                    reported_score=None,
                    frame_ordinals=(0,),
                    reason=None,
                )
                for camera_id in CAMERA_IDS
            }
        ),
        cross_view_hypotheses=(
            CrossViewHypothesis(
                ordinal=0,
                interval=_interval(SECOND, 2 * SECOND),
                action_type="grasp",
                reported_score=None,
            ),
        ),
    )


def _boundary_output(interval: NanosecondInterval) -> BoundaryRefinement:
    return BoundaryRefinement(
        cameras=SixCameraMap[BoundaryCameraClaim].model_validate(
            {
                camera_id: BoundaryCameraClaim(
                    camera_id=camera_id,
                    status=BoundaryStatus.OBSERVED,
                    observed_interval=interval,
                    onset_interval=_interval(900_000_000, 1_100_000_000),
                    offset_interval=_interval(1_900_000_000, 2_100_000_000),
                    reported_score=None,
                    frame_ordinals=(0,),
                    reason=None,
                )
                for camera_id in CAMERA_IDS
            }
        )
    )


def _usage() -> VisionUsage:
    return VisionUsage(
        input_frames=6,
        input_images=6,
        input_tokens=None,
        output_tokens=None,
        cost=None,
        currency=None,
    )


def _success(
    request: VisionInferenceRequest,
    output: QAOutput | EventProposalOutput | ActionEvidence | BoundaryRefinement,
) -> VisionInferenceSuccess:
    return VisionInferenceSuccess(
        schema_version="1.0",
        inference_id=request.inference_id,
        request_id=request.request_id,
        task=request.task,
        status=InferenceStatus.SUCCEEDED,
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        output=output,
        raw_output_sha256=_digest(14),
        schema_valid=True,
        usage=_usage(),
        latency_ms=1,
    )


def _qa_aggregate(
    package: TemporalVisualPackage,
    request: VisionInferenceRequest,
    output: QAOutput,
    *,
    number: int,
) -> QAResultAggregate:
    results: dict[CameraId, CameraQAResult] = {}
    for index, camera_id in enumerate(CAMERA_IDS, start=1):
        results[camera_id] = CameraQAResult(
            qa_result_id=_uuid(300 + number * 10 + index),
            mcap_id=package.mcap_id,
            package_id=package.package_id,
            inference_id=request.inference_id,
            camera_id=camera_id,
            claim=output.cameras[camera_id],
            evidence_frame_ids=(package.cameras[camera_id].frames[0].frame_id,),
        )
    return QAResultAggregate(
        aggregate_id=_uuid(320 + number),
        mcap_id=package.mcap_id,
        scope=package.interval,
        overall_status=RecordingQAStatus.USABLE,
        usable_camera_count=6,
        camera_results=SixCameraMap[CameraQAResult].model_validate(results),
        policy_version="qa-v0",
    )


def _stage_reports() -> tuple[StageReport, ...]:
    return tuple(
        StageReport(
            stage=stage,
            status=StageStatus.SUCCEEDED,
            planned=1,
            succeeded=1,
            failed=0,
            pending=0,
            skipped=0,
            duration_ms=1,
            error_codes=(),
        )
        for stage in MainlineStage
    )


def _bundle() -> MainlineBundle:
    coarse_window = _window()
    candidate_id = _uuid(400)
    dense_window = _window(
        window_id=11,
        purpose=SamplingPurpose.ACTION_DENSE,
        parent_window_id=coarse_window.window_id,
        source_candidate_id=candidate_id,
        generation=1,
    )
    coarse_package = _package(
        coarse_window,
        package_id=20,
        frame_id_base=500,
        digest=10,
    )
    dense_package = _package(
        dense_window,
        package_id=21,
        frame_id_base=600,
        digest=11,
    )

    qa_request = _request(number=1, package=coarse_package, task=VisionTask.QA_COARSE)
    proposal_request = _request(
        number=2,
        package=coarse_package,
        task=VisionTask.EVENT_PROPOSAL,
    )
    dense_qa_request = _request(
        number=3,
        package=dense_package,
        task=VisionTask.QA_DENSE,
    )
    action_request = _request(
        number=4,
        package=dense_package,
        task=VisionTask.ACTION_EVIDENCE,
        candidate_id=candidate_id,
    )
    boundary_request = _request(
        number=5,
        package=dense_package,
        task=VisionTask.BOUNDARY_REFINEMENT,
        candidate_id=candidate_id,
    )

    qa_output = _qa_output(coarse_package.interval)
    dense_qa_output = _qa_output(dense_package.interval)
    proposal_output = _proposal_output(coarse_package.interval)
    action_output = _action_output(dense_package.interval)
    boundary_output = _boundary_output(dense_package.interval)
    candidate = CandidateEvent(
        candidate_event_id=candidate_id,
        mcap_id=coarse_package.mcap_id,
        source_package_id=coarse_package.package_id,
        source_inference_id=proposal_request.inference_id,
        proposal=proposal_output.proposals[0],
        dense_interval=dense_package.interval,
        ontology_version="dev-ontology-v0",
        status=CandidateEventStatus.ACCEPTED,
    )

    coarse_qa_aggregate = _qa_aggregate(
        coarse_package,
        qa_request,
        qa_output,
        number=0,
    )
    dense_qa_aggregate = _qa_aggregate(
        dense_package,
        dense_qa_request,
        dense_qa_output,
        number=1,
    )
    event_provenance: dict[CameraId, CameraEventProvenance] = {}
    for camera_id in CAMERA_IDS:
        event_provenance[camera_id] = CameraEventProvenance(
            camera_id=camera_id,
            claim=action_output.cameras[camera_id],
            package_id=dense_package.package_id,
            inference_id=action_request.inference_id,
            frame_ids=(dense_package.cameras[camera_id].frames[0].frame_id,),
            qa_result_id=dense_qa_aggregate.camera_results[camera_id].qa_result_id,
        )
    event = FusedActionEvent(
        schema_version="1.0",
        event_id=_uuid(401),
        mcap_id=coarse_package.mcap_id,
        candidate_event_ids=(candidate_id,),
        interval=_interval(SECOND, 2_100_000_000),
        action_type="grasp",
        boundary_start_uncertainty_ns=100_000_000,
        boundary_end_uncertainty_ns=100_000_000,
        camera_evidence=SixCameraMap[CameraEventProvenance].model_validate(event_provenance),
        fusion_policy_version="deterministic-fusion-v0",
        boundary_inference_id=boundary_request.inference_id,
        producer_provider="fake",
        status=ActionEventStatus.FINAL,
        production_eligible=False,
        created_at="2026-07-18T12:00:01Z",
    )
    requests = (
        qa_request,
        proposal_request,
        dense_qa_request,
        action_request,
        boundary_request,
    )
    outcomes = (
        _success(qa_request, qa_output),
        _success(proposal_request, proposal_output),
        _success(dense_qa_request, dense_qa_output),
        _success(action_request, action_output),
        _success(boundary_request, boundary_output),
    )
    report = MainlineRunReport(
        schema_version="1.0",
        run_id=_uuid(2),
        source_mcap_id=coarse_package.mcap_id,
        source_recording_identity=_digest(8),
        source_content_sha256=_digest(9),
        video_manifest_artifact_id=_uuid(3),
        video_manifest_sha256=_digest(10),
        video_manifest_semantic_sha256=_digest(11),
        pipeline_version="mainline-v0",
        config_sha256=_digest(12),
        status=RunStatus.PRIMARY_COMPLETE,
        started_at="2026-07-18T12:00:00Z",
        completed_at="2026-07-18T12:00:01Z",
        duration_ms=1_000,
        stages=_stage_reports(),
        window_count=2,
        package_count=2,
        inference_attempt_count=5,
        inference_success_count=5,
        inference_failure_count=0,
        inference_invalid_output_count=0,
        candidate_count=1,
        event_count=1,
        fake_inference_attempt_count=5,
        real_provider_request_count=0,
        error_codes=(),
    )
    return MainlineBundle(
        schema_version="1.0",
        report=report,
        windows=(coarse_window, dense_window),
        packages=(coarse_package, dense_package),
        inference_requests=requests,
        inference_outcomes=outcomes,
        qa_aggregates=(coarse_qa_aggregate, dense_qa_aggregate),
        candidates=(candidate,),
        events=(event,),
    )


def test_full_mainline_bundle_round_trips_and_is_frozen() -> None:
    bundle = _bundle()

    round_tripped = MainlineBundle.model_validate_json(bundle.model_dump_json())

    assert round_tripped == bundle
    assert round_tripped.report.video_manifest_artifact_id == _uuid(3)
    assert round_tripped.report.video_manifest_sha256 == _digest(10)
    assert round_tripped.report.real_provider_request_count == 0
    assert round_tripped.events[0].production_eligible is False
    with pytest.raises(ValidationError, match="Instance is frozen"):
        bundle.report.status = RunStatus.FAILED  # type: ignore[misc]


@pytest.mark.parametrize("mutation", ["missing", "extra", "mismatched_nested_id"])
def test_package_requires_exact_six_camera_slots(mutation: str) -> None:
    package = _bundle().packages[0]
    payload = package.model_dump(mode="json")
    if mutation == "missing":
        payload["cameras"].pop("cam_06")
    elif mutation == "extra":
        payload["cameras"]["cam_07"] = payload["cameras"]["cam_06"]
    else:
        payload["cameras"]["cam_01"]["camera_id"] = "cam_02"

    with pytest.raises(ValidationError):
        TemporalVisualPackage.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("contract", ["qa", "action", "boundary", "event"])
def test_every_six_view_result_rejects_a_missing_camera(contract: str) -> None:
    bundle = _bundle()
    if contract == "qa":
        model: type[Any] = QAOutput
        payload = bundle.inference_outcomes[0].output.model_dump(mode="json")
        field = "cameras"
    elif contract == "action":
        model = ActionEvidence
        payload = bundle.inference_outcomes[3].output.model_dump(mode="json")
        field = "cameras"
    elif contract == "boundary":
        model = BoundaryRefinement
        payload = bundle.inference_outcomes[4].output.model_dump(mode="json")
        field = "cameras"
    else:
        model = FusedActionEvent
        payload = bundle.events[0].model_dump(mode="json")
        field = "camera_evidence"
    payload[field].pop("cam_06")

    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("mutation", ["frame_count", "sampling_count", "outside_interval"])
def test_package_reconciles_frame_counts_and_bounds(mutation: str) -> None:
    package = _bundle().packages[0]
    payload = package.model_dump(mode="json")
    if mutation == "frame_count":
        payload["frame_count_total"] = 7
    elif mutation == "sampling_count":
        payload["cameras"]["cam_01"]["sampling"]["actual_count"] = 0
        payload["cameras"]["cam_01"]["sampling"]["missed_targets"] = 1
        payload["cameras"]["cam_01"]["sampling"]["actual_fps"] = 0.0
    else:
        frame = payload["cameras"]["cam_01"]["frames"][0]
        frame["aligned_timestamp_ns"] = payload["interval"]["end_ns"]
        frame["delta_to_target_ns"] = str(
            int(frame["aligned_timestamp_ns"]) - int(frame["target_timestamp_ns"])
        )

    with pytest.raises(ValidationError):
        TemporalVisualPackage.model_validate_json(json.dumps(payload))


def test_nanoseconds_use_canonical_json_strings_and_half_open_intervals() -> None:
    package = _bundle().packages[0]
    payload = json.loads(package.model_dump_json())

    assert payload["interval"] == {"start_ns": "0", "end_ns": "4000000000"}
    assert isinstance(
        payload["cameras"]["cam_01"]["frames"][0]["aligned_timestamp_ns"],
        str,
    )

    payload["interval"]["start_ns"] = 0
    with pytest.raises(ValidationError):
        TemporalVisualPackage.model_validate_json(json.dumps(payload))

    window_payload = _bundle().windows[0].model_dump(mode="json")
    window_payload["interval"]["end_ns"] = window_payload["interval"]["start_ns"]
    with pytest.raises(ValidationError, match="start_ns must be less than end_ns"):
        TemporalWindow.model_validate_json(json.dumps(window_payload))


@pytest.mark.parametrize(
    ("outcome_index", "forbidden_field"),
    [
        (0, "package_id"),
        (1, "candidate_event_id"),
        (2, "inference_id"),
        (3, "frame_id"),
    ],
)
def test_provider_claims_reject_authoritative_persisted_ids(
    outcome_index: int,
    forbidden_field: str,
) -> None:
    output = _bundle().inference_outcomes[outcome_index].output
    payload = output.model_dump(mode="json")
    payload[forbidden_field] = _uuid(999)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(output).model_validate_json(json.dumps(payload))


def test_no_event_is_observed_negative_evidence_while_missing_is_not() -> None:
    observed_negative = CameraActionClaim(
        camera_id=CameraId.CAM_01,
        status=CameraEvidenceStatus.NO_EVENT,
        event_interval=None,
        observed_interval=_interval(),
        visibility=1.0,
        observed_frame_count=1,
        coverage_fraction=1.0,
        reported_score=None,
        frame_ordinals=(0,),
        reason=None,
    )
    missing = CameraActionClaim(
        camera_id=CameraId.CAM_01,
        status=CameraEvidenceStatus.MISSING,
        event_interval=None,
        observed_interval=None,
        visibility=None,
        observed_frame_count=0,
        coverage_fraction=0.0,
        reported_score=None,
        frame_ordinals=(),
        reason="camera unavailable",
    )

    assert observed_negative.observed_interval is not None
    assert missing.observed_interval is None

    negative_payload = observed_negative.model_dump(mode="python")
    negative_payload["observed_interval"] = None
    with pytest.raises(ValidationError, match="NO_EVENT"):
        CameraActionClaim.model_validate(negative_payload)

    missing_payload = missing.model_dump(mode="python")
    missing_payload["observed_interval"] = _interval()
    with pytest.raises(ValidationError, match="MISSING"):
        CameraActionClaim.model_validate(missing_payload)


def test_inference_outcome_is_discriminated_and_task_typed() -> None:
    bundle = _bundle()
    adapter = TypeAdapter(VisionInferenceOutcome)
    success = bundle.inference_outcomes[0]
    failure = VisionInferenceFailure(
        schema_version="1.0",
        inference_id=_uuid(800),
        request_id=_uuid(801),
        task=VisionTask.QA_COARSE,
        status=InferenceStatus.INVALID_OUTPUT,
        provider="fake",
        model_name="deterministic-fixture",
        model_version="1.0",
        output=None,
        raw_output_sha256=_digest(13),
        schema_valid=False,
        usage=_usage(),
        latency_ms=1,
        failure=InferenceFailureDetail(
            code="INVALID_FIXTURE",
            detail="fixture output did not validate",
            retryability=Retryability.PERMANENT,
        ),
    )

    assert isinstance(adapter.validate_json(success.model_dump_json()), VisionInferenceSuccess)
    assert isinstance(adapter.validate_json(failure.model_dump_json()), VisionInferenceFailure)

    mismatch = success.model_dump(mode="python")
    mismatch["task"] = VisionTask.EVENT_PROPOSAL
    with pytest.raises(ValidationError, match="output type must match task"):
        VisionInferenceSuccess.model_validate(mismatch)

    invalid_failure = failure.model_dump(mode="json")
    invalid_failure["output"] = success.output.model_dump(mode="json")
    invalid_failure["schema_valid"] = True
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(invalid_failure))


@pytest.mark.parametrize(
    "mutation",
    [
        "report_count",
        "package_window",
        "outcome_request",
        "event_action_package",
        "event_action_claim",
        "event_boundary_inference",
        "qa_evidence",
    ],
)
def test_bundle_rejects_broken_lineage_or_accounting(mutation: str) -> None:
    payload = _bundle().model_dump(mode="json")
    if mutation == "report_count":
        payload["report"]["package_count"] = 3
    elif mutation == "package_window":
        payload["packages"][0]["window_id"] = _uuid(999)
    elif mutation == "outcome_request":
        payload["inference_outcomes"][0]["request_id"] = _uuid(999)
    elif mutation == "event_action_package":
        payload["events"][0]["camera_evidence"]["cam_01"]["package_id"] = _uuid(20)
    elif mutation == "event_action_claim":
        payload["events"][0]["camera_evidence"]["cam_01"]["claim"]["reported_score"] = 0.5
    elif mutation == "event_boundary_inference":
        payload["events"][0]["boundary_inference_id"] = payload["inference_requests"][1][
            "inference_id"
        ]
    else:
        payload["qa_aggregates"][1]["camera_results"]["cam_01"]["evidence_frame_ids"] = [_uuid(999)]

    with pytest.raises(ValidationError):
        MainlineBundle.model_validate_json(json.dumps(payload))


def test_complete_status_requires_every_stage_and_all_qa_aggregates() -> None:
    bundle = _bundle()
    report_payload = bundle.report.model_dump(mode="json")
    report_payload["stages"].pop()

    with pytest.raises(ValidationError, match="every mainline stage"):
        MainlineRunReport.model_validate_json(json.dumps(report_payload))

    bundle_payload = bundle.model_dump(mode="json")
    bundle_payload["qa_aggregates"] = []
    with pytest.raises(ValidationError, match="every bundled QA request"):
        MainlineBundle.model_validate_json(json.dumps(bundle_payload))


def test_contracts_are_closed_and_strict() -> None:
    package = _bundle().packages[0]
    payload = package.model_dump(mode="json")
    payload["provider"] = "fake"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TemporalVisualPackage.model_validate_json(json.dumps(payload))

    summary = package.cameras[CameraId.CAM_01].sampling.model_dump(mode="python")
    summary["target_count"] = 1.0
    with pytest.raises(ValidationError):
        SamplingSummary.model_validate(summary)
