from __future__ import annotations

from uuid import UUID

import pytest

from robata.benchmark.evidence import BenchmarkEvidenceContext
from robata.benchmark.metrics import BenchmarkMetricPolicy, QAMetrics
from robata.benchmark.model_comparison import (
    DeploymentRunManifest,
    ExecutionPlacementEvidence,
    ExternalArtifactReference,
    FairLoadDeploymentEvidence,
    FairLoadModelComparisonReport,
    FairLoadQualificationStatus,
    ModelQualityEvidence,
    SaturationOutcomeSummary,
)
from robata.benchmark.provider_qualification import (
    TwoH100ProviderConfiguration,
    TwoH100ProviderQualificationReport,
)
from robata.inference.adapter import ProviderQualificationRequestContract
from robata.inference.experiment_execution import (
    ExperimentComparisonStatus,
    ExperimentPairComparison,
    ExperimentSideOutcome,
)
from robata.inference.models import InferenceStatus
from robata.inference.routing import (
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ModelDeployment,
    ModelRouteRole,
    endpoint_config_digest,
)
from robata.runtime.capacity import CapacityEvidenceClass
from tests.unit.test_provider_qualification import (
    _configuration,
    _point,
    _report,
    _runpod_capabilities,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


_SOURCE_PACKAGE_DIGESTS = (_digest(101),)
_ARRIVAL_SCHEDULE_DIGEST = _digest(100)


def _artifact(name: str, value: int) -> ExternalArtifactReference:
    return ExternalArtifactReference(
        uri=f"object://p18-model-comparison/{name}.json",
        sha256=_digest(value),
    )


def _deployment(*, deployment_id: str, saturation) -> ModelDeployment:
    configuration = saturation.configuration
    capabilities = saturation.capabilities
    return ModelDeployment(
        deployment_id=deployment_id,
        provider=configuration.provider,
        model_name=configuration.model_identifier,
        model_version=configuration.model_version,
        adapter_version=saturation.endpoint_config.adapter_version,
        capability_snapshot_id=capabilities.snapshot_id,
        capability_snapshot_digest=capabilities.snapshot_digest,
        endpoint_config_digest=endpoint_config_digest(saturation.endpoint_config),
        max_concurrent_requests=configuration.max_concurrent_requests,
    )


def _candidate_configuration(
    *,
    request_contracts: tuple[ProviderQualificationRequestContract, ...] | None = None,
    endpoint_url: str | None = None,
) -> TwoH100ProviderConfiguration:
    base = _configuration()
    endpoint = base.endpoint_configuration
    deployment = endpoint.deployment_configuration
    assert deployment is not None
    candidate_endpoint = endpoint.model_copy(
        update={
            "endpoint_url": (
                "https://api.runpod.test/v2/mage-vl-4b/runsync"
                if endpoint_url is None
                else endpoint_url
            ),
            "deployment_configuration": deployment.model_copy(
                update={
                    "model_identifier": "mage-vl-4b",
                    "model_version": "4.0",
                }
            ),
        }
    )
    return TwoH100ProviderConfiguration.create(
        workload_manifest_digest=base.workload_manifest_digest,
        provider=base.provider,
        model_identifier="mage-vl-4b",
        model_version="4.0",
        request_contracts=(
            base.request_contracts if request_contracts is None else request_contracts
        ),
        inference_engine=base.inference_engine,
        precision_or_quantization=base.precision_or_quantization,
        topology=base.topology,
        max_images_per_request=base.max_images_per_request,
        max_input_tokens=base.max_input_tokens,
        max_output_tokens=base.max_output_tokens,
        native_batch_enabled=base.native_batch_enabled,
        native_batch_max_size=base.native_batch_max_size,
        max_concurrent_requests=base.max_concurrent_requests,
        endpoint_configuration=candidate_endpoint,
        retry_policy=base.retry_policy,
        supported_topologies=base.supported_topologies,
    )


def _candidate_report(
    configuration: TwoH100ProviderConfiguration,
    *,
    session_offset: int = 30,
) -> TwoH100ProviderQualificationReport:
    capabilities = _runpod_capabilities().model_copy(
        update={
            "snapshot_id": str(UUID(int=900 + session_offset)),
            "snapshot_digest": _digest(900 + session_offset),
            "model_name": configuration.model_identifier,
            "model_version": configuration.model_version,
        }
    )
    return TwoH100ProviderQualificationReport(
        configuration=configuration,
        endpoint_config=configuration.endpoint_configuration,
        capabilities=capabilities,
        retry_policy=configuration.retry_policy,
        evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
        points=(
            _point(
                configuration,
                session_value=session_offset,
                offered_concurrency=4,
            ),
            _point(
                configuration,
                session_value=session_offset + 1,
                offered_concurrency=8,
                rejected_response_count=1,
            ),
        ),
    )


def _outcomes(saturation) -> tuple[SaturationOutcomeSummary, ...]:
    return tuple(
        SaturationOutcomeSummary(offered_concurrency=point.offered_concurrency)
        for point in saturation.points
    )


def _manifest(
    *,
    deployment: ModelDeployment,
    saturation,
    offset: int,
    worker_pool_identity: str,
    arrival_schedule_sha256: str = _ARRIVAL_SCHEDULE_DIGEST,
    workload_manifest_sha256: str | None = None,
    source_package_digests: tuple[str, ...] = _SOURCE_PACKAGE_DIGESTS,
) -> DeploymentRunManifest:
    return DeploymentRunManifest(
        artifact=_artifact(f"run-manifest-{offset}", 120 + offset),
        endpoint_config_digest=deployment.endpoint_config_digest,
        handler_image=_artifact(f"handler-image-{offset}", 140 + offset),
        source_package_digests=source_package_digests,
        workload_manifest_sha256=(
            saturation.configuration.workload_manifest_digest
            if workload_manifest_sha256 is None
            else workload_manifest_sha256
        ),
        arrival_schedule_sha256=arrival_schedule_sha256,
        placement=ExecutionPlacementEvidence(
            worker_pool_identity=worker_pool_identity,
            artifact=_artifact(f"placement-{offset}", 160 + offset),
        ),
    )


def _evidence(
    *,
    deployment: ModelDeployment,
    saturation,
    offset: int,
    worker_pool_identity: str,
    run_manifest: DeploymentRunManifest | None = None,
    quality: ModelQualityEvidence | None = None,
) -> FairLoadDeploymentEvidence:
    return FairLoadDeploymentEvidence(
        deployment=deployment,
        saturation=saturation,
        run_manifest=(
            _manifest(
                deployment=deployment,
                saturation=saturation,
                offset=offset,
                worker_pool_identity=worker_pool_identity,
            )
            if run_manifest is None
            else run_manifest
        ),
        outcomes=_outcomes(saturation),
        quality=ModelQualityEvidence() if quality is None else quality,
    )


def _contract(
    *,
    control: ModelDeployment,
    candidate: ModelDeployment,
    input_representation: ExperimentInputRepresentation = (
        ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING
    ),
    isolation_profile: ExperimentIsolationProfile = (
        ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE
    ),
) -> ExperimentContract:
    return ExperimentContract(
        experiment_id="model-qualification-fixture",
        contract_version="1.0",
        workload_manifest_sha256=_configuration().workload_manifest_digest,
        arrival_schedule_sha256=_ARRIVAL_SCHEDULE_DIGEST,
        comparison_config_sha256=_digest(103),
        input_representation=input_representation,
        isolation_profile=isolation_profile,
        control=control,
        candidate=candidate,
    )


def _fixture(
    *,
    input_representation: ExperimentInputRepresentation = (
        ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING
    ),
    isolation_profile: ExperimentIsolationProfile = (
        ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE
    ),
    candidate_request_contracts: tuple[ProviderQualificationRequestContract, ...] | None = None,
) -> tuple[ExperimentContract, FairLoadDeploymentEvidence, FairLoadDeploymentEvidence]:
    control_configuration = _configuration()
    control_saturation = _report(control_configuration, session_offset=10)
    candidate_saturation = _candidate_report(
        _candidate_configuration(request_contracts=candidate_request_contracts),
        session_offset=30,
    )
    control_deployment = _deployment(
        deployment_id="control-deployment",
        saturation=control_saturation,
    )
    candidate_deployment = _deployment(
        deployment_id="candidate-deployment",
        saturation=candidate_saturation,
    )
    return (
        _contract(
            control=control_deployment,
            candidate=candidate_deployment,
            input_representation=input_representation,
            isolation_profile=isolation_profile,
        ),
        _evidence(
            deployment=control_deployment,
            saturation=control_saturation,
            offset=0,
            worker_pool_identity="runpod-h100-control-pool",
        ),
        _evidence(
            deployment=candidate_deployment,
            saturation=candidate_saturation,
            offset=1,
            worker_pool_identity="runpod-h100-candidate-pool",
        ),
    )


def _context(value: int) -> BenchmarkEvidenceContext:
    return BenchmarkEvidenceContext.create(
        benchmark_id=str(UUID(int=value)),
        benchmark_manifest_digest=_digest(200 + value),
        governed_corpus_digest=_digest(300 + value),
        ground_truth_manifest_digest=_digest(400 + value),
        grouped_split_manifest_digest=_digest(500 + value),
        data_split="FROZEN_TEST",
        governance_approved=True,
        governance_approval_id=f"approval-{value}",
        governance_approval_digest=_digest(600 + value),
        governance_policy_version="fixture-governance-v1",
    )


def _paired_observation(
    *,
    contract: ExperimentContract,
    control: FairLoadDeploymentEvidence,
    candidate: FairLoadDeploymentEvidence,
) -> ExperimentPairComparison:
    control_inference_id = str(UUID(int=700))
    candidate_inference_id = str(UUID(int=701))
    return ExperimentPairComparison(
        comparison_id=str(UUID(int=702)),
        experiment_id=contract.experiment_id,
        experiment_contract_digest=contract.contract_digest,
        workload_manifest_sha256=contract.workload_manifest_sha256,
        route_id="p18-quality-paired-route",
        route_configuration_digest=_digest(703),
        input_identity_sha256=_digest(704),
        comparison_config_sha256=contract.comparison_config_sha256,
        input_representation=contract.input_representation,
        attempt=1,
        retry_count=0,
        status=ExperimentComparisonStatus.AGREEMENT,
        comparable=True,
        control_inference_id=control_inference_id,
        candidate_inference_id=candidate_inference_id,
        control=ExperimentSideOutcome(
            role=ModelRouteRole.CONTROL,
            deployment_id=control.deployment.deployment_id,
            inference_id=control_inference_id,
            status=InferenceStatus.SUCCEEDED,
            output_valid=True,
        ),
        candidate=ExperimentSideOutcome(
            role=ModelRouteRole.CANDIDATE,
            deployment_id=candidate.deployment.deployment_id,
            inference_id=candidate_inference_id,
            status=InferenceStatus.SUCCEEDED,
            output_valid=True,
        ),
    )


def _quality(
    *,
    context: BenchmarkEvidenceContext,
    value: float,
    observation: ExperimentPairComparison,
    source_package_digests: tuple[str, ...],
    artifact_offset: int,
) -> ModelQualityEvidence:
    policy = BenchmarkMetricPolicy.create(
        policy_version="fixture-metric-v1",
        critical_issue_codes=("BLACK_SCREEN",),
        event_iou_thresholds=(0.5,),
        event_start_end_tolerance_ns=1,
        boundary_tolerance_ns=1,
        calibration_bin_count=2,
        governance_approval_id=context.governance_approval_id,
        governance_approval_digest=context.governance_approval_digest,
        governance_policy_version=context.governance_policy_version,
    )
    qa = QAMetrics(
        measurement_status="MEASURED",
        evidence_context_digest=context.context_digest,
        evidence_context_identity=context.context_identity,
        metric_policy_identity=policy.policy_identity,
        metric_policy_digest=policy.policy_digest,
        metric_policy_version=policy.policy_version,
        per_issue_precision={"BLACK_SCREEN": value},
        per_issue_recall={"BLACK_SCREEN": value},
        per_issue_f1={"BLACK_SCREEN": value},
        macro_f1=value,
        micro_precision=value,
        micro_recall=value,
        micro_f1=value,
        critical_issue_recall=value,
        temporal_iou=value,
        recording_precision=value,
        recording_recall=value,
        false_accept_rate=1.0 - value,
        false_reject_rate=1.0 - value,
        sample_count=12,
    )
    return ModelQualityEvidence(
        measurement_status="MEASURED",
        evidence_context=context,
        quality_artifact=_artifact(f"quality-{artifact_offset}", 800 + artifact_offset),
        source_package_digests=source_package_digests,
        paired_observations=(observation,),
        qa_metrics=qa,
    )


def test_independent_two_model_report_is_fair_load_comparable_without_promotion() -> None:
    contract, control, candidate = _fixture()

    report = FairLoadModelComparisonReport(
        contract=contract,
        control=control,
        candidate=candidate,
    )

    assert control.deployment.model_name != candidate.deployment.model_name
    assert control.deployment.endpoint_config_digest != candidate.deployment.endpoint_config_digest
    assert (
        control.run_manifest.placement.worker_pool_identity
        != candidate.run_manifest.placement.worker_pool_identity
    )
    assert report.qualification_status is FairLoadQualificationStatus.FAIR_LOAD_COMPARABLE
    assert report.capacity_comparison_eligible is True
    assert report.quality_metrics_comparison_eligible is False
    assert all(item.comparable for item in report.capacity_comparisons)
    assert report.production_eligible is False
    assert report.unresolved_external_limits == (
        "CANDIDATE_COST_EVIDENCE_NOT_MEASURED",
        "CANDIDATE_HUMAN_ADJUDICATION_NOT_RECORDED",
        "CANDIDATE_OUTCOME_CLASSIFICATION_NOT_MEASURED",
        "CANDIDATE_QUALITY_EVIDENCE_NOT_MEASURED",
        "CONTROL_COST_EVIDENCE_NOT_MEASURED",
        "CONTROL_HUMAN_ADJUDICATION_NOT_RECORDED",
        "CONTROL_OUTCOME_CLASSIFICATION_NOT_MEASURED",
        "CONTROL_QUALITY_EVIDENCE_NOT_MEASURED",
    )


def test_independent_profile_rejects_shared_endpoint_or_worker_pool() -> None:
    contract, control, candidate = _fixture()

    same_endpoint_deployment = _deployment(
        deployment_id="candidate-shared-endpoint",
        saturation=control.saturation,
    )
    same_endpoint = _evidence(
        deployment=same_endpoint_deployment,
        saturation=control.saturation,
        offset=20,
        worker_pool_identity="runpod-h100-candidate-pool",
    )
    with pytest.raises(ValueError, match="separate endpoints"):
        FairLoadModelComparisonReport(
            contract=_contract(
                control=control.deployment,
                candidate=same_endpoint_deployment,
            ),
            control=control,
            candidate=same_endpoint,
        )

    shared_pool_candidate = candidate.model_copy(
        update={
            "run_manifest": candidate.run_manifest.model_copy(
                update={
                    "placement": candidate.run_manifest.placement.model_copy(
                        update={
                            "worker_pool_identity": (
                                control.run_manifest.placement.worker_pool_identity
                            )
                        }
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="separate worker pools"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control,
            candidate=shared_pool_candidate,
        )


def test_colocated_and_model_specific_blockers_are_retained_together() -> None:
    contract, control, candidate = _fixture(
        input_representation=ExperimentInputRepresentation.MODEL_SPECIFIC_RENDERING,
        isolation_profile=ExperimentIsolationProfile.COLOCATED_SHARED_HARDWARE,
    )

    report = FairLoadModelComparisonReport(
        contract=contract,
        control=control,
        candidate=candidate,
    )

    assert report.qualification_status is FairLoadQualificationStatus.CONTENTION_ONLY
    assert report.capacity_comparison_eligible is False
    assert report.capacity_comparison_blockers == (
        "COLOCATED_SHARED_HARDWARE",
        "MODEL_SPECIFIC_RENDERING",
    )
    assert all(
        item.non_comparable_reasons == ("COLOCATED_SHARED_HARDWARE", "MODEL_SPECIFIC_RENDERING")
        and item.logical_calls_per_wall_hour_ratio is None
        for item in report.capacity_comparisons
    )


def test_arrival_workload_and_nested_p6_hardware_drift_are_rejected() -> None:
    contract, control, candidate = _fixture()

    arrival_drift = candidate.model_copy(
        update={
            "run_manifest": candidate.run_manifest.model_copy(
                update={"arrival_schedule_sha256": _digest(999)}
            )
        }
    )
    with pytest.raises(ValueError, match="arrival schedule"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control,
            candidate=arrival_drift,
        )

    with pytest.raises(ValueError, match="deployment workload"):
        FairLoadModelComparisonReport(
            contract=contract.model_copy(update={"workload_manifest_sha256": _digest(998)}),
            control=control,
            candidate=candidate,
        )

    changed_first_point = candidate.saturation.points[0].model_copy(
        update={
            "telemetry": candidate.saturation.points[0].telemetry.model_copy(
                update={
                    "gpu": candidate.saturation.points[0].telemetry.gpu.model_copy(
                        update={"gpu_sku": "NVIDIA H100 PCIe 80GB"}
                    )
                }
            )
        }
    )
    bypassed_nested_validation = candidate.model_copy(
        update={
            "saturation": candidate.saturation.model_copy(
                update={
                    "points": (
                        changed_first_point,
                        *candidate.saturation.points[1:],
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="same GPU inventory"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control,
            candidate=bypassed_nested_validation,
        )


def test_run_manifest_endpoint_binding_and_identical_rendering_contract_are_enforced() -> None:
    contract, _control, candidate = _fixture()

    with pytest.raises(ValueError, match="run manifest does not match its deployment endpoint"):
        FairLoadDeploymentEvidence(
            deployment=candidate.deployment,
            saturation=candidate.saturation,
            run_manifest=candidate.run_manifest.model_copy(
                update={"endpoint_config_digest": _digest(997)}
            ),
            outcomes=candidate.outcomes,
        )

    changed_request_contract = (
        _configuration().request_contracts[0].model_copy(update={"prompt_sha256": _digest(996)})
    )
    different_contract, different_control, different_candidate = _fixture(
        candidate_request_contracts=(changed_request_contract,)
    )
    with pytest.raises(ValueError, match="prompt and output contracts"):
        FairLoadModelComparisonReport(
            contract=different_contract,
            control=different_control,
            candidate=different_candidate,
        )

    assert contract.input_representation is ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING


def test_measured_quality_requires_matching_p17_evidence_and_frozen_labels() -> None:
    contract, control, candidate = _fixture()
    observation = _paired_observation(
        contract=contract,
        control=control,
        candidate=candidate,
    )
    context = _context(1)
    control_quality = _quality(
        context=context,
        value=0.7,
        observation=observation,
        source_package_digests=control.run_manifest.source_package_digests,
        artifact_offset=0,
    )
    candidate_quality = _quality(
        context=context,
        value=0.8,
        observation=observation,
        source_package_digests=candidate.run_manifest.source_package_digests,
        artifact_offset=1,
    )
    report = FairLoadModelComparisonReport(
        contract=contract,
        control=control.model_copy(update={"quality": control_quality}),
        candidate=candidate.model_copy(update={"quality": candidate_quality}),
    )

    assert report.quality_metrics_comparison_eligible is True

    contract_mismatch = observation.model_copy(update={"experiment_contract_digest": _digest(995)})
    with pytest.raises(
        ValueError, match="quality observation does not match the experiment contract"
    ):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control.model_copy(
                update={
                    "quality": _quality(
                        context=context,
                        value=0.7,
                        observation=contract_mismatch,
                        source_package_digests=control.run_manifest.source_package_digests,
                        artifact_offset=2,
                    )
                }
            ),
            candidate=candidate.model_copy(
                update={
                    "quality": _quality(
                        context=context,
                        value=0.8,
                        observation=contract_mismatch,
                        source_package_digests=candidate.run_manifest.source_package_digests,
                        artifact_offset=3,
                    )
                }
            ),
        )

    with pytest.raises(ValueError, match="one frozen label context"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control.model_copy(
                update={
                    "quality": _quality(
                        context=_context(1),
                        value=0.7,
                        observation=observation,
                        source_package_digests=control.run_manifest.source_package_digests,
                        artifact_offset=4,
                    )
                }
            ),
            candidate=candidate.model_copy(
                update={
                    "quality": _quality(
                        context=_context(2),
                        value=0.8,
                        observation=observation,
                        source_package_digests=candidate.run_manifest.source_package_digests,
                        artifact_offset=5,
                    )
                }
            ),
        )


def test_independent_profile_rejects_distinct_model_with_shared_endpoint_url() -> None:
    control_configuration = _configuration()
    control_saturation = _report(control_configuration, session_offset=50)
    candidate_configuration = _candidate_configuration(
        endpoint_url=str(control_configuration.endpoint_configuration.endpoint_url)
    )
    candidate_saturation = _candidate_report(candidate_configuration, session_offset=70)
    control_deployment = _deployment(
        deployment_id="control-shared-url-deployment",
        saturation=control_saturation,
    )
    candidate_deployment = _deployment(
        deployment_id="candidate-shared-url-deployment",
        saturation=candidate_saturation,
    )
    control = _evidence(
        deployment=control_deployment,
        saturation=control_saturation,
        offset=40,
        worker_pool_identity="runpod-h100-control-pool",
    )
    candidate = _evidence(
        deployment=candidate_deployment,
        saturation=candidate_saturation,
        offset=41,
        worker_pool_identity="runpod-h100-candidate-pool",
    )

    assert control.deployment.model_name != candidate.deployment.model_name
    assert control.deployment.endpoint_config_digest != candidate.deployment.endpoint_config_digest
    assert str(control.saturation.endpoint_config.endpoint_url) == str(
        candidate.saturation.endpoint_config.endpoint_url
    )
    with pytest.raises(ValueError, match="separate endpoint URLs"):
        FairLoadModelComparisonReport(
            contract=_contract(
                control=control.deployment,
                candidate=candidate.deployment,
            ),
            control=control,
            candidate=candidate,
        )


def test_outer_report_revalidates_model_copy_contract_and_outcomes() -> None:
    contract, control, candidate = _fixture()

    invalid_contract = contract.model_copy(update={"isolation_profile": "UNTRUSTED"})
    with pytest.raises(ValueError, match="isolation_profile"):
        FairLoadModelComparisonReport(
            contract=invalid_contract,
            control=control,
            candidate=candidate,
        )

    invalid_outcome = candidate.outcomes[0].model_copy(update={"schema_valid_terminal_count": 1})
    invalid_candidate = candidate.model_copy(
        update={"outcomes": (invalid_outcome, *candidate.outcomes[1:])}
    )
    with pytest.raises(ValueError, match="NOT_MEASURED outcomes cannot retain classified counts"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control,
            candidate=invalid_candidate,
        )


@pytest.mark.parametrize(
    (
        "side_name",
        "side_status",
        "side_output_valid",
        "comparison_status",
        "comparable",
        "expected_error",
    ),
    (
        (
            None,
            None,
            None,
            ExperimentComparisonStatus.BOTH_FAILURE,
            False,
            "quality observation must be a comparable P17 result",
        ),
        (
            "control",
            InferenceStatus.FAILED,
            True,
            ExperimentComparisonStatus.AGREEMENT,
            True,
            "quality observation requires a successful valid model output",
        ),
        (
            "candidate",
            InferenceStatus.FAILED,
            True,
            ExperimentComparisonStatus.AGREEMENT,
            True,
            "quality observation requires a successful valid model output",
        ),
        (
            "control",
            InferenceStatus.SUCCEEDED,
            False,
            ExperimentComparisonStatus.AGREEMENT,
            True,
            "quality observation requires a successful valid model output",
        ),
        (
            "candidate",
            InferenceStatus.SUCCEEDED,
            False,
            ExperimentComparisonStatus.AGREEMENT,
            True,
            "quality observation requires a successful valid model output",
        ),
    ),
)
def test_measured_quality_rejects_noncomparable_and_invalid_p17_sidecars(
    side_name: str | None,
    side_status: InferenceStatus | None,
    side_output_valid: bool | None,
    comparison_status: ExperimentComparisonStatus,
    comparable: bool,
    expected_error: str,
) -> None:
    contract, control, candidate = _fixture()
    observation = _paired_observation(
        contract=contract,
        control=control,
        candidate=candidate,
    ).model_copy(
        update={
            "status": comparison_status,
            "comparable": comparable,
        }
    )
    if side_name is not None:
        side = getattr(observation, side_name)
        assert isinstance(side, ExperimentSideOutcome)
        observation = observation.model_copy(
            update={
                side_name: side.model_copy(
                    update={
                        "status": side_status,
                        "output_valid": side_output_valid,
                    }
                )
            }
        )

    context = _context(30)
    control_quality = _quality(
        context=context,
        value=0.7,
        observation=observation,
        source_package_digests=control.run_manifest.source_package_digests,
        artifact_offset=30,
    )
    candidate_quality = _quality(
        context=context,
        value=0.8,
        observation=observation,
        source_package_digests=candidate.run_manifest.source_package_digests,
        artifact_offset=31,
    )
    with pytest.raises(ValueError, match=expected_error):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control.model_copy(update={"quality": control_quality}),
            candidate=candidate.model_copy(update={"quality": candidate_quality}),
        )


def test_outer_report_revalidates_model_copy_quality_metric_and_input_identity() -> None:
    contract, control, candidate = _fixture()
    observation = _paired_observation(
        contract=contract,
        control=control,
        candidate=candidate,
    )
    context = _context(40)
    control_quality = _quality(
        context=context,
        value=0.7,
        observation=observation,
        source_package_digests=control.run_manifest.source_package_digests,
        artifact_offset=40,
    )
    candidate_quality = _quality(
        context=context,
        value=0.8,
        observation=observation,
        source_package_digests=candidate.run_manifest.source_package_digests,
        artifact_offset=41,
    )

    assert control_quality.qa_metrics is not None
    negative_sample_quality = control_quality.model_copy(
        update={"qa_metrics": control_quality.qa_metrics.model_copy(update={"sample_count": -1})}
    )
    with pytest.raises(ValueError, match="sample_count"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control.model_copy(update={"quality": negative_sample_quality}),
            candidate=candidate.model_copy(update={"quality": candidate_quality}),
        )

    repeated_input_observation = observation.model_copy(
        update={"comparison_id": str(UUID(int=999))}
    )
    assert repeated_input_observation.comparison_id != observation.comparison_id
    assert repeated_input_observation.input_identity_sha256 == observation.input_identity_sha256
    repeated_input_quality = control_quality.model_copy(
        update={"paired_observations": (observation, repeated_input_observation)}
    )
    with pytest.raises(
        ValueError,
        match="quality observations must not repeat an experiment input",
    ):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control.model_copy(update={"quality": repeated_input_quality}),
            candidate=candidate.model_copy(update={"quality": candidate_quality}),
        )


def test_outer_report_revalidates_nested_artifact_and_terminal_identity() -> None:
    contract, control, candidate = _fixture()

    invalid_manifest = candidate.run_manifest.model_copy(
        update={
            "handler_image": candidate.run_manifest.handler_image.model_copy(
                update={"sha256": "not-a-digest"}
            )
        }
    )
    with pytest.raises(ValueError, match="sha256"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control,
            candidate=candidate.model_copy(update={"run_manifest": invalid_manifest}),
        )

    observation = _paired_observation(
        contract=contract,
        control=control,
        candidate=candidate,
    )
    assert observation.candidate is not None
    reused_terminal_observation = observation.model_copy(
        update={
            "candidate_inference_id": observation.control_inference_id,
            "candidate": observation.candidate.model_copy(
                update={"inference_id": observation.control_inference_id}
            ),
        }
    )
    context = _context(50)
    control_quality = _quality(
        context=context,
        value=0.7,
        observation=reused_terminal_observation,
        source_package_digests=control.run_manifest.source_package_digests,
        artifact_offset=50,
    )
    candidate_quality = _quality(
        context=context,
        value=0.8,
        observation=reused_terminal_observation,
        source_package_digests=candidate.run_manifest.source_package_digests,
        artifact_offset=51,
    )
    with pytest.raises(ValueError, match="cannot reuse one inference terminal"):
        FairLoadModelComparisonReport(
            contract=contract,
            control=control.model_copy(update={"quality": control_quality}),
            candidate=candidate.model_copy(update={"quality": candidate_quality}),
        )


def test_independent_profile_normalizes_endpoint_host_identity() -> None:
    control_configuration = _configuration()
    control_saturation = _report(control_configuration, session_offset=80)
    candidate_saturation = _candidate_report(
        _candidate_configuration(endpoint_url="https://API.RUNPOD.TEST/v2/qualified/runsync"),
        session_offset=90,
    )
    control_deployment = _deployment(
        deployment_id="control-case-endpoint-deployment",
        saturation=control_saturation,
    )
    candidate_deployment = _deployment(
        deployment_id="candidate-case-endpoint-deployment",
        saturation=candidate_saturation,
    )
    control = _evidence(
        deployment=control_deployment,
        saturation=control_saturation,
        offset=60,
        worker_pool_identity="runpod-h100-control-pool",
    )
    candidate = _evidence(
        deployment=candidate_deployment,
        saturation=candidate_saturation,
        offset=61,
        worker_pool_identity="runpod-h100-candidate-pool",
    )

    with pytest.raises(ValueError, match="separate endpoint URLs"):
        FairLoadModelComparisonReport(
            contract=_contract(
                control=control.deployment,
                candidate=candidate.deployment,
            ),
            control=control,
            candidate=candidate,
        )
