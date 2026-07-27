from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest

from robata.application.canonical.qualification_evidence import (
    CanonicalRecoveryEvidenceClass,
    CanonicalRecoveryQualificationEvidence,
    CanonicalRecoveryReceiptEvidence,
    CanonicalRecoveryScenario,
)
from robata.benchmark.evidence import BenchmarkEvidenceContext
from robata.benchmark.promotion import (
    GateCategory,
    GateResult,
    PromotionDecision,
    PromotionGate,
    PromotionGateRegistry,
)
from robata.benchmark.provider_qualification import (
    ProviderSaturationPoint,
    TwoH100ProviderQualificationReport,
)
from robata.benchmark.qualification import (
    ExternalQualificationGateEvidence,
    GovernedQualityQualificationEvidence,
    ProductionQualificationScope,
    RepresentativeCostResourceEvidence,
    RepresentativeDeadlineEvidence,
    RepresentativeMediaProfile,
    RepresentativeOperationsEvidence,
    RepresentativeProductionQualificationReport,
    RepresentativeQualityCoverageEvidence,
    RepresentativeQueueObservation,
    RepresentativeServiceCapacityEvidence,
)
from robata.contracts.qa import ProductQAIssue
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityInput,
    ProviderMode,
    build_measured_capacity_report,
)
from tests.unit.test_provider_qualification import (
    _DIGEST,
    _HOUR_NS,
    _configuration,
    _endpoint_config,
    _runpod_capabilities,
    _session,
    _telemetry,
)

_DAY_NS = 86_400_000_000_000


def _uuid(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"robata:p10:{label}"))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _capacity_for_rtf(recording_rtf: float):
    return build_measured_capacity_report(
        MeasuredCapacityInput(
            workload_fingerprint=_DIGEST,
            evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
            provider_mode=ProviderMode.NETWORK_PROVIDER,
            execution_mode="FRESH",
            recording_count=1,
            recording_worker_count=1,
            camera_count=6,
            recording_duration_ns=_HOUR_NS,
            wall_time_ns=round(_HOUR_NS / recording_rtf),
            provider_images=120,
            logical_calls=12,
            http_requests=12,
            retries=0,
            input_tokens=1_200,
            output_tokens=120,
            output_token_responses=12,
        )
    )


def _service_capacity_report(
    recording_rtf: float,
    run_duration_ns: int = _DAY_NS,
):
    return build_measured_capacity_report(
        MeasuredCapacityInput(
            workload_fingerprint=_DIGEST,
            evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
            provider_mode=ProviderMode.NETWORK_PROVIDER,
            execution_mode="FRESH",
            recording_count=1,
            recording_worker_count=4,
            camera_count=6,
            recording_duration_ns=round(recording_rtf * run_duration_ns),
            wall_time_ns=run_duration_ns,
        )
    )


def _provider_report() -> tuple[
    TwoH100ProviderQualificationReport,
    ProviderSaturationPoint,
    ProviderSaturationPoint,
]:
    configuration = _configuration()
    preferred_session = _session(configuration, 2_001)
    saturation_session = _session(configuration, 2_002)
    preferred = ProviderSaturationPoint(
        configuration_digest=configuration.configuration_digest,
        qualification_session=preferred_session,
        run_namespace=preferred_session.run_namespace,
        offered_concurrency=4,
        capacity=_capacity_for_rtf(21.0),
        telemetry=_telemetry(preferred_session),
    )
    saturation = ProviderSaturationPoint(
        configuration_digest=configuration.configuration_digest,
        qualification_session=saturation_session,
        run_namespace=saturation_session.run_namespace,
        offered_concurrency=8,
        capacity=_capacity_for_rtf(30.0),
        telemetry=_telemetry(saturation_session),
    )
    return (
        TwoH100ProviderQualificationReport(
            configuration=configuration,
            endpoint_config=_endpoint_config(configuration),
            capabilities=_runpod_capabilities(),
            retry_policy=configuration.retry_policy,
            points=(preferred, saturation),
            evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
        ),
        preferred,
        saturation,
    )


def _scope() -> ProductionQualificationScope:
    return ProductionQualificationScope.create(
        qualification_id=_uuid("qualification"),
        qualification_run_namespace="p10-qualification-run",
        workload_manifest_digest=_DIGEST,
        benchmark_manifest_digest=_digest(1),
        governed_corpus_digest=_digest(11),
        ground_truth_manifest_digest=_digest(12),
        grouped_split_manifest_digest=_digest(13),
        provider_configuration_digest=_configuration().configuration_digest,
        code_revision_sha256=_digest(2),
        schema_catalog_sha256=_digest(3),
        sampler_policy_sha256=_digest(4),
        media_manifest_sha256=_digest(5),
        calibration_profile_sha256=_digest(9),
        preprocess_policy_sha256=_digest(10),
        arrival_distribution_sha256=_digest(6),
        provider_adapter_version="runpod-adapter-v1",
        storage_adapter_version="object-store-v1",
        storage_configuration_sha256=_digest(7),
        broker_adapter_version="broker-v1",
        broker_configuration_sha256=_digest(8),
        host_resource_inventory_uri="object://qualification/h100-inventory.json",
        host_resource_inventory_sha256="d" * 64,
    )


def _service_capacity(
    scope: ProductionQualificationScope,
    recording_rtf: float = 26.0,
    backlog_start_count: int = 1,
    run_duration_ns: int = _DAY_NS,
) -> RepresentativeServiceCapacityEvidence:
    return RepresentativeServiceCapacityEvidence(
        run_namespace="p10-qualification-run",
        qualification_scope_sha256=scope.scope_sha256,
        arrival_distribution_sha256=scope.arrival_distribution_sha256,
        arrival_observation_uri="object://qualification/arrival-observation.json",
        arrival_observation_sha256=_digest(17),
        arrival_recording_count=500,
        arrival_recording_hours=500.0,
        arrival_camera_hours=3_000.0,
        provider_configuration_digest=_provider_report()[0].configuration.configuration_digest,
        code_revision_sha256=scope.code_revision_sha256,
        schema_catalog_sha256=scope.schema_catalog_sha256,
        sampler_policy_sha256=scope.sampler_policy_sha256,
        media_manifest_sha256=scope.media_manifest_sha256,
        calibration_profile_sha256=scope.calibration_profile_sha256,
        preprocess_policy_sha256=scope.preprocess_policy_sha256,
        storage_adapter_version=scope.storage_adapter_version,
        storage_configuration_sha256=scope.storage_configuration_sha256,
        broker_adapter_version=scope.broker_adapter_version,
        broker_configuration_sha256=scope.broker_configuration_sha256,
        capacity=_service_capacity_report(recording_rtf, run_duration_ns),
        run_duration_ns=run_duration_ns,
        queues=(
            RepresentativeQueueObservation(
                name="ingress",
                capacity=64,
                high_watermark=64,
                end_depth=0,
            ),
            RepresentativeQueueObservation(
                name="provider",
                capacity=32,
                high_watermark=31,
                end_depth=0,
            ),
            RepresentativeQueueObservation(
                name="publish",
                capacity=16,
                high_watermark=16,
                end_depth=0,
            ),
        ),
        arrival_peak_observed=True,
        backlog_drained_after_peak=True,
        backlog_start_count=backlog_start_count,
        backlog_end_count=0,
        outbox_delivery_p95_ns=1,
        outbox_delivery_observation_uri="object://qualification/outbox-latency.json",
        outbox_delivery_observation_sha256=_digest(15),
        outbox_delivery_observation_count=1,
        review_delivery_p95_ns=1,
        review_delivery_observation_uri="object://qualification/review-latency.json",
        review_delivery_observation_sha256=_digest(16),
        review_delivery_observation_count=1,
        evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
        production_eligible=False,
    )


def _gate(category: GateCategory, label: str) -> PromotionGate:
    return PromotionGate(
        gate_id=_uuid(label),
        category=category,
        metric_definition=label,
        threshold=0.9,
        margin=0.0,
        comparison="GTE",
        denominator="records",
        required_strata=("all",),
        data_split="FROZEN_TEST",
        owner="quality-owner",
        effective_date="2026-07-25",
        failure_action="block",
        version="p10-gates-v1",
    )


def _quality(scope: ProductionQualificationScope) -> GovernedQualityQualificationEvidence:
    context = BenchmarkEvidenceContext.create(
        benchmark_id=_uuid("benchmark"),
        benchmark_manifest_digest=scope.benchmark_manifest_digest,
        governed_corpus_digest=_digest(11),
        ground_truth_manifest_digest=_digest(12),
        grouped_split_manifest_digest=_digest(13),
        data_split="FROZEN_TEST",
        governance_approved=True,
        governance_approval_id="approved-p10-labels",
        governance_approval_digest=_digest(14),
        governance_policy_version="1.0",
    )
    gates = (
        _gate(GateCategory.DATA_LINEAGE, "data-lineage"),
        _gate(GateCategory.ALIGNMENT, "alignment"),
        _gate(GateCategory.QA, "qa-quality"),
        _gate(GateCategory.QA, "qa-calibration"),
        _gate(GateCategory.EVENT_PROPOSAL, "event-proposal"),
        _gate(GateCategory.ACTION_BOUNDARY, "action-boundary"),
        _gate(GateCategory.STRUCTURED_OUTPUT, "structured-output"),
        _gate(GateCategory.PRIMARY_REGRESSION, "primary-regression"),
        _gate(GateCategory.SHADOW_ISOLATION, "shadow-isolation"),
        _gate(GateCategory.CAPACITY, "capacity"),
        _gate(GateCategory.COST, "cost"),
    )
    frozen_at = datetime(2026, 7, 25, tzinfo=UTC)
    registry = PromotionGateRegistry(
        registry_id=_uuid("registry"),
        gates=gates,
        benchmark_id=context.benchmark_id,
        evidence_context_digest=context.context_digest,
        frozen_at=frozen_at,
    )
    approved = tuple(
        GateResult(
            category=gate.category,
            passed=True,
            evidence={
                "gate_id": gate.gate_id,
                "metric_definition": gate.metric_definition,
                "comparison": gate.comparison,
                "margin": gate.margin,
                "denominator": gate.denominator,
                "required_strata": list(gate.required_strata),
                "data_split": gate.data_split,
                "evidence_context_digest": context.context_digest,
                "evidence_context_identity": context.context_identity,
                "measurement_status": "MEASURED",
                "stratum_values": {"all": 0.95},
            },
            threshold=gate.threshold,
            actual_value=0.95,
        )
        for gate in gates
    )
    decision = PromotionDecision(
        approved=True,
        rejected_gates=(),
        approved_gates=approved,
        timestamp=frozen_at,
        validation_errors=(),
    )
    return GovernedQualityQualificationEvidence(
        context=context,
        qualification_scope_sha256=scope.scope_sha256,
        code_revision_sha256=scope.code_revision_sha256,
        schema_catalog_sha256=scope.schema_catalog_sha256,
        sampler_policy_sha256=scope.sampler_policy_sha256,
        media_manifest_sha256=scope.media_manifest_sha256,
        calibration_profile_sha256=scope.calibration_profile_sha256,
        preprocess_policy_sha256=scope.preprocess_policy_sha256,
        registry=registry,
        decision=decision,
        qa_gate_ids=(gates[2].gate_id,),
        event_gate_ids=(gates[4].gate_id, gates[5].gate_id),
        calibration_gate_ids=(gates[3].gate_id,),
    )


def _quality_coverage(scope: ProductionQualificationScope) -> RepresentativeQualityCoverageEvidence:
    return RepresentativeQualityCoverageEvidence(
        run_namespace="p10-qualification-run",
        qualification_scope_sha256=scope.scope_sha256,
        qa_class_ids=tuple(ProductQAIssue),
        media_profiles=(
            RepresentativeMediaProfile(
                codec="h264",
                width=1920,
                height=1080,
                frames_per_second=30.0,
                gop_frames=30,
            ),
        ),
        observation_uri="object://qualification/quality-coverage.json",
        observation_sha256=_digest(17),
    )


def _deadlines(scope: ProductionQualificationScope) -> RepresentativeDeadlineEvidence:
    return RepresentativeDeadlineEvidence(
        run_namespace="p10-qualification-run",
        qualification_scope_sha256=scope.scope_sha256,
        qa_observation_count=500,
        qa_deadline_miss_count=0,
        qa_completion_p50_ns=1,
        qa_completion_p95_ns=2,
        qa_completion_p99_ns=3,
        annotation_observation_count=500,
        annotation_deadline_miss_count=0,
        annotation_completion_p50_ns=1,
        annotation_completion_p95_ns=2,
        annotation_completion_p99_ns=3,
        observation_uri="object://qualification/deadlines.json",
        observation_sha256=_digest(18),
    )


def _cost_resources(scope: ProductionQualificationScope) -> RepresentativeCostResourceEvidence:
    return RepresentativeCostResourceEvidence(
        run_namespace="p10-qualification-run",
        qualification_scope_sha256=scope.scope_sha256,
        recording_hours=500.0,
        camera_hours=3_000.0,
        provider_cost_usd=1.0,
        gpu_cost_usd=2.0,
        object_storage_cost_usd=3.0,
        object_egress_cost_usd=4.0,
        database_cost_usd=5.0,
        queue_cost_usd=6.0,
        total_cost_usd=21.0,
        cpu_seconds=7.0,
        gpu_seconds=8.0,
        nvme_read_bytes=9,
        nvme_write_bytes=10,
        object_storage_bytes=11,
        object_egress_bytes=12,
        database_operation_count=13,
        queue_operation_count=14,
        observation_uri="object://qualification/cost-resources.json",
        observation_sha256=_digest(19),
    )


def _operations(scope: ProductionQualificationScope) -> RepresentativeOperationsEvidence:
    return RepresentativeOperationsEvidence(
        run_namespace="p10-qualification-run",
        qualification_scope_sha256=scope.scope_sha256,
        runbook_uri="object://qualification/runbook.md",
        runbook_sha256=_digest(20),
        security_retention_uri="object://qualification/security-retention.md",
        security_retention_sha256=_digest(21),
        incident_response_uri="object://qualification/incident-response.md",
        incident_response_sha256=_digest(22),
    )


def _external_gates() -> tuple[ExternalQualificationGateEvidence, ...]:
    statuses = (
        "NOT_MEASURED",
        "NOT_MEASURED",
        "NOT_MEASURED",
        "NOT_MEASURED",
        "NOT_MEASURED",
        "NOT_MEASURED",
        "PENDING_INDEPENDENT_REVIEW",
    )
    return tuple(
        ExternalQualificationGateEvidence(
            gate_id=f"E{ordinal}",
            status=status,
            unresolved_reason=(
                "independent review pending"
                if status == "PENDING_INDEPENDENT_REVIEW"
                else "external evidence is not measured"
            ),
        )
        for ordinal, status in enumerate(statuses)
    )


def _recovery_evidence(
    scope: ProductionQualificationScope,
) -> tuple[CanonicalRecoveryQualificationEvidence, ...]:
    failure_scenarios = {
        CanonicalRecoveryScenario.RESTART_REPLAY,
        CanonicalRecoveryScenario.PROCESS_CRASH,
        CanonicalRecoveryScenario.LEASE_EXPIRY,
        CanonicalRecoveryScenario.DUPLICATE_INJECTION,
        CanonicalRecoveryScenario.PROVIDER_RETRY,
        CanonicalRecoveryScenario.PROVIDER_TIMEOUT,
        CanonicalRecoveryScenario.BROKER_FAILURE,
        CanonicalRecoveryScenario.OBJECT_STORE_FAILURE,
    }
    evidence: list[CanonicalRecoveryQualificationEvidence] = []
    for ordinal, scenario in enumerate(CanonicalRecoveryScenario, start=100):
        completion = _digest(ordinal)
        outbox_id = f"outbox-{ordinal}"
        review_task_id = f"review-{ordinal}"
        common = {
            "run_id": f"p10-{scenario.value.lower()}",
            "recording_identity": f"recording-{ordinal}",
            "status": "SUCCEEDED",
            "command_sha256": _digest(ordinal + 100),
            "completion_semantic_sha256": completion,
            "event_ids": (),
            "revision_ids": (),
            "outbox_ids": (outbox_id,),
            "review_task_id": review_task_id,
            "evidence_class": CanonicalRecoveryEvidenceClass.PRODUCTION_QUALIFICATION,
            "production_eligible": False,
        }
        fresh = CanonicalRecoveryReceiptEvidence(
            **common,
            outbox_delivery_outcome="DELIVERED",
            review_routing_disposition="ENQUEUED",
            replayed=False,
        )
        replay = CanonicalRecoveryReceiptEvidence(
            **common,
            outbox_delivery_outcome="DELIVERED",
            review_routing_disposition="ALREADY_ENQUEUED",
            replayed=True,
        )
        evidence.append(
            CanonicalRecoveryQualificationEvidence(
                workload_fingerprint=_DIGEST,
                run_namespace="p10-qualification-run",
                qualification_scope_sha256=scope.scope_sha256,
                scenario=scenario,
                scenario_evidence_sha256=_digest(ordinal + 200),
                fresh=fresh,
                replay=replay,
                failure_observed=scenario in failure_scenarios,
                recovery_completed=True,
                authoritative_terminal_ids=(completion,),
                authoritative_terminal_count=1,
                duplicate_terminal_count=0,
                outbox_delivery_ids=(outbox_id,),
                outbox_delivery_count=1,
                duplicate_outbox_delivery_count=0,
                review_task_ids=(review_task_id,),
                review_task_count=1,
                duplicate_review_task_count=0,
                duplicate_injection_count=(
                    1 if scenario is CanonicalRecoveryScenario.DUPLICATE_INJECTION else 0
                ),
                backlog_start_count=(
                    2 if scenario is CanonicalRecoveryScenario.BACKLOG_DRAIN else 0
                ),
                backlog_end_count=0,
                run_duration_ns=(
                    _DAY_NS if scenario is CanonicalRecoveryScenario.SOAK else _HOUR_NS
                ),
                evidence_class=CanonicalRecoveryEvidenceClass.PRODUCTION_QUALIFICATION,
                production_eligible=False,
            )
        )
    return tuple(evidence)


def _report_inputs() -> dict[str, object]:
    scope = _scope()
    provider_report, preferred, saturation = _provider_report()
    return {
        "scope": scope,
        "service_capacity": _service_capacity(scope),
        "provider_saturation": provider_report,
        "preferred_operating_point_run_namespace": preferred.run_namespace,
        "saturation_supporting_point_run_namespace": saturation.run_namespace,
        "quality": _quality(scope),
        "recovery_evidence": _recovery_evidence(scope),
        "quality_coverage": _quality_coverage(scope),
        "deadlines": _deadlines(scope),
        "cost_resources": _cost_resources(scope),
        "operations": _operations(scope),
        "external_gates": _external_gates(),
    }


def test_representative_report_binds_all_p10_evidence_without_promoting_it() -> None:
    report = RepresentativeProductionQualificationReport.create(**_report_inputs())

    assert report.technical_requirements_satisfied is False
    assert report.production_eligible is False
    assert report.report_sha256 != "0" * 64
    assert report.service_capacity.capacity.recording_hours_per_wall_hour >= 25.0
    with pytest.raises(ValueError, match="report_sha256"):
        report.model_copy(update={"report_sha256": "0" * 64}).validate_report()


def test_service_capacity_rejects_a_subtarget_24_hour_run() -> None:
    with pytest.raises(ValueError, match="25"):
        _service_capacity(_scope(), recording_rtf=24.9)


def test_service_capacity_rejects_an_underfilled_arrival_observation() -> None:
    with pytest.raises(ValueError, match="500 recording-hours"):
        _service_capacity(_scope()).model_copy(
            update={"arrival_recording_hours": 499.9}
        ).validate_service_capacity()


def test_service_capacity_requires_500_recording_hours_per_elapsed_day() -> None:
    with pytest.raises(ValueError, match="500 recording-hours per 24 hours"):
        _service_capacity(_scope(), run_duration_ns=2 * _DAY_NS)


def test_representative_report_rejects_local_recovery_provenance() -> None:
    inputs = _report_inputs()
    recoveries = list(inputs["recovery_evidence"])
    first = recoveries[0]
    local_fresh = first.fresh.model_copy(
        update={"evidence_class": CanonicalRecoveryEvidenceClass.LOCAL_CONFORMANCE}
    )
    local_replay = first.replay.model_copy(
        update={"evidence_class": CanonicalRecoveryEvidenceClass.LOCAL_CONFORMANCE}
    )
    recoveries[0] = first.model_copy(
        update={
            "fresh": local_fresh,
            "replay": local_replay,
            "evidence_class": CanonicalRecoveryEvidenceClass.LOCAL_CONFORMANCE,
        }
    )

    with pytest.raises(ValueError, match="production"):
        RepresentativeProductionQualificationReport.create(
            **{**inputs, "recovery_evidence": tuple(recoveries)}
        )


def test_representative_report_rejects_an_overutilized_preferred_point() -> None:
    inputs = _report_inputs()
    provider_report = inputs["provider_saturation"]
    assert isinstance(provider_report, TwoH100ProviderQualificationReport)
    preferred_namespace = inputs["preferred_operating_point_run_namespace"]
    points: list[ProviderSaturationPoint] = []
    for point in provider_report.points:
        if point.run_namespace == preferred_namespace:
            gpu = point.telemetry.gpu.model_copy(update={"gpu_utilization_fraction": 0.71})
            telemetry = point.telemetry.model_copy(update={"gpu": gpu})
            point = point.model_copy(update={"telemetry": telemetry})
        points.append(point)

    with pytest.raises(ValueError, match="utilization"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "provider_saturation": provider_report.model_copy(update={"points": tuple(points)}),
            }
        )


def test_representative_report_rejects_evidence_from_another_qualification_run() -> None:
    inputs = _report_inputs()
    service_capacity = inputs["service_capacity"]
    assert isinstance(service_capacity, RepresentativeServiceCapacityEvidence)

    with pytest.raises(ValueError, match="run namespace"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "service_capacity": service_capacity.model_copy(
                    update={"run_namespace": "another-qualification-run"}
                ),
            }
        )


def test_representative_report_rejects_arrival_distribution_mismatch() -> None:
    inputs = _report_inputs()
    service_capacity = inputs["service_capacity"]
    assert isinstance(service_capacity, RepresentativeServiceCapacityEvidence)

    with pytest.raises(ValueError, match="arrival distribution"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "service_capacity": service_capacity.model_copy(
                    update={"arrival_distribution_sha256": _digest(98)}
                ),
            }
        )


def test_representative_report_rejects_quality_gate_with_failing_actual_value() -> None:
    inputs = _report_inputs()
    quality = inputs["quality"]
    assert isinstance(quality, GovernedQualityQualificationEvidence)
    rejected_result = quality.decision.approved_gates[0].model_copy(update={"actual_value": 0.0})
    decision = quality.decision.model_copy(
        update={
            "approved_gates": (
                rejected_result,
                *quality.decision.approved_gates[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="frozen threshold"):
        GovernedQualityQualificationEvidence(
            context=quality.context,
            qualification_scope_sha256=quality.qualification_scope_sha256,
            code_revision_sha256=quality.code_revision_sha256,
            schema_catalog_sha256=quality.schema_catalog_sha256,
            sampler_policy_sha256=quality.sampler_policy_sha256,
            media_manifest_sha256=quality.media_manifest_sha256,
            calibration_profile_sha256=quality.calibration_profile_sha256,
            preprocess_policy_sha256=quality.preprocess_policy_sha256,
            registry=quality.registry,
            decision=decision,
            qa_gate_ids=quality.qa_gate_ids,
            event_gate_ids=quality.event_gate_ids,
            calibration_gate_ids=quality.calibration_gate_ids,
        )


def test_representative_report_rejects_quality_labels_outside_frozen_scope() -> None:
    inputs = _report_inputs()
    scope = inputs["scope"]
    assert isinstance(scope, ProductionQualificationScope)

    changed_scope = ProductionQualificationScope.create(
        **{
            **scope.model_dump(mode="python", exclude={"scope_sha256"}),
            "ground_truth_manifest_digest": _digest(99),
        }
    )
    with pytest.raises(ValueError, match="scope does not match"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "scope": changed_scope,
                "service_capacity": _service_capacity(changed_scope),
                "recovery_evidence": _recovery_evidence(changed_scope),
            }
        )


def test_service_capacity_requires_observed_peak_and_backlog_drain() -> None:
    scope = _scope()
    with pytest.raises(ValueError, match="nonzero backlog"):
        _service_capacity(scope, backlog_start_count=0)

    service_capacity = _service_capacity(scope)
    zero_peaks = tuple(
        queue.model_copy(update={"high_watermark": 0}) for queue in service_capacity.queues
    )
    with pytest.raises(ValueError, match="queue peak"):
        service_capacity.model_copy(update={"queues": zero_peaks}).validate_service_capacity()

    with pytest.raises(ValueError, match="positive sample counts"):
        service_capacity.model_copy(
            update={"outbox_delivery_observation_count": 0}
        ).validate_service_capacity()


def test_representative_report_rejects_reused_recovery_evidence() -> None:
    inputs = _report_inputs()
    recoveries = list(inputs["recovery_evidence"])
    recoveries[1] = recoveries[1].model_copy(
        update={"scenario_evidence_sha256": recoveries[0].scenario_evidence_sha256}
    )

    with pytest.raises(ValueError, match="distinct evidence"):
        RepresentativeProductionQualificationReport.create(
            **{**inputs, "recovery_evidence": tuple(recoveries)}
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "code_revision_sha256",
        "schema_catalog_sha256",
        "sampler_policy_sha256",
        "media_manifest_sha256",
        "calibration_profile_sha256",
        "preprocess_policy_sha256",
    ),
)
def test_representative_report_rejects_quality_frozen_configuration_drift(
    field_name: str,
) -> None:
    inputs = _report_inputs()
    quality = inputs["quality"]
    assert isinstance(quality, GovernedQualityQualificationEvidence)

    with pytest.raises(ValueError, match=field_name):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "quality": quality.model_copy(update={field_name: _digest(99)}),
            }
        )


def test_representative_report_rejects_service_capacity_frozen_configuration_drift() -> None:
    inputs = _report_inputs()
    service_capacity = inputs["service_capacity"]
    assert isinstance(service_capacity, RepresentativeServiceCapacityEvidence)

    with pytest.raises(ValueError, match="service capacity media_manifest_sha256"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "service_capacity": service_capacity.model_copy(
                    update={"media_manifest_sha256": _digest(99)}
                ),
            }
        )


def test_representative_report_renders_explicit_p10_operating_evidence() -> None:
    report = RepresentativeProductionQualificationReport.create(**_report_inputs())

    payload = report.as_dict()
    reloaded = RepresentativeProductionQualificationReport.model_validate_json(json.dumps(payload))
    assert reloaded == report
    assert payload["production_eligible"] is False
    assert payload["quality_coverage"]["qa_class_ids"]
    markdown = report.render_markdown()
    assert "QA classes: 21" in markdown
    assert "Production eligible: NO" in markdown
    assert "E6: PENDING_INDEPENDENT_REVIEW" in markdown


def test_representative_report_rejects_deadline_cost_and_external_gate_drift() -> None:
    inputs = _report_inputs()
    deadlines = inputs["deadlines"]
    cost_resources = inputs["cost_resources"]
    assert isinstance(deadlines, RepresentativeDeadlineEvidence)
    assert isinstance(cost_resources, RepresentativeCostResourceEvidence)

    with pytest.raises(ValueError, match="QA deadline evidence"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "deadlines": deadlines.model_copy(update={"qa_deadline_miss_count": 1}),
            }
        )
    with pytest.raises(ValueError, match="cost evidence total"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "cost_resources": cost_resources.model_copy(update={"total_cost_usd": 99.0}),
            }
        )
    with pytest.raises(ValueError, match="external gates"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "external_gates": tuple(reversed(inputs["external_gates"])),
            }
        )


def test_representative_report_binds_cost_and_deadline_populations_to_arrivals() -> None:
    inputs = _report_inputs()
    deadlines = inputs["deadlines"]
    cost_resources = inputs["cost_resources"]
    assert isinstance(deadlines, RepresentativeDeadlineEvidence)
    assert isinstance(cost_resources, RepresentativeCostResourceEvidence)

    with pytest.raises(ValueError, match="cost/resource recording-hours"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "cost_resources": cost_resources.model_copy(
                    update={"recording_hours": 1.0, "camera_hours": 6.0}
                ),
            }
        )
    with pytest.raises(ValueError, match="deadline QA observations"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "deadlines": deadlines.model_copy(
                    update={"qa_observation_count": 1, "annotation_observation_count": 1}
                ),
            }
        )


def test_external_gate_measurement_state_requires_matching_artifacts() -> None:
    with pytest.raises(ValueError, match="supporting artifact"):
        ExternalQualificationGateEvidence(
            gate_id="E0",
            status="MEASURED_PENDING_INDEPENDENT_REVIEW",
            unresolved_reason="independent review pending",
        )
    with pytest.raises(ValueError, match="cannot claim"):
        ExternalQualificationGateEvidence(
            gate_id="E1",
            status="NOT_MEASURED",
            unresolved_reason="representative labels unavailable",
            supporting_artifact_uri="object://qualification/unmeasured.json",
            supporting_artifact_sha256=_digest(90),
        )


def test_quality_coverage_requires_the_canonical_product_vocabulary() -> None:
    coverage = _quality_coverage(_scope())

    with pytest.raises(ValueError, match="all 21 product QA classes"):
        coverage.model_copy(
            update={"qa_class_ids": tuple(reversed(ProductQAIssue))}
        ).validate_coverage()


def test_representative_report_requires_duplicate_injection_proof() -> None:
    inputs = _report_inputs()
    recoveries = list(inputs["recovery_evidence"])
    duplicate_index = next(
        index
        for index, evidence in enumerate(recoveries)
        if evidence.scenario is CanonicalRecoveryScenario.DUPLICATE_INJECTION
    )
    recoveries[duplicate_index] = recoveries[duplicate_index].model_copy(
        update={"duplicate_injection_count": 0}
    )

    with pytest.raises(ValueError, match="duplicate-injection"):
        RepresentativeProductionQualificationReport.create(
            **{
                **inputs,
                "recovery_evidence": tuple(recoveries),
            }
        )
