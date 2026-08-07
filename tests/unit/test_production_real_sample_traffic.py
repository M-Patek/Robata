"""Production worker traffic-admission and component reconciliation tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

import robata.application.canonical.production_real_sample_worker as worker_module
from robata.application.canonical.production_real_sample_worker import (
    ParticipationStatus,
    ProductionRealSampleWorker,
    ProductionRealSampleWorkerConfig,
    ProductionRealSampleWorkerResult,
    ProductionSampleContext,
    ProductionStage,
    WorkerExecutionObservation,
    WorkerStageParticipation,
)
from robata.application.canonical.production_routing import (
    ModelDeployment as CanonicalModelDeployment,
)
from robata.application.canonical.production_routing import (
    ProductionRoute as CanonicalProductionRoute,
)
from robata.application.canonical.production_routing import (
    ProductionRouteAuthorization as CanonicalProductionRouteAuthorization,
)
from robata.application.canonical.production_runtime import ProductionCanonicalRuntime
from robata.application.canonical.production_traffic import ProductionTrafficBridge
from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.routing import (
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ExperimentRoute,
    ModelDeployment,
    RouteMode,
)
from robata.runtime.observability import runtime_span


def _digest(number: int) -> str:
    return f"{number:064x}"


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _canonical_primary() -> CanonicalProductionRoute:
    deployment = CanonicalModelDeployment(
        deployment_id="qwen-control",
        provider="runpod",
        model_name="Qwen3-VL-4B-Instruct",
        model_version="2026.08.05",
        adapter_version="runpod-adapter-v1",
        capability_snapshot_id=_uuid(1),
        capability_snapshot_digest=_digest(1),
        endpoint_config_digest=_digest(2),
        max_concurrent_requests=1,
    )
    return CanonicalProductionRoute(
        route_id="production-qwen-control",
        policy_version="1.0",
        deployment=deployment,
        authorization=CanonicalProductionRouteAuthorization(
            qualification_report_ref="r2://reports/qwen.json",
            qualification_report_sha256=_digest(3),
            release_decision_ref="r2://releases/qwen.json",
            release_decision_sha256=_digest(4),
        ),
    )


class _PrimaryVerifier:
    def __init__(self, result: bool = True) -> None:
        self.result = result

    def verify_primary(self, *, authorization: object, deployment: object) -> bool:
        del authorization, deployment
        return self.result


class _Driver:
    def __init__(self, *, omitted: set[ProductionStage] | None = None) -> None:
        self.omitted = omitted or set()
        self.context: ProductionSampleContext | None = None

    def preflight(self, *, runtime: object) -> None:
        del runtime

    def execute(self, *, context: ProductionSampleContext) -> WorkerExecutionObservation:
        self.context = context
        spans = {
            ProductionStage.SCHEDULING: "scheduler.production.execute",
            ProductionStage.INFERENCE: "inference.production.execute",
            ProductionStage.EVIDENCE: "evidence.production.persist",
            ProductionStage.REDUCTION: "reduction.production.execute",
            ProductionStage.COMPLETION: "completion.production.commit",
            ProductionStage.OUTBOX: "outbox.production.deliver",
            ProductionStage.PUBLICATION: "publication.production.publish",
        }
        participation: list[WorkerStageParticipation] = []
        for stage, span_name in spans.items():
            if stage in self.omitted:
                status = ParticipationStatus.BYPASSED
                reason = "test intentionally omitted component"
            else:
                with runtime_span(context.observer, span_name):
                    pass
                status = ParticipationStatus.PARTICIPATING
                reason = "test component executed"
            participation.append(
                WorkerStageParticipation(stage=stage, status=status, reason=reason)
            )
        return WorkerExecutionObservation(
            execution_driver="traffic-test-driver",
            canonical_run_id="canonical-test-run",
            output_refs=("r2://bucket/result",),
            participation=tuple(participation),
        )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        work_scheduler=object(),
        capture_authority=object(),
        primary_adapter=object(),
        inference_evidence=object(),
        barrier_storage=object(),
        primary_completion=object(),
        outbox_delivery=object(),
        read_model=object(),
        r2_object_store=object(),
    )


def _config(tmp_path: Path) -> tuple[ProductionRealSampleWorkerConfig, bytes]:
    source = tmp_path / "sample.mcap"
    source_bytes = b"traffic-sidecar-test"
    source.write_bytes(source_bytes)
    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}", encoding="utf-8")
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text("{}", encoding="utf-8")
    return (
        ProductionRealSampleWorkerConfig(
            bootstrap_config_path=str(bootstrap),
            mapping_config_path=str(mapping),
            output_directory=str(tmp_path / "out"),
            state_directory=str(tmp_path / "state"),
            source_path=str(source),
            source_sha256=exact_bytes_sha256(source_bytes),
            source_byte_count=len(source_bytes),
            run_id="traffic-run-1",
        ),
        source_bytes,
    )


@pytest.fixture
def harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[ProductionRealSampleWorkerConfig, bytes]:
    profile = SimpleNamespace(profile_id="profile-v1", approval_status="APPROVED")
    monkeypatch.setattr(
        worker_module,
        "load_production_runtime_bootstrap_configuration",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        worker_module,
        "authorize_mcap_mapping",
        lambda path, allow_unapproved_profile: SimpleNamespace(profile=profile),
    )
    monkeypatch.setattr(
        worker_module,
        "load_canonical_mcap_source",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    return _config(tmp_path)


def _worker(
    config: ProductionRealSampleWorkerConfig,
    driver: _Driver,
    *,
    traffic_bridge: ProductionTrafficBridge | None = None,
) -> ProductionRealSampleWorkerResult:
    return ProductionRealSampleWorker(
        config=config,
        environment={},
        runtime_builder=lambda bootstrap, environment, observer: cast(
            ProductionCanonicalRuntime, _runtime()
        ),
        execution_driver=driver,
        traffic_bridge=traffic_bridge,
    ).run()


def test_ready_primary_writes_route_sidecars_and_binds_context(
    harness: tuple[ProductionRealSampleWorkerConfig, bytes],
) -> None:
    config, source_bytes = harness
    bridge = ProductionTrafficBridge(
        canonical_primary_route=_canonical_primary(),
        primary_release_verifier=_PrimaryVerifier(),
    )
    driver = _Driver()
    result = _worker(config, driver, traffic_bridge=bridge)

    assert result.report.status == "SUCCEEDED"
    readiness_ref = result.report.traffic_readiness
    route_plan_ref = result.report.traffic_route_plan
    readiness_path = result.traffic_readiness_path
    route_plan_path = result.traffic_route_plan_path
    result_readiness_ref = result.traffic_readiness_ref
    result_route_plan_ref = result.traffic_route_plan_ref
    assert readiness_ref is not None
    assert route_plan_ref is not None
    assert readiness_path is not None and readiness_path.is_file()
    assert route_plan_path is not None and route_plan_path.is_file()
    assert result_readiness_ref is not None
    assert result_route_plan_ref is not None
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    route_plan = json.loads(route_plan_path.read_text(encoding="utf-8"))
    assert readiness["state"] == "READY"
    assert route_plan["input_identity_sha256"] == exact_bytes_sha256(source_bytes)
    assert route_plan["shadow_observation_only"] is True
    assert readiness_ref.sha256 == result_readiness_ref.sha256
    assert route_plan_ref.sha256 == result_route_plan_ref.sha256
    context = driver.context
    assert context is not None
    assert context.traffic_readiness_ref is not None
    assert context.traffic_route_plan_ref is not None
    assert context.traffic_route_plan is not None
    assert context.traffic_readiness_ref.sha256 == readiness_ref.sha256
    assert context.traffic_route_plan_ref.sha256 == route_plan_ref.sha256
    assert context.traffic_route_plan.canonical_authoritative_deployment_id == "qwen-control"


def test_unverified_primary_fails_closed_and_keeps_readiness_evidence(
    harness: tuple[ProductionRealSampleWorkerConfig, bytes],
) -> None:
    config, _ = harness
    bridge = ProductionTrafficBridge(canonical_primary_route=_canonical_primary())
    result = _worker(config, _Driver(), traffic_bridge=bridge)

    assert result.report.status == "FAILED"
    assert result.report.failure_code == "TRAFFIC_NOT_READY"
    assert result.report.traffic_readiness is not None
    assert result.report.traffic_route_plan is None
    assert result.traffic_readiness_path is not None and result.traffic_readiness_path.is_file()
    readiness = json.loads(result.traffic_readiness_path.read_text(encoding="utf-8"))
    assert readiness["state"] == "NOT_READY"
    assert "PRIMARY_ROUTE_RELEASE_UNVERIFIED" in readiness["blockers"]
    assert result.traffic_route_plan_path is None


def test_shadow_without_submit_hook_fails_closed(
    harness: tuple[ProductionRealSampleWorkerConfig, bytes],
) -> None:
    config, _ = harness
    primary = _canonical_primary()
    inference_primary = ProductionTrafficBridge(
        canonical_primary_route=primary,
        primary_release_verifier=_PrimaryVerifier(),
    ).primary_route.deployment
    candidate = ModelDeployment(
        deployment_id="mage-shadow",
        provider="runpod",
        model_name="Mage-VL-4B",
        model_version="2026.08.05",
        adapter_version="runpod-adapter-v1",
        capability_snapshot_id=_uuid(30),
        capability_snapshot_digest=_digest(31),
        endpoint_config_digest=_digest(32),
        max_concurrent_requests=1,
    )
    shadow = ExperimentRoute(
        route_id="mage-shadow-route",
        policy_version="1.0",
        mode=RouteMode.SHADOW,
        sample_ratio=1.0,
        contract=ExperimentContract(
            experiment_id="mage-shadow-01",
            contract_version="1.0",
            workload_manifest_sha256=_digest(40),
            arrival_schedule_sha256=_digest(41),
            comparison_config_sha256=_digest(42),
            input_representation=ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING,
            isolation_profile=ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
            control=inference_primary,
            candidate=candidate,
        ),
    )
    bridge = ProductionTrafficBridge(
        canonical_primary_route=primary,
        primary_release_verifier=_PrimaryVerifier(),
        shadow_route=shadow,
    )
    result = _worker(config, _Driver(), traffic_bridge=bridge)

    assert result.report.status == "FAILED"
    assert result.report.failure_code == "TRAFFIC_NOT_READY"
    readiness_path = result.traffic_readiness_path
    assert readiness_path is not None
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert "SHADOW_SUBMIT_HOOK_UNBOUND" in readiness["blockers"]


def test_traffic_sidecar_references_are_content_bound(
    harness: tuple[ProductionRealSampleWorkerConfig, bytes],
) -> None:
    config, _ = harness
    bridge = ProductionTrafficBridge(
        canonical_primary_route=_canonical_primary(),
        primary_release_verifier=_PrimaryVerifier(),
    )
    result = _worker(config, _Driver(), traffic_bridge=bridge)
    readiness_path = result.traffic_readiness_path
    route_plan_path = result.traffic_route_plan_path
    readiness_ref = result.report.traffic_readiness
    route_plan_ref = result.report.traffic_route_plan
    assert readiness_path is not None
    assert route_plan_path is not None
    assert readiness_ref is not None
    assert route_plan_ref is not None
    readiness_payload = readiness_path.read_bytes()
    route_payload = route_plan_path.read_bytes()
    assert exact_bytes_sha256(readiness_payload) == readiness_ref.sha256
    assert exact_bytes_sha256(route_payload) == route_plan_ref.sha256
    component = json.loads(result.component_participation_path.read_text(encoding="utf-8"))
    assert component["traffic_readiness_sha256"] == readiness_ref.sha256
    assert component["traffic_route_plan_sha256"] == route_plan_ref.sha256


def test_success_requires_canonical_run_and_output_receipts(
    harness: tuple[ProductionRealSampleWorkerConfig, bytes],
) -> None:
    config, _ = harness

    class _NoReceiptDriver(_Driver):
        def execute(self, *, context: ProductionSampleContext) -> WorkerExecutionObservation:
            observed = super().execute(context=context)
            return observed.model_copy(update={"canonical_run_id": None, "output_refs": ()})

    result = _worker(config, _NoReceiptDriver())
    assert result.report.status == "FAILED"
    assert result.report.failure_code == "CANONICAL_EXECUTION_FAILED"
    assert "canonical_run_id and output_refs" in (result.report.failure_detail or "")


def test_completion_and_outbox_omission_cannot_hide_behind_publication_span(
    harness: tuple[ProductionRealSampleWorkerConfig, bytes],
) -> None:
    config, _ = harness
    result = _worker(
        config,
        _Driver(omitted={ProductionStage.COMPLETION, ProductionStage.OUTBOX}),
    )

    assert result.report.status == "FAILED"
    assert result.report.failure_code == "E2E_COVERAGE_INCOMPLETE"
    component = json.loads(result.component_participation_path.read_text(encoding="utf-8"))
    assert any("PUBLICATION" in issue for issue in component["component_reconciliation_issues"])
