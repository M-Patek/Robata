"""Focused tests for the fail-closed production real-sample launcher."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import robata.application.canonical.production_real_sample_worker as worker_module
from robata.application.canonical.production_real_sample_worker import (
    ParticipationStatus,
    ProductionRealSampleWorker,
    ProductionRealSampleWorkerConfig,
    ProductionStage,
    WorkerExecutionObservation,
    WorkerStageParticipation,
)
from robata.contracts.hashing import exact_bytes_sha256
from robata.runtime.e2e_participation import (
    E2EParticipationBoundary,
    E2EParticipationDeclaration,
    E2EParticipationState,
)
from robata.runtime.observability import runtime_span

_PRODUCTION_STAGES = (
    ProductionStage.SCHEDULING,
    ProductionStage.INFERENCE,
    ProductionStage.EVIDENCE,
    ProductionStage.REDUCTION,
    ProductionStage.COMPLETION,
    ProductionStage.OUTBOX,
    ProductionStage.PUBLICATION,
)
_STAGE_SPAN_NAMES = {
    ProductionStage.SCHEDULING: "scheduler.production.execute",
    ProductionStage.INFERENCE: "inference.production.execute",
    ProductionStage.EVIDENCE: "evidence.production.persist",
    ProductionStage.REDUCTION: "reduction.production.execute",
    ProductionStage.COMPLETION: "completion.production.commit",
    ProductionStage.OUTBOX: "outbox.production.deliver",
    ProductionStage.PUBLICATION: "publication.production.publish",
}


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


def _declarations(*, reduction_bypassed: bool = False) -> tuple[E2EParticipationDeclaration, ...]:
    values: list[E2EParticipationDeclaration] = []
    for boundary in E2EParticipationBoundary:
        if reduction_bypassed and boundary is E2EParticipationBoundary.REDUCTION:
            values.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=E2EParticipationState.BYPASSED,
                    required=False,
                    reason="reduction is intentionally omitted for this bounded sample",
                )
            )
        else:
            values.append(
                E2EParticipationDeclaration(
                    boundary=boundary,
                    state=E2EParticipationState.PARTICIPATING,
                    required=True,
                )
            )
    return tuple(values)


class _FakeDriver:
    def __init__(self, *, omitted: set[ProductionStage] | None = None) -> None:
        self.omitted = set() if omitted is None else omitted
        self.preflight_calls = 0
        self.execute_calls = 0

    def preflight(self, *, runtime: object) -> None:
        del runtime
        self.preflight_calls += 1

    def execute(self, *, context: object) -> WorkerExecutionObservation:
        assert hasattr(context, "observer")
        self.execute_calls += 1
        participation: list[WorkerStageParticipation] = []
        for stage in _PRODUCTION_STAGES:
            if stage not in self.omitted:
                with runtime_span(context.observer, _STAGE_SPAN_NAMES[stage]):
                    pass
                status = ParticipationStatus.PARTICIPATING
                reason = "test driver executed production boundary"
            else:
                status = ParticipationStatus.BYPASSED
                reason = "test driver intentionally omitted optional boundary"
            participation.append(
                WorkerStageParticipation(stage=stage, status=status, reason=reason)
            )
        return WorkerExecutionObservation(
            execution_driver="test-driver",
            canonical_run_id="canonical-test-run",
            output_refs=("r2://bucket/result",),
            participation=tuple(participation),
        )


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "sample.mcap"
    source_bytes = b"pinned-mcap-test"
    source.write_bytes(source_bytes)
    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}", encoding="utf-8")
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text("{}", encoding="utf-8")
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
    config = ProductionRealSampleWorkerConfig(
        bootstrap_config_path=str(bootstrap),
        mapping_config_path=str(mapping),
        output_directory=str(tmp_path / "out"),
        state_directory=str(tmp_path / "state"),
        source_path=str(source),
        source_sha256=exact_bytes_sha256(source_bytes),
        source_byte_count=len(source_bytes),
        run_id="run-test-1",
    )
    return {"config": config, "source": source, "tmp_path": tmp_path}


def test_default_driver_fails_closed_and_writes_audit_bundle(
    harness: dict[str, object],
) -> None:
    config = harness["config"]
    assert isinstance(config, ProductionRealSampleWorkerConfig)
    result = ProductionRealSampleWorker(
        config=config,
        environment={},
        runtime_builder=lambda bootstrap, environment, observer: _runtime(),
    ).run()

    assert result.report.status == "FAILED"
    assert result.report.failure_code == "CANONICAL_EXECUTION_BRIDGE_UNAVAILABLE"
    assert result.report.production_eligible is False
    assert result.report.canonical_authority is False
    assert result.report.source is not None
    assert result.report.source.mapping_profile_id == "profile-v1"
    assert result.report.participation_coverage == "FAILED"
    stages = {item.stage: item for item in result.report.participation}
    assert stages[ProductionStage.SOURCE].status is ParticipationStatus.PARTICIPATING
    assert stages[ProductionStage.SCHEDULING].status is ParticipationStatus.NOT_CONFIGURED
    assert stages[ProductionStage.PUBLICATION].status is ParticipationStatus.NOT_CONFIGURED
    assert result.report_path.is_file()
    assert result.trace_path.is_file()
    assert result.participation_path.is_file()
    assert result.component_participation_path.is_file()
    payload = json.loads(result.participation_path.read_text(encoding="utf-8"))
    assert payload["coverage"] == "FAILED"
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert any(span["name"] == "source.mcap.prepare" for span in trace["runtime_profile"]["spans"])
    component = json.loads(result.component_participation_path.read_text(encoding="utf-8"))
    assert component["status"] == "FAILED"


def test_complete_driver_binds_generic_manifest_and_component_sidecar(
    harness: dict[str, object],
) -> None:
    config = harness["config"]
    assert isinstance(config, ProductionRealSampleWorkerConfig)
    driver = _FakeDriver()
    result = ProductionRealSampleWorker(
        config=config,
        environment={},
        runtime_builder=lambda bootstrap, environment, observer: _runtime(),
        execution_driver=driver,
    ).run()

    assert result.report.status == "SUCCEEDED"
    assert result.report.participation_coverage == "COMPLETE"
    assert result.report.execution is not None
    assert driver.preflight_calls == 1
    assert driver.execute_calls == 1
    stages = {item.stage: item for item in result.report.participation}
    assert all(
        stages[stage].status is ParticipationStatus.PARTICIPATING for stage in _PRODUCTION_STAGES
    )
    manifest = json.loads(result.participation_path.read_text(encoding="utf-8"))
    assert manifest["coverage"] == "COMPLETE"
    assert all(item["observed_measurement_status"] == "MEASURED" for item in manifest["boundaries"])
    component = json.loads(result.component_participation_path.read_text(encoding="utf-8"))
    assert component["participation_sha256"] == result.report.participation_manifest.sha256


def test_missing_required_span_fails_coverage_closed(
    harness: dict[str, object],
) -> None:
    config = harness["config"]
    assert isinstance(config, ProductionRealSampleWorkerConfig)
    result = ProductionRealSampleWorker(
        config=config,
        environment={},
        runtime_builder=lambda bootstrap, environment, observer: _runtime(),
        execution_driver=_FakeDriver(omitted={ProductionStage.INFERENCE}),
    ).run()

    assert result.report.status == "FAILED"
    assert result.report.failure_code == "E2E_COVERAGE_INCOMPLETE"
    assert result.report.participation_coverage == "PARTIAL"
    manifest = json.loads(result.participation_path.read_text(encoding="utf-8"))
    inference = next(item for item in manifest["boundaries"] if item["boundary"] == "INFERENCE")
    assert inference["observed_measurement_status"] == "NOT_MEASURED"
    assert any(issue["code"] == "PARTICIPATING_NOT_MEASURED" for issue in manifest["issues"])


def test_optional_reduction_can_be_bypassed_without_invalidating_manifest(
    harness: dict[str, object],
) -> None:
    config = harness["config"]
    assert isinstance(config, ProductionRealSampleWorkerConfig)
    config = config.model_copy(update={"e2e_participation": _declarations(reduction_bypassed=True)})
    driver = _FakeDriver(omitted={ProductionStage.REDUCTION})
    result = ProductionRealSampleWorker(
        config=config,
        environment={},
        runtime_builder=lambda bootstrap, environment, observer: _runtime(),
        execution_driver=driver,
    ).run()

    assert result.report.status == "SUCCEEDED"
    assert result.report.participation_coverage == "COMPLETE"
    reduction = next(
        item
        for item in json.loads(result.participation_path.read_text(encoding="utf-8"))["boundaries"]
        if item["boundary"] == "REDUCTION"
    )
    assert reduction["state"] == "BYPASSED"
    assert reduction["observed_measurement_status"] == "NOT_MEASURED"
    worker_reduction = next(
        item for item in result.report.participation if item.stage is ProductionStage.REDUCTION
    )
    assert worker_reduction.status is ParticipationStatus.BYPASSED


def test_source_digest_mismatch_fails_before_source_loader(
    harness: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = harness["config"]
    assert isinstance(config, ProductionRealSampleWorkerConfig)
    monkeypatch.setattr(worker_module, "load_canonical_mcap_source", pytest.fail)
    config = config.model_copy(update={"source_sha256": "0" * 64})
    result = ProductionRealSampleWorker(
        config=config,
        environment={},
        runtime_builder=lambda bootstrap, environment, observer: _runtime(),
    ).run()

    assert result.report.status == "FAILED"
    assert result.report.failure_code == "SOURCE_UNPINNED"
    assert result.report.source is None


def test_config_requires_explicit_source_pin(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProductionRealSampleWorkerConfig(
            bootstrap_config_path=str(tmp_path / "bootstrap.json"),
            mapping_config_path=str(tmp_path / "mapping.json"),
            output_directory=str(tmp_path / "out"),
            state_directory=str(tmp_path / "state"),
            source_path=str(tmp_path / "sample.mcap"),
        )
