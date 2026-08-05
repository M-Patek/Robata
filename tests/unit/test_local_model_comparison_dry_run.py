from __future__ import annotations

import asyncio
import json

from robata.benchmark.local_model_comparison_dry_run import (
    LOCAL_MODEL_COMPARISON_EXTERNAL_GATES,
    run_local_model_comparison_dry_run,
)
from robata.inference.experiment_execution import ExperimentComparisonStatus
from robata.inference.routing import DispatchDisposition, RouteMode, RoutePlane
from scripts.run_local_model_comparison_dry_run import main


def test_local_dry_run_exercises_authoritative_and_observational_planes() -> None:
    report = asyncio.run(run_local_model_comparison_dry_run())

    assert report.execution_class == "LOCAL_CONFORMANCE"
    assert report.production_eligible is False
    assert report.network_call_count == 0
    assert report.deterministic_adapter_call_count == 3
    assert report.production_decision.plane is RoutePlane.PRODUCTION
    assert report.production_decision.mode is RouteMode.PRIMARY
    assert report.production_decision.dispatches[0].disposition is DispatchDisposition.AUTHORITATIVE
    assert report.experiment_decision.plane is RoutePlane.EXPERIMENT
    assert report.experiment_decision.mode is RouteMode.PAIRED
    assert all(
        dispatch.disposition is DispatchDisposition.OBSERVATION
        for dispatch in report.experiment_decision.dispatches
    )
    assert report.production_terminal.shadow is False
    assert report.control_terminal.shadow is True
    assert report.candidate_terminal.shadow is True


def test_local_dry_run_retains_the_p17_sidecar_without_claiming_p18_evidence() -> None:
    report = asyncio.run(run_local_model_comparison_dry_run())

    assert report.comparison_status is ExperimentComparisonStatus.DIFFERENCE
    assert report.comparison.comparison_id == report.comparison_id
    assert report.comparison.status is report.comparison_status
    assert report.comparison.comparable is True
    assert [delta.path for delta in report.comparison.field_deltas] == ["label"]
    assert report.comparison_field_delta_count == 1
    assert report.p18_readiness.comparison_id == report.comparison_id
    assert report.p18_readiness.comparison_status is report.comparison_status
    assert report.p18_readiness.fair_load_evidence_status == "NOT_MEASURED"
    assert report.p18_readiness.quality_evidence_status == "NOT_MEASURED"
    assert report.p18_readiness.cost_evidence_status == "NOT_MEASURED"
    assert report.p18_readiness.fair_load_report_emitted is False
    assert report.p18_readiness.candidate_authority is False
    assert report.p18_readiness.unresolved_external_gates == LOCAL_MODEL_COMPARISON_EXTERNAL_GATES


def test_local_dry_run_command_materializes_the_same_non_promotional_report(
    tmp_path, capsys
) -> None:
    output = tmp_path / "local-model-comparison.json"

    assert main(["--output", str(output)]) == 0

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert printed == persisted
    assert persisted["network_call_count"] == 0
    assert persisted["production_eligible"] is False
    assert persisted["p18_readiness"]["unresolved_external_gates"] == list(
        LOCAL_MODEL_COMPARISON_EXTERNAL_GATES
    )
