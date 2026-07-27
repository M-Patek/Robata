from __future__ import annotations

import pytest

from robata.benchmark import (
    LocalQualificationContext,
    LocalRecoveryScenario,
    build_local_quality_capacity_qualification_package,
    build_local_sampling_dense_pareto_report,
)
from robata.contracts.hashing import semantic_sha256
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityInput,
    ProviderMode,
    build_measured_capacity_report,
)
from tests.unit.test_benchmark_pareto import _policy_matrix

_HOUR_NS = 3_600_000_000_000


def _digest(label: str) -> str:
    return semantic_sha256({"local-qualification-test": label})


def _fixtures() -> tuple[
    LocalQualificationContext,
    object,
    object,
    tuple[LocalRecoveryScenario, ...],
]:
    workload_digest = _digest("workload")
    context = LocalQualificationContext(
        workload_manifest_digest=workload_digest,
        recording_count=1,
        recording_duration_ns=_HOUR_NS,
        camera_count=6,
        model_identifier="local-fixture-provider",
        provider_mode=ProviderMode.LOCAL_OFFLINE_FIXTURE,
        provider_concurrency=2,
        hardware_identifier="local-test-host",
        run_duration_ns=_HOUR_NS // 2,
    )
    pareto = build_local_sampling_dense_pareto_report(
        fixture_manifest_digest=workload_digest,
        pipeline_version="canonical-fixture-v1",
        model_identifier=context.model_identifier,
        prompt_version="qa-event-prompt-v1",
        policies=_policy_matrix(),
    )
    capacity = build_measured_capacity_report(
        MeasuredCapacityInput(
            workload_fingerprint=workload_digest,
            evidence_class=CapacityEvidenceClass.LOCAL_CONFORMANCE,
            provider_mode=ProviderMode.LOCAL_OFFLINE_FIXTURE,
            execution_mode="FRESH",
            recording_count=1,
            recording_worker_count=1,
            camera_count=6,
            recording_duration_ns=_HOUR_NS,
            wall_time_ns=_HOUR_NS // 2,
            windows=4,
            unique_images=10,
            coarse_unique_images=8,
            dense_unique_images=4,
            provider_images=25,
            logical_calls=5,
            call_parts=8,
            call_splits=3,
            http_requests=5,
            retries=2,
            batches=2,
            batch_requests=5,
            input_tokens=1_000,
            output_tokens=250,
            output_token_responses=5,
            dense_logical_calls=2,
            dense_provider_images=5,
        )
    )
    scenarios = tuple(
        LocalRecoveryScenario(
            scenario_id=scenario_id,
            terminal_reconciled=True,
            outbox_reconciled=True,
        )
        for scenario_id in (
            "RESTART_REPLAY",
            "PROVIDER_RETRY",
            "PROVIDER_TIMEOUT",
            "OUTBOX_RECONCILIATION",
        )
    )
    return context, pareto, capacity, scenarios


def test_local_qualification_package_is_deterministic_and_explicitly_local() -> None:
    context, pareto, capacity, scenarios = _fixtures()

    first = build_local_quality_capacity_qualification_package(
        context=context,
        pareto=pareto,
        capacity=capacity,
        recovery_scenarios=scenarios,
    )
    second = build_local_quality_capacity_qualification_package(
        context=context,
        pareto=pareto,
        capacity=capacity,
        recovery_scenarios=reversed(scenarios),
    )

    assert first.as_dict() == second.as_dict()
    assert first.render_markdown() == second.render_markdown()
    assert tuple(item.scenario_id for item in first.recovery_scenarios) == (
        "OUTBOX_RECONCILIATION",
        "PROVIDER_RETRY",
        "PROVIDER_TIMEOUT",
        "RESTART_REPLAY",
    )
    payload = first.as_dict()
    assert payload["evidence_class"] == "LOCAL_CONFORMANCE"
    assert payload["measurement_status"] == "NOT_MEASURED"
    assert payload["capacity"]["provider_mode"] == "LOCAL_OFFLINE_FIXTURE"
    assert payload["production_eligible"] is False

    markdown = first.render_markdown()
    assert "## Pareto frontier" in markdown
    assert "Provider images" in markdown
    assert "OUTBOX_RECONCILIATION" in markdown
    assert "Production eligible: NO" in markdown


def test_local_qualification_package_rejects_mismatched_workload_and_incomplete_recovery() -> None:
    context, pareto, capacity, scenarios = _fixtures()

    with pytest.raises(ValueError, match="Pareto fixture manifest"):
        build_local_quality_capacity_qualification_package(
            context=context.model_copy(update={"workload_manifest_digest": _digest("other")}),
            pareto=pareto,
            capacity=capacity,
            recovery_scenarios=scenarios,
        )

    incomplete = (*scenarios[:-1], scenarios[-2])
    with pytest.raises(ValueError, match="each recovery scenario exactly once"):
        build_local_quality_capacity_qualification_package(
            context=context,
            pareto=pareto,
            capacity=capacity,
            recovery_scenarios=incomplete,
        )
