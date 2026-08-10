from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robata.benchmark.mage_25x import (
    MageCapacityEvidenceError,
    MageProviderV2LocalBaseline,
    build_mage_25x_capacity_report,
    load_provider_v2_local_baseline,
    required_aggregate_realtime_factor,
)
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = (
    REPOSITORY_ROOT / "docs" / "mage-dcvc-provider-v2-local-qualification-2026-08-09.json"
)
SOURCE_EXACT_SHA256 = "7298d21fb05f0ecbc4bc1e11481f67abf2c82b4b13380227177edfbbbaa24287"
SOURCE_SEMANTIC_SHA256 = "ea659e3e78243e43e4c1f921ff0898c64f18c4e68993c9c219d2425c8a25b0d8"
TRACKED_CAPACITY_REPORT = (
    REPOSITORY_ROOT / "docs" / "mage-25x-local-capacity-baseline-2026-08-09.json"
)
TRACKED_CAPACITY_REPORT_EXACT_SHA256 = (
    "1a6ff7cd3fc56f582027ac950e2dc0419eb5963a2d4836bbe0cfa8ae35e6d7b7"
)


def _baseline() -> MageProviderV2LocalBaseline:
    return load_provider_v2_local_baseline(
        path=SOURCE_REPORT,
        expected_exact_sha256=SOURCE_EXACT_SHA256,
        expected_semantic_sha256=SOURCE_SEMANTIC_SHA256,
    )


def test_retained_baseline_reproduces_exact_stage_and_capacity_values() -> None:
    baseline = _baseline()

    assert baseline.media_seconds == 40.0
    assert baseline.camera_count == 1
    assert baseline.segment_count == 5
    assert baseline.worker_count == 1
    assert baseline.preparation_wall_seconds == pytest.approx(50.31025060000138)
    assert baseline.preparation_worker_sum_seconds == pytest.approx(37.407738199999586)
    assert baseline.preparation_orchestration_delta_seconds == pytest.approx(12.902512400001797)
    assert baseline.generation_stream_wall_seconds == pytest.approx(21.96186850000049)
    assert baseline.observation_sum_seconds == pytest.approx(20.68564569999762)
    assert baseline.warm_observation_mean_seconds == pytest.approx(3.150942349999241)
    assert baseline.generation_sum_seconds == pytest.approx(19.957743899998604)
    assert baseline.first_generation_seconds == pytest.approx(7.671305199999453)
    assert baseline.warm_generation_mean_seconds == pytest.approx(3.0716096749993085)
    assert baseline.warm_time_to_first_token_mean_seconds == pytest.approx(0.39576164999971297)
    assert baseline.processor_sum_seconds == pytest.approx(0.3119143999974767)
    assert baseline.input_materialization_sum_seconds == pytest.approx(0.03534639999998035)
    assert baseline.decode_sum_seconds == pytest.approx(0.005493699998623924)

    scenarios = {item.scenario_id: item for item in baseline.scenarios()}
    serial = scenarios["local_serial_retained_whole_stage_walls"]
    overlap = scenarios["local_two_stage_overlap_retained_whole_stage_walls"]
    lower_bound = scenarios["local_two_stage_overlap_measured_job_sums_lower_bound"]
    assert serial.wall_seconds == pytest.approx(72.27211910000186)
    assert serial.realtime_factor == pytest.approx(0.5534637768771189)
    assert serial.required_logical_lanes == 46
    assert overlap.realtime_factor == pytest.approx(0.7950666021925739)
    assert overlap.required_logical_lanes == 32
    assert lower_bound.realtime_factor == pytest.approx(1.0692974749272717)
    assert lower_bound.required_logical_lanes == 24
    assert lower_bound.includes_full_orchestration_wall is False


def test_tracked_capacity_report_is_exactly_reproducible() -> None:
    baseline = load_provider_v2_local_baseline(
        path=SOURCE_REPORT,
        expected_exact_sha256=SOURCE_EXACT_SHA256,
        expected_semantic_sha256=SOURCE_SEMANTIC_SHA256,
        source_reference="docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json",
    )
    expected = build_mage_25x_capacity_report(baseline=baseline)
    tracked_bytes = TRACKED_CAPACITY_REPORT.read_bytes()

    assert hashlib.sha256(tracked_bytes).hexdigest() == TRACKED_CAPACITY_REPORT_EXACT_SHA256
    assert json.loads(tracked_bytes) == expected


def test_capacity_formula_is_camera_hours_with_headroom_over_24_hours() -> None:
    assert required_aggregate_realtime_factor(
        daily_camera_hours=500.0,
        headroom=1.20,
    ) == pytest.approx(25.0)
    with pytest.raises(MageCapacityEvidenceError, match="headroom must be at least"):
        required_aggregate_realtime_factor(daily_camera_hours=500.0, headroom=0.99)


def test_target_multiplier_scenario_is_explicitly_unmeasured() -> None:
    report = build_mage_25x_capacity_report(
        baseline=_baseline(),
        codec_multiplier=8.0,
        decoder_multiplier=6.0,
    )

    scenario = report["scenarios"][-1]  # type: ignore[index]
    assert scenario["scenario_id"] == "target_separate_device_multiplier_scenario"
    assert scenario["evidence_class"] == "UNMEASURED_SCENARIO"
    assert scenario["measured"] is False
    assert report["decision"]["state"] == "HOLD"  # type: ignore[index]
    target = report["target"]  # type: ignore[assignment]
    assert target["cycle_assumption"]["required_aggregate_realtime_factor"] == 25.0
    assert (
        target["repository_requirement_conflict"]["independent_camera_stream_equivalent_rtf"]
        == 150.0
    )
    assert target["repository_requirement_conflict"]["resolved"] is False
    assert report["production_eligible"] is False
    semantic = str(report.pop("semantic_sha256"))
    assert semantic == semantic_sha256(report)


def test_target_multiplier_requires_a_pair() -> None:
    with pytest.raises(MageCapacityEvidenceError, match="must be supplied together"):
        _baseline().scenarios(codec_multiplier=3.0)


def test_exact_hash_mismatch_fails_before_capacity_calculation() -> None:
    with pytest.raises(MageCapacityEvidenceError, match="exact SHA-256 differs"):
        load_provider_v2_local_baseline(
            path=SOURCE_REPORT,
            expected_exact_sha256="0" * 64,
        )


def test_missing_generation_measurement_is_not_coerced_to_zero(tmp_path: Path) -> None:
    document = json.loads(SOURCE_REPORT.read_bytes())
    bounded = document["variants"]["provider_v2_bounded"][0]
    del bounded["generation"]["measurement"]["stream_run_wall_seconds"]
    document.pop("semantic_sha256")
    document["semantic_sha256"] = semantic_sha256(document)
    changed = tmp_path / "missing.json"
    changed.write_bytes(canonical_json_bytes(document))

    with pytest.raises(MageCapacityEvidenceError, match="must be numeric"):
        load_provider_v2_local_baseline(path=changed)
