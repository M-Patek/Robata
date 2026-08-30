from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.atomic_experiment_reports import (
    HistoricalAtomicExperimentArm,
    LegacyEvidenceAvailability,
    LegacyExperimentEvidenceError,
    build_historical_a0_a3_ledger,
    infer_historical_arm,
    project_legacy_case_record,
    project_legacy_experiment_report,
)


def _legacy_quality() -> dict[str, object]:
    # Deliberately not a valid current score shape: P34 must retain it rather
    # than run any newer scoring logic over historic model text.
    return {"legacy_metric": "stored-verbatim", "direction_match": True}


def _base_case(**changes: object) -> dict[str, object]:
    case: dict[str, object] = {
        "uid": "P37_102_33",
        "context_frame_indices": [10, 12, 14],
        "focus_frame_indices": [16, 18, 20],
        "raw_output_text": "The hand turns the faucet on.",
        "quality": _legacy_quality(),
        "generation_seconds": 1.25,
    }
    case.update(changes)
    return case


def test_a0_projection_preserves_legacy_facts_and_marks_missing_observation_fields() -> None:
    report = {
        "input_profile": "context-focus",
        "cases": [_base_case()],
    }
    original = deepcopy(report)

    ledger = project_legacy_experiment_report(report)
    case = ledger.cases[0]

    assert ledger.arm is HistoricalAtomicExperimentArm.A0
    assert case.source_frames.availability is LegacyEvidenceAvailability.KNOWN
    assert case.source_frames.value == {
        "context_frame_indices": [10, 12, 14],
        "focus_frame_indices": [16, 18, 20],
        "source_frame_timestamps": None,
    }
    assert case.raw_output.availability is LegacyEvidenceAvailability.KNOWN
    assert case.lexical_outcome.value is report["cases"][0]["quality"]
    assert case.transform_trace.availability is LegacyEvidenceAvailability.UNKNOWN
    assert case.roi.availability is LegacyEvidenceAvailability.UNKNOWN
    assert case.final_thumbnail_geometry.availability is LegacyEvidenceAvailability.UNKNOWN
    assert case.processor_grid.availability is LegacyEvidenceAvailability.UNKNOWN
    assert case.processor_tensor_shape.availability is LegacyEvidenceAvailability.UNKNOWN
    assert ledger.stop_outcome.availability is LegacyEvidenceAvailability.UNKNOWN
    assert report == original


def test_a3_input_size_is_retained_but_not_mistaken_for_final_thumbnail_geometry() -> None:
    report = {
        "input_profile": "context-focus-microburst-hybrid-roi",
        "cases": [
            _base_case(
                hybrid_focus={
                    "roi_xyxy": [12, 18, 240, 180],
                    "input_size": [448, 448],
                }
            )
        ],
    }

    ledger = project_legacy_experiment_report(report)
    case = ledger.cases[0]

    assert ledger.arm is HistoricalAtomicExperimentArm.A3
    assert case.roi.availability is LegacyEvidenceAvailability.KNOWN
    assert case.roi.value == [12, 18, 240, 180]
    assert case.legacy_pre_runtime_input_size.availability is LegacyEvidenceAvailability.KNOWN
    assert case.legacy_pre_runtime_input_size.value == [448, 448]
    assert case.transform_trace.availability is LegacyEvidenceAvailability.UNKNOWN
    assert case.final_thumbnail_geometry.availability is LegacyEvidenceAvailability.UNKNOWN
    assert "pre-runtime" in case.final_thumbnail_geometry.note


def test_explicit_current_fields_are_known_without_reinterpreting_legacy_fields() -> None:
    case = project_legacy_case_record(
        _base_case(
            per_frame_transform_trace=[{"role": "focus", "crop": [1, 2, 3, 4]}],
            final_thumbnail_geometry=[[224, 224], [224, 224]],
            processor_observation={
                "video_grid_thw": [8, 16, 16],
                "tensor_shapes": {"pixel_values_videos": [8, 3, 224, 224]},
            },
        )
    )

    assert case.transform_trace.availability is LegacyEvidenceAvailability.KNOWN
    assert case.final_thumbnail_geometry.availability is LegacyEvidenceAvailability.KNOWN
    assert case.processor_grid.value == [8, 16, 16]
    assert case.processor_tensor_shape.value == {"pixel_values_videos": [8, 3, 224, 224]}


def test_recorded_comparison_stop_outcome_can_be_attached_without_changing_raw_reports() -> None:
    a0 = {"input_profile": "context-focus", "cases": [_base_case(uid="a0")]}
    a3 = {
        "input_profile": "context-focus-microburst-hybrid-roi",
        "cases": [_base_case(uid="a3")],
    }
    raw_a3_before = deepcopy(a3)
    comparison = {"decision": "STOP_BEFORE_D12"}

    ledger = build_historical_a0_a3_ledger(
        {"A0": a0, "A3": a3},
        stop_outcomes={"A3": comparison},
    )

    a3_row = next(row for row in ledger["arms"] if row["arm"] == "A3")
    assert a3_row["stop_outcome"] == {
        "availability": "known",
        "value": "STOP_BEFORE_D12",
        "note": "recorded historical branch outcome",
    }
    assert ledger["stop_outcomes"]["A0"]["availability"] == "unknown"
    assert ledger["case_count"] == 2
    assert ledger["unknown_case_field_counts"]["processor_grid"] == 2
    assert a3 == raw_a3_before
    assert comparison == {"decision": "STOP_BEFORE_D12"}


def test_arm_inference_only_uses_explicit_labels_or_known_profiles() -> None:
    assert infer_historical_arm({"input_profile": "context-focus-microburst"}) is (
        HistoricalAtomicExperimentArm.A1
    )
    assert infer_historical_arm({}, arm="a2") is HistoricalAtomicExperimentArm.A2
    assert infer_historical_arm({"input_profile": "unknown-profile"}) is None
    assert infer_historical_arm({"arm": "custom-historical-arm"}) == "custom-historical-arm"


def test_malformed_case_container_is_rejected_but_stored_lexical_data_is_not_revalidated() -> None:
    with pytest.raises(LegacyExperimentEvidenceError, match="cases must be a sequence"):
        project_legacy_experiment_report({"cases": {"uid": "not-a-list"}})

    ledger = project_legacy_experiment_report(
        {"cases": [_base_case(quality={"old": "unrecognized-shape"})]}
    )
    assert ledger.cases[0].lexical_outcome.availability is LegacyEvidenceAvailability.KNOWN
