from __future__ import annotations

import pytest

from robata.benchmark.mage_small_encoder import (
    aggregate_small_encoder_shadow_run,
    evaluate_small_encoder_pair,
    normalize_compact_action_label,
    parse_compact_output,
)


def test_parse_compact_output_keeps_action_and_boundary() -> None:
    parsed = parse_compact_output(
        '{"observations":[{"action":"Fold Green Cloth","interval":'
        '{"start_offset_seconds":1.0,"end_offset_seconds":2.5}}]}'
    )
    assert parsed.json_valid is True
    assert parsed.normalized_labels == ("fold_green_cloth",)
    assert parsed.actions[0].end_offset_seconds == 2.5


def test_pair_evaluation_counts_false_silence() -> None:
    result = evaluate_small_encoder_pair(
        native_output_text=(
            '{"observations":[{"action":"fold cloth","interval":'
            '{"start_offset_seconds":1,"end_offset_seconds":2}}]}'
        ),
        candidate_output_text='{"observations":[]}',
    )
    assert result.false_silence is True
    assert result.exact_label_recall == 0.0
    assert result.exact_label_precision is None


def test_pair_evaluation_matches_duplicates_in_temporal_order() -> None:
    native = (
        '{"observations":['
        '{"action":"fold cloth","interval":{"start_offset_seconds":1,"end_offset_seconds":2}},'
        '{"action":"fold cloth","interval":{"start_offset_seconds":4,"end_offset_seconds":5}}]}'
    )
    candidate = (
        '{"observations":['
        '{"action":"fold cloth","interval":{"start_offset_seconds":1.25,"end_offset_seconds":2.5}},'
        '{"action":"fold cloth","interval":{"start_offset_seconds":4.5,"end_offset_seconds":5.5}},'
        '{"action":"fold cloth","interval":{"start_offset_seconds":6,"end_offset_seconds":7}}]}'
    )
    result = evaluate_small_encoder_pair(
        native_output_text=native,
        candidate_output_text=candidate,
    )
    assert result.exact_label_match_count == 2
    assert result.candidate_repeated_label_excess_count == 2
    assert result.boundary_start_mae_seconds == 0.375
    assert result.boundary_end_mae_seconds == 0.5


def test_invalid_json_is_not_silence() -> None:
    result = evaluate_small_encoder_pair(
        native_output_text='{"observations":[]}',
        candidate_output_text="not-json",
    )
    assert result.candidate_json_valid is False
    assert result.false_silence is False


def test_label_normalization_is_lexical_not_semantic() -> None:
    assert normalize_compact_action_label("Fold green cloth!") == "fold_green_cloth"
    assert normalize_compact_action_label("fold green shirt") != normalize_compact_action_label(
        "fold green cloth"
    )


def test_aggregate_shadow_run_applies_quality_and_latency_gates() -> None:
    exact = evaluate_small_encoder_pair(
        native_output_text=(
            '{"observations":[{"action":"fold cloth","interval":'
            '{"start_offset_seconds":1,"end_offset_seconds":2}}]}'
        ),
        candidate_output_text=(
            '{"observations":[{"action":"fold cloth","interval":'
            '{"start_offset_seconds":1.25,"end_offset_seconds":2.5}}]}'
        ),
    )
    duplicate = evaluate_small_encoder_pair(
        native_output_text=(
            '{"observations":[{"action":"wipe table","interval":'
            '{"start_offset_seconds":3,"end_offset_seconds":4}}]}'
        ),
        candidate_output_text=(
            '{"observations":['
            '{"action":"use mouse","interval":{"start_offset_seconds":3,"end_offset_seconds":4}},'
            '{"action":"use mouse","interval":{"start_offset_seconds":4,"end_offset_seconds":5}}]}'
        ),
    )
    summary = aggregate_small_encoder_shadow_run(
        evaluations=(exact, duplicate),
        native_generation_seconds=10.0,
        candidate_generation_seconds=11.0,
        candidate_preparation_seconds=1.0,
    )
    assert summary.exact_label_recall == 0.5
    assert summary.candidate_repeated_label_excess_rate == pytest.approx(1 / 3)
    assert summary.generation_plus_preparation_speedup == pytest.approx(10 / 12)
    assert summary.matched_boundary_start_mae_seconds == 0.25
    assert summary.matched_boundary_end_mae_seconds == 0.5
    assert summary.gates == {
        "all_json_syntax_valid": True,
        "all_compact_contract_valid": True,
        "false_silence_zero": True,
        "exact_label_recall_at_least_0_90": False,
        "exact_label_precision_at_least_0_90": False,
        "matched_boundary_measurement_complete": True,
        "matched_boundary_start_mae_at_most_0_50_seconds": True,
        "matched_boundary_end_mae_at_most_0_50_seconds": True,
        "candidate_faster_than_native": False,
    }
    assert summary.qualified is False


def test_aggregate_shadow_run_can_qualify_only_when_all_gates_pass() -> None:
    exact = evaluate_small_encoder_pair(
        native_output_text=(
            '{"observations":[{"action":"hold cup","interval":'
            '{"start_offset_seconds":0,"end_offset_seconds":1}}]}'
        ),
        candidate_output_text=(
            '{"observations":[{"action":"hold cup","interval":'
            '{"start_offset_seconds":0,"end_offset_seconds":1}}]}'
        ),
    )
    summary = aggregate_small_encoder_shadow_run(
        evaluations=(exact,),
        native_generation_seconds=2.0,
        candidate_generation_seconds=1.0,
        candidate_preparation_seconds=0.25,
    )
    assert summary.qualified is True
    assert all(summary.gates.values())


def test_aggregate_shadow_run_rejects_large_boundary_drift() -> None:
    pair = evaluate_small_encoder_pair(
        native_output_text=(
            '{"observations":[{"action":"hold cup","interval":'
            '{"start_offset_seconds":0,"end_offset_seconds":1}}]}'
        ),
        candidate_output_text=(
            '{"observations":[{"action":"hold cup","interval":'
            '{"start_offset_seconds":100,"end_offset_seconds":101}}]}'
        ),
    )
    summary = aggregate_small_encoder_shadow_run(
        evaluations=(pair,),
        native_generation_seconds=2.0,
        candidate_generation_seconds=1.0,
        candidate_preparation_seconds=0.25,
    )
    assert summary.matched_boundary_start_mae_seconds == 100.0
    assert summary.matched_boundary_end_mae_seconds == 100.0
    assert summary.gates["matched_boundary_start_mae_at_most_0_50_seconds"] is False
    assert summary.gates["matched_boundary_end_mae_at_most_0_50_seconds"] is False
    assert summary.qualified is False


def test_aggregate_shadow_run_rejects_empty_or_negative_inputs() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        aggregate_small_encoder_shadow_run(
            evaluations=(),
            native_generation_seconds=1.0,
            candidate_generation_seconds=1.0,
            candidate_preparation_seconds=0.0,
        )
    evaluation = evaluate_small_encoder_pair(
        native_output_text='{"observations":[]}',
        candidate_output_text='{"observations":[]}',
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        aggregate_small_encoder_shadow_run(
            evaluations=(evaluation,),
            native_generation_seconds=-1.0,
            candidate_generation_seconds=1.0,
            candidate_preparation_seconds=0.0,
        )


def test_compact_contract_rejects_misspelled_interval_key_but_keeps_syntax_diagnostic() -> None:
    parsed = parse_compact_output(
        '{"observations":[{"action":"reach shirt","interval":'
        '{"start_offset offset_seconds":1,"end_offset_seconds":2}}]}'
    )
    assert parsed.json_syntax_valid is True
    assert parsed.compact_contract_valid is False
    assert parsed.normalized_labels == ("reach_shirt",)
