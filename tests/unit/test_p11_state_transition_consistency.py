from __future__ import annotations

import copy

import pytest

from robata.benchmark.p11_state_transition_consistency import (
    StateTransitionError,
    build_public_result,
    derive_transition,
    evaluate_posthoc,
    normalize_state,
)


def _candidate() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for variant, pre, post, direction in (
        ("normal", "off", "on", "on_to_off"),
        ("reverse", "on", "off", "on_to_off"),
        ("pre_pre", "off", "on", "on_to_off"),
        ("post_post", "on", "off", "on_to_off"),
    ):
        rows.append(
            {
                "case_id": "case-1",
                "video_group": "group-a",
                "variant": variant,
                "raw_model_output": (
                    '{"object":"faucet","active_part":"handle",'
                    f'"pre_state":"{pre}","post_state":"{post}",'
                    f'"direction":"{direction}","state_relation":"change",'
                    '"confidence":0.9,"evidence":["visible endpoint"]}'
                ),
            }
        )
    return {
        "artifact_version": "p9-clean-qwen-temporal-invariance-raw-v1",
        "label_blind_inference": True,
        "production_eligible": False,
        "hash_or_sha_used": False,
        "heldout_100_opened": False,
        "cases": rows,
    }


def test_normalize_and_derive_are_conservative() -> None:
    assert normalize_state("water flowing") == ("on_off", "on")
    assert normalize_state("opening the door") is None
    assert derive_transition("off", "on")["direction"] == "off_to_on"
    assert derive_transition("open", "closed")["direction"] == "open_to_closed"
    assert derive_transition("off", "off")["relation"] == "no_change"


def test_public_projection_repairs_direction_field_and_freezes_controls() -> None:
    result = build_public_result(_candidate())
    normal = next(row for row in result["rows"] if row["variant"] == "normal")
    assert normal["projection"]["direction"] == "off_to_on"
    assert normal["projection"]["direction_consistent_with_raw"] is False
    assert "candidate_direction_conflicts_with_state_pair" in normal["projection"]["reasons"]
    freeze = [row for row in result["rows"] if row["variant"] in {"pre_pre", "post_post"}]
    assert all(row["projection"]["relation"] == "no_change" for row in freeze)
    assert result["summary"]["freeze_no_change_rate"] == 1.0


def test_feature_null_vetoes_nonfreeze_candidate() -> None:
    feature = {
        "artifact_version": "p10-frozen-visual-feature-audit-public-v1",
        "label_blind_input": True,
        "production_eligible": False,
        "rows": [
            {"case_id": "case-1", "variant": "normal", "temporal_metrics": {"delta_norm": 0.0}},
        ],
    }
    result = build_public_result(_candidate(), feature)
    normal = next(row for row in result["rows"] if row["variant"] == "normal")
    assert normal["projection"]["relation"] == "no_change"
    assert normal["projection"]["disposition"] == "abstain_no_change"


def test_p10_feature_artifact_structural_expected_arms_is_accepted() -> None:
    """P11 may consume P10 metadata without treating counts as semantic labels."""

    feature = {
        "artifact_version": "p10-frozen-visual-feature-audit-v1",
        "label_blind_inference": True,
        "production_eligible": False,
        "expected_arms": 20,
        "rows": [
            {
                "case_id": "case-1",
                "variant": "normal",
                "temporal_metrics": {"delta_norm": 1.0},
            },
        ],
    }
    result = build_public_result(_candidate(), feature)
    normal = next(row for row in result["rows"] if row["variant"] == "normal")
    assert normal["feature_evidence"]["change_present"] is True


def test_p10_case_metric_alias_is_consumed() -> None:
    """The real P10 metric name must not be dropped by the P11 join."""

    feature = {
        "artifact_version": "p10-frozen-visual-feature-audit-v1",
        "label_blind_inference": True,
        "production_eligible": False,
        "case_metrics": [
            {
                "case_id": "case-1",
                "normal_endpoint_delta": 0.25,
            }
        ],
    }
    result = build_public_result(_candidate(), feature)
    normal = next(row for row in result["rows"] if row["variant"] == "normal")
    assert normal["feature_evidence"]["available"] is True
    assert normal["feature_evidence"]["change_present"] is True
    assert normal["feature_evidence"]["delta_norm"] == 0.25


def test_named_p10_surface_projects_compact_metrics_onto_all_arms() -> None:
    rows = [
        {
            "case_id": "case-1",
            "variant": variant,
            "temporal_metrics": {"normal_endpoint_delta": 0.0},
        }
        for variant in ("normal", "reverse", "pre_pre", "post_post")
    ]
    feature = {
        "artifact_version": "p10-frozen-visual-feature-audit-v1",
        "label_blind": True,
        "production_eligible": False,
        "cases": rows,
        "case_metrics_by_surface": {
            "last_hidden_state": [
                {
                    "case_id": "case-1",
                    "normal_endpoint_delta": 0.001,
                    "pre_pre_duplicate_delta": 0.0,
                    "post_post_duplicate_delta": 0.0,
                }
            ],
            "pooler_output": [
                {
                    "case_id": "case-1",
                    "normal_endpoint_delta": 0.25,
                    "pre_pre_duplicate_delta": 0.0,
                    "post_post_duplicate_delta": 0.0,
                }
            ],
        },
    }
    result = build_public_result(_candidate(), feature, feature_surface="pooler_output")
    normal = next(row for row in result["rows"] if row["variant"] == "normal")
    freeze = [row for row in result["rows"] if row["variant"] in {"pre_pre", "post_post"}]
    assert result["feature_surface"] == "pooler_output"
    assert normal["feature_evidence"]["delta_norm"] == 0.25
    assert all(row["feature_evidence"]["delta_norm"] == 0.0 for row in freeze)


def test_named_p10_surface_must_exist() -> None:
    feature = {
        "artifact_version": "p10-frozen-visual-feature-audit-v1",
        "label_blind": True,
        "production_eligible": False,
        "case_metrics_by_surface": {"pooler_output": []},
    }
    with pytest.raises(StateTransitionError, match="unavailable"):
        build_public_result(_candidate(), feature, feature_surface="missing")


def test_p10_repeated_case_metrics_do_not_mark_freeze_controls_changed() -> None:
    """Arm-specific duplicate deltas must win over the repeated normal delta."""

    rows = [
        {
            "case_id": "case-1",
            "variant": variant,
            "temporal_metrics": {
                "normal_endpoint_delta": 0.25,
                "pre_pre_duplicate_delta": 0.0,
                "post_post_duplicate_delta": 0.0,
            },
        }
        for variant in ("normal", "reverse", "pre_pre", "post_post")
    ]
    feature = {
        "artifact_version": "p10-frozen-visual-feature-audit-v1",
        "label_blind_inference": True,
        "production_eligible": False,
        "cases": rows,
    }
    result = build_public_result(_candidate(), feature)
    freeze = [row for row in result["rows"] if row["variant"] in {"pre_pre", "post_post"}]
    assert all(row["feature_evidence"]["delta_norm"] == 0.0 for row in freeze)
    assert all(row["projection"]["relation"] == "no_change" for row in freeze)
    assert result["summary"]["freeze_no_change_rate"] == 1.0


def test_public_boundary_rejects_oracle_and_posthoc_does_not_mutate() -> None:
    candidate = _candidate()
    candidate["cases"][0]["expected_object"] = "faucet"  # type: ignore[index]
    with pytest.raises(StateTransitionError, match="private"):
        build_public_result(candidate)

    public = build_public_result(_candidate())
    before = copy.deepcopy(public)
    posthoc = evaluate_posthoc(
        public,
        {
            "cases": [
                {
                    "case_id": "case-1",
                    "variant": "normal",
                    "expected_object": "faucet",
                    "expected_active_part": "handle",
                    "expected_change_direction": "off_to_on",
                    "expected_state_relation": "change",
                }
            ]
        },
    )
    assert public == before
    assert posthoc["posthoc_only"] is True
    assert posthoc["summary"]["derived_direction_accuracy"] == 1.0
