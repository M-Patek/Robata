from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "scripts" / "check_wemm_epic_baseline_completeness.py"
    spec = importlib.util.spec_from_file_location("wemm_baseline_completeness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report():
    check = _module()
    metrics = {
        mode: {
            "recall_at_1": 0.1,
            "recall_at_3": 0.2,
            "recall_at_5": 0.3,
            "recall_at_10": 0.4,
            "mrr": 0.2,
        }
        for mode in check.EXPECTED_MODES
    }
    return {
        "normal_metrics_27_ontology": {
            variant: dict(metrics) for variant in check.EXPECTED_VARIANTS
        },
        "nearest_neighbor_cases_canonical": [
            {
                "row": 0,
                "modes": {
                    mode: {"top5": [{"rank": 1, "action_key": [1, 2]}]}
                    for mode in check.EXPECTED_MODES
                },
            }
        ],
        "mapper_boundary": {
            "retrieval_only": True,
            "existing_mapper_invoked": False,
        },
        "scope": {"production_paths_touched": False},
    }


def test_check_covers_variants_modes_metrics_hard_negatives_and_boundaries() -> None:
    result = _module().inspect(_report(), source="baseline.json")

    assert result["complete"] is True
    assert result["coverage"]["metrics"]["block_count"] == 9
    assert result["coverage"]["hard_negatives"]["case_count"] == 1
    assert result["coverage"]["hard_negatives"]["cases_with_hard_negative"] == 1
    boundaries = result["coverage"]["boundaries"]
    assert boundaries["mapper"]["retrieval_only"] is True
    assert boundaries["mapper"]["existing_mapper_invoked"] is False
    assert boundaries["resolver"]["resolver_invoked"] is False
    assert boundaries["resolver"]["quality_claim"] is False
    assert result["controls"]["hash_or_sha_used"] is False


def test_check_fails_when_one_metric_or_hard_negative_mode_is_missing() -> None:
    check = _module()
    report = _report()
    del report["normal_metrics_27_ontology"]["natural"]["hybrid"]["mrr"]
    del report["nearest_neighbor_cases_canonical"][0]["modes"]["text"]

    result = check.inspect(report)

    assert result["complete"] is False
    assert "natural/hybrid/MRR" in result["coverage"]["metrics"]["invalid_values"]
    assert result["coverage"]["hard_negatives"]["passed"] is False


def test_markdown_states_resolver_is_not_part_of_baseline() -> None:
    result = _module().inspect(_report())
    rendered = _module().markdown(result)

    assert "Resolver invoked: `False`" in rendered
    assert "Mapper retrieval-only: `True`" in rendered


def test_check_fails_closed_if_a_resolver_invocation_is_declared() -> None:
    report = _report()
    report["controls"] = {"resolver_invoked": True}

    result = _module().inspect(report)

    assert result["complete"] is False
    assert result["coverage"]["boundaries"]["resolver"]["resolver_invoked"] is True
