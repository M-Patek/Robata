from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_wemm_epic_retrieval as summary  # noqa: E402


def test_projection_metrics_uses_row_audit_stable_key_before_ordinal_fallback() -> None:
    report = {
        "input": {"row_input_audit": [{"row_key": "case-a"}]},
        "results": {
            "canonical": {
                "metrics": {"case_deltas": [{"ground_truth": [1, 2]}]},
                "projections": {
                    "case-a": {
                        "visual": {
                            "status": "MAPPED",
                            "joint_selected": {"verb_id": 1, "noun_id": 2},
                        }
                    }
                },
            }
        },
    }

    metrics = summary._projection_metrics(report, "canonical", "visual")

    assert metrics["query_count"] == 1
    assert metrics["accepted_count"] == 1
    assert metrics["accepted_precision"] == 1.0


def test_markdown_includes_recall_at_three_in_normal_results() -> None:
    metric = {
        "recall_at_k": {"1": 0.1, "3": 0.2, "5": 0.3, "10": 0.4},
        "mrr": 0.2,
        "top1_accuracy": 0.1,
    }
    variants = {variant: {mode: metric for mode in summary.MODES} for variant in summary.VARIANTS}
    intervention = {
        arm: {
            "canonical": {
                mode: {
                    "top1_accuracy": 0.1,
                    "recall_at_5": 0.3,
                    "recall_at_10": 0.4,
                    "mrr": 0.2,
                }
                for mode in summary.MODES
            }
        }
        for arm in summary.ARMS
    }
    deltas = {
        arm: {
            "canonical": {
                mode: {"top1_changed_fraction": 0.0, "delta_top1_accuracy": 0.0}
                for mode in summary.MODES
            }
        }
        for arm in summary.ARMS[1:]
    }
    projection = {
        mode: {
            "accepted_coverage": 1.0,
            "accepted_precision": 0.1,
            "raw_joint_accuracy": 0.1,
        }
        for mode in summary.MODES
    }
    payload = {
        "experiment": {
            "model": {"identity": "fake", "dimension": 2},
            "case_count": 1,
            "frame_count": 4,
            "catalog_size": 2,
        },
        "normal_metrics": variants,
        "intervention_metrics": intervention,
        "intervention_deltas_vs_normal": deltas,
        "candidate_projection": {"default_threshold_metrics": {"canonical": projection}},
    }

    rendered = summary.markdown(payload)

    assert "| Variant | Mode | R@1 | R@3 | R@5 | R@10 | MRR | Top-1 |" in rendered
    assert "| canonical | visual | 0.100 | 0.200 | 0.300 | 0.400 | 0.200 | 0.100 |" in rendered
