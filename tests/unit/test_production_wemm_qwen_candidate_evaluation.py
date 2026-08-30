from __future__ import annotations

from robata.benchmark.production_wemm_qwen_candidate_evaluation import (
    evaluate_wemm_qwen_candidate_verifier,
    render_markdown,
)


def _reference() -> dict[str, object]:
    return {
        "items": [
            {
                "window_id": "w00",
                "recommendation": "EDIT",
                "segments": [{"verb": "spread", "noun": "garment"}],
            },
            {
                "window_id": "w01",
                "recommendation": "ABSTAIN",
                "segments": [{"verb": "fold", "noun": "garment"}],
            },
        ]
    }


def _pack() -> dict[str, object]:
    return {
        "windows": [
            {
                "window_id": "w00",
                "model_context": {
                    "wemm": {
                        "top_k": [
                            {"rank": 1, "verb": "pick up", "noun": "garment", "score": 0.8},
                            {"rank": 2, "verb": "spread", "noun": "garment", "score": 0.7},
                        ]
                    }
                },
            },
            {
                "window_id": "w01",
                "model_context": {
                    "wemm": {
                        "top_k": [
                            {"rank": 1, "verb": "fold", "noun": "garment", "score": 0.8},
                        ]
                    }
                },
            },
        ]
    }


def _joined() -> dict[str, object]:
    return {
        "windows": [
            {
                "window_id": "w00",
                "decision": "accept",
                "parsed_verification": {
                    "decision": "accept",
                    "selected_rank": 2,
                    "candidate_verdicts": [{"rank": 2, "support": "supported"}],
                },
            },
            {
                "window_id": "w01",
                "decision": "accept",
                "parsed_verification": {
                    "decision": "accept",
                    "selected_rank": 1,
                    "candidate_verdicts": [{"rank": 1, "support": "supported"}],
                },
            },
        ]
    }


def _qwen_sidecar() -> dict[str, object]:
    return {
        "model": {"load_seconds": 3.5},
        "windows": [
            {"window_id": "w00", "output_tokens": 10, "generation_seconds": 1.0},
            {"window_id": "w00", "output_tokens": 14, "generation_seconds": 2.0},
            {"window_id": "w01", "output_tokens": 8, "generation_seconds": 3.0},
        ],
        "elapsed_seconds": 9.5,
    }


def test_evaluation_separates_retrieval_from_selected_verifier() -> None:
    report = evaluate_wemm_qwen_candidate_verifier(_reference(), _pack(), _joined())
    assert report["metrics"]["denominator_windows"] == 1
    assert report["metrics"]["retrieval"]["recall_at_1"] == 0.0
    assert report["metrics"]["retrieval"]["recall_at_3"] == 1.0
    assert report["metrics"]["verifier"]["accepted_precision"] == 1.0
    assert report["windows"][0]["selected_action"] == "spread garment"
    assert report["windows"][0]["selected_match"] is True


def test_evaluation_markdown_is_non_gold() -> None:
    report = evaluate_wemm_qwen_candidate_verifier(_reference(), _pack(), _joined())
    markdown = render_markdown(report)
    assert "SURROGATE_ONLY" in markdown
    assert "NOT_MEASURED" in markdown
    assert report["accuracy_status"] == "NOT_MEASURED"
    assert "Accuracy status" in markdown
    assert "Separate metrics" in markdown


def test_evaluation_records_optional_sidecar_cost_without_inference() -> None:
    report = evaluate_wemm_qwen_candidate_verifier(
        _reference(), _pack(), _joined(), _qwen_sidecar()
    )
    cost = report["metrics"]["cost"]
    assert cost["status"] == "RECORDED"
    assert cost["model_load_seconds"] == 3.5
    assert cost["generation_seconds_total"] == 6.0
    assert cost["generation_seconds_mean"] == 2.0
    assert cost["generation_seconds_median"] == 2.0
    assert cost["output_tokens_total"] == 32
    assert cost["camera_rows"] == 3
    assert cost["window_rows"] == 2
    assert cost["elapsed_seconds"] == 9.5


def test_evaluation_without_sidecar_marks_cost_not_measured() -> None:
    report = evaluate_wemm_qwen_candidate_verifier(_reference(), _pack(), _joined())
    assert report["metrics"]["cost"]["status"] == "NOT_MEASURED"
    assert report["cost"]["generation_seconds_total"] is None
