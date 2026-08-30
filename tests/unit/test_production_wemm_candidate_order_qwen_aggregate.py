from __future__ import annotations

from robata.benchmark.production_wemm_candidate_order_qwen_aggregate import (
    aggregate_candidate_order_qwen_diagnostic,
    render_markdown,
)


def _sidecar() -> dict[str, object]:
    def row(
        mode: str,
        *,
        rank: int = 1,
        decision: str = "accept",
        evidence: str = "hand moves",
    ) -> dict[str, object]:
        return {
            "window_id": "w01",
            "recording_id": "r01",
            "mode": mode,
            "status": "SUCCEEDED",
            "generation_seconds": 1.0,
            "output_tokens": 10,
            "parsed_verification": {
                "parse_status": "PARSED",
                "decision": decision,
                "selected_rank": rank,
                "accept_contract_ok": True,
                "candidate_verdicts": [
                    {"rank": rank, "support": "supported", "evidence": [evidence]}
                ],
            },
        }

    return {
        "format": "robata-production-wemm-candidate-order-qwen-diagnostic-v1",
        "modes": ["as_is", "reverse", "shuffle"],
        "rows": [
            row("as_is"),
            row("reverse"),
            row("shuffle", decision="abstain", evidence="hand moves slowly"),
        ],
    }


def test_aggregate_exposes_rank_invariance_and_decision_flip() -> None:
    report = aggregate_candidate_order_qwen_diagnostic(_sidecar())
    assert report["status"] == "COMPLETE"
    assert report["conclusion"]["result"] == "ORDER_SENSITIVE"
    metrics = report["metrics"]
    assert metrics["window_count"] == 1
    assert metrics["valid_complete_window_count"] == 1
    assert metrics["rank_changed_windows"] == 0
    assert metrics["decision_changed_windows"] == 1
    assert metrics["evidence_changed_windows"] == 1
    assert metrics["pairwise"]["as_is_vs_shuffle"]["difference_windows"] == 1


def test_aggregate_reports_parse_failure_and_incomplete_window() -> None:
    sidecar = _sidecar()
    sidecar["rows"] = [
        sidecar["rows"][0],
        {"window_id": "w02", "mode": "as_is", "status": "FAILED"},
    ]
    report = aggregate_candidate_order_qwen_diagnostic(sidecar)
    assert report["status"] == "COMPLETE"
    assert report["metrics"]["parse_failure_count"] == 1
    assert report["metrics"]["complete_window_count"] == 0
    assert report["metrics"]["valid_complete_window_count"] == 0


def test_markdown_contains_order_metrics() -> None:
    report = aggregate_candidate_order_qwen_diagnostic(_sidecar())
    markdown = render_markdown(report)
    assert "candidate-order" in markdown
    assert "Decision" in markdown
    assert "as_is_vs_shuffle" in markdown
