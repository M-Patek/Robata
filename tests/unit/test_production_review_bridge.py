from __future__ import annotations

import pytest

from robata.benchmark.production_review_bridge import (
    ProductionReviewBridgeError,
    apply_review_decisions,
    build_decision_template,
)


def _blank_pack() -> dict[str, object]:
    return {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "source": {"path": "sample.mcap"},
        "items": [
            {
                "window_id": "w00",
                "gold": {
                    "status": "PENDING_HUMAN_REVIEW",
                    "segments": [],
                    "provenance": {},
                },
                "model_outputs": {"wemm": {"status": "NOT_RUN"}},
                "adjudication": {},
            },
            {
                "window_id": "w01",
                "gold": {
                    "status": "PENDING_HUMAN_REVIEW",
                    "segments": [],
                    "provenance": {},
                },
                "model_outputs": {"qwen": {"status": "NOT_RUN"}},
                "adjudication": {},
            },
        ],
        "controls": {
            "labels_inferred": False,
            "model_predictions_copied": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "sha_or_digest_computed": False,
        },
    }


def _agent_pack() -> dict[str, object]:
    return {
        "format": "robata-production-agent-reviewed-segment-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "items": [
            {
                "window_id": "w00",
                "accepted_as_gold": False,
                "segments": [
                    {
                        "verb": "fold",
                        "noun": "garment",
                        "attributes": None,
                        "location": "table",
                        "hand": "both hands",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                    }
                ],
            },
            {"window_id": "w01", "accepted_as_gold": False, "segments": []},
        ],
    }


def _decisions(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "format": "robata-production-review-decisions-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source_agent_pack": "agent.json",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-08-27T12:00:00Z",
        "decisions": list(rows),
    }


def test_template_is_pending_and_contains_every_agent_window() -> None:
    template = build_decision_template(_agent_pack())
    assert template["format"] == "robata-production-review-decisions-v1"
    assert [row["window_id"] for row in template["decisions"]] == ["w00", "w01"]  # type: ignore[index]
    assert all(row["decision"] == "pending" for row in template["decisions"])  # type: ignore[index]


def test_accept_copies_only_agent_suggestion_and_adds_provenance() -> None:
    result = apply_review_decisions(
        _blank_pack(),
        _agent_pack(),
        _decisions({"window_id": "w00", "decision": "accept"}),
    )
    item = result["items"][0]  # type: ignore[index]
    assert item["gold"]["status"] == "ACCEPTED"  # type: ignore[index]
    assert item["gold"]["segments"][0]["verb"] == "fold"  # type: ignore[index]
    assert item["gold"]["provenance"]["reviewer_id"] == "reviewer-1"  # type: ignore[index]
    assert item["gold"]["provenance"]["agent_suggestion_used"] is True  # type: ignore[index]
    assert result["controls"]["model_predictions_copied"] is False  # type: ignore[index]
    assert result["items"][1]["gold"]["status"] == "PENDING_HUMAN_REVIEW"  # type: ignore[index]


def test_edit_and_split_require_explicit_segments() -> None:
    with pytest.raises(ProductionReviewBridgeError, match="requires segments"):
        apply_review_decisions(
            _blank_pack(),
            _agent_pack(),
            _decisions({"window_id": "w00", "decision": "edit"}),
        )
    result = apply_review_decisions(
        _blank_pack(),
        _agent_pack(),
        _decisions(
            {
                "window_id": "w00",
                "decision": "split",
                "segments": [
                    {"verb": "smooth", "noun": "garment", "start_seconds": 0.5, "end_seconds": 1.0},
                    {"verb": "fold", "noun": "garment", "start_seconds": 1.0, "end_seconds": 2.0},
                ],
            }
        ),
    )
    assert len(result["items"][0]["gold"]["segments"]) == 2  # type: ignore[index]


def test_reject_and_abstain_do_not_enter_accepted_denominator() -> None:
    result = apply_review_decisions(
        _blank_pack(),
        _agent_pack(),
        _decisions(
            {"window_id": "w00", "decision": "reject"},
            {"window_id": "w01", "decision": "abstain"},
        ),
    )
    assert result["items"][0]["gold"]["status"] == "REJECTED"  # type: ignore[index]
    assert result["items"][1]["gold"]["status"] == "ABSTAIN"  # type: ignore[index]


def test_unknown_window_and_model_payload_fail_closed() -> None:
    with pytest.raises(ProductionReviewBridgeError, match="absent from blank"):
        apply_review_decisions(
            _blank_pack(),
            _agent_pack(),
            _decisions({"window_id": "unknown", "decision": "pending"}),
        )
    with pytest.raises(ProductionReviewBridgeError, match="model or gold payload"):
        apply_review_decisions(
            _blank_pack(),
            _agent_pack(),
            _decisions({"window_id": "w00", "decision": "pending", "predictions": []}),
        )
