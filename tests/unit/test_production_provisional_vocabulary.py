from __future__ import annotations

import pytest

from robata.benchmark.production_provisional_vocabulary import (
    ProvisionalVocabularyError,
    build_provisional_vocabulary,
    compare_provisional_vocabulary,
)


def _review_pack() -> dict[str, object]:
    return {
        "format": "robata-production-agent-reviewed-segment-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "review_contract": {
            "accepted_as_gold": False,
            "official_gold_status": "PENDING_HUMAN_REVIEW",
            "not_an_evaluator_input": True,
        },
        "items": [
            {
                "window_id": "w00",
                "segments": [
                    {
                        "verb": "picking",
                        "noun": "garment",
                        "label_text": "picking garment",
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                    }
                ],
            }
        ],
        "controls": {
            "gold_read": False,
            "gold_written": False,
            "model_predictions_copied": False,
        },
    }


def test_build_keeps_production_vocab_provisional() -> None:
    result = build_provisional_vocabulary(_review_pack())

    assert result["status"] == "PROVISIONAL_UNAPPROVED"
    assert result["production_eligible"] is False
    assert result["quality"]["measurement_status"] == "NOT_MEASURED"  # type: ignore[index]
    assert result["ontology_projection"]["unresolved_record_count"] == 1  # type: ignore[index]
    assert result["controls"]["gold_written"] is False  # type: ignore[index]


def test_build_rejects_pack_that_claims_gold() -> None:
    pack = _review_pack()
    pack["production_eligible"] = True

    with pytest.raises(ProvisionalVocabularyError, match="production eligible"):
        build_provisional_vocabulary(pack)


def test_compare_reports_routing_overlap_without_quality_claim() -> None:
    vocabulary = build_provisional_vocabulary(_review_pack())
    comparison = compare_provisional_vocabulary(
        vocabulary,
        {
            "status": "NON_GOLD_EXPLORATORY",
            "routes": {
                "qwen": {
                    "candidate_windows": {
                        "w00": [
                            {"verb": "pick", "noun": "cloth"},
                            {"verb": "put", "noun": "cloth"},
                        ]
                    }
                }
            },
        },
    )

    route = comparison["route_metrics"]["qwen"]  # type: ignore[index]
    assert comparison["quality_claim"] is False
    assert route["metrics"]["family_pair"]["at_k"] == 1.0  # type: ignore[index]
    assert route["metrics"]["strict_pair"]["at_k"] == 0.0  # type: ignore[index]
