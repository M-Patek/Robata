from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.production_structured_annotation import (
    build_structured_annotation_envelope,
)
from robata.benchmark.production_structured_review_adapter import (
    STRUCTURED_REVIEW_DRAFT_VERSION,
    StructuredReviewAdapterError,
    build_structured_review_draft,
)


def _envelope() -> dict[str, object]:
    wemm_prediction = {
        "rank": 1,
        "action_key": [1, 2],
        "verb": "open",
        "noun": "drawer",
        "score": 0.91,
    }
    return build_structured_annotation_envelope(
        {
            "wemm": {
                "format": "robata-production-wemm-shadow-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "source": {"path": "source.mcap", "camera_count": 1},
                "windows": [
                    {
                        "window_id": "w00",
                        "ordinal": 0,
                        "start_seconds": 0.0,
                        "end_seconds": 4.0,
                        "model": {
                            "status": "SUCCEEDED",
                            "predictions": [wemm_prediction],
                        },
                    }
                ],
            },
            "qwen": {
                "format": "robata-production-qwen-structured-native-shadow-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "source": {"manifest": "cohort.json", "camera_count": 1},
                "windows": [
                    {
                        "window_id": "w00",
                        "ordinal": 0,
                        "interval": [0.0, 4.0],
                        "camera_id": "cam_01",
                        "status": "SUCCEEDED",
                        "segments": [
                            {
                                "start_time_sec": 1.0,
                                "end_time_sec": 2.0,
                                "structured_labels": {
                                    "verb": "close",
                                    "noun": "door",
                                    "attributes": None,
                                    "location": None,
                                    "hand": None,
                                },
                                "confidence": 0.8,
                                "evidence": ["door moves shut"],
                            }
                        ],
                        "candidates": [
                            {"verb": "close", "noun": "door", "confidence": 0.8},
                            {"verb": "open", "noun": "door", "confidence": 0.2},
                        ],
                        "raw_text": '{"segments":[]}',
                    }
                ],
            },
        },
        source_path="source.mcap",
        window_specs=[
            {
                "ordinal": 0,
                "window_id": "w00",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
            }
        ],
        camera_count=1,
    )


def test_draft_preserves_top_k_and_marks_window_boundary_non_action() -> None:
    envelope = _envelope()
    before = deepcopy(envelope)
    draft = build_structured_review_draft(envelope)

    assert envelope == before
    assert draft["format"] == STRUCTURED_REVIEW_DRAFT_VERSION
    window = draft["windows"][0]
    assert window["annotation_draft"]["status"] == "PROVISIONAL"
    claim = window["annotation_draft"]["segments"][0]
    assert claim["model"] == "qwen"
    assert claim["claim"]["structured_labels"]["verb"]["value"] == "close"
    assert window["annotation_draft"]["window_boundary"]["is_action_boundary"] is False
    assert "not an action boundary" in window["annotation_draft"]["window_boundary"]["note"]

    assert window["model_top_k"]["wemm"] == [
        {
            "rank": 1,
            "action_key": [1, 2],
            "verb": "open",
            "noun": "drawer",
            "score": 0.91,
        }
    ]
    assert window["model_top_k"]["qwen"] == [
        {"verb": "close", "noun": "door", "confidence": 0.8},
        {"verb": "open", "noun": "door", "confidence": 0.2},
    ]
    assert window["model_top_k"]["mage"] == []
    assert "TOP1_MODEL_DISAGREEMENT" in window["conflicts"]
    assert window["abstention"]["decision"] == "review"
    assert draft["controls"]["gold_written"] is False


def test_missing_structured_claims_explicitly_abstain_but_retain_candidates() -> None:
    envelope = _envelope()
    qwen = envelope["windows"][0]["models"]["qwen"]  # type: ignore[index]
    # Keep the envelope internally consistent: an empty segment list is an
    # explicit non-measurement, not a MEASURED section with missing claims.
    qwen["segments"] = []  # type: ignore[index]
    qwen["measurement_status"] = "NOT_MEASURED"  # type: ignore[index]
    draft = build_structured_review_draft(envelope)
    window = draft["windows"][0]
    assert window["annotation_draft"]["status"] == "ABSTAIN"
    assert window["annotation_draft"]["segments"] == []
    assert window["abstention"]["decision"] == "abstain"
    assert "NO_STRUCTURED_SEGMENTS" in window["abstention"]["reason_codes"]
    assert window["model_top_k"]["qwen"]


def test_invalid_qwen_parse_observations_reach_conflict_and_abstention_codes() -> None:
    envelope = _envelope()
    qwen = envelope["windows"][0]["models"]["qwen"]  # type: ignore[index]
    # An invalid runner row is retained as provenance with no promoted
    # segments.  The model status may still be SUCCEEDED because the model
    # call completed; parser/mapping diagnostics must therefore be surfaced
    # independently of that transport status.
    qwen["segments"] = []  # type: ignore[index]
    qwen["measurement_status"] = "NOT_MEASURED"  # type: ignore[index]
    qwen["parse_observations"] = [  # type: ignore[index]
        {
            "camera_id": "cam_01",
            "parse_status": "INVALID",
            "errors": [
                "TIMESTAMP_MAPPING_FAILED",
                "segments[0] end_time_sec must exceed start_time_sec",
            ],
            "warnings": ["FILLER_VERB_PRESENT:adjust"],
            "generation_warnings": [],
            "timestamp_mapping_status": "FAILED",
        }
    ]

    draft = build_structured_review_draft(envelope)
    window = draft["windows"][0]
    assert window["annotation_draft"]["status"] == "ABSTAIN"
    expected = {
        "PARSE_INVALID",
        "TIMESTAMP_MAPPING_FAILED",
        "PARSE_ERROR:segments[0] end_time_sec must exceed start_time_sec",
        "PARSE_WARNING:FILLER_VERB_PRESENT:adjust",
    }
    assert expected.issubset(set(window["conflicts"]))
    assert expected.issubset(set(window["abstention"]["reason_codes"]))


def test_parse_warnings_are_review_diagnostics_without_promoting_gold() -> None:
    envelope = _envelope()
    qwen = envelope["windows"][0]["models"]["qwen"]  # type: ignore[index]
    qwen["parse_observations"] = [  # type: ignore[index]
        {
            "camera_id": "cam_01",
            "parse_status": "PARSED",
            "errors": [],
            "warnings": ["FILLER_VERB_PRESENT:reaches"],
            "generation_warnings": ["MAX_NEW_TOKENS_REACHED"],
        }
    ]

    draft = build_structured_review_draft(envelope)
    window = draft["windows"][0]
    assert window["annotation_draft"]["status"] == "PROVISIONAL"
    assert "PARSE_WARNING:FILLER_VERB_PRESENT:reaches" in window["conflicts"]
    assert "PARSE_WARNING:MAX_NEW_TOKENS_REACHED" in window["conflicts"]
    assert "PARSE_WARNING:FILLER_VERB_PRESENT:reaches" in window["abstention"]["reason_codes"]
    assert draft["controls"]["gold_written"] is False


def test_multiple_segments_in_one_model_are_not_cross_model_disagreement() -> None:
    envelope = _envelope()
    qwen = envelope["windows"][0]["models"]["qwen"]  # type: ignore[index]
    second = deepcopy(qwen["segments"][0])  # type: ignore[index]
    second["start_time_sec"] = 2.0
    second["end_time_sec"] = 3.0
    second["structured_labels"]["verb"]["value"] = "open"  # type: ignore[index]
    qwen["segments"].append(second)  # type: ignore[index]

    draft = build_structured_review_draft(envelope)
    conflicts = draft["windows"][0]["conflicts"]  # type: ignore[index]

    assert "STRUCTURED_SEGMENT_DISAGREEMENT" not in conflicts
    assert len(draft["windows"][0]["annotation_draft"]["segments"]) == 2  # type: ignore[index]


def test_single_qwen_sidecar_is_wrapped_and_missing_mage_is_explicitly_blocked() -> None:
    sidecar = {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {"manifest": "cohort.json", "camera_count": 1},
        "windows": [
            {
                "window_id": "w00",
                "ordinal": 0,
                "interval": [0.0, 4.0],
                "camera_id": "cam_01",
                "status": "SUCCEEDED",
                "segments": [],
                "candidates": [{"verb": "open", "noun": "drawer"}],
            }
        ],
    }
    draft = build_structured_review_draft(sidecar)
    assert draft["windows"][0]["model_top_k"]["qwen"] == [{"verb": "open", "noun": "drawer"}]
    assert draft["windows"][0]["model_top_k"]["wemm"] == []
    assert "MODEL_BLOCKED:mage" in draft["windows"][0]["conflicts"]


def test_rejects_non_structured_input() -> None:
    with pytest.raises(StructuredReviewAdapterError, match="structured annotation envelope"):
        build_structured_review_draft({"format": "unknown", "windows": []})
