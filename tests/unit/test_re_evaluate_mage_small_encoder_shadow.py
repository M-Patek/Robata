from __future__ import annotations

import hashlib

import pytest

from scripts.re_evaluate_mage_small_encoder_shadow import (
    _canonical_sha256,
    evaluate_report_payload,
)


def _report() -> dict[str, object]:
    report: dict[str, object] = {
        "report_version": "mage-small-encoder-shadow-report-v2",
        "authority": "MAGE_NATIVE",
        "candidate": {"policy_semantic_sha256": "a" * 64},
        "segments": [
            {
                "native": {
                    "output_text": (
                        '{"observations":[{"action":"hold cup","interval":'
                        '{"start_offset_seconds":0,"end_offset_seconds":1}}]}'
                    ),
                    "generation_seconds": 2.0,
                },
                "small_encoder_shadow": {
                    "output_text": (
                        '{"observations":[{"action":"hold cup","interval":'
                        '{"start_offset_seconds":0,"end_offset_seconds":1}}]}'
                    ),
                    "generation_seconds": 1.0,
                    "telemetry": {"total_seconds": 0.25},
                },
            }
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def test_re_evaluate_report_recomputes_embedded_identity_and_qualification() -> None:
    result = evaluate_report_payload(
        _report(), source_report_file_sha256=hashlib.sha256(b"report").hexdigest()
    )
    assert result["source_report_embedded_sha256_valid"] is True
    assert result["analysis_version"] == "mage-small-encoder-shadow-analysis-v3"
    assert result["verdict"] == "SHADOW_QUALIFIED_FOR_NEXT_CANARY_ONLY"
    assert result["qualification"]["qualified"] is True
    assert result["qualification"]["gates"]["source_report_identity_valid"] is True
    assert result["evaluator"]["version"] == "mage-small-encoder-shadow-evaluator-v3"
    assert len(result["evaluator"]["source_sha256"]) == 64
    assert len(result["analysis_sha256"]) == 64


def test_re_evaluate_report_rejects_missing_segments() -> None:
    report = _report()
    report["segments"] = []
    with pytest.raises(ValueError, match="non-empty segments"):
        evaluate_report_payload(report)


def test_re_evaluate_report_preserves_invalid_embedded_identity() -> None:
    report = _report()
    report["report_sha256"] = "0" * 64
    result = evaluate_report_payload(report)
    assert result["source_report_embedded_sha256_valid"] is False
    assert result["qualification"]["qualified"] is False
    assert result["qualification"]["gates"]["source_report_identity_valid"] is False
    assert result["verdict"] == "INVALID_SOURCE_REPORT_IDENTITY"
