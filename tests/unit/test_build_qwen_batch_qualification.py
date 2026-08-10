from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from scripts.build_qwen_batch_qualification import (
    QwenBatchQualificationError,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "qwen-native-batch-qualification-2026-08-09.json"
REPORT_EXACT_SHA256 = "0844c9b7c43bf1db7396977c91eb8575755a971ebf2c1d5b28310fff641084fc"
REPORT_SEMANTIC_SHA256 = "319f334ba4def2cd5446ad0000f0726ee1c947b3cce914b984c7915928f4b956"


def _report() -> dict[str, object]:
    value = json.loads(REPORT.read_bytes())
    assert isinstance(value, dict)
    return value


def _rehash(report: dict[str, object]) -> None:
    report.pop("semantic_sha256", None)
    report["semantic_sha256"] = semantic_sha256(report)


def test_tracked_qwen_batch_report_is_canonical_and_nonproduction() -> None:
    report = _report()

    validate(report)
    assert REPORT.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_EXACT_SHA256
    assert report["semantic_sha256"] == REPORT_SEMANTIC_SHA256
    assert report["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert report["production_eligible"] is False
    assert report["decision"]["batch_candidate"] == (
        "ACCEPT_BATCH4_HYBRID_FOR_VERSIONED_LOCAL_ENDPOINT_INTEGRATION"
    )
    assert report["decision"]["qwen_route_state"] == "HOLD_FULL_PRODUCTION_QUALIFICATION"


def test_hybrid_is_repeatable_exact_and_reduces_recurring_wall() -> None:
    report = _report()
    runs = report["runs"]["batch4_hybrid_recomputations"]

    assert len(runs) == 2
    assert [run["quality"]["normalized_exact_match_count"] for run in runs] == [51, 51]
    assert [run["quality"]["raw_exact_match_count"] for run in runs] == [51, 51]
    assert [run["quality"]["quality_gate_pass"] for run in runs] == [True, True]
    assert report["comparison"]["hybrid_speedup_median"] == pytest.approx(3.7810767980162585)
    assert report["comparison"]["hybrid_wall_reduction_fraction"] == pytest.approx(
        0.7355250756809145
    )
    assert report["comparison"]["hybrid_local_equivalent_lanes_for_25x_camera_hours"] == 7


def test_all_native_candidates_remain_quality_gated() -> None:
    report = _report()

    assert report["runs"]["naive_batch2"]["quality"]["normalized_exact_match_count"] == 50
    assert report["runs"]["grouped_batch4"]["quality"]["normalized_exact_match_count"] == 50
    assert report["runs"]["grouped_batch8"]["quality"]["normalized_exact_match_count"] == 49
    assert (
        report["runs"]["grouped_batch8"]["execution"]["wall_seconds"]
        > report["runs"]["grouped_batch4"]["execution"]["wall_seconds"]
    )


def test_validator_rejects_production_promotion() -> None:
    report = deepcopy(_report())
    report["production_eligible"] = True
    _rehash(report)

    with pytest.raises(QwenBatchQualificationError, match="non-production"):
        validate(report)


def test_validator_rejects_lane_formula_tamper() -> None:
    report = deepcopy(_report())
    report["comparison"]["hybrid_local_equivalent_lanes_for_25x_camera_hours"] = 6
    _rehash(report)

    with pytest.raises(QwenBatchQualificationError, match="25x lane formula"):
        validate(report)


def test_validator_rejects_hybrid_quality_tamper() -> None:
    report = deepcopy(_report())
    report["runs"]["batch4_hybrid_recomputations"][0]["quality"]["quality_gate_pass"] = False
    _rehash(report)

    with pytest.raises(QwenBatchQualificationError, match="exact quality parity"):
        validate(report)
