from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from scripts.build_mage_temporal_memory_qualification import (
    TemporalMemoryEvidenceError,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "mage-temporal-memory-qualification-2026-08-09.json"
REPORT_EXACT_SHA256 = "9e06dba340f3b51fcf385811cd4b10cb0442a4680d97570a284576bf0c1be6dd"
REPORT_SEMANTIC_SHA256 = "1f3d020be441e1b87ddede0754959494c09b72b452770db5b00c64a6890216a8"


def _report() -> dict[str, object]:
    value = json.loads(REPORT.read_bytes())
    assert isinstance(value, dict)
    return value


def _rehash(report: dict[str, object]) -> None:
    report.pop("semantic_sha256", None)
    report["semantic_sha256"] = semantic_sha256(report)


def test_temporal_memory_report_is_canonical_and_rejects_candidate() -> None:
    report = _report()

    validate(report)
    assert REPORT.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_EXACT_SHA256
    assert report["semantic_sha256"] == REPORT_SEMANTIC_SHA256
    assert report["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert report["production_eligible"] is False
    assert report["decision"] == {
        "mage_route": "HOLD",
        "qwen_batch_hedge": "ACTIVATE",
        "reason": (
            "Temporal memory improves local wall time only by collapsing the action sequence; "
            "it fails semantic non-inferiority and cannot contribute accepted capacity."
        ),
        "rollback": "Use full-v1 8x131K control; do not select temporal-memory-v1.",
        "temporal_memory_v1": "REJECT",
    }
    assert report["quality"]["candidate_action_repeat_count"] == 5
    assert report["quality"]["candidate_distinct_action_count"] == 1
    assert report["comparison"]["control_local_lanes_for_25x"] == 16
    assert report["comparison"]["candidate_unaccepted_lanes_for_25x"] == 14
    assert report["comparison"]["stream_wall_reduction_percent"] == pytest.approx(
        12.988538002881727
    )
    assert report["comparison"]["prompt_token_increase_percent"] == pytest.approx(
        19.522633744855966
    )
    assert report["comparison"]["peak_vram_increase_mib"] == 668
    assert report["candidate"]["durability"]["memory_artifact_count"] == 5
    assert report["candidate"]["durability"]["memory_link_artifact_count"] == 5


def test_validator_rejects_temporal_candidate_promotion() -> None:
    report = deepcopy(_report())
    report["decision"]["temporal_memory_v1"] = "GO"
    _rehash(report)

    with pytest.raises(TemporalMemoryEvidenceError, match="must remain rejected"):
        validate(report)


def test_validator_rejects_quality_disposition_tamper() -> None:
    report = deepcopy(_report())
    report["quality"]["disposition"] = "ACCEPT"
    _rehash(report)

    with pytest.raises(TemporalMemoryEvidenceError, match="quality disposition"):
        validate(report)


def test_validator_rejects_capacity_formula_tamper() -> None:
    report = deepcopy(_report())
    report["comparison"]["candidate_unaccepted_lanes_for_25x"] = 13
    _rehash(report)

    with pytest.raises(TemporalMemoryEvidenceError, match="lane formula"):
        validate(report)
