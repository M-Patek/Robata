from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from scripts.build_mage_traditional_generation_qualification import (
    EvidenceError,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "mage-traditional-codec-generation-qualification-2026-08-09.json"
REPORT_EXACT_SHA256 = "f1b5db9d838152f8ebd775e5d68c62139421556456e60f91caa1686e061cd910"
REPORT_SEMANTIC_SHA256 = "3d2bc915a5cb7247193f6503a5148f2bb4d6ec8ef29bbbaad846e76f76d8972e"


def _report() -> dict[str, object]:
    value = json.loads(REPORT.read_bytes())
    assert isinstance(value, dict)
    return value


def _rehash(report: dict[str, object]) -> None:
    report.pop("semantic_sha256", None)
    report["semantic_sha256"] = semantic_sha256(report)


def test_tracked_report_is_canonical_and_preserves_local_decisions() -> None:
    report = _report()

    validate(report)
    assert REPORT.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_EXACT_SHA256
    assert report["semantic_sha256"] == REPORT_SEMANTIC_SHA256
    assert report["production_eligible"] is False
    assert report["decision"] == {
        "state": "HOLD_TRADITIONAL",
        "traditional_target_canvas_8": "HOLD",
        "traditional_target_canvas_16": "STOP",
        "dcvc_role": "RETAINED_CONTROL",
        "next_gate": "P20_COMPACT_DECODER_BUDGET_AND_QUALITY_AB",
        "reasons": report["decision"]["reasons"],
    }

    traditional = report["traditional_target_canvas_8"]["generation"]
    assert traditional["lifecycle"]["model_load_seconds"] == pytest.approx(36.03563010000107)
    assert traditional["lifecycle"]["model_load_included_in_stream_run_wall"] is False
    assert traditional["timing"]["hot_end_to_end_realtime_factor"] == pytest.approx(
        1.2482115702634684
    )
    assert traditional["timing"]["warm_generation_realtime_factor"] == pytest.approx(
        2.3688050290223783
    )
    assert [item["output_tokens"] for item in traditional["per_segment"]] == [203, 33, 37, 41, 43]
    assert [item["generation_seconds"] for item in traditional["per_segment"]] == pytest.approx(
        [
            16.62041059999865,
            2.9260390999988886,
            3.150640999998359,
            3.5496741000006296,
            3.882562999999209,
        ]
    )

    quality16 = report["traditional_target_canvas_16"]
    assert quality16["generation"]["prompt_tokens"] == 1663
    assert quality16["generation"]["output_tokens"] == 256
    assert quality16["generation"]["output_budget_exhausted"] is True
    assert quality16["generation"]["strict_json"] is False
    assert report["capacity_target"]["capacity_unit_conflict"]["resolved"] is False
    assert report["comparison"]["local_logical_lanes_for_25x_camera_target"] == 21
    assert report["comparison"]["local_logical_lanes_for_150x_recording_target"] == 121


def test_validator_rejects_production_eligibility_tamper() -> None:
    report = deepcopy(_report())
    report["production_eligible"] = True
    _rehash(report)

    with pytest.raises(EvidenceError, match="authority"):
        validate(report)


def test_validator_rejects_capacity_formula_tamper() -> None:
    report = deepcopy(_report())
    report["comparison"]["local_logical_lanes_for_25x_camera_target"] = 20
    _rehash(report)

    with pytest.raises(EvidenceError, match="camera-hour lane formula"):
        validate(report)


def test_validator_rejects_quality16_decision_drift() -> None:
    report = deepcopy(_report())
    report["traditional_target_canvas_16"]["generation"]["strict_json"] = True
    _rehash(report)

    with pytest.raises(EvidenceError, match="sixteen-canvas decision"):
        validate(report)
