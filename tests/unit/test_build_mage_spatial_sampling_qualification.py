from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from scripts.build_mage_spatial_sampling_qualification import SpatialEvidenceError, validate

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs" / "mage-spatial-sampling-qualification-2026-08-09.json"
REPORT_EXACT_SHA256 = "5975e82ea9b445bb435c0f3994b56d1d0212bb81518ff68c93768a4f2d95820a"
REPORT_SEMANTIC_SHA256 = "34b7c283b9ac05787b4b7bd912a00097dddb0e592a1ed67f751a297db5501681"


def _report() -> dict[str, object]:
    value = json.loads(REPORT.read_bytes())
    assert isinstance(value, dict)
    return value


def _rehash(report: dict[str, object]) -> None:
    report.pop("semantic_sha256", None)
    report["semantic_sha256"] = semantic_sha256(report)


def test_tracked_spatial_report_is_canonical_and_holds_production() -> None:
    report = _report()

    validate(report)
    assert REPORT.read_bytes() == canonical_json_bytes(report) + b"\n"
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == REPORT_EXACT_SHA256
    assert report["semantic_sha256"] == REPORT_SEMANTIC_SHA256
    assert report["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert report["production_eligible"] is False
    assert report["decision"]["state"] == "HOLD_MAGE_SPATIAL"
    assert report["decision"]["selected_local_profile"] == "traditional_8x131k"
    assert report["six_by_131k"]["state"] == "NOT_RUN_NOT_JUSTIFIED_BY_CURRENT_QUALITY_GATE"

    candidate = report["profiles"]["traditional_8x131k"]
    runs = candidate["fresh_recomputations"]
    assert len(runs) == 2
    assert candidate["semantic_output_recomputation_stable"] is True
    assert {run["generation"]["output_text_sequence_sha256"] for run in runs} == {
        "755ac127a8e617eb1809cc8f9effea3a88705e0f9407852eebff33d0bed1c170"
    }
    assert [run["timing"]["end_to_end_realtime_factor"] for run in runs] == pytest.approx(
        [1.686021163966118, 1.6457420440221628]
    )
    assert report["comparison"]["traditional_8x131k"]["local_lanes_for_25x"] == 16
    assert (
        report["comparison"]["traditional_8x131k"][
            "local_lanes_for_150x_if_six_independent_cameras"
        ]
        == 92
    )


def test_validator_rejects_production_promotion() -> None:
    report = deepcopy(_report())
    report["production_eligible"] = True
    _rehash(report)

    with pytest.raises(SpatialEvidenceError, match="non-production"):
        validate(report)


def test_validator_rejects_recomputation_stability_tamper() -> None:
    report = deepcopy(_report())
    report["profiles"]["traditional_8x131k"]["semantic_output_recomputation_stable"] = False
    _rehash(report)

    with pytest.raises(SpatialEvidenceError, match="not stable"):
        validate(report)


def test_validator_rejects_capacity_formula_tamper() -> None:
    report = deepcopy(_report())
    report["comparison"]["traditional_8x131k"]["local_lanes_for_25x"] = 15
    _rehash(report)

    with pytest.raises(SpatialEvidenceError, match="25x lane formula"):
        validate(report)
