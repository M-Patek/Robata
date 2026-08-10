from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robata.contracts.hashing import semantic_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPOSITORY_ROOT / "docs" / "robata-25x-route-decision-2026-08-09.json"
HUMAN_REPORT_PATH = REPOSITORY_ROOT / "docs" / "ROBATA_25X_ROUTE_DECISION_2026-08-09.md"


def test_route_decision_binds_rfc8785_identity_and_all_evidence() -> None:
    document = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    recorded_semantic_sha256 = document.pop("semantic_sha256")

    assert semantic_sha256(document) == recorded_semantic_sha256
    assert document["production_eligible"] is False
    assert document["decision"]["production_go"] is False

    evidence = document["evidence"]
    assert len(evidence) == 11
    assert len({item["id"] for item in evidence}) == len(evidence)

    for item in evidence:
        artifact_path = (REPOSITORY_ROOT / item["path"]).resolve()
        assert artifact_path.is_relative_to(REPOSITORY_ROOT / "docs")
        assert artifact_path.is_file()
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == item["exact_sha256"]


def test_human_route_decision_keeps_intended_utf8_symbols() -> None:
    report = HUMAN_REPORT_PATH.read_text(encoding="utf-8")

    assert "?" not in report
    assert "\u00d7" in report
    assert "\u2014" in report
    assert "\u2192" in report
