from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robata.benchmark.production_review_candidate_pack import (
    FORMAT,
    ProductionReviewCandidatePackError,
    build_production_review_candidate_pack,
    render_markdown,
)


def _identity_row(action: str = "fold garment", camera_id: str = "cam_01") -> dict[str, object]:
    return {
        "window_id": "w00",
        "ordinal": 0,
        "interval": [0.0, 4.0],
        "camera_id": camera_id,
        "status": "SUCCEEDED",
        "parsed_identity": {
            "parse_status": "PARSED",
            "action": action,
            "confidence": 0.9,
            "evidence": ["hands move the garment inward"],
        },
    }


def _boundary_row(
    *,
    camera_id: str = "cam_01",
    action: tuple[str, str] = ("folding", "garment"),
    mapped: bool = True,
) -> dict[str, object]:
    return {
        "window_id": "w00",
        "ordinal": 0,
        "interval": [0.0, 4.0],
        "camera_id": camera_id,
        "status": "SUCCEEDED",
        "timestamp_mapping_status": "MAPPED" if mapped else "UNMAPPED",
        "segments": [
            {
                "start_time_sec": 1.0 if mapped else 0.0,
                "end_time_sec": 2.0 if mapped else 0.0,
                "timestamp_basis": "source_absolute_seconds"
                if mapped
                else "window_relative_seconds",
                "timestamp_mapping_status": "MEASURED" if mapped else "UNMAPPED",
                "structured_labels": {
                    "verb": {"value": action[0], "status": "MEASURED"},
                    "noun": {"value": action[1], "status": "MEASURED"},
                    "attributes": {"value": "blue", "status": "MEASURED"},
                    "location": {"value": "table", "status": "MEASURED"},
                    "hand": {"value": "both hands", "status": "MEASURED"},
                },
                "confidence": 0.8,
                "evidence": ["edges move inward"],
            }
        ],
    }


def _wemm() -> dict[str, object]:
    return {
        "format": "robata-production-wemm-vocabulary-shadow-v1",
        "source": {"path": "sample.mcap"},
        "windows": [
            {
                "window_id": "w00",
                "ordinal": 0,
                "start_seconds": 0.0,
                "end_seconds": 4.0,
                "model": {
                    "status": "SUCCEEDED",
                    "predictions": [
                        {
                            "rank": 1,
                            "label_text": "fold garment",
                            "verb": "fold",
                            "noun": "garment",
                            "score": 0.91,
                        },
                        {
                            "rank": 2,
                            "label_text": "take cloth",
                            "verb": "take",
                            "noun": "cloth",
                            "score": 0.9,
                        },
                    ],
                },
            }
        ],
    }


def test_pack_separates_dimensions_and_keeps_unmapped_wemm() -> None:
    report = build_production_review_candidate_pack(
        {"windows": [_identity_row(), _identity_row(camera_id="cam_02")]},
        {"windows": [_boundary_row(), _boundary_row(camera_id="cam_02", mapped=False)]},
        _wemm(),
        expected_camera_count=2,
    )
    assert report["format"] == FORMAT
    assert report["official_gold_status"] == "NOT_ESTABLISHED"
    assert report["production_eligible"] is False
    window = report["windows"][0]
    assert window["status"] == "REVIEW_REQUIRED"
    candidate = window["annotation_candidates"][0]
    assert candidate["label_text"] == "fold garment"
    assert candidate["start_seconds"] == 1.0
    assert candidate["end_seconds"] == 2.0
    # Production consumers use the *_time_sec names while the sidecar
    # contract historically exposed *_seconds.  The pack deliberately keeps
    # both aliases in lockstep so downstream readers do not need to guess.
    assert candidate["start_time_sec"] == candidate["start_seconds"] == 1.0
    assert candidate["end_time_sec"] == candidate["end_seconds"] == 2.0
    assert candidate["sources"]["qwen_identity"]["camera_support"] == 2
    assert candidate["sources"]["qwen_boundary"]["measured_camera_support"] == 1
    assert window["dimensions"]["identity"]["agreement_fraction"] == 1.0
    assert window["dimensions"]["boundary"]["measurement_rate"] == 0.5
    assert window["dimensions"]["semantic"]["status"] == "NOT_CHECKED"
    top_k = window["model_context"]["wemm"]["top_k"]
    assert top_k[0]["mapped_action"] == "fold garment"
    assert top_k[1]["mapped_action"] is None
    assert top_k[1]["mapping_status"] == "UNMAPPED_EPIC_OR_FOREIGN_LABEL"
    assert window["annotation_candidates"][0]["field_status"]["attributes"] == "MEASURED"


def test_missing_boundary_keeps_times_null_and_is_reviewable() -> None:
    report = build_production_review_candidate_pack(
        {"windows": [_identity_row()]},
        {"windows": []},
        None,
    )
    candidate = report["windows"][0]["annotation_candidates"][0]
    assert candidate["start_seconds"] is None
    assert candidate["end_seconds"] is None
    assert candidate["start_time_sec"] is None
    assert candidate["end_time_sec"] is None
    assert candidate["boundary_status"] == "NOT_MEASURED"
    assert "BOUNDARY_SUPPORT_LT_2_CAMERAS" in candidate["reason_codes"]
    assert report["windows"][0]["dimensions"]["timestamp"]["status"] == "NOT_MEASURED"


def test_frame_ordinal_boundary_is_explicitly_projected_to_source_clock() -> None:
    identity_row = _identity_row(action="spread garment")
    identity_row["interval"] = [4.0, 8.0]
    identity = {"windows": [identity_row]}
    boundary = {
        "windows": [
            {
                "window_id": "w00",
                "ordinal": 0,
                "interval": [4.0, 8.0],
                "camera_id": "cam_01",
                "coordinate_mode": "sampled_frame_ordinal",
                "identity_context": {"action": "spread garment"},
                "parsed_boundary": {
                    "coordinate_mode": "sampled_frame_ordinal",
                    "timestamp_mapping_status": "MAPPED_FROM_FRAME_ORDINAL",
                    "mapped_timestamp_basis": "window_relative_seconds",
                    "start_frame_ordinal": 1,
                    "end_frame_ordinal": 3,
                    "start_time_sec": 0.5,
                    "end_time_sec": 2.0,
                    "confidence": 0.9,
                    "evidence": "visible spreading",
                },
            }
        ]
    }
    report = build_production_review_candidate_pack(identity, boundary, None)
    candidate = report["windows"][0]["annotation_candidates"][0]
    assert candidate["label_text"] == "spread garment"
    assert candidate["start_seconds"] == 4.5
    assert candidate["end_seconds"] == 6.0
    assert candidate["timestamp_basis"] == "source_absolute_seconds"
    projection = candidate["raw_identity_observations"][0]["raw_boundary"]
    assert projection["parsed_boundary"]["timestamp_mapping_status"] == (
        "MAPPED_FROM_FRAME_ORDINAL"
    )
    assert candidate["boundary_status"] == "MEASURED"


def test_plain_relative_boundary_is_not_projected_without_frame_contract() -> None:
    identity = {"windows": [_identity_row(action="spread garment")]}
    boundary = {
        "windows": [
            {
                "window_id": "w00",
                "interval": [4.0, 8.0],
                "camera_id": "cam_01",
                "parsed_boundary": {
                    "timestamp_basis": "window_relative_seconds",
                    "start_time_sec": 0.5,
                    "end_time_sec": 2.0,
                    "confidence": 0.9,
                    "structured_labels": {"verb": "spread", "noun": "garment"},
                },
            }
        ]
    }
    report = build_production_review_candidate_pack(identity, boundary, None)
    candidate = report["windows"][0]["annotation_candidates"][0]
    assert candidate["start_seconds"] is None
    assert candidate["end_seconds"] is None


def test_rejects_gold_key_in_model_sidecar() -> None:
    with pytest.raises(ProductionReviewCandidatePackError, match="gold"):
        build_production_review_candidate_pack(
            {"windows": [_identity_row()]},
            {"windows": []},
            {"windows": [], "gold": {"status": "ACCEPTED"}},
        )


def test_foreign_noun_alias_is_not_projected_to_production_label() -> None:
    wemm = _wemm()
    wemm["windows"][0]["model"]["predictions"][0]["label_text"] = "fold cloth"  # type: ignore[index]
    wemm["windows"][0]["model"]["predictions"][0]["noun"] = "cloth"  # type: ignore[index]
    report = build_production_review_candidate_pack(
        {"windows": [_identity_row()]},
        {"windows": []},
        wemm,
    )
    top_k = report["windows"][0]["model_context"]["wemm"]["top_k"]
    assert top_k[0]["mapped_action"] is None
    assert top_k[0]["mapping_status"] == "UNMAPPED_EPIC_OR_FOREIGN_LABEL"


def test_markdown_and_cli_smoke(tmp_path: Path) -> None:
    identity = tmp_path / "identity.json"
    boundary = tmp_path / "boundary.json"
    wemm = tmp_path / "wemm.json"
    output = tmp_path / "pack.json"
    identity.write_text(json.dumps({"windows": [_identity_row()]}), encoding="utf-8")
    boundary.write_text(json.dumps({"windows": [_boundary_row()]}), encoding="utf-8")
    wemm.write_text(json.dumps(_wemm()), encoding="utf-8")
    script = Path(__file__).parents[2] / "scripts" / "build_production_review_candidate_pack.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--identity",
            str(identity),
            "--boundary",
            str(boundary),
            "--wemm",
            str(wemm),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT
    assert "Dimensions" in output.with_suffix(".md").read_text(encoding="utf-8")
    assert "review-only" in render_markdown(payload)
