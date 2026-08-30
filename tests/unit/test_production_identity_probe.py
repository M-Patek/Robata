from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robata.benchmark.production_identity_probe import (
    ProductionIdentityProbeError,
    evaluate_production_identity_probe,
    project_identity_pending_candidates,
    render_markdown,
)


def _row(
    window_id: str,
    *,
    action: str = "pick up garment",
    confidence: object = 0.9,
    evidence: object = "A hand grasps the garment.",
    parsed: str = "PARSED",
    native: bool = True,
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "status": "SUCCEEDED",
        "input_mode": "native_video",
        "native_video_complete": native,
        "visual_input": {
            "content_sequence": ["video", "instruction"],
            "processor_tensor_shapes": {
                "input_ids": [1, 10],
                "pixel_values_videos": [1280, 1536],
            },
        },
        "parsed_identity": {
            "parse_status": parsed,
            "action": action,
            "confidence": confidence,
            "evidence": evidence,
            "errors": [],
            "warnings": [],
        },
    }


def _sidecar(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "status": "SUCCEEDED",
        "model": {
            "identifier": "Qwen3-VL-4B-Instruct",
            "label_profile": "production_identity_only",
            "prompt_version": "qwen-production-identity-only-v1",
            "native_route": "complete_native_video",
        },
        "windows": list(rows),
    }


def _owner_reference() -> dict[str, object]:
    return {
        "format": "robata-production-owner-confirmation-v1",
        "status": "OWNER_CONFIRMED_SCOPED_NON_GOLD",
        "official_gold_status": "NOT_ESTABLISHED",
        "human_adjudication": "NOT_PERFORMED",
        "production_eligible": False,
        "official_gold": False,
        "accepted_as_gold": False,
        "windows": [
            {
                "window_id": "w00",
                "decision": "accept",
                "segments": [{"verb": "pick up", "noun": "garment"}],
            },
            {
                "window_id": "w01",
                "decision": "abstain",
                "segments": [],
            },
        ],
    }


def test_identity_probe_reports_contract_metrics_without_running_model() -> None:
    report = evaluate_production_identity_probe(
        _sidecar(
            _row("w00", action="pick up garment"),
            _row("w01", action="uncertain"),
        )
    )

    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["quality_claim"] is False
    assert report["metrics"]["window_count"] == 2
    assert report["metrics"]["parse"]["parsed"] == 2
    assert report["metrics"]["parse"]["rate"] == 1.0
    assert report["metrics"]["action"]["recognized"] == 2
    assert report["metrics"]["action"]["positive"] == 1
    assert report["metrics"]["action"]["abstentions"] == 1
    assert report["metrics"]["evidence"]["nonempty"] == 2
    assert report["metrics"]["confidence"]["valid_count"] == 2
    assert report["metrics"]["confidence"]["mean"] == pytest.approx(0.9)
    assert report["metrics"]["native_completeness"]["complete"] == 2
    assert report["reference_overlap"]["status"] == "NOT_SUPPLIED"
    assert report["controls"]["model_invoked"] is False
    assert report["controls"]["media_decoded"] is False
    assert report["controls"]["gold_written"] is False


def test_identity_probe_separates_invalid_fields_and_native_metadata() -> None:
    report = evaluate_production_identity_probe(
        _sidecar(
            _row("w00"),
            _row(
                "w01",
                action="not an allowed action",
                confidence=2.0,
                evidence="",
                parsed="INVALID",
                native=False,
            ),
        )
    )
    metrics = report["metrics"]
    assert metrics["parse"]["parsed"] == 1
    assert metrics["parse"]["invalid"] == 1
    assert metrics["action"]["recognized"] == 1
    assert metrics["action"]["invalid_or_missing"] == 1
    assert metrics["evidence"]["nonempty"] == 1
    assert metrics["confidence"]["valid_count"] == 1
    assert metrics["native_completeness"]["complete"] == 1
    assert metrics["native_completeness"]["complete_flag"] == 1


def test_owner_overlap_is_explicitly_surrogate_and_never_gold() -> None:
    report = evaluate_production_identity_probe(
        _sidecar(_row("w00", action="pick up garment"), _row("w01", action="uncertain")),
        _owner_reference(),
    )
    overlap = report["reference_overlap"]
    assert overlap["status"] == "SURROGATE_NON_GOLD"
    assert overlap["measurement_status"] == "SURROGATE_NON_GOLD"
    assert overlap["official_quality_status"] == "NOT_MEASURED"
    assert overlap["official_gold"] is False
    assert overlap["eligible_window_count"] == 1
    assert overlap["exact_action_hits"] == 1
    assert overlap["exact_action_rate"] == 1.0
    assert overlap["primary_action_hits"] == 1
    assert overlap["primary_action_rate"] == 1.0
    assert report["quality_claim"] is False
    assert "SURROGATE_NON_GOLD" in render_markdown(report)


def test_owner_overlap_honors_recommendation_alias_for_abstain_windows() -> None:
    reference = _owner_reference()
    reference["windows"] = [
        {
            "window_id": "w00",
            "recommendation": "EDIT",
            "segments": [{"verb": "pick up", "noun": "garment"}],
        },
        {
            "window_id": "w01",
            "recommendation": "ABSTAIN",
            "segments": [{"verb": "fold", "noun": "garment"}],
        },
    ]
    report = evaluate_production_identity_probe(
        _sidecar(_row("w00", action="pick up garment"), _row("w01", action="fold garment")),
        reference,
    )
    overlap = report["reference_overlap"]
    assert overlap["eligible_window_count"] == 1
    assert overlap["per_window"][0]["window_id"] == "w00"


def test_owner_overlap_does_not_treat_omitted_decision_as_accept() -> None:
    reference = _owner_reference()
    reference["windows"] = [
        {
            "window_id": "w00",
            "segments": [{"verb": "pick up", "noun": "garment"}],
        }
    ]
    report = evaluate_production_identity_probe(_sidecar(_row("w00")), reference)
    overlap = report["reference_overlap"]
    assert overlap["eligible_window_count"] == 0
    assert overlap["exact_action_hits"] == 0
    assert overlap["prediction_coverage"] is None


def test_owner_reference_gold_claim_is_rejected() -> None:
    reference = _owner_reference()
    reference["official_gold_status"] = "ESTABLISHED"
    with pytest.raises(ProductionIdentityProbeError, match="non-gold"):
        evaluate_production_identity_probe(_sidecar(_row("w00")), reference)


def test_identity_probe_rejects_wrong_profile() -> None:
    sidecar = _sidecar(_row("w00"))
    sidecar["model"] = {"label_profile": "production_coarse"}
    with pytest.raises(ProductionIdentityProbeError, match="label_profile"):
        evaluate_production_identity_probe(sidecar)


def test_identity_probe_accepts_disambiguated_profile_and_multi_camera_rows() -> None:
    first = _row("w00", action="pick up garment")
    second = _row("w00", action="spread garment")
    first["camera_id"] = "cam_01"
    second["camera_id"] = "cam_02"
    sidecar = _sidecar(first, second)
    sidecar["model"] = {"label_profile": "production_identity_disambiguated"}
    report = evaluate_production_identity_probe(sidecar, _owner_reference())
    assert report["source"]["window_count"] == 2
    assert report["source"]["unique_window_count"] == 1
    assert report["source"]["camera_count"] == 2
    overlap = report["reference_overlap"]
    assert overlap["eligible_window_count"] == 1
    assert overlap["exact_action_hits"] == 1
    assert overlap["per_window"][0]["predicted_actions"] == [
        "pick up garment",
        "spread garment",
    ]
    assert overlap["per_window"][0]["primary_action_tie"] is True
    assert overlap["primary_action_hits"] == 0


def test_identity_projection_emits_only_clean_pending_candidates_and_preserves_raw() -> None:
    clean = _row("w00", action="pick up garment")
    clean["camera_id"] = "cam_01"
    uncertain = _row("w01", action="uncertain")
    uncertain["camera_id"] = "cam_01"
    incomplete = _row("w02", action="fold garment", native=False)
    incomplete["camera_id"] = "cam_01"
    sidecar = _sidecar(clean, uncertain, incomplete)

    report = project_identity_pending_candidates(sidecar, input_path="identity.json")

    assert report["status"] == "REVIEW_REQUIRED"
    assert report["production_eligible"] is False
    assert report["automatic_eligible"] is False
    assert report["contract"]["boundaries_measured"] is False
    assert report["controls"]["model_invoked"] is False
    assert report["raw_sidecar"] == sidecar
    assert report["metrics"]["annotation_candidate_count"] == 1
    assert report["metrics"]["rejected_identity_row_count"] == 2

    windows = {item["window_id"]: item for item in report["windows"]}
    candidate = windows["w00"]["annotation_candidates"][0]
    assert candidate["status"] == "PENDING_HUMAN_REVIEW"
    assert candidate["automatic_eligible"] is False
    assert candidate["verb"] == "pick up"
    assert candidate["noun"] == "garment"
    assert candidate["start_seconds"] is None
    assert candidate["end_seconds"] is None
    assert candidate["boundary_status"] == "NOT_MEASURED"
    assert "IDENTITY_ONLY_NO_BOUNDARY" in candidate["reason_codes"]
    assert windows["w01"]["annotation_candidates"] == []
    assert "IDENTITY_UNCERTAIN" in windows["w01"]["rejected_identity_rows"][0]["reason_codes"]
    assert "NATIVE_VIDEO_INCOMPLETE" in windows["w02"]["rejected_identity_rows"][0]["reason_codes"]


def test_identity_projection_abstains_when_no_clean_rows_exist() -> None:
    report = project_identity_pending_candidates(_sidecar(_row("w00", action="uncertain")))
    assert report["status"] == "ABSTAIN"
    assert report["metrics"]["annotation_candidate_count"] == 0
    assert report["metrics"]["abstained_window_count"] == 1


def test_identity_projection_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "identity.json"
    output = tmp_path / "projection.json"
    output_md = tmp_path / "projection.md"
    source.write_text(json.dumps(_sidecar(_row("w00"))), encoding="utf-8")
    script = (
        Path(__file__).parents[2] / "scripts" / "project_production_qwen_identity_candidates.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--sidecar",
            str(source),
            "--output-json",
            str(output),
            "--output-md",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"].endswith("identity-candidate-projection-v1")
    assert payload["production_eligible"] is False
    assert payload["metrics"]["annotation_candidate_count"] == 1
    assert "pending" in output_md.read_text(encoding="utf-8").casefold()
