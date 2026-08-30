from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_annotation_draft import (
    FORMAT,
    ProductionWemmAnnotationDraftError,
    build_wemm_annotation_draft,
    validate_wemm_annotation_draft,
)


def _proposal(
    *,
    label: str = "pick up garment",
    proposal_status: str = "PROPOSED",
    interval_status: str = "NOT_MEASURED",
    start: float | None = None,
    end: float | None = None,
    split_hint: bool = False,
) -> dict[str, object]:
    verb, noun = label.rsplit(" ", 1)
    return {
        "proposal_id": "w00-p01",
        "proposal_status": proposal_status,
        "label_text": label,
        "structured_labels": {
            "verb": verb,
            "noun": noun,
            "attributes": None,
            "location": "on table",
            "hand": "right hand",
        },
        "proposal_interval": {
            "start_seconds": start,
            "end_seconds": end,
            "status": interval_status,
        },
        "confidence": 0.82,
        "evidence": [{"camera_id": "cam_01", "text": "hands lift garment"}],
        "camera_support": ["cam_01", "cam_02"],
        "top_k": [
            {"rank": 1, "label_text": label, "score": 0.82},
            {"rank": 2, "label_text": "fold garment", "score": 0.71},
        ],
        "margin": 0.11,
        "split_hint": split_hint,
    }


def _aggregate(proposals: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format": "robata-production-wemm-review-pack-aggregate-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "production_eligible": False,
        "source": {"recording_count": 1, "camera_count": 6},
        "items": [
            {
                "window_id": "w00",
                "ordinal": 0,
                "recording_id": "recording-01",
                "source_interval": {
                    "start_seconds": 8.0,
                    "end_seconds": 16.0,
                    "status": "WINDOW_CONTEXT_ONLY",
                },
                "window_status": "PROPOSALS_AVAILABLE",
                "window_decision": "pending",
                "proposals": proposals,
                "raw_candidates": [{"label_text": "kept raw"}],
                "source_ref": {"recording_id": "recording-01", "source_path": "sample.mcap"},
                "provenance": {
                    "qa_status": "PASS",
                    "source_preflight_status": "PASS",
                    "review_pack_path": "review.json",
                },
            }
        ],
    }


def test_wemm_proposal_becomes_editable_segment_without_fabricating_window_boundary() -> None:
    result = build_wemm_annotation_draft(_aggregate([_proposal()]))

    assert result["format"] == FORMAT
    assert result["production_eligible"] is False
    assert result["controls"]["gold_written"] is False
    assert result["review_contract"]["provenance_fields"] == [
        "qa_status",
        "source_preflight_status",
        "review_pack_path",
        "archive_member",
        "source_path",
    ]
    assert set(result["review_contract"]["required_fields"]) == {
        "start_seconds",
        "end_seconds",
        "verb",
        "noun",
        "attributes",
        "location",
        "hand",
        "confidence",
        "evidence",
        "camera_support",
        "top_k",
        "margin",
    }
    assert result["review_contract"]["status_fields"][-2:] == ["decision", "split_hint"]
    window = result["windows"][0]
    assert window["status"] == "PROVISIONAL"
    assert window["decision"] == "pending"
    assert window["source_interval"] == {
        "start_seconds": 8.0,
        "end_seconds": 16.0,
        "status": "WINDOW_CONTEXT_ONLY",
        "is_action_boundary": False,
    }
    segment = window["annotation_draft"]["segments"][0]
    assert segment["editable"] is True
    assert segment["start_seconds"] is None
    assert segment["end_seconds"] is None
    assert segment["boundary_status"] == "NOT_MEASURED"
    assert segment["window_context"]["is_action_boundary"] is False
    assert segment["verb"] == "pick up"
    assert segment["noun"] == "garment"
    assert segment["location"] == "on table"
    assert segment["camera_support"] == ["cam_01", "cam_02"]
    assert segment["decision"] == "pending"
    assert segment["automatic_eligible"] is False
    assert segment["review_required"] is True
    assert len(segment["top_k"]) == 2
    assert segment["margin"] == pytest.approx(0.11)
    assert result["metrics"]["unmeasured_boundary_segment_count"] == 1
    assert result["metrics"]["qa_status_counts"] == {"PASS": 1}
    assert result["metrics"]["source_preflight_status_counts"] == {"PASS": 1}
    assert result["windows"][0]["source_provenance"]["review_pack_path"] == "review.json"
    assert result["windows"][0]["source_provenance"]["archive_member"] is None


def test_explicit_source_bound_proposal_interval_is_retained() -> None:
    proposal = _proposal(
        interval_status="SOURCE_BOUND",
        start=9.25,
        end=10.5,
    )
    result = build_wemm_annotation_draft(_aggregate([proposal]))
    segment = result["windows"][0]["annotation_draft"]["segments"][0]
    assert segment["start_seconds"] == pytest.approx(9.25)
    assert segment["end_seconds"] == pytest.approx(10.5)
    assert segment["boundary_status"] == "SOURCE_BOUND"
    assert result["metrics"]["measured_boundary_segment_count"] == 1


def test_unknown_abstain_and_split_are_first_class() -> None:
    unknown = _aggregate([])
    unknown["items"][0]["window_status"] = "UNKNOWN"  # type: ignore[index]
    result = build_wemm_annotation_draft(unknown)
    assert result["windows"][0]["status"] == "UNKNOWN"
    assert result["windows"][0]["unknown"] is True
    assert result["windows"][0]["annotation_draft"]["segments"] == []

    abstain = _aggregate([])
    abstain["items"][0]["window_status"] = "ABSTAIN"  # type: ignore[index]
    result = build_wemm_annotation_draft(abstain)
    assert result["windows"][0]["status"] == "ABSTAIN"
    assert result["windows"][0]["abstain"] is True

    split = _aggregate(
        [
            _proposal(proposal_status="SPLIT", split_hint=True),
            {
                **_proposal(label="fold garment"),
                "proposal_id": "w00-p02",
                "proposal_status": "SPLIT",
                "split_hint": True,
            },
        ]
    )
    split_result = build_wemm_annotation_draft(split)
    split_window = split_result["windows"][0]
    assert split_window["status"] == "SPLIT_REVIEW"
    assert split_window["split_requested"] is True
    assert len(split_window["annotation_draft"]["segments"]) == 2
    assert split_result["metrics"]["split_review_window_count"] == 1


def test_unknown_proposal_is_retained_as_candidate_context() -> None:
    payload = _aggregate([_proposal(proposal_status="UNKNOWN")])
    payload["items"][0]["window_status"] = "UNKNOWN"  # type: ignore[index]
    result = build_wemm_annotation_draft(payload)
    window = result["windows"][0]
    assert window["status"] == "UNKNOWN"
    assert window["annotation_draft"]["segments"] == []
    assert len(window["candidate_proposals"]) == 1
    assert window["candidate_proposals"][0]["proposal_status"] == "UNKNOWN"


def test_mapping_camera_support_is_normalized() -> None:
    proposal = _proposal()
    proposal["camera_support"] = {"cam_02": {}, "cam_01": {}}
    result = build_wemm_annotation_draft(_aggregate([proposal]))
    assert result["windows"][0]["annotation_draft"]["segments"][0]["camera_support"] == [
        "cam_01",
        "cam_02",
    ]


def test_aggregate_recording_source_metadata_is_attached() -> None:
    payload = _aggregate([_proposal()])
    payload["recordings"] = [
        {
            "recording_id": "recording-01",
            "source": {
                "archive_member": "file/recording-01.mcap",
                "archive_path": "source.zip",
                "source": {
                    "camera_count": 6,
                    "media_type": "application/x-mcap",
                    "path_lifecycle": "STAGED_PATH_REMOVED_AFTER_RECORDING",
                },
            },
        }
    ]
    result = build_wemm_annotation_draft(payload)
    window = result["windows"][0]
    assert window["recording_source"]["archive_path"] == "source.zip"
    assert window["recording_source"]["camera_count"] == 6
    assert window["source_provenance"]["archive_path"] == "source.zip"
    assert window["source_ref"]["archive_path"] == "source.zip"


def test_gold_payload_is_rejected_and_input_is_not_mutated() -> None:
    payload = _aggregate([_proposal()])
    before = json.loads(json.dumps(payload))
    payload["items"][0]["gold"] = {"status": "PENDING"}  # type: ignore[index]
    with pytest.raises(ProductionWemmAnnotationDraftError, match=r"gold|official"):
        build_wemm_annotation_draft(payload)
    # The converter never mutates the valid portion of the input before the
    # provenance error is raised.
    assert payload["items"][0]["proposals"] == before["items"][0]["proposals"]  # type: ignore[index]


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    source = tmp_path / "aggregate.json"
    output = tmp_path / "draft.json"
    markdown = tmp_path / "draft.md"
    source.write_text(json.dumps(_aggregate([_proposal()])), encoding="utf-8")
    script = Path(__file__).parents[2] / "scripts" / "build_production_wemm_annotation_draft.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            "--output",
            str(output),
            "--output-md",
            str(markdown),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT
    assert "not production gold" in markdown.read_text(encoding="utf-8")


def test_validator_checks_context_boundary_top_k_and_provenance() -> None:
    result = build_wemm_annotation_draft(_aggregate([_proposal()]))
    detached = validate_wemm_annotation_draft(result)
    assert detached == result

    broken = json.loads(json.dumps(result))
    broken["windows"][0]["annotation_draft"]["segments"][0]["start_seconds"] = 8.0
    with pytest.raises(ProductionWemmAnnotationDraftError, match="context-only"):
        validate_wemm_annotation_draft(broken)

    broken = json.loads(json.dumps(result))
    del broken["windows"][0]["source_provenance"]["qa_status"]
    with pytest.raises(ProductionWemmAnnotationDraftError, match="source_provenance"):
        validate_wemm_annotation_draft(broken)


def test_validator_cli_accepts_real_aggregate_draft_when_available(tmp_path: Path) -> None:
    aggregate = Path(".agent_tmp/production_wemm_full_postprocess_20260828/review_aggregate.json")
    if not aggregate.exists():
        pytest.skip("local production aggregate is not available")
    draft = tmp_path / "draft.json"
    build = Path(__file__).parents[2] / "scripts" / "build_production_wemm_annotation_draft.py"
    built = subprocess.run(
        [sys.executable, str(build), str(aggregate), "--output", str(draft)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    audit = Path(__file__).parents[2] / "scripts" / "audit_production_wemm_annotation_draft.py"
    checked = subprocess.run(
        [sys.executable, str(audit), str(draft)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    summary = json.loads(checked.stdout)
    assert summary["status"] == "VALID"
    assert summary["windows"] == 788
