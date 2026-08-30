from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_ambiguity_selector import (
    AMBIGUITY_SELECTION_FORMAT,
    ProductionWemmAmbiguitySelectorError,
    render_markdown,
    select_production_wemm_ambiguities,
)
from robata.benchmark.production_wemm_preannotation import (
    build_preannotation_envelope,
    build_review_pack,
)
from robata.benchmark.production_wemm_review_pack_aggregate import (
    aggregate_production_wemm_review_packs,
)
from scripts.select_production_wemm_ambiguities import main as select_main

CAMERAS = ["cam_01", "cam_02", "cam_03"]


def _candidate(
    label: str,
    verb: str,
    noun: str,
    score: float,
    ranks: dict[str, int],
) -> dict[str, object]:
    return {
        "label_text": label,
        "verb": verb,
        "noun": noun,
        "score": score,
        "evidence": [
            {
                "camera_id": camera,
                "rank": rank,
                "score": score,
                "label_text": label,
            }
            for camera, rank in ranks.items()
        ],
    }


def _source() -> dict[str, object]:
    return {
        "recording_id": "recording-selector-01",
        "path": "recording-selector-01.mcap",
        "archive_path": "production.zip",
        "archive_member": "file/recording-selector-01.mcap",
        "camera_count": len(CAMERAS),
        "camera_ids": CAMERAS,
        "window_count": 3,
        "source_preflight_status": "PASS",
        "qa_status": "PENDING",
    }


def _proposal(
    label: str,
    verb: str,
    *,
    score: float,
    second: dict[str, object] | None = None,
    ranks: dict[str, int] | None = None,
    confidence: float | None = 0.8,
    fields: dict[str, object] | None = None,
    interval: dict[str, object] | None = None,
) -> dict[str, object]:
    labels: dict[str, object] = {
        "verb": verb,
        "noun": "garment",
        "attributes": None,
        "location": None,
        "hand": None,
    }
    if fields:
        labels.update(fields)
    first_ranks = ranks or {camera: 1 for camera in CAMERAS}
    candidates: list[dict[str, object]] = [
        _candidate(label, verb, "garment", score, first_ranks),
    ]
    if second is not None:
        candidates.append(
            _candidate(
                str(second["label_text"]),
                str(second["verb"]),
                str(second.get("noun", "garment")),
                float(second["score"]),
                second.get("ranks", {camera: 2 for camera in CAMERAS}),  # type: ignore[arg-type]
            )
        )
    result: dict[str, object] = {
        "label_text": label,
        "structured_labels": labels,
        "confidence": confidence,
        "camera_support": CAMERAS,
        "evidence": [
            {
                "camera_id": camera,
                "rank": rank,
                "score": score,
                "label_text": label,
            }
            for camera, rank in first_ranks.items()
        ],
        "top_k": candidates,
    }
    if interval is not None:
        result["proposal_interval"] = interval
    return result


def _envelope() -> dict[str, object]:
    return build_preannotation_envelope(
        _source(),
        [
            {
                "window_id": "recording-selector-01-w0000",
                "ordinal": 0,
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "camera_ids": CAMERAS,
                "proposals": [
                    _proposal(
                        "pick up garment",
                        "pick up",
                        score=0.80,
                        second={
                            "label_text": "fold garment",
                            "verb": "fold",
                            "score": 0.795,
                            "ranks": {"cam_01": 2, "cam_02": 2, "cam_03": 1},
                        },
                        ranks={"cam_01": 1, "cam_02": 1, "cam_03": 2},
                    )
                ],
            },
            {
                "window_id": "recording-selector-01-w0001",
                "ordinal": 1,
                "start_seconds": 8.0,
                "end_seconds": 16.0,
                "camera_ids": CAMERAS,
                "proposals": [
                    _proposal(
                        "pick up garment",
                        "pick up",
                        score=0.90,
                        second={
                            "label_text": "fold garment",
                            "verb": "fold",
                            "score": 0.60,
                        },
                        ranks={camera: 1 for camera in CAMERAS},
                    )
                ],
            },
            {
                "window_id": "recording-selector-01-w0002",
                "ordinal": 2,
                "start_seconds": 16.0,
                "end_seconds": 24.0,
                "camera_ids": CAMERAS,
                "proposals": [
                    {
                        "label_text": "unknown garment event",
                        "structured_labels": {
                            "verb": {"status": "NOT_MEASURED"},
                            "noun": {"status": "NOT_MEASURED"},
                        },
                        "confidence": None,
                        "camera_support": [],
                        "evidence": [],
                        "top_k": [],
                    }
                ],
            },
        ],
        model={"name": "WeMM-Embedding-2B", "route": "video_embedding"},
    )


def test_selects_low_margin_camera_conflict_and_keeps_context_only_boundary() -> None:
    envelope = _envelope()
    report = select_production_wemm_ambiguities(
        envelope,
        camera_consensus_threshold=0.8,
        include_recording_edges=False,
        include_adjacent_transitions=False,
    )

    assert report["format"] == AMBIGUITY_SELECTION_FORMAT
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["production_eligible"] is False
    assert report["controls"]["model_invoked"] is False  # type: ignore[index]
    assert report["controls"]["qwen_read"] is False  # type: ignore[index]
    assert report["summary"]["input_window_count"] == 3  # type: ignore[index]
    selected = {row["window_id"]: row for row in report["windows"]}  # type: ignore[index]
    assert set(selected) == {
        "recording-selector-01-w0000",
        "recording-selector-01-w0002",
    }
    reasons = selected["recording-selector-01-w0000"]["reason_codes"]
    assert "LOW_TOP1_TOP2_MARGIN" in reasons
    assert "TOP_K_NEAR_TIE_VERB_CONFLICT" in reasons
    assert "LOW_CAMERA_TOP1_CONSENSUS" in reasons
    assert selected["recording-selector-01-w0000"]["source_interval"]["is_action_boundary"] is False
    assert (
        "SOURCE_INTERVAL_CONTEXT_ONLY"
        in selected["recording-selector-01-w0000"]["observed_reason_codes"]
    )
    # The high-margin/unanimous row is not routed by its systematic optional
    # field and unmeasured proposal-boundary gaps in the default policy.
    assert report["summary"]["unselected_window_count"] == 1  # type: ignore[index]

    # Routing must not discard source review state while compacting the row.
    selected_row = selected["recording-selector-01-w0000"]
    source_window = envelope["windows"][0]  # type: ignore[index]
    assert selected_row["window_status"] == source_window["window_status"]
    assert selected_row["window_decision"] == source_window["window_decision"]
    assert selected_row["raw_candidates"] == source_window["raw_candidates"]
    assert selected_row["review_contract"] == envelope["review_contract"]
    assert report["review_contract"] == envelope["review_contract"]
    assert report["source_contracts"] == [envelope["review_contract"]]

    # Detached copies prevent a caller from mutating the source envelope via
    # the selector result.
    selected_row["raw_candidates"].append({"mutated": True})
    assert len(source_window["raw_candidates"]) < len(selected_row["raw_candidates"])


def test_boundary_and_optional_switches_are_explicit_and_separate() -> None:
    envelope = _envelope()
    baseline = select_production_wemm_ambiguities(
        envelope,
        include_recording_edges=False,
        include_adjacent_transitions=False,
    )
    with_boundary = select_production_wemm_ambiguities(
        envelope,
        include_unmeasured_boundaries=True,
        include_recording_edges=False,
        include_adjacent_transitions=False,
    )
    with_optional = select_production_wemm_ambiguities(
        envelope,
        margin_threshold=0.0,
        include_optional_field_gaps=True,
        include_recording_edges=False,
        include_adjacent_transitions=False,
    )
    baseline_ids = {row["window_id"] for row in baseline["windows"]}  # type: ignore[index]
    boundary_ids = {row["window_id"] for row in with_boundary["windows"]}  # type: ignore[index]
    optional_ids = {row["window_id"] for row in with_optional["windows"]}  # type: ignore[index]
    assert "recording-selector-01-w0001" not in baseline_ids
    assert "recording-selector-01-w0001" in boundary_ids
    assert "recording-selector-01-w0001" in optional_ids
    row = next(
        row for row in with_boundary["windows"] if row["window_id"] == "recording-selector-01-w0001"
    )
    assert "PROPOSAL_BOUNDARY_UNMEASURED" in row["reason_codes"]
    proposal = row["proposal_diagnostics"][0]
    assert proposal["boundary_status"] == "NOT_MEASURED"
    assert proposal["missing_optional_fields"] == ["attributes", "location", "hand"]


def test_review_aggregate_source_ref_and_raw_sidecar_are_preserved() -> None:
    envelope = _envelope()
    original = json.dumps(envelope, sort_keys=True)
    review_pack = build_review_pack(envelope)
    aggregate = aggregate_production_wemm_review_packs([review_pack])
    report = select_production_wemm_ambiguities(
        aggregate,
        camera_consensus_threshold=0.8,
        include_recording_edges=False,
        include_adjacent_transitions=False,
    )
    row = next(
        item
        for item in report["windows"]  # type: ignore[index]
        if item["window_id"] == "recording-selector-01-w0000"
    )
    assert row["source_ref"]["recording_id"] == "recording-selector-01"  # type: ignore[index]
    assert row["source_ref"]["archive_member"] == "file/recording-selector-01.mcap"  # type: ignore[index]
    assert row["provenance"]["input_format"] == aggregate["format"]  # type: ignore[index]
    assert row["raw_top_k_retained_in_source_sidecar"] is True
    assert row["window_status"] == envelope["windows"][0]["window_status"]  # type: ignore[index]
    assert row["window_decision"] == envelope["windows"][0]["window_decision"]  # type: ignore[index]
    assert row["raw_candidates"] == envelope["windows"][0]["raw_candidates"]  # type: ignore[index]
    assert row["review_contract"] == aggregate["review_contract"]
    assert report["review_contract"] == aggregate["review_contract"]
    assert envelope["format"] == "robata-production-wemm-preannotation-v1"
    assert json.dumps(envelope, sort_keys=True) == original
    json.dumps(report)
    assert "ROUTING-ONLY" in render_markdown(report)


def test_edges_and_neighbor_transitions_are_contextual_routing_signals() -> None:
    envelope = _envelope()
    # Alter only the final Top-1 phrase; this is a neighboring-window context
    # change, not a synthesized action boundary.
    final_proposal = envelope["windows"][2]["proposals"][0]  # type: ignore[index]
    final_proposal["label_text"] = "fold garment"  # type: ignore[index]
    final_proposal["structured_labels"]["verb"] = {  # type: ignore[index]
        "value": "fold",
        "status": "MEASURED",
    }
    final_proposal["structured_labels"]["noun"] = {  # type: ignore[index]
        "value": "garment",
        "status": "MEASURED",
    }
    final_proposal["confidence"] = 0.8  # type: ignore[index]
    final_proposal["camera_support"] = CAMERAS  # type: ignore[index]
    final_proposal["evidence"] = [  # type: ignore[index]
        {"camera_id": camera, "rank": 1, "score": 0.8, "label_text": "fold garment"}
        for camera in CAMERAS
    ]
    final_proposal["top_k"] = [  # type: ignore[index]
        _candidate(
            "fold garment",
            "fold",
            "garment",
            0.8,
            {camera: 1 for camera in CAMERAS},
        ),
        _candidate(
            "pick up garment",
            "pick up",
            "garment",
            0.7,
            {camera: 2 for camera in CAMERAS},
        ),
    ]
    report = select_production_wemm_ambiguities(
        envelope,
        include_recording_edges=True,
        include_adjacent_transitions=True,
    )
    ids = {row["window_id"] for row in report["windows"]}  # type: ignore[index]
    assert "recording-selector-01-w0000" in ids  # leading edge
    assert "recording-selector-01-w0002" in ids  # trailing edge/transition
    trailing = next(
        row for row in report["windows"] if row["window_id"] == "recording-selector-01-w0002"
    )
    assert "RECORDING_TRAILING_CONTEXT_EDGE" in trailing["reason_codes"]
    assert "ADJACENT_TOP1_TRANSITION" in trailing["reason_codes"]
    assert trailing["source_context_is_action_boundary"] is False


def test_invalid_policy_is_rejected_and_cli_writes_queue(tmp_path: Path) -> None:
    with pytest.raises(ProductionWemmAmbiguitySelectorError, match="margin_threshold"):
        select_production_wemm_ambiguities(_envelope(), margin_threshold=-1.0)
    input_path = tmp_path / "preannotation.json"
    input_path.write_text(json.dumps(_envelope()), encoding="utf-8")
    output_path = tmp_path / "selection.json"
    assert (
        select_main(
            [
                str(input_path),
                "--output-json",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format"] == AMBIGUITY_SELECTION_FORMAT
    assert payload["summary"]["selected_window_count"] == 2
    assert output_path.with_suffix(".md").exists()


def test_cli_context_switches_are_opt_in_and_legacy_excludes_are_accepted(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "preannotation.json"
    input_path.write_text(json.dumps(_envelope()), encoding="utf-8")

    default_output = tmp_path / "default.json"
    assert select_main([str(input_path), "--output-json", str(default_output)]) == 0
    default_payload = json.loads(default_output.read_text(encoding="utf-8"))
    assert default_payload["policy"]["include_recording_edges"] is False
    assert default_payload["policy"]["include_adjacent_transitions"] is False

    excluded_output = tmp_path / "excluded.json"
    assert (
        select_main(
            [
                str(input_path),
                "--output-json",
                str(excluded_output),
                "--exclude-recording-edges",
                "--exclude-adjacent-transitions",
            ]
        )
        == 0
    )
    excluded_payload = json.loads(excluded_output.read_text(encoding="utf-8"))
    assert excluded_payload["policy"]["include_recording_edges"] is False
    assert excluded_payload["policy"]["include_adjacent_transitions"] is False

    included_output = tmp_path / "included.json"
    assert (
        select_main(
            [
                str(input_path),
                "--output-json",
                str(included_output),
                "--include-recording-edges",
                "--include-adjacent-transitions",
            ]
        )
        == 0
    )
    included_payload = json.loads(included_output.read_text(encoding="utf-8"))
    assert included_payload["policy"]["include_recording_edges"] is True
    assert included_payload["policy"]["include_adjacent_transitions"] is True

    # The compatibility switches must not permit an accidental opt-in when
    # paired with their positive forms.
    with pytest.raises(SystemExit) as conflict:
        select_main(
            [
                str(input_path),
                "--output-json",
                str(tmp_path / "conflict.json"),
                "--include-recording-edges",
                "--exclude-recording-edges",
            ]
        )
    assert conflict.value.code == 2


def test_selector_rejects_non_wemm_document_without_invoking_other_routes() -> None:
    report = select_production_wemm_ambiguities(
        {"format": "robata-production-qwen-native-v1", "windows": []}
    )
    assert report["summary"]["input_window_count"] == 0  # type: ignore[index]
    assert report["summary"]["selected_window_count"] == 0  # type: ignore[index]
    assert report["input_artifacts"]["rejected_inputs"][0]["reason"] == (  # type: ignore[index]
        "UNSUPPORTED_OR_NON_WEMM_FORMAT"
    )
    assert report["controls"]["qwen_read"] is False  # type: ignore[index]


def test_empty_and_bad_paths_are_visible_without_a_model_call(tmp_path: Path) -> None:
    missing = select_production_wemm_ambiguities(tmp_path / "does-not-exist")
    assert missing["summary"]["input_window_count"] == 0  # type: ignore[index]
    assert missing["input_artifacts"]["rejected_inputs"][0]["reason"] == (  # type: ignore[index]
        "PATH_NOT_FOUND"
    )
    malformed = tmp_path / "preannotations"
    malformed.mkdir()
    (malformed / "broken.json").write_text("{not-json", encoding="utf-8")
    report = select_production_wemm_ambiguities(malformed)
    assert report["summary"]["input_window_count"] == 0  # type: ignore[index]
    assert report["input_artifacts"]["invalid_inputs"][0]["reason"].startswith(  # type: ignore[index]
        "could not read JSON"
    )
    assert report["controls"]["media_decoded"] is False  # type: ignore[index]


def test_duplicate_window_keys_are_reported_and_max_selected_is_deterministic() -> None:
    envelope = _envelope()
    duplicate = json.loads(json.dumps(envelope))
    duplicate["source"]["recording_id"] = "recording-selector-01"  # type: ignore[index]
    report = select_production_wemm_ambiguities(
        [envelope, duplicate],
        include_recording_edges=False,
        include_adjacent_transitions=False,
        max_selected=1,
    )
    assert report["summary"]["selection_truncated_count"] == 3  # type: ignore[index]
    assert report["summary"]["duplicate_selection_keys"]  # type: ignore[index]
    assert len(report["windows"]) == 1  # type: ignore[index]
    assert report["summary"]["windows_are_action_segments"] is False  # type: ignore[index]


def test_mixed_source_contracts_are_not_silently_merged() -> None:
    first = _envelope()
    second = json.loads(json.dumps(first))
    second["review_contract"]["decision_options"] = ["accept", "reject"]  # type: ignore[index]
    report = select_production_wemm_ambiguities(
        [first, second],
        include_recording_edges=False,
        include_adjacent_transitions=False,
    )
    assert report["review_contract"] is None
    assert len(report["source_contracts"]) == 2  # type: ignore[arg-type]
    assert report["source_contracts"][0] != report["source_contracts"][1]  # type: ignore[index]
