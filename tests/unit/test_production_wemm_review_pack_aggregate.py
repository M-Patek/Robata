from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robata.benchmark.production_wemm_preannotation import (
    build_preannotation_envelope,
    build_review_pack,
)
from robata.benchmark.production_wemm_review_pack_aggregate import (
    AGGREGATE_FORMAT,
    ProductionWemmReviewPackAggregateError,
    aggregate_production_wemm_review_packs,
    render_markdown,
)
from scripts.aggregate_production_wemm_review_packs import main as aggregate_review_packs_main


def _review_pack(recording_id: str, *, windows: int = 1) -> dict[str, object]:
    source = {
        "recording_id": recording_id,
        "path": f"{recording_id}.mcap",
        "archive_member": f"file/{recording_id}.mcap",
        "archive_path": "source.zip",
        "camera_count": 6,
        "source_preflight_status": "PASS",
        "qa_status": "PENDING",
        "manifest_format": "robata-production-shaped-cohort-v1",
    }
    rows: list[dict[str, object]] = []
    for ordinal in range(windows):
        rows.append(
            {
                "window_id": f"{recording_id}-w{ordinal:04d}",
                "ordinal": ordinal,
                "start_seconds": ordinal * 8.0,
                "end_seconds": (ordinal + 1) * 8.0,
                "camera_ids": [f"cam_{index:02d}" for index in range(1, 7)],
                "proposals": [
                    {
                        "proposal_id": f"p-{ordinal}",
                        "label_text": "pick up garment",
                        "structured_labels": {"verb": "pick up", "noun": "garment"},
                        "confidence": 0.8,
                        "top_k": [
                            {"label_text": "pick up garment", "score": 0.8},
                            {"label_text": "fold garment", "score": 0.7},
                        ],
                    }
                ],
            }
        )
    envelope = build_preannotation_envelope(
        source,
        rows,
        model={"name": "WeMM-Embedding-2B", "route": "video_embedding"},
        raw_model_output={
            "catalog": {
                "format": "robata-production-open-phrase-catalog-v1",
                "phrase_count": 6,
                "epic_ontology_used": False,
                "mapper_used": False,
                "provisional": True,
            }
        },
    )
    return build_review_pack(envelope)


def test_aggregate_flattens_items_and_preserves_lineage_and_retrieval() -> None:
    report = aggregate_production_wemm_review_packs(
        [_review_pack("rec-a", windows=2), _review_pack("rec-b")]
    )

    assert report["format"] == AGGREGATE_FORMAT
    assert report["status"] == "PENDING_REVIEW"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["production_eligible"] is False
    assert report["summary"]["recording_count"] == 2  # type: ignore[index]
    assert report["summary"]["window_count"] == 3  # type: ignore[index]
    assert report["summary"]["proposal_count"] == 3  # type: ignore[index]
    assert report["summary"]["camera_window_input_count"] == 18  # type: ignore[index]
    item = report["items"][0]  # type: ignore[index]
    assert item["source_ref"]["recording_id"] == "rec-a"  # type: ignore[index]
    assert item["source_ref"]["archive_member"] == "file/rec-a.mcap"  # type: ignore[index]
    assert item["decision_options"] == [
        "accept",
        "edit",
        "split",
        "reject",
        "abstain",
    ]
    proposal = item["proposals"][0]  # type: ignore[index]
    assert proposal["top_k"][0]["label_text"] == "pick up garment"  # type: ignore[index]
    assert proposal["margin"] == pytest.approx(0.1)  # type: ignore[index]
    assert item["window_status"] == "PROPOSALS_AVAILABLE"
    assert item["window_decision"] == "pending"
    assert report["summary"]["window_status_counts"] == {  # type: ignore[index]
        "PROPOSALS_AVAILABLE": 3
    }
    assert report["summary"]["window_decision_counts"] == {  # type: ignore[index]
        "pending": 3
    }
    assert report["controls"]["gold_read"] is False  # type: ignore[index]
    assert report["controls"]["qwen_read"] is False  # type: ignore[index]
    assert report["provenance"]["model_names"] == {"WeMM-Embedding-2B": 2}  # type: ignore[index]
    assert report["provenance"]["catalog_formats"] == {  # type: ignore[index]
        "robata-production-open-phrase-catalog-v1": 2
    }
    assert report["review_contract"]["fixed_windows_are_action_boundaries"] is False  # type: ignore[index]
    assert {
        "confidence",
        "evidence",
        "camera_support",
        "top_k",
        "margin",
    }.issubset(report["review_contract"]["required_fields"])  # type: ignore[index]
    markdown = render_markdown(report)
    assert "not action spans" in markdown
    assert "NOT_MEASURED" in markdown


def test_aggregate_preserves_model_temporal_sidecar_with_recording_lineage() -> None:
    pack = _review_pack("rec-temporal")
    temporal = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "status": "PROPOSALS_ONLY",
        "mode": "dense_score",
        "production_eligible": False,
        "parameters": {
            "start_threshold": 0.72,
            "stop_threshold": 0.60,
            "boundary_mode": "midpoint",
        },
        "diagnostics": {"context_window_count": 4},
        "segments": [
            {
                "segment_id": "open-cupboard@1.0-2.0",
                "provisional_id": "open-cupboard",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "boundary_status": "MODEL_PROBE_BOUND",
                "review_required": True,
                "automatic_eligible": False,
                "supporting_window_ids": ["rec-temporal-w0000"],
                "top_k": [
                    {
                        "window_id": "rec-temporal-w0000",
                        "candidates": [{"provisional_id": "open-cupboard", "score": 0.8}],
                    }
                ],
            }
        ],
    }
    pack["temporal_resolution"] = temporal
    pack["temporal_segments"] = list(temporal["segments"])  # type: ignore[index]

    report = aggregate_production_wemm_review_packs([pack])
    assert report["summary"]["temporal_sidecar_recording_count"] == 1  # type: ignore[index]
    assert report["summary"]["temporal_segment_count"] == 1  # type: ignore[index]
    assert report["summary"]["temporal_boundary_status_counts"] == {  # type: ignore[index]
        "MODEL_PROBE_BOUND": 1
    }
    assert report["review_contract"]["temporal_segments_review_only"] is True  # type: ignore[index]
    assert report["review_contract"]["temporal_segments_are_action_boundary_proposals"] is True  # type: ignore[index]

    segment = report["temporal_segments"][0]  # type: ignore[index]
    assert segment["recording_id"] == "rec-temporal"
    assert segment["temporal_segment_key"].startswith("rec-temporal::temporal::")
    assert segment["source_ref"]["archive_member"] == "file/rec-temporal.mcap"
    assert segment["provenance"]["model_boundary_sidecar"] is True
    temporal_summary = report["temporal_resolution"]  # type: ignore[index]
    assert temporal_summary["recordings"][0]["mode"] == "dense_score"
    assert temporal_summary["recordings"][0]["segment_count"] == 1
    markdown = render_markdown(report)
    assert "Temporal interval sidecar" in markdown
    assert "Model-derived interval proposals" in markdown


def test_aggregate_rejects_inconsistent_temporal_sidecar_alias() -> None:
    pack = _review_pack("rec-temporal-mismatch")
    pack["temporal_resolution"] = {
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "segments": [
            {
                "segment_id": "one",
                "review_required": True,
                "automatic_eligible": False,
                "boundary_status": "MODEL_PROBE_BOUND",
            }
        ],
    }
    pack["temporal_segments"] = [
        {
            "segment_id": "two",
            "review_required": True,
            "automatic_eligible": False,
            "boundary_status": "MODEL_PROBE_BOUND",
        }
    ]
    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    assert "does not match temporal_resolution.segments" in report["invalid_inputs"][0]["reason"]  # type: ignore[index]


def test_aggregate_rejects_automatic_temporal_segment() -> None:
    pack = _review_pack("rec-temporal-automatic")
    pack["temporal_segments"] = [
        {
            "segment_id": "one",
            "review_required": False,
            "automatic_eligible": True,
        }
    ]
    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    assert "review_required must be true" in report["invalid_inputs"][0]["reason"]  # type: ignore[index]


def test_directory_discovers_only_review_packs_and_reports_bad_inputs() -> None:
    with tempfile.TemporaryDirectory(prefix="robata-review-pack-") as temp_dir:
        root = Path(temp_dir) / "run"
        review = root / "review"
        pre = root / "preannotations"
        review.mkdir(parents=True)
        pre.mkdir(parents=True)
        (review / "good.json").write_text(json.dumps(_review_pack("rec-dir")), encoding="utf-8")
        (review / "bad.json").write_text("{not-json", encoding="utf-8")
        (review / "checkpoint.json").write_text(
            json.dumps({"format": "robata-production-wemm-batch-run-v1"}), encoding="utf-8"
        )
        # This is intentionally a preannotation-shaped file and must not be merged.
        (pre / "pre.json").write_text(
            json.dumps({"format": "robata-production-wemm-preannotation-v1"}),
            encoding="utf-8",
        )
        report = aggregate_production_wemm_review_packs(root)
        assert report["summary"]["recording_count"] == 1  # type: ignore[index]
        assert report["summary"]["window_count"] == 1  # type: ignore[index]
        assert report["provenance"]["input_pack_count"] == 1  # type: ignore[index]
        reasons = " ".join(  # type: ignore[index]
            row["reason"] for row in report["invalid_inputs"]
        )
        assert "could not read JSON" in reasons
        rejected = " ".join(  # type: ignore[index]
            row["reason"] for row in report["rejected_inputs"]
        )
        assert "ROOT_NOT_REVIEW_PACK" in rejected


def test_duplicate_recording_and_item_keys_are_visible() -> None:
    report = aggregate_production_wemm_review_packs([_review_pack("same"), _review_pack("same")])
    assert report["duplicates"]["recording_ids"] == ["same"]  # type: ignore[index]
    assert report["duplicates"]["item_keys"] == ["same::same-w0000"]  # type: ignore[index]
    assert report["summary"]["window_count"] == 2  # type: ignore[index]


def test_gold_or_epic_flags_are_rejected() -> None:
    pack = _review_pack("rec")
    pack["gold"] = {"status": "ACCEPTED"}
    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    assert "gold/review data" in report["invalid_inputs"][0]["reason"]  # type: ignore[index]

    epic = _review_pack("epic")
    epic["label_space"] = {"epic_ontology_used": True, "mapper_used": False}
    report = aggregate_production_wemm_review_packs([epic])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    assert "EPIC ontology" in report["invalid_inputs"][0]["reason"]  # type: ignore[index]


def test_invalid_camera_count_is_rejected() -> None:
    with pytest.raises(ProductionWemmReviewPackAggregateError, match="positive integer"):
        aggregate_production_wemm_review_packs([], expected_camera_count=0)


def test_multiple_roots_and_checkpoint_merge_preserve_context_and_provenance(
    tmp_path: Path,
) -> None:
    """B1 + remaining can be joined without reading non-review artifacts."""

    b1_root = tmp_path / "b1"
    remaining_root = tmp_path / "remaining"
    (b1_root / "review").mkdir(parents=True)
    (remaining_root / "review").mkdir(parents=True)
    b1_pack_path = b1_root / "review" / "b1.json"
    remaining_pack_path = remaining_root / "review" / "remaining.json"
    b1_pack_path.write_text(json.dumps(_review_pack("b1")), encoding="utf-8")
    remaining_pack_path.write_text(json.dumps(_review_pack("remaining")), encoding="utf-8")

    # The checkpoint is intentionally incomplete: only COMPLETE items may be
    # followed, so a running/planned item cannot contribute a partially written
    # review file to the unified queue.
    checkpoint_path = remaining_root / "batch-run.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "format": "robata-production-wemm-batch-run-v1",
                "status": "RUNNING",
                "summary": {"complete_count": 1, "running_count": 1},
                "items": [
                    {
                        "status": "COMPLETE",
                        "review_path": "review/remaining.json",
                    },
                    {"status": "RUNNING", "review_path": None},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = aggregate_production_wemm_review_packs([b1_root, checkpoint_path])
    assert report["summary"]["recording_count"] == 2  # type: ignore[index]
    assert report["summary"]["window_count"] == 2  # type: ignore[index]
    assert report["summary"]["checkpoint_count"] == 1  # type: ignore[index]
    assert report["summary"]["unresolved_checkpoint_item_count"] == 1  # type: ignore[index]
    assert report["summary"]["all_checkpoint_items_complete"] is False  # type: ignore[index]
    assert report["provenance"]["input_pack_count"] == 2  # type: ignore[index]
    assert report["provenance"]["partial_input"] is True  # type: ignore[index]
    sources = report["provenance"]["input_sources"]  # type: ignore[index]
    assert any(source["kind"] == "batch_checkpoint" for source in sources)
    assert any(source["path"].endswith("b1") for source in sources)
    assert any(
        "CHECKPOINT_ITEM_NOT_COMPLETE:RUNNING" in row["reason"]
        for row in report["rejected_inputs"]  # type: ignore[index]
    )

    item = report["items"][0]  # type: ignore[index]
    assert item["source_interval"]["status"] == "WINDOW_CONTEXT_ONLY"  # type: ignore[index]
    proposal = item["proposals"][0]  # type: ignore[index]
    assert proposal["margin"] == pytest.approx(0.1)  # type: ignore[index]
    assert proposal["top_k"][0]["label_text"] == "pick up garment"  # type: ignore[index]
    assert proposal["raw"]["top_k"][1]["label_text"] == "fold garment"  # type: ignore[index]
    assert item["provenance"]["window_context_only"] is True  # type: ignore[index]

    # Exercise the CLI's variadic input surface as well as the library API.
    output_path = tmp_path / "unified.json"
    assert (
        aggregate_review_packs_main(
            [str(b1_root), str(checkpoint_path), "--json-output", str(output_path)]
        )
        == 0
    )
    cli_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert cli_report["summary"]["recording_count"] == 2
