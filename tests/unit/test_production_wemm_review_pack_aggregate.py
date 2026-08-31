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


def _coarse_temporal_segment(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "segment_id": "temporal-segment",
        "provisional_id": "open-cupboard",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "boundary_status": "MODEL_PROBE_BOUND",
        "boundary_source": "wemm_temporal_score",
        "boundary_method": "probe_center_midpoint",
        "context_only": True,
        "window_context_only": True,
        "is_action_boundary": False,
        "action_boundary": False,
        "review_required": True,
        "automatic_eligible": False,
    }
    row.update(overrides)
    return row


def _refined_temporal_segment(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "segment_id": "refined-segment",
        "provisional_id": "open faucet",
        "coarse_interval": {"start_seconds": 1.0, "end_seconds": 3.0},
        "start_seconds": 1.4,
        "end_seconds": 2.6,
        "boundary_status": "MODEL_REFINED",
        "boundary_source": "wemm_short_refinement",
        "boundary_method": "short_probe_model",
        "context_only": True,
        "window_context_only": True,
        "is_action_boundary": False,
        "action_boundary": False,
        "review_required": True,
        "automatic_eligible": False,
    }
    row.update(overrides)
    return row


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


def test_aggregate_preserves_item_lineage_extensions() -> None:
    pack = _review_pack("rec-item-lineage")
    item = pack["items"][0]  # type: ignore[index]
    item["source_ref"] = {
        "recording_id": "stale-recording",
        "review_pack_path": "stale-pack.json",
        "custom_locator": "camera-grid-7",
    }
    item["provenance"] = {
        "review_pack_path": "stale-pack.json",
        "custom_stage": "upstream-review",
    }

    report = aggregate_production_wemm_review_packs([pack])
    flattened = report["items"][0]  # type: ignore[index]
    assert flattened["source_ref"]["recording_id"] == "rec-item-lineage"
    assert flattened["source_ref"]["review_pack_path"] == "<sequence>[0]"
    assert flattened["source_ref"]["custom_locator"] == "camera-grid-7"
    assert flattened["source_ref"]["upstream_lineage"] == {
        "recording_id": "stale-recording",
        "review_pack_path": "stale-pack.json",
    }
    assert flattened["provenance"]["review_pack_path"] == "<sequence>[0]"
    assert flattened["provenance"]["custom_stage"] == "upstream-review"
    assert flattened["provenance"]["upstream_lineage"] == {
        "review_pack_path": "stale-pack.json",
    }


def test_aggregate_preserves_model_temporal_sidecar_with_recording_lineage() -> None:
    pack = _review_pack("rec-temporal")
    temporal = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "mode": "dense_score",
        "production_eligible": False,
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 8.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "parameters": {
            "start_threshold": 0.72,
            "stop_threshold": 0.60,
            "boundary_mode": "midpoint",
        },
        "diagnostics": {"context_window_count": 4},
        "segments": [
            _coarse_temporal_segment(
                segment_id="open-cupboard@1.0-2.0",
                supporting_window_ids=["rec-temporal-w0000"],
                top_k=[
                    {
                        "window_id": "rec-temporal-w0000",
                        "candidates": [{"provisional_id": "open-cupboard", "score": 0.8}],
                    }
                ],
            )
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


def test_aggregate_canonical_lineage_overrides_stale_temporal_source_ref() -> None:
    pack = _review_pack("rec-lineage")
    segment = _coarse_temporal_segment(
        source_ref={
            "recording_id": "stale-recording",
            "review_pack_path": "stale-pack.json",
            "archive_member": "stale-member.mcap",
            "upstream_note": "retain me",
        }
    )
    temporal = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [segment],
    }
    pack["temporal_resolution"] = temporal
    pack["temporal_segments"] = [segment]

    report = aggregate_production_wemm_review_packs([pack])
    projected = report["temporal_segments"][0]  # type: ignore[index]
    source_ref = projected["source_ref"]
    assert source_ref["recording_id"] == "rec-lineage"
    assert source_ref["review_pack_path"] == "<sequence>[0]"
    assert source_ref["archive_member"] == "file/rec-lineage.mcap"
    assert source_ref["upstream_lineage"] == {
        "recording_id": "stale-recording",
        "review_pack_path": "stale-pack.json",
        "archive_member": "stale-member.mcap",
    }
    assert source_ref["upstream_note"] == "retain me"


def test_aggregate_normalizes_legacy_temporal_marker_omissions() -> None:
    """Old sidecars remain reviewable while new output has explicit markers."""

    pack = _review_pack("rec-legacy-temporal")
    segment = _coarse_temporal_segment(segment_id="legacy")
    for key in ("context_only", "window_context_only", "is_action_boundary", "action_boundary"):
        segment.pop(key)
    temporal = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "context_interval": {"start_seconds": 0.0, "end_seconds": 4.0, "context_only": True},
        "segments": [segment],
    }
    pack["temporal_resolution"] = temporal
    pack["temporal_segments"] = [segment]

    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "PENDING_REVIEW"
    projected = report["temporal_segments"][0]  # type: ignore[index]
    assert projected["context_only"] is True
    assert projected["window_context_only"] is True
    assert projected["is_action_boundary"] is False
    assert projected["action_boundary"] is False
    assert (
        report["temporal_resolution"]["recordings"][0]["sidecar"][  # type: ignore[index]
            "context_interval"
        ]["action_boundary"]
        is False
    )


def test_aggregate_rejects_inconsistent_temporal_sidecar_alias() -> None:
    pack = _review_pack("rec-temporal-mismatch")
    pack["temporal_resolution"] = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "production_eligible": False,
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 4.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [_coarse_temporal_segment(segment_id="one")],
    }
    pack["temporal_segments"] = [_coarse_temporal_segment(segment_id="two")]
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update({"review_required": False}),
        lambda item: item.update({"automatic_eligible": True}),
    ],
)
def test_aggregate_rejects_non_review_item_flags(mutate) -> None:
    pack = _review_pack("rec-item-flags")
    mutate(pack["items"][0])  # type: ignore[index]
    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    reason = report["invalid_inputs"][0]["reason"]  # type: ignore[index]
    assert "must be true" in reason or "must be false" in reason


@pytest.mark.parametrize(
    ("sidecar_key", "segment"),
    [
        (
            "temporal_resolution",
            {
                "status": "PROPOSALS_ONLY",
                "production_eligible": False,
                "segments": [
                    {
                        "segment_id": "missing-review-flag",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "automatic_eligible": False,
                        "boundary_status": "MODEL_PROBE_BOUND",
                    }
                ],
            },
        ),
        (
            "temporal_segments",
            [
                {
                    "segment_id": "bad-interval",
                    "start_seconds": 2.0,
                    "end_seconds": 2.0,
                    "review_required": True,
                    "automatic_eligible": False,
                    "boundary_status": "MODEL_PROBE_BOUND",
                }
            ],
        ),
    ],
)
def test_aggregate_rejects_malformed_temporal_segment_contract(
    sidecar_key: str, segment: object
) -> None:
    pack = _review_pack("rec-temporal-malformed")
    pack[sidecar_key] = segment
    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    reason = report["invalid_inputs"][0]["reason"]  # type: ignore[index]
    assert any(
        marker in reason
        for marker in (
            "format must be",
            "review_required must be true",
            "context_only must be true",
            "interval must satisfy",
        )
    )


def test_aggregate_requires_explicit_nonproduction_temporal_resolution() -> None:
    pack = _review_pack("rec-temporal-production-flag")
    pack["temporal_resolution"] = {
        "format": "robata-production-wemm-temporal-resolver-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "PROPOSALS_ONLY",
        "context_interval": {
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "context_only": True,
            "is_action_boundary": False,
            "action_boundary": False,
        },
        "segments": [],
    }
    report = aggregate_production_wemm_review_packs([pack])
    assert report["status"] == "NO_VALID_REVIEW_PACKS"
    assert "production_eligible must be false" in report["invalid_inputs"][0]["reason"]  # type: ignore[index]


def test_aggregate_preserves_adaptive_sidecars_and_separates_measured_rows() -> None:
    pack = _review_pack("rec-adaptive")
    measured = _refined_temporal_segment(segment_id="seg-measured")
    pending = _refined_temporal_segment(
        segment_id="seg-pending",
        provisional_id="close faucet",
        coarse_interval={"start_seconds": 4.0, "end_seconds": 6.0},
        start_seconds=None,
        end_seconds=None,
        boundary_status="MODEL_REFINEMENT_PENDING",
    )
    rows = [measured, pending]
    pack.update(
        {
            "temporal_refinement_plan": {
                "format": "robata-production-wemm-temporal-refinement-plan-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "requests": [],
            },
            "temporal_refinement_fine_plan": {
                "format": "robata-production-wemm-temporal-score-refinement-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "requests": [],
            },
            "temporal_refinement_score_resolution": {
                "format": "robata-production-wemm-temporal-score-result-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "results": [],
            },
            "temporal_refinement": {
                "format": "robata-production-wemm-temporal-refinement-review-v1",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
                "production_eligible": False,
                "refined_segments": rows,
            },
            "refined_segments": rows,
            "refined_temporal_segments": rows,
        }
    )
    report = aggregate_production_wemm_review_packs([pack])
    assert report["summary"]["temporal_refinement_recording_count"] == 1  # type: ignore[index]
    assert report["summary"]["refined_temporal_segment_count"] == 2  # type: ignore[index]
    assert report["summary"]["refined_temporal_interval_proposal_count"] == 1  # type: ignore[index]
    assert report["summary"]["refined_temporal_boundary_status_counts"] == {  # type: ignore[index]
        "MODEL_REFINED": 1,
        "MODEL_REFINEMENT_PENDING": 1,
    }
    assert report["temporal_refinement"]["recordings"][0]["fine_plan"]["requests"] == []  # type: ignore[index]
    assert report["temporal_refinement_segments"][1]["start_seconds"] is None  # type: ignore[index]
    refined_proposal = report["refined_temporal_interval_proposals"][0]  # type: ignore[index]
    assert refined_proposal["recording_id"] == "rec-adaptive"
    assert refined_proposal["provenance"]["model_refinement_sidecar"] is True


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
